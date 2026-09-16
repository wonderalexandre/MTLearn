import math
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch

from .prepared_batch import PreparedBatch
from .prepared_morphology import PreparedMorphology
from ._loader_memory import tensor_buffers


class _Array(np.ndarray):
    def __array_finalize__(self, other):
        self.lease = getattr(other, "lease", None)


class SharedBatchLease:
    def __init__(self, size):
        self.memory = SharedMemory(create=True, size=max(1, size))
        self.size = size
        self.unlinked = False

    def unlink(self):
        if not self.unlinked:
            self.memory.unlink()
            self.unlinked = True

    def close(self):
        self.unlink()
        self.memory.close()

    def __del__(self):
        if hasattr(self, "memory"):
            self.close()


def write_shared_batch(value, memory):
    tensor_buffers(value)
    locations, sizes = {}, []
    cursor = 0

    def encode(item):
        nonlocal cursor
        if torch.is_tensor(item):
            storage = item.untyped_storage()
            key = storage.data_ptr(), storage.nbytes()
            if key not in locations:
                start = (cursor + 63) // 64 * 64
                end = start + storage.nbytes()
                if end > memory.size:
                    raise ValueError("Shared batch exceeds its reserved transport capacity.")
                raw = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
                memory.buf[start:end] = memoryview(raw.numpy())
                locations[key] = len(sizes)
                sizes.append((start, storage.nbytes()))
                cursor = end
            return ("tensor", locations[key], str(item.dtype).removeprefix("torch."),
                    tuple(item.shape), tuple(item.stride()), item.storage_offset())
        if isinstance(item, PreparedBatch):
            samples = []
            for sample in item.samples:
                channels = []
                for channel in sample:
                    channels.append({key: (encode(dict(prepared.info)),
                        [(attribute.name, encode(tensor)) for attribute, tensor in prepared.raw_attributes.items()],
                        prepared.input_id, prepared.format_version) for key, prepared in channel.items()})
                samples.append(channels)
            return ("batch", item.sample_ids, samples)
        if isinstance(item, dict):
            if any(type(key) not in (str, int) for key in item):
                raise TypeError("Shared target mappings require string or integer keys.")
            return ("dict", [(key, encode(value)) for key, value in item.items()])
        if isinstance(item, (tuple, list)):
            return ("tuple" if isinstance(item, tuple) else "list", [encode(value) for value in item])
        if item is None or type(item) in (bool, int, float, str):
            return ("scalar", item)
        if hasattr(item, "name") and type(item).__module__.startswith("mtlearn"):
            return ("enum", item.name)
        raise TypeError(f"Unsupported shared target or metadata type: {type(item).__name__}.")

    try:
        content = encode(value)
    finally:
        encode = None
    return {"content": content, "storages": sizes, "used_bytes": cursor}


def read_shared_batch(packet, lease, preprocessor, sample_ids):
    if type(packet) is not dict or set(packet) != {"content", "storages", "used_bytes"}:
        raise ValueError("Invalid shared batch descriptor.")
    used = packet["used_bytes"]
    if type(used) is not int or not 0 <= used <= lease.size:
        raise ValueError("Invalid shared batch size.")
    buffers = []
    previous_end = 0
    for start, size in packet["storages"]:
        if (type(start) is not int or type(size) is not int or start < previous_end or size < 0
                or start % 64 or start + size > used):
            raise ValueError("Invalid shared storage bounds.")
        if size:
            array = np.ndarray((size,), dtype=np.uint8, buffer=lease.memory.buf, offset=start).view(_Array)
            array.lease = lease
            buffers.append(torch.from_numpy(array))
        else:
            buffers.append(torch.empty(0, dtype=torch.uint8))
        previous_end = start + size

    def decode(item, tree_key=None):
        kind = item[0]
        if kind == "tensor":
            _, index, name, shape, stride, offset = item
            if type(index) is not int or not 0 <= index < len(buffers):
                raise ValueError("Invalid shared storage index.")
            dtype = getattr(torch, name, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError("Invalid shared tensor dtype.")
            if (len(shape) != len(stride) or type(offset) is not int or offset < 0
                    or any(type(n) is not int or n < 0 for n in (*shape, *stride))):
                raise ValueError("Invalid shared tensor layout.")
            base = buffers[index].view(dtype)
            maximum = offset + sum((n - 1) * step for n, step in zip(shape, stride)) + 1
            if math.prod(shape) and maximum > base.numel():
                raise ValueError("Shared tensor view exceeds its storage.")
            return base.as_strided(shape, stride, offset)
        if kind == "batch":
            _, ids, samples = item
            if tuple(ids) != tuple(sample_ids):
                raise ValueError("Shared batch sample IDs do not match the scheduled batch.")
            result = []
            for sample in samples:
                channels = []
                for channel in sample:
                    trees = {}
                    for key, (info, raw, input_id, version) in channel.items():
                        if key not in preprocessor.tree_specs:
                            raise ValueError("Shared batch has an unexpected tree.")
                        attributes = {attribute.name: attribute for attribute in preprocessor.features[key].attributes}
                        trees[key] = PreparedMorphology(preprocessor.tree_specs[key], preprocessor.features[key],
                            decode(info, key), {attributes[name]: decode(tensor) for name, tensor in raw},
                            input_id, version)
                    channels.append(trees)
                result.append(channels)
            return PreparedBatch(tuple(result), tuple(ids))
        if kind == "dict":
            return {key: decode(value, tree_key) for key, value in item[1]}
        if kind in ("tuple", "list"):
            values = [decode(value, tree_key) for value in item[1]]
            return tuple(values) if kind == "tuple" else values
        if kind == "enum" and tree_key is not None:
            tree_type = preprocessor.tree_specs[tree_key].tree_type
            if item[1] != tree_type.name:
                raise ValueError("Shared tree type does not match the preprocessor.")
            return tree_type
        if kind == "scalar" and (item[1] is None or type(item[1]) in (bool, int, float, str)):
            return item[1]
        raise ValueError("Invalid shared batch value descriptor.")

    try:
        return decode(packet["content"])
    finally:
        buffers.clear()
        decode = None
