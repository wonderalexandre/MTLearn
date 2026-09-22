import threading
import weakref

import numpy as np
import torch

from .prepared_batch import PreparedBatch


def tensor_buffers(value):
    result = {}
    if isinstance(value, PreparedBatch):
        for sample in value.samples:
            for channel in sample:
                for prepared in channel.values():
                    result.update(tensor_buffers(tuple(prepared.info.values())))
                    result.update(tensor_buffers(tuple(prepared.raw_attributes.values())))
    elif torch.is_tensor(value):
        if value.device.type != "cpu" or value.requires_grad or value.layout != torch.strided:
            raise ValueError("Prepared loading requires detached dense CPU tensors, including source and targets.")
        storage = value.untyped_storage()
        result[("torch", storage.data_ptr())] = storage.nbytes()
    elif isinstance(value, np.ndarray):
        root = value
        while isinstance(root.base, np.ndarray):
            root = root.base
        result[("numpy", root.__array_interface__["data"][0])] = root.nbytes
    elif isinstance(value, dict):
        for item in value.values():
            result.update(tensor_buffers(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            result.update(tensor_buffers(item))
    return result


class ActiveBuffers:
    def __init__(self):
        self.lock = threading.RLock()
        self.owners = {}

    def track(self, value):
        if isinstance(value, PreparedBatch) or torch.is_tensor(value) or isinstance(value, np.ndarray):
            key = id(value)
            buffers = tensor_buffers(value)
            def release(reference):
                with self.lock:
                    if key in self.owners and self.owners[key][0] is reference:
                        del self.owners[key]
            with self.lock:
                self.owners[key] = weakref.ref(value, release), buffers
        elif isinstance(value, dict):
            for item in value.values():
                self.track(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.track(item)

    def buffers(self):
        with self.lock:
            result = {}
            for reference, buffers in self.owners.values():
                result.update(buffers)
            return result


class CheckedSource:
    def __init__(self, source):
        self.source = source
        self.limit = None

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        value = self.source[index]
        size = sum(tensor_buffers(value).values())
        if self.limit is not None and size > self.limit:
            raise ValueError(f"Source sample {index} has {size} tensor bytes, exceeding source_memory_bytes={self.limit}.")
        return value
