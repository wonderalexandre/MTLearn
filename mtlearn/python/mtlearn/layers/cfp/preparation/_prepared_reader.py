import json
import operator

from ._identity import validate_identity, tree_config
from .prepared_dataset import PreparedDataset, collate_prepared
from ._loader_memory import CheckedSource, tensor_buffers

CONTRACT_FIELDS = ("config", "source_version", "preprocessing_version", "split", "sample_count")


def contract(store, manifest):
    row = store._manifest_record(manifest)
    return tuple(row[field] for field in CONTRACT_FIELDS)


def describe(store, dataset, indices, source_memory_bytes, workspace_bytes, max_batch_bytes, mmap):
    generation = store._read_generation()
    if contract(store, dataset.manifest) != dataset._manifest_contract:
        raise ValueError("Manifest contract changed; reopen the prepared loader.")
    rows, ids, limits, shapes = [], [], [], []
    payload_bytes = 0
    for index in indices:
        index = operator.index(index)
        if index < 0:
            index += len(dataset)
        if not 0 <= index < len(dataset):
            raise IndexError(f"Manifest position {index} is outside the dataset.")
        row = store._db.execute("SELECT * FROM samples WHERE manifest=? AND position=?",
                                (dataset.manifest, index)).fetchone()
        if row is None:
            raise ValueError(f"Manifest sample at position {index} is missing.")
        shape, bindings = json.loads(row["shape"]), json.loads(row["bindings"])
        if (not isinstance(shape, list) or len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape)
                or not isinstance(bindings, list) or len(bindings) != shape[0]
                or any(not isinstance(trees, dict) or set(trees) != set(dataset.preprocessor.tree_specs) for trees in bindings)):
            raise ValueError(f"Invalid prepared metadata for sample {row['sample_id']!r}.")
        if shapes and shape != shapes[0]:
            raise ValueError("A prepared batch requires matching channel/spatial shapes; use batch_size=1 for variable resolution.")
        shapes.append(shape)
        for trees in bindings:
            for tree_key, key in trees.items():
                entry = store._db.execute("SELECT * FROM entries WHERE key=?", (key,)).fetchone()
                if entry is None or entry["state"] != "valid":
                    raise ValueError(f"Sample {row['sample_id']!r}: incomplete entry {key}; resume preparation.")
                identity, summary = json.loads(entry["identity"]), json.loads(entry["summary"])
                prep = dataset.preprocessor
                if (validate_identity(identity) != key or identity["shape"] != shape[1:]
                        or identity["tree"] != tree_config(prep.tree_specs[tree_key])
                        or identity["attributes"] != [a.name for a in prep.features[tree_key].attributes]
                        or identity["attribute_dtype"] != prep.attribute_dtype.name
                        or set(summary) != set(identity["attributes"])):
                    raise ValueError(f"Prepared identity/summary does not match sample {row['sample_id']!r}.")
                counts = {value["count"] for value in summary.values()}
                if len(counts) != 1 or any(type(n) is not int or n <= 0 for n in counts):
                    raise ValueError("Prepared node counts are inconsistent.")
                logical = 4 * shape[1] * shape[2] + next(iter(counts)) * (28 + len(summary) * prep.attribute_dtype.itemsize)
                size = entry["size_bytes"]
                if type(size) is not int or size < logical or store._entry_path(key).stat().st_size != size:
                    raise ValueError(f"Prepared size metadata is invalid for sample {row['sample_id']!r}, entry {key}.")
                payload_bytes += size
        limit = source_memory_bytes(index) if callable(source_memory_bytes) else source_memory_bytes
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("source_memory_bytes must return a nonnegative integer per source sample.")
        limits.append(limit)
        ids.append(row["sample_id"])
        rows.append((index, tuple(row)))
    reserved = (payload_bytes * (1 if mmap else 2) + 2 * sum(n or 0 for n in limits) + workspace_bytes)
    if max_batch_bytes is not None and reserved > max_batch_bytes:
        raise ValueError(f"Samples {tuple(ids)!r} need {reserved} reserved bytes, exceeding max_batch_bytes={max_batch_bytes}.")
    if store._read_generation() != generation:
        raise ValueError("Manifest changed during batch planning; reopen the loader.")
    return {"indices": tuple(row[0] for row in rows), "rows": tuple(rows), "sample_ids": tuple(ids),
            "source_limits": tuple(limits), "reserved_bytes": reserved, "payload_bytes": payload_bytes,
            "workspace_bytes": workspace_bytes}


def open_dataset(store, manifest, source, expected_contract):
    dataset = PreparedDataset(store, manifest, None if source is None else CheckedSource(source))
    if dataset._manifest_contract != expected_contract:
        raise ValueError("Manifest contract changed before reader startup; reopen the loader.")
    return dataset


def load_batch(dataset, description):
    items = []
    for (index, expected), limit in zip(description["rows"], description["source_limits"]):
        current = dataset.store._db.execute("SELECT * FROM samples WHERE manifest=? AND position=?",
                                            (dataset.manifest, index)).fetchone()
        if current is None or tuple(current) != expected:
            raise ValueError(f"Manifest sample {index} changed after scheduling; reopen the loader.")
        if dataset.source is not None:
            dataset.source.limit = limit
        items.append(dataset[index])
    result = collate_prepared(items)
    batch = result[0] if isinstance(result, tuple) else result
    if batch.sample_ids != description["sample_ids"]:
        raise ValueError("Prepared sample IDs changed during loading.")
    if sum(tensor_buffers(batch).values()) > description["payload_bytes"]:
        raise ValueError("Prepared tensor storages exceed their verified file-size bounds.")
    if all(n is not None for n in description["source_limits"]):
        targets = result[1] if isinstance(result, tuple) else None
        if sum(tensor_buffers(targets).values()) > 2 * sum(description["source_limits"]):
            raise ValueError("Collated targets exceed the declared source-memory bounds.")
    return result
