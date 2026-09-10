"""Preparation coordinator and deterministic manifest reductions."""
import hashlib
import json
from collections.abc import Mapping

import numpy as np
import torch

from ..._helpers import to_numpy_u8
from ..normalization import AttributeNormalizer
from ..normalization.statistics_snapshot import StatisticsSnapshot
from ..runtime.cache_input_contract import validate_cfp_cache_batch_x
from ._identity import canonical_json, image_identity, identity_key, preprocessor_config, preprocessor_from_config
from .preparation_result import PreparationResult


def statistics_contract(preprocessor, options):
    expected_keys = tuple(sorted(f"{key}::{attr.name}" for key, features in preprocessor.features.items()
                                 for attr in features.attributes))
    if options is True:
        options = {}
    if not isinstance(options, Mapping):
        raise TypeError("collect_stats must be a normalization mapping, True, or None.")
    options = dict(options)
    dtype = options.pop("attribute_dtype", preprocessor.attribute_dtype.name)
    keys = tuple(options.pop("stat_keys", expected_keys))
    if dtype != preprocessor.attribute_dtype.name or keys != expected_keys:
        raise ValueError("Statistical contract does not match preparation trees, attributes or dtype.")
    normalizer = AttributeNormalizer(**options)
    return {"scale_mode": normalizer.scale_mode, "eps": normalizer.eps,
            "clipped_zscore_radius": normalizer.clipped_zscore_radius,
            "clipped_zscore_floor": normalizer.clipped_zscore_floor,
            "attribute_dtype": dtype, "stat_keys": keys}


def sample_image(item):
    target = None
    if isinstance(item, (tuple, list)) and len(item) == 2:
        item, target = item
    x = item if torch.is_tensor(item) else torch.as_tensor(np.asarray(item))
    if x.ndim == 2:
        x = x.unsqueeze(0)
    validate_cfp_cache_batch_x(x.unsqueeze(0))
    return x, target


def image_bindings(preprocessor, image):
    bindings = []
    for channel in image:
        canonical = np.ascontiguousarray(to_numpy_u8(channel))
        digest = hashlib.sha256(memoryview(canonical)).hexdigest()
        bindings.append({key: identity_key(image_identity(preprocessor, canonical, key, pixel_digest=digest))
                         for key in preprocessor.tree_specs})
    return bindings


def manifest_record(store, name, *, complete=True):
    store._check()
    row = store._db.execute("SELECT * FROM manifests WHERE name=?", (name,)).fetchone()
    if row is None:
        raise KeyError(f"Unknown prepared manifest {name!r}.")
    if complete and row["state"] != "complete":
        raise RuntimeError(f"Manifest {name!r} is {row['state']}; resume preparation before consumption.")
    return row


def _begin_manifest(store, name, preprocessor, count, source_version, preprocessing_version, split):
    if any(not isinstance(v, str) or not v for v in (name, source_version, preprocessing_version)):
        raise ValueError("Persistent preparation requires manifest, source_version and preprocessing_version strings.")
    config = canonical_json(preprocessor_config(preprocessor))
    values = (config, source_version, preprocessing_version, split, count)
    row = store._db.execute("SELECT * FROM manifests WHERE name=?", (name,)).fetchone()
    if row is not None and tuple(row[k] for k in ("config", "source_version", "preprocessing_version", "split", "sample_count")) != values:
        raise ValueError("Manifest contract/source version changed; use a new manifest name to reuse compatible content safely.")
    with store._db:
        store._db.execute("""INSERT INTO manifests VALUES (?,?,?,?,?,?,'preparing',NULL)
            ON CONFLICT(name) DO UPDATE SET state='preparing',error=NULL""", (name, *values))


def _bind_sample(store, name, position, sample_id, bindings, shape):
    binding_text = canonical_json(bindings)
    with store._db:
        store._db.execute("INSERT OR IGNORE INTO samples VALUES (?,?,?,?,?)",
                          (name, position, sample_id, binding_text, canonical_json(shape)))
        row = store._db.execute("SELECT * FROM samples WHERE manifest=? AND position=?", (name, position)).fetchone()
        if row is None or row["sample_id"] != sample_id or row["bindings"] != binding_text or json.loads(row["shape"]) != list(shape):
            raise ValueError("Logical IDs must be unique and stable across preparation retries.")
        for channel, trees in enumerate(bindings):
            for tree_key, key in trees.items():
                entry = store._db.execute("SELECT state FROM entries WHERE key=?", (key,)).fetchone()
                if entry is None or entry["state"] != "valid":
                    raise RuntimeError("Cannot bind an incomplete preparation to a logical sample.")
                store._db.execute("INSERT OR IGNORE INTO contributions VALUES (?,?,?,?,?)",
                                  (name, position, channel, tree_key, key))


def statistics_from_manifest(store, name, options):
    row = manifest_record(store, name)
    if row["split"] != "train":
        raise ValueError("Statistics can only be fitted from a train manifest.")
    preprocessor = preprocessor_from_config(json.loads(row["config"]))
    contract = statistics_contract(preprocessor, options)
    normalizer = AttributeNormalizer(contract["scale_mode"])
    query = """SELECT c.tree_key,e.summary,e.state FROM contributions c JOIN entries e ON c.entry_key=e.key
               WHERE c.manifest=? ORDER BY c.position,c.channel,c.tree_key"""
    for contribution in store._db.execute(query, (name,)):
        if contribution["state"] != "valid":
            raise RuntimeError("Statistics reference an invalid entry; resume preparation first.")
        if contract["scale_mode"] == "none":
            continue
        mode_fields = ("amin", "amax") if contract["scale_mode"] == "dataset_minmax01" else ("count", "sum", "sumsq")
        for attr, values in json.loads(contribution["summary"]).items():
            dtype = torch.float64 if contract["attribute_dtype"] == "float64" else torch.float32
            summary = {field: torch.tensor(values[field], dtype=(torch.int64 if field == "count" else
                       dtype if field in ("amin", "amax") else torch.float64)) for field in mode_fields}
            normalizer.merge({f"{contribution['tree_key']}::{attr}": summary})
    return StatisticsSnapshot(contract, normalizer.ds_stats, sample_count=row["sample_count"])


def statistics_identity(store, name, contract):
    digest = hashlib.sha256(canonical_json(contract).encode())
    for row in store._db.execute("SELECT sample_id,bindings FROM samples WHERE manifest=? ORDER BY position", (name,)):
        digest.update(canonical_json(list(row)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_source(preprocessor, source, *, store=None, manifest=None, source_version=None,
                   preprocessing_version=None, split="train", sample_ids=None, collect_stats=None, cancel=None,
                   num_workers=0, max_in_flight=None, worker_threads=1, max_sample_bytes=64 * 1024**2, max_retries=0):
    from ..storage import NullStore
    if not hasattr(source, "__len__") or not hasattr(source, "__getitem__"):
        raise TypeError("prepare requires a finite map-style dataset/sequence; use a Subset to select/order samples.")
    count = len(source)
    if count <= 0:
        raise ValueError("prepare requires a nonempty source.")
    if split not in ("train", "evaluation"):
        raise ValueError("split must be 'train' or 'evaluation'.")
    collect = collect_stats is not None and collect_stats is not False
    if collect and split != "train":
        raise ValueError("collect_stats requires split='train'.")
    contract = statistics_contract(preprocessor, collect_stats) if collect else None
    if sample_ids is not None and not callable(sample_ids) and len(sample_ids) != count:
        raise ValueError("sample_ids must have one string per source item.")
    store = NullStore() if store is None else store
    from ._parallel_preparation import validate_parallel_options, prepare_parallel
    max_in_flight = validate_parallel_options(store, num_workers, max_in_flight, worker_threads, max_sample_bytes, max_retries)
    persistent = getattr(store, "persistent", False)
    if persistent:
        store._check(write=True)
        _begin_manifest(store, manifest, preprocessor, count, source_version, preprocessing_version, split)
        # A resumed pass must revalidate durable files, even if this process has
        # already borrowed valid CPU buffers from a now damaged/missing file.
        store.clear()
    normalizer = AttributeNormalizer(contract["scale_mode"]) if collect and not persistent else None
    completed = 0
    try:
        if num_workers:
            completed, cancelled = prepare_parallel(preprocessor, source, store, manifest, sample_ids, cancel,
                num_workers=num_workers, max_in_flight=max_in_flight, worker_threads=worker_threads,
                max_sample_bytes=max_sample_bytes, max_retries=max_retries)
            if cancelled:
                with store._db:
                    store._db.execute("UPDATE manifests SET state='cancelled' WHERE name=?", (manifest,))
                return PreparationResult(str(store.path), manifest, completed, "cancelled")
        for index in range(count) if not num_workers else ():
            if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
                if persistent:
                    with store._db:
                        store._db.execute("UPDATE manifests SET state='cancelled' WHERE name=?", (manifest,))
                return PreparationResult(str(store.path) if persistent else None, manifest, completed, "cancelled")
            sample_id = str(index) if sample_ids is None else (sample_ids(index) if callable(sample_ids) else sample_ids[index])
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("sample_ids must be nonempty strings.")
            item = source[index]
            image, target = sample_image(item)
            if persistent:
                bindings = image_bindings(preprocessor, image)
                old = store._db.execute("SELECT * FROM samples WHERE manifest=? AND position=?", (manifest, index)).fetchone()
                if old is not None and (old["sample_id"] != sample_id or old["bindings"] != canonical_json(bindings)):
                    raise ValueError("Source content/order changed under the same manifest version; use a new manifest name.")
            batch = preprocessor.prepare_batch(image.unsqueeze(0), store=store, sample_ids=(sample_id,))
            if persistent:
                _bind_sample(store, manifest, index, sample_id, bindings, list(image.shape))
            elif collect:
                for channel in batch.samples[0]:
                    for key, prepared in channel.items():
                        for attr, values in prepared.raw_attributes.items():
                            normalizer.merge(normalizer.summarize(f"{key}::{attr.name}", values.view(-1)))
                # Loop locals must not retain the previous preparation.
                del channel, prepared, values
            completed += 1
            del batch, image, target, item
        if persistent:
            with store._db:
                store._db.execute("UPDATE manifests SET state='complete',error=NULL WHERE name=?", (manifest,))
            snapshot = store.statistics(manifest, contract) if collect else None
            fingerprint = statistics_identity(store, manifest, contract) if collect else None
        else:
            snapshot = StatisticsSnapshot(contract, normalizer.ds_stats, sample_count=count) if collect else None
            fingerprint = None
        return PreparationResult(str(store.path) if persistent else None, manifest, count, "complete", snapshot, fingerprint)
    except BaseException as exc:
        if persistent:
            with store._db:
                store._db.execute("UPDATE manifests SET state='failed',error=? WHERE name=?", (str(exc), manifest))
        raise
