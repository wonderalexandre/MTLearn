import json
import math
from pathlib import Path
import shutil

from ..storage import DiskStore
from ._identity import canonical_json, preprocessor_config, validate_identity, tree_config
from ._persistent_preparation import statistics_contract, statistics_identity
from .preparation_result import DiskPreparationResult


class _Cancelled(Exception):
    def __init__(self, count):
        self.count = count


class _PreparationStore(DiskStore):
    def __init__(self, *args, **kwargs):
        self._actions_ready = False
        self.actions = {"reused": 0, "prepared": 0, "repaired": 0, "recovered": 0}
        super().__init__(*args, **kwargs)
        try:
            self._db.execute("PRAGMA temp_store=FILE")
            with self._db:
                self._db.execute("CREATE TEMP TABLE preparation_actions (key TEXT PRIMARY KEY, action TEXT)")
                self._db.execute("INSERT INTO preparation_actions SELECT key,NULL FROM entries")
            self._actions_ready = True
        except BaseException:
            self.close()
            raise

    def _record(self, key, *, published=False):
        row = self._db.execute("SELECT action FROM preparation_actions WHERE key=?", (key,)).fetchone()
        old = None if row is None else row[0]
        if old in ("prepared", "repaired"):
            return
        action = ("prepared" if row is None else "repaired") if published else "reused"
        if old == action:
            return
        with self._db:
            self._db.execute("INSERT INTO preparation_actions VALUES (?,?) ON CONFLICT(key) DO UPDATE SET action=excluded.action",
                             (key, action))
        if old is not None:
            self.actions[old] -= 1
        self.actions[action] += 1

    def get(self, key):
        prepared = super().get(key)
        if prepared is not None and self._actions_ready:
            self._record(key)
        return prepared

    def _register_valid(self, key, identity, path, summary, expected_digest):
        super()._register_valid(key, identity, path, summary, expected_digest)
        if self._actions_ready:
            self._record(key, published=True)
        else:
            self.actions["recovered"] += 1


def _free_bytes(path):
    existing = path
    while not existing.exists():
        existing = existing.parent
    return shutil.disk_usage(existing).free


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _header(store, preprocessor, manifest, source_version, preprocessing_version, split, count):
    row = store._db.execute("SELECT * FROM manifests WHERE name=?", (manifest,)).fetchone()
    if row is None:
        return None
    expected = {"config": canonical_json(preprocessor_config(preprocessor)), "source_version": source_version,
                "preprocessing_version": preprocessing_version, "split": split}
    if count is not None:
        expected["sample_count"] = count
    differences = [name for name, value in expected.items() if row[name] != value]
    if differences:
        raise ValueError(f"Manifest {manifest!r} is incompatible: {', '.join(differences)} changed. "
                         "Use the matching contract or a new manifest name; cache metadata must not be rewritten.")
    return row


def _scan(store, preprocessor, header, sample_ids, *, complete, cancel, progress=None):
    generation = store._read_generation()
    name, total = header["name"], header["sample_count"]
    if complete and header["state"] != "complete":
        raise RuntimeError(f"Manifest {name!r} is {header['state']}; use mode='prepare_missing' with the source and a quota.")
    checked = ready = maximum = 0
    for row in store._db.execute("SELECT * FROM samples WHERE manifest=? ORDER BY position", (name,)):
        if cancel():
            raise _Cancelled(checked)
        position, sample_id = row["position"], row["sample_id"]
        if not 0 <= position < total or (complete and position != checked):
            raise ValueError(f"Manifest {name!r} has inconsistent sample positions.")
        if sample_ids is not None:
            expected_id = sample_ids(position) if callable(sample_ids) else sample_ids[position]
            if not isinstance(expected_id, str) or expected_id != sample_id:
                raise ValueError(f"Manifest {name!r} sample ID/order differs at position {position}.")
        bindings, shape = json.loads(row["bindings"]), json.loads(row["shape"])
        if (not isinstance(sample_id, str) or not sample_id or not isinstance(shape, list) or len(shape) != 3
                or any(type(n) is not int or n <= 0 for n in shape) or not isinstance(bindings, list)
                or len(bindings) != shape[0]
                or any(not isinstance(trees, dict) or set(trees) != set(preprocessor.tree_specs) for trees in bindings)):
            raise ValueError(f"Manifest {name!r} has invalid shape/tree/channel bindings for sample {sample_id!r}.")
        expected = {(channel, tree_key, key) for channel, trees in enumerate(bindings) for tree_key, key in trees.items()}
        actual = {tuple(item) for item in store._db.execute(
            "SELECT channel,tree_key,entry_key FROM contributions WHERE manifest=? AND position=?", (name, position))}
        if expected != actual:
            raise ValueError(f"Manifest {name!r} has inconsistent statistical contributions for sample {sample_id!r}.")
        valid, size = True, 0
        for channel, tree_key, key in expected:
            path = store._entry_path(key)
            entry = store._db.execute("SELECT * FROM entries WHERE key=?", (key,)).fetchone()
            if entry is not None:
                identity = json.loads(entry["identity"])
                if (validate_identity(identity) != key or identity["shape"] != shape[1:]
                        or identity["tree"] != tree_config(preprocessor.tree_specs[tree_key])
                        or identity["attributes"] != [a.name for a in preprocessor.features[tree_key].attributes]
                        or identity["attribute_dtype"] != preprocessor.attribute_dtype.name):
                    raise ValueError(f"Prepared identity does not match sample {sample_id!r}, entry {key}.")
            try:
                size_on_disk = path.stat().st_size if path.is_file() else None
            except FileNotFoundError:
                size_on_disk = None
            entry_valid = (entry is not None and entry["state"] == "valid" and type(entry["size_bytes"]) is int
                           and entry["size_bytes"] > 0 and size_on_disk == entry["size_bytes"])
            if not entry_valid:
                valid = False
                if complete:
                    raise RuntimeError(f"Sample {sample_id!r}, entry {key} is missing, incomplete or has changed size; "
                                       "use mode='prepare_missing' with the matching source and quota.")
            else:
                size += size_on_disk
        checked += 1
        if valid:
            ready += 1
            maximum = max(maximum, size)
        if progress is not None:
            progress({"phase": "inspect", "status": "running", "manifest": name,
                      "sample_count": checked, "total_samples": total, "position": position, "sample_id": sample_id})
    if complete and checked != total:
        raise RuntimeError(f"Manifest {name!r} has incomplete sample bindings; resume preparation.")
    if store._read_generation() != generation:
        raise RuntimeError("Manifest changed while checking reuse; retry with a stable cache.")
    unique = store._db.execute("SELECT COUNT(DISTINCT entry_key) FROM contributions WHERE manifest=?", (name,)).fetchone()[0]
    return {"ready_samples": ready, "observed_samples": checked, "max_sample_disk_bytes": maximum,
            "unique_entries": unique}


def _resources(path, store, scan, count, quota, reserve, factor, pilot_count, *, writable):
    if store is None:
        used = 0
    elif writable:
        used = store._tensor_file_bytes()
    else:
        used = store._db.execute("SELECT COALESCE(SUM(size_bytes),0) FROM entries WHERE state='valid'").fetchone()[0]
    remaining = max(0, count - scan["ready_samples"])
    estimate = (math.ceil(scan["max_sample_disk_bytes"] * remaining * factor)
                if scan["max_sample_disk_bytes"] else 0 if not remaining else None)
    return {**scan, "pilot_samples": pilot_count, "disk_bytes": used,
            "disk_bytes_kind": "tensor_files" if writable else "registered_valid_entries",
            "max_disk_bytes": quota, "quota_remaining_bytes": None if quota is None else max(0, quota-used),
            "free_disk_bytes": _free_bytes(path), "min_free_disk_bytes": reserve,
            "additional_disk_bytes_estimate": estimate, "disk_safety_factor": factor}


def prepare_or_reuse(preprocessor, source, *, path, manifest, source_version, preprocessing_version,
                     split, sample_ids, collect_stats, mode, max_disk_bytes, min_free_disk_bytes,
                     pilot_samples, disk_safety_factor, cancel, progress, num_workers, max_in_flight,
                     worker_threads, max_sample_bytes, max_retries):
    if mode not in ("reuse", "prepare_missing"):
        raise ValueError("mode must be 'reuse' or 'prepare_missing'.")
    for name, value in (("manifest", manifest), ("source_version", source_version),
                        ("preprocessing_version", preprocessing_version)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty string.")
    if split not in ("train", "evaluation"):
        raise ValueError("split must be 'train' or 'evaluation'.")
    if source is not None and (not hasattr(source, "__len__") or not hasattr(source, "__getitem__")):
        raise TypeError("source must be a finite map-style dataset or None in reuse mode.")
    count = None if source is None else len(source)
    if count is not None and count <= 0:
        raise ValueError("source must be nonempty.")
    if mode == "prepare_missing" and (source is None or max_disk_bytes is None):
        raise ValueError("mode='prepare_missing' requires a source and explicit max_disk_bytes.")
    for name, value in (("min_free_disk_bytes", min_free_disk_bytes), ("pilot_samples", pilot_samples)):
        _integer(name, value)
    if max_disk_bytes is not None:
        _integer("max_disk_bytes", max_disk_bytes)
    for name, value, minimum in (("num_workers", num_workers, 0), ("worker_threads", worker_threads, 1),
                                 ("max_sample_bytes", max_sample_bytes, 1), ("max_retries", max_retries, 0)):
        _integer(name, value, minimum)
    if max_in_flight is not None:
        _integer("max_in_flight", max_in_flight, 1)
    if not num_workers and (max_in_flight is not None or max_retries):
        raise ValueError("max_in_flight/max_retries require num_workers > 0.")
    if type(disk_safety_factor) not in (int, float) or not math.isfinite(disk_safety_factor) or disk_safety_factor < 1:
        raise ValueError("disk_safety_factor must be finite and >= 1.")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None.")
    if cancel is not None and not callable(cancel) and not callable(getattr(cancel, "is_set", None)):
        raise TypeError("cancel must be callable, an event, or None.")
    collect = collect_stats is not None and collect_stats is not False
    if collect and split != "train":
        raise ValueError("collect_stats requires split='train'; reuse the training snapshot for evaluation.")
    contract = statistics_contract(preprocessor, collect_stats) if collect else None
    path = Path(path).expanduser().resolve()
    reason = None
    phase = "inspect"
    completed = pilot_count = 0
    scan = {"ready_samples": 0, "observed_samples": 0, "max_sample_disk_bytes": 0, "unique_entries": 0}
    resources = {}
    store = None

    def stopped():
        nonlocal reason
        if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
            reason = "cancelled_by_user"
            return True
        if phase in ("pilot", "prepare") and min_free_disk_bytes and _free_bytes(path) < min_free_disk_bytes:
            reason = "free_space_reserve"
            return True
        return False

    def notify(event):
        nonlocal completed
        if event["phase"] == "prepare":
            completed = event["sample_count"]
        if progress is not None:
            event = dict(event, mode=mode)
            if event["phase"] == "prepare":
                event["phase"] = phase
                if phase == "pilot" and event["status"] == "cancelled" and reason is None:
                    event["status"] = "complete"
            if isinstance(store, _PreparationStore):
                event.update({name + "_entries": value for name, value in store.actions.items()})
            progress(event)

    def result(status, statistics=None, fingerprint=None):
        actions = store.actions if isinstance(store, _PreparationStore) else {
            "reused": scan["unique_entries"] if status == "complete" else 0,
            "prepared": 0, "repaired": 0, "recovered": 0}
        value = DiskPreparationResult(str(path), manifest, completed, status, statistics, fingerprint,
            mode=mode, total_samples=count or 0, reused_entries=actions["reused"],
            prepared_entries=actions["prepared"], repaired_entries=actions["repaired"], recovered_entries=actions["recovered"],
            statistics_source="manifest_summaries" if statistics is not None else None,
            reason=reason, resources=dict(resources))
        notify({"phase": "result", "status": status, "manifest": manifest,
                "sample_count": completed, "total_samples": count or 0, "reason": reason})
        return value

    try:
        if (path / "manifest.sqlite3").exists() or mode == "reuse":
            with DiskStore(path, readonly=True) as reader:
                generation = reader._read_generation()
                header = _header(reader, preprocessor, manifest, source_version, preprocessing_version, split, count)
                if header is None and mode == "reuse":
                    raise KeyError(f"Unknown prepared manifest {manifest!r}; use mode='prepare_missing' with a source and quota.")
                if header is not None:
                    count = header["sample_count"]
                    if sample_ids is not None and not callable(sample_ids) and len(sample_ids) != count:
                        raise ValueError("sample_ids must have one string per manifest sample.")
                    scan = _scan(reader, preprocessor, header, sample_ids, complete=mode == "reuse",
                                 cancel=stopped, progress=notify)
                resources = _resources(path, reader, scan, count, max_disk_bytes, min_free_disk_bytes,
                                       disk_safety_factor, 0, writable=False)
                if mode == "reuse":
                    if stopped():
                        raise _Cancelled(0)
                    if collect:
                        notify({"phase": "statistics", "status": "running", "manifest": manifest,
                                "sample_count": count, "total_samples": count})
                    statistics = reader.statistics(manifest, contract) if collect else None
                    fingerprint = statistics_identity(reader, manifest, contract) if collect else None
                    if reader._read_generation() != generation:
                        raise RuntimeError("Manifest changed while recovering statistics; retry with a stable cache.")
                    completed = count
                    return result("complete", statistics, fingerprint)
        if sample_ids is not None and not callable(sample_ids) and len(sample_ids) != count:
            raise ValueError("sample_ids must have one string per source item.")
        phase = "prepare"
        if not resources:
            resources = _resources(path, None, scan, count, max_disk_bytes, min_free_disk_bytes,
                                   disk_safety_factor, 0, writable=True)
        if stopped():
            return result("resource_limited" if reason == "free_space_reserve" else "cancelled")
        store = _PreparationStore(path, max_disk_bytes=max_disk_bytes, min_free_disk_bytes=min_free_disk_bytes)
        from ._parallel_preparation import validate_parallel_options
        validate_parallel_options(store, num_workers, max_in_flight, worker_threads, max_sample_bytes, max_retries)
        options = dict(store=store, manifest=manifest, source_version=source_version,
            preprocessing_version=preprocessing_version, split=split, sample_ids=sample_ids, progress=notify)
        prepared_result = None
        if pilot_samples and scan["ready_samples"] < count:
            phase = "pilot"
            target = min(pilot_samples, count)
            prepared_result = preprocessor.prepare(source, **options,
                collect_stats=contract if collect and target == count else None,
                cancel=lambda: stopped() or completed >= target)
            pilot_count = completed
        header = _header(store, preprocessor, manifest, source_version, preprocessing_version, split, count)
        if header is not None:
            scan = _scan(store, preprocessor, header, sample_ids, complete=False, cancel=lambda: False)
        resources = _resources(path, store, scan, count, max_disk_bytes, min_free_disk_bytes,
                               disk_safety_factor, pilot_count, writable=True)
        notify({"phase": "resources", "status": "running", "manifest": manifest,
                "sample_count": completed, "total_samples": count, **resources})
        if reason is not None:
            return result("resource_limited" if reason == "free_space_reserve" else "cancelled")
        estimate = resources["additional_disk_bytes_estimate"]
        if estimate is not None and estimate > resources["quota_remaining_bytes"]:
            reason = "estimated_quota_insufficient"
            return result("resource_limited")
        if estimate is not None and estimate > max(0, resources["free_disk_bytes"] - min_free_disk_bytes):
            reason = "estimated_free_space_insufficient"
            return result("resource_limited")
        if prepared_result is None or prepared_result.status != "complete":
            phase = "prepare"
            completed = 0
            prepared_result = preprocessor.prepare(source, **options, collect_stats=contract,
                cancel=stopped, num_workers=num_workers, max_in_flight=max_in_flight,
                worker_threads=worker_threads, max_sample_bytes=max_sample_bytes, max_retries=max_retries)
        header = _header(store, preprocessor, manifest, source_version, preprocessing_version, split, count)
        scan = _scan(store, preprocessor, header, sample_ids, complete=prepared_result.status == "complete", cancel=lambda: False)
        resources = _resources(path, store, scan, count, max_disk_bytes, min_free_disk_bytes,
                               disk_safety_factor, pilot_count, writable=True)
        status = "resource_limited" if reason == "free_space_reserve" else prepared_result.status
        return result(status, prepared_result.statistics, prepared_result.statistics_id)
    except _Cancelled as exc:
        completed = exc.count
        return result("cancelled")
    except OSError as exc:
        if isinstance(store, _PreparationStore) and any(text in str(exc) for text in ("quota exceeded", "free-space reserve")):
            reason = "quota_exceeded" if "quota exceeded" in str(exc) else "free_space_reserve"
            resources = _resources(path, store, scan, count, max_disk_bytes, min_free_disk_bytes,
                                   disk_safety_factor, pilot_count, writable=True)
            return result("resource_limited")
        raise
    finally:
        if store is not None:
            store.close()
