import gc
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer
from mtlearn.layers.cfp import CFPPreprocessor, DiskStore, PreparedDataset
from mtlearn.layers.cfp.storage import disk_store as disk_module
from mtlearn.layers.cfp.storage._writer_lock import _WriterLock

pytestmark = pytest.mark.integration


@pytest.fixture
def cache(tmp_path):
    layer = Layer(1, [{"tree_type": "max-tree", "attributes": [
        morphology.AttributeType.AREA, morphology.AttributeType.GRAY_LEVEL_HEIGHT]}],
        scale_mode="dataset_clipped_zscore01")
    images = torch.randint(0, 256, (3, 1, 6, 7), dtype=torch.uint8,
                           generator=torch.Generator().manual_seed(42))
    source = TensorDataset(images, (images > 100).float())
    path = tmp_path / "cache"
    with DiskStore(path, max_disk_bytes=1024**2) as writer:
        result = CFPPreprocessor.from_layer(layer).prepare(source, store=writer, manifest="train",
            source_version="source-v1", preprocessing_version="uint8-v1", sample_ids=["a", "b", "c"],
            collect_stats=layer.get_statistics_contract())
        layer.set_stats(result.statistics)
        keys = [row[0] for row in writer._db.execute("SELECT entry_key FROM contributions ORDER BY position")]
    return path, layer, source, keys


def session(path, **kwargs):
    if os.name != "posix":
        pytest.skip("Session validation requires POSIX shared locks")
    return DiskStore(path, readonly=True, immutable=True, validation="session", **kwargs)


def forbid(*args, **kwargs):
    raise AssertionError("This operation must not inspect or rebuild tensor files")


@pytest.mark.parametrize("options", [
    {"validation": "sometimes"}, {"validation": "session"},
    {"validation": "session", "immutable": True},
    {"validation": "session", "readonly": True}, {"immutable": 1},
    {"max_validation_entries": -1}, {"max_validation_entries": True},
    {"max_validation_entries": 1.5},
])
def test_invalid_read_options_do_not_create_cache(tmp_path, options):
    with pytest.raises(ValueError):
        DiskStore(tmp_path / "new", max_disk_bytes=1024, **options)
    assert not (tmp_path / "new").exists()


def test_inspection_is_paginated_and_does_not_read_payloads(cache, monkeypatch):
    path, _, _, _ = cache
    with DiskStore(path, readonly=True) as store:
        monkeypatch.setattr(store, "_load_file", forbid)
        monkeypatch.setattr(disk_module, "digest_file", forbid)
        first = store.inspect_manifest("train", limit=2)
        second = store.inspect_manifest("train", offset=first["next_offset"], limit=2)
        assert [s["sample_id"] for s in first["samples"] + second["samples"]] == ["a", "b", "c"]
        assert first["sample_count"] == first["bound_samples"] == first["unique_entries"] == 3
        assert first["compatible"] and first["state"] == "complete"
        assert first["registered_disk_bytes"] > 0 and first["entry_states"] == {"valid": 3}
        assert first["samples"][0]["shape"] == [1, 6, 7]
        assert first["samples"][0]["entries"][0]["size_bytes"] > 0
        assert second["next_offset"] is None
        assert store.inspect_manifest("train", offset=100)["samples"] == []
        with pytest.raises(KeyError):
            store.inspect_manifest("unknown")


@pytest.mark.parametrize("options", [{"offset": -1}, {"offset": True}, {"limit": 0}, {"limit": 1001}])
def test_inspection_rejects_invalid_page(cache, options):
    with DiskStore(cache[0], readonly=True) as store:
        with pytest.raises(ValueError):
            store.inspect_manifest("train", **options)


def test_inspection_reports_incompatible_manifest_config(cache):
    with DiskStore(cache[0], max_disk_bytes=1024**2) as writer:
        config = json.loads(writer._db.execute("SELECT config FROM manifests").fetchone()[0])
        config["format_version"] = -1
        with writer._db:
            writer._db.execute("UPDATE manifests SET config=?", (json.dumps(config),))
        result = writer.inspect_manifest("train")
        assert not result["compatible"] and result["compatibility_error"]


def test_counters_do_not_scan_files_or_query_sql(cache, monkeypatch):
    with DiskStore(cache[0], readonly=True) as store:
        statements = []
        store._db.set_trace_callback(statements.append)
        monkeypatch.setattr(disk_module.os, "scandir", forbid)
        counters = store.counters()
        assert not statements and counters["checksum_reads"] == 0
        assert counters["max_validation_entries"] == 1024


@pytest.mark.parametrize("mmap", [False, True])
def test_session_checksums_once_and_reopen_checks_again(cache, mmap):
    path, _, _, keys = cache
    with session(path, mmap=mmap) as store:
        for _ in range(3):
            prepared = store.get(keys[0])
            assert prepared.info["node_of_pixel"].dtype == torch.uint32
            del prepared
        assert store.counters()["checksum_reads"] == 1
        assert store.counters()["validation_hits"] == 2
        assert store.counters()["retained_bytes"] == 0
    with session(path, mmap=mmap) as reopened:
        reopened.get(keys[0])
        assert reopened.counters()["checksum_reads"] == 1


@pytest.mark.parametrize("ram_bytes,expected", [(0, 3), (1024**2, 1)])
def test_always_policy_preserves_ram_hit_behavior(cache, ram_bytes, expected):
    with DiskStore(cache[0], readonly=True, max_ram_bytes=ram_bytes) as store:
        for _ in range(3):
            store.get(cache[3][0])
        assert store.counters()["checksum_reads"] == expected


@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("changed", [None, "path", "descriptor", "identity"])
def test_read_checks_path_and_descriptor_signatures_independently(cache, monkeypatch, mmap, changed):
    path, _, _, keys = cache
    with DiskStore(path, readonly=True, mmap=mmap) as store:
        entry = store._entry_path(keys[0])
        original_stat = Path.stat
        original_fstat = os.fstat
        original_load = store._load_file
        loaded = False

        def adjusted(value, **kwargs):
            attributes = {name: getattr(value, name) for name in dir(value) if name.startswith("st_")}
            return SimpleNamespace(**(attributes | kwargs))

        def path_stat(filename, *args, **kwargs):
            value = original_stat(filename, *args, **kwargs)
            if filename == entry:
                return adjusted(value, st_ctime_ns=value.st_ctime_ns + 1000 + int(loaded and changed == "path"),
                    st_ino=value.st_ino + int(changed == "identity"))
            return value

        def fstat(descriptor):
            value = original_fstat(descriptor)
            return adjusted(value, st_ctime_ns=value.st_ctime_ns + int(loaded and changed == "descriptor"))

        def load(filename, **kwargs):
            nonlocal loaded
            prepared = original_load(filename, **kwargs)
            loaded = True
            return prepared

        monkeypatch.setattr(Path, "stat", path_stat)
        monkeypatch.setattr(disk_module, "os", SimpleNamespace(**(vars(os) | {"fstat": fstat})))
        monkeypatch.setattr(disk_module, "descriptor_path", lambda handle: None)
        monkeypatch.setattr(store, "_load_file", load)
        if changed is None:
            assert store.get(keys[0]) is not None
            assert loaded
        else:
            with pytest.raises(ValueError, match="changed during"):
                store.get(keys[0])
            assert loaded == (changed != "identity")


@pytest.mark.parametrize("capacity,expected", [(0, 4), (1, 3), (2, 2)])
def test_session_metadata_is_bounded_and_eviction_revalidates(cache, capacity, expected):
    with session(cache[0], max_validation_entries=capacity) as store:
        for index in (0, 0, 1, 0):
            store.get(cache[3][index])
        assert store.counters()["checksum_reads"] == expected
        assert store.counters()["validation_entries"] <= capacity


def test_session_blocks_writer_but_allows_readers(cache):
    with session(cache[0]):
        with session(cache[0]), DiskStore(cache[0], readonly=True):
            with pytest.raises(RuntimeError, match="writer"):
                DiskStore(cache[0], max_disk_bytes=1024**2)
    with DiskStore(cache[0], max_disk_bytes=1024**2):
        with pytest.raises(RuntimeError, match="immutable read session"):
            session(cache[0])


def test_session_rejects_orphan_active_worker(cache):
    worker_path = cache[0] / (".worker." + "a" * 32 + ".lock")
    worker = _WriterLock(worker_path)
    try:
        with pytest.raises(RuntimeError, match="worker"):
            session(cache[0])
    finally:
        worker.close()
    with session(cache[0]):
        assert worker_path.exists()


@pytest.mark.parametrize("ram_bytes", [0, 1024**2])
def test_session_detects_changed_or_missing_file_even_with_ram_hit(cache, ram_bytes):
    path, _, _, keys = cache
    entry = path / "entries" / f"{keys[0]}.pt"
    with session(path, max_ram_bytes=ram_bytes) as store:
        active = store.get(keys[0])
        before = active.raw_attributes[morphology.AttributeType.AREA].clone()
        with entry.open("ab") as handle:
            handle.write(b"invalid")
        with pytest.raises(ValueError, match="Invalid prepared entry"):
            store.get(keys[0])
        assert store.counters()["validation_entries"] == 0
        assert store.counters()["retained_bytes"] == 0
        torch.testing.assert_close(active.raw_attributes[morphology.AttributeType.AREA], before)
        entry.unlink()
        assert store.get(keys[0]) is None


def test_session_revalidates_replaced_identical_entry(cache):
    path, _, _, keys = cache
    entry = path / "entries" / f"{keys[0]}.pt"
    with session(path) as store:
        store.get(keys[0])
        replacement = path / "replacement.pt"
        shutil.copy2(entry, replacement)
        os.replace(replacement, entry)
        store.get(keys[0])
        assert store.counters()["checksum_reads"] == 2


def test_loading_uses_validated_descriptor_and_detects_path_replacement(cache, monkeypatch):
    path, _, _, keys = cache
    entry = path / "entries" / f"{keys[0]}.pt"
    with session(path) as store:
        original = store._load_file
        def replace_before_mapping(filename, **kwargs):
            replacement = path / "replacement.pt"
            shutil.copy2(entry, replacement)
            os.replace(replacement, entry)
            assert Path(filename) != entry
            return original(filename, **kwargs)
        monkeypatch.setattr(store, "_load_file", replace_before_mapping)
        with pytest.raises(ValueError, match="changed during"):
            store.get(keys[0])
        assert store.counters()["validation_entries"] == 0


def test_manifest_change_invalidates_session_and_prepared_dataset(cache):
    path, _, source, keys = cache
    with session(path, max_ram_bytes=1024**2) as store:
        dataset = PreparedDataset(store, "train", source)
        dataset[0]
        with sqlite3.connect(path / "manifest.sqlite3") as connection:
            connection.execute("UPDATE manifests SET source_version='changed'")
        with pytest.raises(ValueError, match="Manifest contract changed"):
            dataset[0]
        assert store.counters()["validation_entries"] == 0
        assert store.counters()["retained_bytes"] == 0
        store.get(keys[0])
        assert store.counters()["checksum_reads"] == 2


def test_manifest_header_cache_refreshes_after_same_connection_write(cache):
    with DiskStore(cache[0], max_disk_bytes=1024**2) as store:
        PreparedDataset(store, "train")
        with store._db:
            store._db.execute("UPDATE manifests SET state='cancelled'")
        with pytest.raises(RuntimeError, match="cancelled"):
            PreparedDataset(store, "train")


def test_header_queries_are_cached_but_source_pixels_still_checked(cache):
    path, _, source, _ = cache
    with DiskStore(path, readonly=True) as store:
        dataset = PreparedDataset(store, "train", source)
        statements = []
        store._db.set_trace_callback(statements.append)
        dataset[0]
        dataset[1]
        assert not any("SELECT * FROM manifests" in sql for sql in statements)
        source.tensors[0][0].zero_()
        with pytest.raises(ValueError, match="Source pixels/shape changed"):
            dataset[0]


def test_validation_forces_checksums_and_full_structure_without_writing(cache):
    path, _, _, keys = cache
    events = []
    with session(path, max_ram_bytes=1024**2) as store:
        store.get(keys[0])
        result = store.validate_manifest("train", progress=events.append)
        assert result["status"] == "valid" and result["fully_validated"]
        assert result["checked_samples"] == result["checked_entries"] == result["valid_entries"] == 3
        assert store.counters()["checksum_reads"] == 4
        assert store.counters()["writes"] == 0
        assert events[0]["status"] == "running" and events[-1]["status"] == "valid"
        assert [event["sample_id"] for event in events[1:-1]] == ["a", "b", "c"]
        assert all(not torch.is_tensor(value) for event in events for value in event.values())


def test_validation_collects_bounded_errors_without_repairing_writer(cache):
    path, _, _, keys = cache
    with DiskStore(path, max_disk_bytes=1024**2) as store:
        before = [tuple(row) for row in store._db.execute("SELECT * FROM entries ORDER BY key")]
        for key in keys:
            (path / "entries" / f"{key}.pt").unlink()
        result = store.validate_manifest("train", max_errors=1)
        after = [tuple(row) for row in store._db.execute("SELECT * FROM entries ORDER BY key")]
        assert before == after
        assert result["status"] == "invalid" and not result["fully_validated"]
        assert result["error_count"] == 3 and len(result["errors"]) == 1
        assert result["errors_truncated"] and result["errors"][0]["sample_id"] == "a"


def test_validation_subset_and_cancellation_do_not_claim_full_validation(cache):
    with DiskStore(cache[0], readonly=True) as store:
        result = store.validate_manifest("train", offset=1, limit=1)
        assert result["status"] == "valid" and not result["fully_validated"]
        assert result["checked_samples"] == 1
        cancel = threading.Event()
        def progress(event):
            if event["checked_samples"] == 1:
                cancel.set()
        result = store.validate_manifest("train", cancel=cancel, progress=progress)
        assert result["status"] == "cancelled" and result["checked_samples"] == 1
        assert not result["fully_validated"]


def test_validation_rejects_changed_manifest_during_callback(cache):
    with DiskStore(cache[0], max_disk_bytes=1024**2) as store:
        def progress(event):
            if event["status"] == "running" and event["checked_samples"] == 1:
                with store._db:
                    store._db.execute("UPDATE manifests SET source_version='changed'")
        result = store.validate_manifest("train", progress=progress)
        assert result["status"] == "invalid" and result["error_count"] == 1


def test_validation_detects_inconsistent_contributions(cache):
    with DiskStore(cache[0], max_disk_bytes=1024**2) as store:
        with store._db:
            store._db.execute("DELETE FROM contributions WHERE position=0")
        result = store.validate_manifest("train")
        assert result["status"] == "invalid"
        assert "contributions" in result["errors"][0]["message"]


def test_validation_propagates_callback_errors_and_releases_payloads(cache):
    with DiskStore(cache[0], readonly=True) as store:
        def progress(event):
            if event["checked_samples"]:
                raise RuntimeError("callback failed")
        with pytest.raises(RuntimeError, match="callback failed"):
            store.validate_manifest("train", progress=progress)
        assert store.counters()["active_bytes"] == 0


def test_session_close_preserves_active_backward(cache):
    path, layer, source, _ = cache
    store = session(path)
    batch, target = PreparedDataset(store, "train", source)[0]
    reference = weakref.ref(batch)
    output = layer(batch)
    store.close()
    del batch
    gc.collect()
    assert reference() is not None
    ((output / 255 - target[None])**2).mean().backward()
    assert reference() is None


@pytest.mark.parametrize("warm", [False, True])
def test_session_detects_same_size_corruption(cache, warm):
    path, _, _, keys = cache
    entry = path / "entries" / f"{keys[0]}.pt"
    with session(path, max_ram_bytes=1024**2) as store:
        if warm:
            store.get(keys[0])
        previous = entry.stat()
        with entry.open("r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            value = handle.read(1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes([value[0] ^ 1]))
        os.utime(entry, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        with pytest.raises(ValueError, match="checksum"):
            store.get(keys[0])
        assert store.counters()["checksum_reads"] == 1 + int(warm)
        assert store.counters()["retained_bytes"] == 0


@pytest.mark.parametrize("replacement", [False, True])
def test_session_rejects_removed_or_replaced_database(cache, replacement):
    path, _, _, keys = cache
    database = path / "manifest.sqlite3"
    with session(path, max_ram_bytes=1024**2) as store:
        store.get(keys[0])
        if replacement:
            other = path / "other.sqlite3"
            shutil.copy2(database, other)
            os.replace(other, database)
        else:
            database.unlink()
        with pytest.raises(RuntimeError, match="reopen DiskStore"):
            store.get(keys[0])
        assert store.counters()["validation_entries"] == store.counters()["retained_bytes"] == 0


def test_session_rejects_changed_backend_contract(cache):
    path, _, _, keys = cache
    with session(path, max_ram_bytes=1024**2) as store:
        store.get(keys[0])
        with sqlite3.connect(path / "manifest.sqlite3") as connection:
            connection.execute("UPDATE metadata SET value='{}' WHERE key='contract'")
        for _ in range(2):
            with pytest.raises(ValueError, match="format/backend contract changed"):
                store.get(keys[0])
        assert store.counters()["validation_entries"] == store.counters()["retained_bytes"] == 0


def test_manifest_header_metadata_has_a_fixed_capacity(cache):
    with DiskStore(cache[0], max_disk_bytes=1024**2, max_validation_entries=2) as store:
        with store._db:
            for name in ("other1", "other2"):
                store._db.execute("""INSERT INTO manifests SELECT ?,config,source_version,
                    preprocessing_version,split,sample_count,state,error FROM manifests WHERE name='train'""", (name,))
        for name in ("train", "other1", "other2"):
            store.inspect_manifest(name, limit=1)
        assert store.counters()["manifest_cache_entries"] == 2


def test_full_validation_checks_tensor_hashes_beyond_file_checksum(cache):
    path, _, _, keys = cache
    entry = path / "entries" / f"{keys[0]}.pt"
    with DiskStore(path, max_disk_bytes=1024**2) as store:
        data = torch.load(entry, weights_only=True)
        data["attributes"]["AREA"].add_(1)
        torch.save(data, entry)
        with store._db:
            store._db.execute("UPDATE entries SET size_bytes=?,sha256=? WHERE key=?",
                (entry.stat().st_size, disk_module.digest_file(entry), keys[0]))
        result = store.validate_manifest("train")
        assert result["status"] == "invalid" and result["error_count"] == 1
        assert "tensor checksum" in result["errors"][0]["message"]


def test_validation_incomplete_manifest_and_deduplicated_entries(cache):
    with DiskStore(cache[0], max_disk_bytes=1024**2) as store:
        with store._db:
            bindings = store._db.execute("SELECT bindings FROM samples WHERE position=0").fetchone()[0]
            key = cache[3][0]
            store._db.execute("UPDATE samples SET bindings=? WHERE position=1", (bindings,))
            store._db.execute("UPDATE contributions SET entry_key=? WHERE position=1", (key,))
            store._db.execute("UPDATE manifests SET state='cancelled'")
        assert store.inspect_manifest("train")["unique_entries"] == 2
        result = store.validate_manifest("train")
        assert result["status"] == "incomplete" and not result["fully_validated"]
        assert result["checked_samples"] == result["valid_entries"] == 3


def test_session_lock_blocks_a_writer_in_another_process(cache):
    script = """import sys
from mtlearn.layers.cfp import DiskStore
try:
    with DiskStore(sys.argv[1], max_disk_bytes=1024**2):
        pass
except RuntimeError:
    sys.exit(23)
"""
    def writer():
        return subprocess.run([sys.executable, "-c", script, str(cache[0])], capture_output=True, timeout=30)
    with session(cache[0]):
        assert writer().returncode == 23
    result = writer()
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize("mmap", [False, True])
def test_mmap_uses_filename_supported_by_older_torch(cache, monkeypatch, mmap):
    original_load = torch.load

    def load(path, **kwargs):
        if kwargs.get("mmap") and not isinstance(path, str):
            raise ValueError("f must be a string filename in order to use mmap argument")
        return original_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    with DiskStore(cache[0], readonly=True, mmap=mmap) as store:
        prepared = store.get(cache[3][0])
        prepared.validate(full=True)
        assert prepared.image_shape == (6, 7)
