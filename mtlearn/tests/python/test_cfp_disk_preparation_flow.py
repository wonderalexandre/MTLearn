import contextlib
import hashlib
import io
import os
import sqlite3
import threading
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer
from mtlearn.layers.cfp import CFPPreprocessor, DiskStore, DiskPreparationResult, PreparedDataset
from mtlearn.layers.cfp.preparation import _disk_preparation as flow
from mtlearn.layers.cfp.storage import _disk_format, _writer_lock

pytestmark = pytest.mark.integration
QUOTA = 1024**2


def forbid(*args, **kwargs):
    raise AssertionError("This operation must not build, load, summarize or read images")


@pytest.fixture
def experiment():
    layer = Layer(1, [{"tree_type": "max-tree", "attributes": [
        morphology.AttributeType.AREA, morphology.AttributeType.GRAY_LEVEL_HEIGHT]}],
        scale_mode="dataset_clipped_zscore01")
    images = torch.randint(0, 256, (4, 1, 6, 7), dtype=torch.uint8,
                           generator=torch.Generator().manual_seed(42))
    source = TensorDataset(images, (images > 100).float())
    return CFPPreprocessor.from_layer(layer), layer, source


def options(path):
    return dict(path=path, manifest="train", source_version="source-v1", preprocessing_version="uint8-v1",
                sample_ids=["a", "b", "c", "d"])


@pytest.fixture
def cache(tmp_path, experiment):
    preprocessor, layer, source = experiment
    path = tmp_path / "cache"
    result = preprocessor.prepare_or_reuse(source, **options(path), mode="prepare_missing", max_disk_bytes=QUOTA,
        collect_stats=layer.get_statistics_contract())
    assert result.status == "complete"
    return path, result


def assert_stats(actual, expected):
    assert actual.sample_count == expected.sample_count
    assert actual.contract == expected.contract
    for key, values in expected.statistics.items():
        for name, value in values.items():
            torch.testing.assert_close(actual.statistics[key][name], value, rtol=0, atol=0)


def files(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path / "entries").glob("*.pt")}


def test_reuse_never_loads_payloads_or_source_and_merges_statistics_once(cache, experiment, monkeypatch):
    path, prepared = cache
    preprocessor, layer, source = experiment
    before = files(path)
    class Source:
        def __len__(self):
            return len(source)
        __getitem__ = forbid
    calls = []
    original = DiskStore.statistics
    def statistics(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(DiskStore, "statistics", statistics)
    monkeypatch.setattr(CFPPreprocessor, "prepare_u8", forbid)
    monkeypatch.setattr(DiskStore, "_load_file", forbid)
    monkeypatch.setattr(_disk_format, "summarize", forbid)
    events = []
    result = preprocessor.prepare_or_reuse(Source(), **options(path), collect_stats=layer.get_statistics_contract(),
        max_disk_bytes=0, min_free_disk_bytes=10**30, progress=events.append)
    assert isinstance(result, DiskPreparationResult) and result.mode == "reuse" and result.status == "complete"
    assert result.sample_count == result.total_samples == result.reused_entries == 4
    assert result.prepared_entries == result.repaired_entries == 0
    assert result.statistics_source == "manifest_summaries" and result.statistics_id == prepared.statistics_id
    assert_stats(result.statistics, prepared.statistics)
    assert calls == [1] and files(path) == before
    assert events[-1]["status"] == "complete" and events[-1]["phase"] == "result"
    assert all(not torch.is_tensor(v) for e in events for v in e.values())
    assert any(e["phase"] == "statistics" for e in events)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_prepare_then_reuse_preserves_statistics_and_unique_action_counts(tmp_path, experiment, num_workers):
    preprocessor, layer, source = experiment
    source.tensors[0][3].copy_(source.tensors[0][0])
    events = []
    result = preprocessor.prepare_or_reuse(source, **options(tmp_path / "cache"), mode="prepare_missing",
        max_disk_bytes=QUOTA, collect_stats=layer.get_statistics_contract(), num_workers=num_workers, progress=events.append)
    reference = preprocessor.prepare(source, collect_stats=layer.get_statistics_contract())
    assert result.status == "complete" and result.prepared_entries == 3 and result.reused_entries == 0
    assert_stats(result.statistics, reference.statistics)
    assert sorted(e["sample_id"] for e in events if e.get("phase") == "prepare" and e.get("position") is not None) == ["a", "b", "c", "d"]
    again = preprocessor.prepare_or_reuse(source, **options(tmp_path / "cache"), mode="prepare_missing",
        max_disk_bytes=QUOTA, collect_stats=layer.get_statistics_contract(), num_workers=num_workers)
    assert again.reused_entries == 3 and again.prepared_entries == again.repaired_entries == 0
    assert_stats(again.statistics, reference.statistics)


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_prepare_missing_repairs_only_damaged_entry(cache, experiment, monkeypatch, damage, num_workers):
    path, previous = cache
    preprocessor, layer, source = experiment
    with DiskStore(path, readonly=True) as store:
        key = store.inspect_manifest("train", limit=1)["samples"][0]["entries"][0]["entry_key"]
    entry = path / "entries" / (key + ".pt")
    if damage == "missing":
        entry.unlink()
    else:
        with entry.open("r+b") as handle:
            handle.write(b"bad!")
    before = files(path)
    result = preprocessor.prepare_or_reuse(source, **options(path), mode="prepare_missing", max_disk_bytes=QUOTA,
        collect_stats=layer.get_statistics_contract(), num_workers=num_workers)
    assert result.status == "complete" and result.repaired_entries == 1 and result.reused_entries == 3
    assert result.prepared_entries == 0
    after = files(path)
    assert all(value == after[name] for name, value in before.items() if name != entry.name)
    assert_stats(result.statistics, previous.statistics)


def test_pilot_stops_on_estimated_quota_and_resumes_without_duplicate_stats(cache, tmp_path, experiment):
    path, reference = cache
    preprocessor, layer, source = experiment
    with DiskStore(path, readonly=True) as reader:
        size = reader.inspect_manifest("train", limit=1)["samples"][0]["entries"][0]["size_bytes"]
    new_path = tmp_path / "pilot"
    result = preprocessor.prepare_or_reuse(source, **options(new_path), mode="prepare_missing",
        max_disk_bytes=size*2, pilot_samples=1, collect_stats=layer.get_statistics_contract())
    assert result.status == "resource_limited" and result.reason == "estimated_quota_insufficient"
    assert result.sample_count == result.prepared_entries == result.resources["pilot_samples"] == 1
    assert result.resources["additional_disk_bytes_estimate"] > result.resources["quota_remaining_bytes"]
    assert result.statistics is None and len(files(new_path)) == 1
    resumed = preprocessor.prepare_or_reuse(source, **options(new_path), mode="prepare_missing",
        max_disk_bytes=QUOTA, collect_stats=layer.get_statistics_contract())
    assert resumed.status == "complete" and resumed.reused_entries == 1 and resumed.prepared_entries == 3
    assert_stats(resumed.statistics, reference.statistics)


@pytest.mark.parametrize("pilot_samples", [1, 4, 9])
def test_pilot_is_reused_and_statistics_reduced_once(tmp_path, experiment, monkeypatch, pilot_samples):
    preprocessor, layer, source = experiment
    calls = []
    original = DiskStore.statistics
    def statistics(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(DiskStore, "statistics", statistics)
    result = preprocessor.prepare_or_reuse(source, **options(tmp_path / "cache"), mode="prepare_missing",
        max_disk_bytes=QUOTA, pilot_samples=pilot_samples, collect_stats=layer.get_statistics_contract())
    assert result.status == "complete" and result.prepared_entries == 4 and result.reused_entries == 0
    assert result.resources["pilot_samples"] == min(pilot_samples, 4)
    assert calls == [1]


def test_cancel_after_sample_preserves_entries_and_resume(cache, tmp_path, experiment):
    _, reference = cache
    preprocessor, layer, source = experiment
    path = tmp_path / "cancel"
    event = threading.Event()
    def progress(message):
        if message["phase"] == "prepare" and message["sample_count"] == 2:
            event.set()
    result = preprocessor.prepare_or_reuse(source, **options(path), mode="prepare_missing", max_disk_bytes=QUOTA,
        progress=progress, cancel=event, collect_stats=layer.get_statistics_contract())
    assert result.status == "cancelled" and result.sample_count == result.prepared_entries == 2
    assert result.statistics is None and result.reason == "cancelled_by_user"
    resumed = preprocessor.prepare_or_reuse(source, **options(path), mode="prepare_missing", max_disk_bytes=QUOTA,
        collect_stats=layer.get_statistics_contract())
    assert resumed.reused_entries == resumed.prepared_entries == 2
    assert_stats(resumed.statistics, reference.statistics)


def test_resource_preflight_does_not_create_cache(tmp_path, experiment, monkeypatch):
    preprocessor, _, source = experiment
    monkeypatch.setattr(flow, "_free_bytes", lambda path: 100)
    path = tmp_path / "new"
    result = preprocessor.prepare_or_reuse(source, **options(path), mode="prepare_missing",
        max_disk_bytes=QUOTA, min_free_disk_bytes=101)
    assert result.status == "resource_limited" and result.reason == "free_space_reserve"
    assert result.resources["free_disk_bytes"] == 100 and not path.exists()


def test_reserve_during_serialization_leaves_no_partial_payload(tmp_path, experiment, monkeypatch):
    preprocessor, _, source = experiment
    path = tmp_path / "new"
    monkeypatch.setattr(flow, "_free_bytes", lambda path: QUOTA)
    monkeypatch.setattr(_writer_lock.shutil, "disk_usage", lambda path: SimpleNamespace(free=100))
    result = preprocessor.prepare_or_reuse(source, **options(path), mode="prepare_missing",
        max_disk_bytes=QUOTA, min_free_disk_bytes=100)
    assert result.status == "resource_limited" and result.reason == "free_space_reserve"
    assert not list((path / "entries").iterdir())


def test_quota_failure_is_resumable_without_pilot(tmp_path, experiment):
    preprocessor, _, source = experiment
    path = tmp_path / "new"
    result = preprocessor.prepare_or_reuse(source, **options(path), mode="prepare_missing", max_disk_bytes=1)
    assert result.status == "resource_limited" and result.reason == "quota_exceeded"
    assert result.prepared_entries == 0 and not list((path / "entries").iterdir())
    resumed = preprocessor.prepare_or_reuse(source, **options(path), mode="prepare_missing", max_disk_bytes=QUOTA)
    assert resumed.status == "complete" and resumed.repaired_entries == 1 and resumed.prepared_entries == 3


@pytest.mark.parametrize("changes,match", [
    ({"source_version": "v2"}, "source_version"), ({"preprocessing_version": "v2"}, "preprocessing_version"),
    ({"split": "evaluation"}, "split"), ({"sample_ids": ["b", "a", "c", "d"]}, "ID/order"),
])
def test_incompatible_contract_is_rejected_before_preparation(cache, experiment, monkeypatch, changes, match):
    preprocessor, _, source = experiment
    monkeypatch.setattr(CFPPreprocessor, "prepare_u8", forbid)
    with pytest.raises(ValueError, match=match):
        preprocessor.prepare_or_reuse(source, **(options(cache[0]) | changes), mode="prepare_missing", max_disk_bytes=QUOTA)


def test_reuse_rejects_incomplete_manifest_and_missing_files(cache, experiment):
    preprocessor, _, _ = experiment
    with sqlite3.connect(cache[0] / "manifest.sqlite3") as db:
        db.execute("UPDATE manifests SET state='cancelled'")
    with pytest.raises(RuntimeError, match="prepare_missing"):
        preprocessor.prepare_or_reuse(**options(cache[0]))
    with sqlite3.connect(cache[0] / "manifest.sqlite3") as db:
        db.execute("UPDATE manifests SET state='complete'")
    next((cache[0] / "entries").glob("*.pt")).unlink()
    with pytest.raises(RuntimeError, match="prepare_missing"):
        preprocessor.prepare_or_reuse(**options(cache[0]))


def test_reuse_is_metadata_check_and_dataset_still_checks_live_pixels(cache, experiment):
    preprocessor, _, source = experiment
    source.tensors[0][0].zero_()
    result = preprocessor.prepare_or_reuse(source, **options(cache[0]))
    assert result.status == "complete"
    with DiskStore(cache[0], readonly=True) as store:
        with pytest.raises(ValueError, match="pixels/shape changed"):
            PreparedDataset(store, "train", source)[0]


@pytest.mark.parametrize("bad", [
    {"mode": "rebuild"}, {"pilot_samples": -1}, {"pilot_samples": True}, {"disk_safety_factor": 0.9},
    {"disk_safety_factor": float("inf")}, {"min_free_disk_bytes": -1}, {"num_workers": -1},
    {"max_sample_bytes": 0}, {"max_in_flight": 0}, {"progress": 1}, {"cancel": 1},
])
def test_invalid_options_do_not_create_cache(tmp_path, experiment, bad):
    preprocessor, _, source = experiment
    path = tmp_path / "new"
    kwargs = options(path) | dict(mode="prepare_missing", max_disk_bytes=QUOTA) | bad
    with pytest.raises((TypeError, ValueError)):
        preprocessor.prepare_or_reuse(source, **kwargs)
    assert not path.exists()


def test_callback_failure_keeps_completed_payload_and_closes_writer(tmp_path, experiment):
    preprocessor, _, source = experiment
    path = tmp_path / "new"
    def progress(event):
        if event.get("position") == 0:
            raise RuntimeError("callback failed")
    with pytest.raises(RuntimeError, match="callback failed"):
        preprocessor.prepare_or_reuse(source, **options(path), mode="prepare_missing", max_disk_bytes=QUOTA, progress=progress)
    assert len(files(path)) == 1
    with DiskStore(path, max_disk_bytes=QUOTA):
        pass


def test_evaluation_never_fits_statistics(cache, experiment):
    preprocessor, layer, source = experiment
    with pytest.raises(ValueError, match="training snapshot"):
        preprocessor.prepare_or_reuse(source, **(options(cache[0]) | {"split": "evaluation"}),
            collect_stats=layer.get_statistics_contract())


def test_worker_serializer_enforces_free_space_reserve(tmp_path, experiment, monkeypatch):
    from mtlearn.layers.cfp.preparation._parallel_preparation import _write_entry
    from mtlearn.layers.cfp.preparation._identity import image_identity
    preprocessor, _, source = experiment
    image = source.tensors[0][0, 0].numpy()
    key = next(iter(preprocessor.tree_specs))
    temporary = tmp_path / "temporary.pt"
    entry = {"tree_key": key, "identity": image_identity(preprocessor, image, key),
             "temporary": str(temporary), "min_free_disk_bytes": 100}
    budget = SimpleNamespace(value=0, get_lock=contextlib.nullcontext)
    monkeypatch.setattr(_writer_lock.shutil, "disk_usage", lambda path: SimpleNamespace(free=100))
    with pytest.raises(OSError, match="free-space reserve"):
        _write_entry(preprocessor, image, entry, budget, QUOTA)
    assert budget.value == 0 and not temporary.exists()


def test_quota_writer_checks_the_next_write_against_reserve(monkeypatch):
    handle = io.BytesIO()
    free = [101]
    monkeypatch.setattr(_writer_lock.shutil, "disk_usage", lambda path: SimpleNamespace(free=free[0]))
    writer = _writer_lock._QuotaWriter(handle, 1000, min_free_disk_bytes=100, disk_path="unused")
    assert writer.write(b"x") == 1
    free[0] = 100
    with pytest.raises(OSError, match="reserve"):
        writer.write(b"y")
    assert writer.reserve_exceeded and not writer.exceeded and handle.getvalue() == b"x"


def test_reuse_rejects_manifest_changes_during_statistics_event(cache, experiment):
    preprocessor, layer, _ = experiment
    def progress(event):
        if event["phase"] == "statistics":
            with sqlite3.connect(cache[0] / "manifest.sqlite3") as db:
                db.execute("UPDATE manifests SET source_version='changed'")
    with pytest.raises(RuntimeError, match="Manifest changed"):
        preprocessor.prepare_or_reuse(**options(cache[0]), collect_stats=layer.get_statistics_contract(), progress=progress)


def test_cancel_during_pilot_is_not_treated_as_pilot_completion(tmp_path, experiment):
    preprocessor, _, source = experiment
    cancel = threading.Event()
    def progress(event):
        if event["phase"] == "pilot" and event["sample_count"] == 1:
            cancel.set()
    result = preprocessor.prepare_or_reuse(source, **options(tmp_path / "cache"), mode="prepare_missing",
        max_disk_bytes=QUOTA, pilot_samples=2, cancel=cancel, progress=progress)
    assert result.status == "cancelled" and result.prepared_entries == 1
    assert result.resources["pilot_samples"] == 1 and result.reason == "cancelled_by_user"


def test_reserve_drop_between_samples_preserves_completed_entry(tmp_path, experiment, monkeypatch):
    preprocessor, _, source = experiment
    free = [QUOTA]
    monkeypatch.setattr(flow, "_free_bytes", lambda path: free[0])
    def progress(event):
        if event["phase"] == "prepare" and event["sample_count"] == 1:
            free[0] = 9
    result = preprocessor.prepare_or_reuse(source, **options(tmp_path / "cache"), mode="prepare_missing",
        max_disk_bytes=QUOTA, min_free_disk_bytes=10, progress=progress)
    assert result.status == "resource_limited" and result.reason == "free_space_reserve"
    assert result.sample_count == result.prepared_entries == len(files(tmp_path / "cache")) == 1


@pytest.mark.skipif(os.name != "posix", reason="Session validation requires POSIX locks")
def test_complete_reuse_does_not_change_sqlite_or_acquire_writer_lock(cache, experiment):
    preprocessor, _, _ = experiment
    database = cache[0] / "manifest.sqlite3"
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    with DiskStore(cache[0], readonly=True, validation="session", immutable=True):
        result = preprocessor.prepare_or_reuse(**options(cache[0]))
    assert result.status == "complete"
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_existing_cache_inventory_is_preserved_in_resource_preflight(cache, experiment, monkeypatch):
    preprocessor, _, source = experiment
    monkeypatch.setattr(flow, "_free_bytes", lambda path: 100)
    result = preprocessor.prepare_or_reuse(source, **options(cache[0]), mode="prepare_missing",
        max_disk_bytes=QUOTA, min_free_disk_bytes=101)
    assert result.status == "resource_limited" and result.resources["disk_bytes"] > 0
    assert result.resources["disk_bytes_kind"] == "registered_valid_entries"
    assert result.resources["ready_samples"] == 4
