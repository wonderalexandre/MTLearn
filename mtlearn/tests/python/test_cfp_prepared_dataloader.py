import gc
import json
import os
from pathlib import Path
import sqlite3
import threading
import weakref

import pytest
import torch
from torch.utils.data import TensorDataset

from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer, load_checkpoint
from mtlearn.layers.cfp import CFPPreprocessor, DiskStore, build_prepared_dataloader, PreparedDataLoadingError
from mtlearn.layers.cfp.preparation import _threaded_reader as threaded

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).parent / "fixtures/cfp_cache_p0_mmcfilters_v5_2_0"
REFERENCE = json.loads((FIXTURES / "manifest.json").read_text())


@pytest.fixture
def cache(tmp_path):
    layer = Layer(1, [{"tree_type": "max-tree", "attributes": [
        morphology.AttributeType.AREA, morphology.AttributeType.GRAY_LEVEL_HEIGHT]}], scale_mode="none")
    image = torch.arange(42, dtype=torch.uint8).reshape(1, 6, 7)
    images = torch.stack([image] * 5)
    source = TensorDataset(images, images.float() / 255)
    path = tmp_path / "cache"
    CFPPreprocessor.from_layer(layer).prepare_or_reuse(source, path=path, manifest="train",
        source_version="v1", preprocessing_version="uint8", mode="prepare_missing", max_disk_bytes=1024**2)
    return path, layer, source


def loader_options(source=None, **kwargs):
    return dict(source=source, prefetch=True, max_prefetch_bytes=64*1024, max_batch_bytes=128*1024,
                max_prefetch_tasks=2, read_workspace_bytes=1024, source_memory_bytes=4096,
                source_thread_safe=True) | kwargs


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_loader_outputs_gradients_and_lifetime_match_p0(tmp_path, prefetch, mmap, device, name):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    baseline = torch.load(FIXTURES / "baseline.pt", weights_only=True)
    case = baseline["cases"][name]
    layer, _ = load_checkpoint(FIXTURES / case["checkpoint"],
        lambda configs: Layer.from_config(configs[""], device=device), device=device)
    source = TensorDataset(baseline["images"][3:], baseline["targets"][3:])
    CFPPreprocessor.from_layer(layer).prepare_or_reuse(source, path=tmp_path, manifest="test",
        source_version="v1", preprocessing_version="v1", split="evaluation", mode="prepare_missing", max_disk_bytes=1024**2)
    with build_prepared_dataloader(tmp_path, "test", batch_size=2, mmap=mmap,
                                  **loader_options(source, prefetch=prefetch,
                                      max_prefetch_bytes=512*1024, max_batch_bytes=512*1024)) as loader:
        iterator = iter(loader)
        batch, target = next(iterator)
        reference = weakref.ref(batch)
        output = layer(batch)
        del batch
        assert iterator.close()
        assert reference() is not None and loader.counters()["active_bytes"] > 0
        loss = ((output / 255 - target.to(device))**2).mean()
        loss.backward(retain_graph=True)
        assert reference() is not None
        for parameter in layer.parameters():
            parameter.grad = None
        loss.backward()
        assert reference() is None
        tolerance = REFERENCE["cpu_mps_tolerance" if device == "mps" else "same_device_tolerance"]
        torch.testing.assert_close(output.cpu(), case["output"], **tolerance)
        torch.testing.assert_close(loss.cpu(), case["loss"], **tolerance)
        for key, parameter in layer.named_parameters():
            torch.testing.assert_close(parameter.grad.cpu(), case["gradients"][key], **tolerance)


@pytest.mark.parametrize("prefetch", [False, True])
def test_order_short_batch_multiple_epochs_and_rng(cache, prefetch):
    path, _, source = cache
    with build_prepared_dataloader(path, "train", batch_size=2, sampler=[3, 1, 4, 0, 2],
                                  **loader_options(source, prefetch=prefetch)) as loader:
        assert len(loader) == 3
        for _ in range(2):
            found = []
            for batch, targets in loader:
                found.extend(batch.sample_ids)
                torch.testing.assert_close(targets, torch.stack([source[int(i)][1] for i in batch.sample_ids]))
            assert found == ["3", "1", "4", "0", "2"]
            assert batch.shape[0] == 1
        assert not loader.counters()["thread_alive"]


def test_prefetch_preserves_training_rng_with_dedicated_shuffle_generator(cache):
    path, _, source = cache
    captures = []
    for prefetch in (False, True):
        torch.manual_seed(55)
        with build_prepared_dataloader(path, "train", shuffle=True,
                generator=torch.Generator().manual_seed(77), **loader_options(source, prefetch=prefetch)) as loader:
            captures.append([(batch.sample_ids, torch.rand(4), torch.get_rng_state()) for batch, _ in loader])
    for first, second in zip(*captures):
        assert first[0] == second[0]
        assert torch.equal(first[1], second[1]) and torch.equal(first[2], second[2])


def test_slow_consumer_limits_queue_bytes_and_tasks(cache):
    path, _, source = cache
    with build_prepared_dataloader(path, "train", **loader_options(source)) as loader:
        iterator = iter(loader)
        first = next(iterator)
        with iterator.state.condition:
            assert iterator.state.condition.wait_for(lambda: len(iterator.state.results) == 2, timeout=5)
        info = loader.counters()
        assert info["queued_batches"] == 2 and info["building_batches"] == 0
        assert info["peak_prefetch_tasks"] == 2
        assert 0 < info["prefetch_reserved_bytes"] <= 64*1024
        assert info["queued_bytes"] > 0 and info["active_bytes"] > 0
        assert iterator.close()
        closed = loader.counters()
        assert closed["prefetch_reserved_bytes"] == closed["queued_bytes"] == 0
        assert not closed["thread_alive"] and first[0].shape == (1, 1, 6, 7)


def test_shared_storages_are_counted_once_across_active_queued_and_ram(cache):
    path, _, source = cache
    with build_prepared_dataloader(path, "train", max_ram_bytes=64*1024, **loader_options(source)) as loader:
        iterator = iter(loader)
        first = next(iterator)
        with iterator.state.condition:
            assert iterator.state.condition.wait_for(lambda: len(iterator.state.results) == 2, timeout=5)
        info = loader.counters()
        raw = next(iter(first[0].samples[0][0].values())).nbytes
        target_bytes = first[1].untyped_storage().nbytes()
        assert info["retained_bytes"] == raw
        assert info["known_tensor_bytes"] == raw + 3 * target_bytes
        assert info["active_bytes"] == raw + target_bytes


def test_oversized_batches_drain_and_use_caller_thread(cache):
    path, _, source = cache
    threads = []
    class Source:
        def __len__(self):
            return len(source)
        def __getitem__(self, index):
            threads.append(threading.get_ident())
            return source[index]
    with build_prepared_dataloader(path, "train", **loader_options(Source(), max_prefetch_bytes=1)) as loader:
        for batch, target in loader:
            pass
        assert loader.counters()["sync_fallback_batches"] == 5
        assert loader.counters()["peak_prefetch_reserved_bytes"] == 0
    assert set(threads) == {threading.get_ident()}


def test_mixed_prefetch_and_oversized_batch_preserve_order(cache):
    path, _, source = cache
    def memory(index):
        return 30000 if index == 2 else 4096
    with build_prepared_dataloader(path, "train", **loader_options(source,
            max_prefetch_bytes=30000, source_memory_bytes=memory)) as loader:
        assert [batch.sample_ids[0] for batch, target in loader] == ["0", "1", "2", "3", "4"]
        assert loader.counters()["sync_fallback_batches"] == 1


def test_batch_limit_fails_before_loading_source(cache):
    path, _, source = cache
    class Source:
        def __len__(self):
            return len(source)
        def __getitem__(self, index):
            raise AssertionError("Source must not be loaded")
    with build_prepared_dataloader(path, "train", **loader_options(Source(), max_batch_bytes=1)) as loader:
        with pytest.raises(PreparedDataLoadingError, match="max_batch_bytes"):
            next(iter(loader))
        assert not loader.counters()["thread_alive"]


def test_source_exceeding_declared_storage_is_rejected(cache):
    with build_prepared_dataloader(cache[0], "train", **loader_options(cache[2], source_memory_bytes=1)) as loader:
        with pytest.raises(PreparedDataLoadingError, match="source_memory_bytes"):
            next(iter(loader))
        assert loader.counters()["prefetch_reserved_bytes"] == 0


def test_untrusted_thread_source_and_lazy_sampler_fall_back_to_sync(cache):
    path, _, source = cache
    with build_prepared_dataloader(path, "train", source=source, prefetch=True) as loader:
        assert not loader.prefetch and loader.prefetch_reason == "source_not_declared_thread_safe"
        assert len(list(loader)) == 5
    class Sampler:
        def __iter__(self):
            yield from (3, 0)
        def __len__(self):
            return 2
    with build_prepared_dataloader(path, "train", sampler=Sampler(), prefetch=True) as loader:
        assert not loader.prefetch and "sampler" in loader.prefetch_reason
        assert [batch.sample_ids[0] for batch in loader] == ["3", "0"]


def test_reader_opens_and_closes_sqlite_in_its_own_thread(cache, monkeypatch):
    events = []
    original_init, original_close = DiskStore.__init__, DiskStore.close
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        events.append((id(self), "open", threading.get_ident()))
    def close(self):
        if self._db is not None:
            events.append((id(self), "close", threading.get_ident()))
        return original_close(self)
    monkeypatch.setattr(DiskStore, "__init__", initialize)
    monkeypatch.setattr(DiskStore, "close", close)
    with build_prepared_dataloader(cache[0], "train", **loader_options(cache[2])) as loader:
        list(loader)
    readers = {}
    for key, action, thread in events:
        readers.setdefault(key, []).append((action, thread))
    assert any(thread != threading.get_ident() for key, action, thread in events)
    for values in readers.values():
        assert values[0][0] == "open" and values[-1][0] == "close"
        assert len({thread for action, thread in values}) == 1


def test_close_during_blocked_io_reports_pending_and_releases_afterward(cache):
    path, _, source = cache
    entered, release = threading.Event(), threading.Event()
    class Source:
        def __len__(self):
            return len(source)
        def __getitem__(self, index):
            if index == 1:
                entered.set()
                assert release.wait(5)
            return source[index]
    loader = build_prepared_dataloader(path, "train", close_timeout=0.01, **loader_options(Source()))
    iterator = iter(loader)
    try:
        first = next(iterator)
        assert entered.wait(5)
        assert not loader.close()
        assert loader.counters()["thread_alive"] and loader.counters()["building_reserved_bytes"] > 0
        with pytest.raises(RuntimeError):
            iter(loader)
    finally:
        release.set()
        iterator.state.thread.join(5)
        loader.close()
    assert not loader.counters()["thread_alive"]
    assert loader.counters()["prefetch_reserved_bytes"] == 0
    assert first[0].sample_ids == ("0",)


def test_source_error_is_delivered_in_order_without_retaining_exception_payloads(cache):
    path, _, source = cache
    refs = []
    class Source:
        def __len__(self):
            return len(source)
        def __getitem__(self, index):
            if index == 2:
                temporary = torch.zeros(100)
                refs.append(weakref.ref(temporary))
                raise ValueError("injected read error")
            return source[index]
    with build_prepared_dataloader(path, "train", **loader_options(Source())) as loader:
        iterator = iter(loader)
        assert next(iterator)[0].sample_ids == ("0",)
        assert next(iterator)[0].sample_ids == ("1",)
        with pytest.raises(PreparedDataLoadingError, match="'2'.*injected read error") as error:
            next(iterator)
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert not loader.counters()["thread_alive"]


def test_break_context_closes_queue_and_backward_survives(cache):
    path, layer, source = cache
    with build_prepared_dataloader(path, "train", **loader_options(source)) as loader:
        for batch, target in loader:
            output = layer(batch)
            reference = weakref.ref(batch)
            break
    del batch
    assert not loader.counters()["thread_alive"] and loader.counters()["prefetch_reserved_bytes"] == 0
    assert reference() is not None
    output.sum().backward()
    assert reference() is None


def test_previous_epoch_active_buffers_remain_counted(cache):
    path, layer, source = cache
    with build_prepared_dataloader(path, "train", sampler=[0], **loader_options(source)) as loader:
        first = list(loader)[0]
        before = loader.counters()["active_bytes"]
        second = list(loader)[0]
        assert loader.counters()["active_bytes"] > before
        del first, second
        gc.collect()
        assert loader.counters()["active_bytes"] == 0


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("validation", ["always", "session"])
def test_snapshot_without_source_drop_last_and_session(cache, prefetch, validation):
    with build_prepared_dataloader(cache[0], "train", batch_size=2, drop_last=True,
            validation=validation, immutable=validation == "session",
            **loader_options(prefetch=prefetch)) as loader:
        assert len(loader) == 2
        assert [batch.sample_ids for batch in loader] == [("0", "1"), ("2", "3")]
        assert loader.counters()["checksum_reads"] == (1 if validation == "session" else 4)


@pytest.mark.parametrize("failure", ["remove", "corrupt", "sample_metadata"])
def test_changed_entry_after_scheduling_fails_in_order(cache, monkeypatch, failure):
    path, _, source = cache
    original = threaded.load_batch
    def change_then_load(dataset, description):
        if description["indices"] == (2,):
            row = dataset.store._db.execute("SELECT bindings FROM samples WHERE manifest='train' AND position=2").fetchone()
            key = next(iter(json.loads(row[0])[0].values()))
            entry = dataset.store._entry_path(key)
            if failure == "remove":
                entry.unlink()
            elif failure == "corrupt":
                with entry.open("r+b") as stream:
                    stream.seek(-1, 2)
                    stream.write(b"x")
            else:
                with sqlite3.connect(path / "manifest.sqlite3") as db:
                    db.execute("UPDATE samples SET sample_id='changed' WHERE manifest='train' AND position=2")
        return original(dataset, description)
    monkeypatch.setattr(threaded, "load_batch", change_then_load)
    with build_prepared_dataloader(path, "train", **loader_options(source, max_prefetch_tasks=1)) as loader:
        iterator = iter(loader)
        assert next(iterator)[0].sample_ids == ("0",)
        assert next(iterator)[0].sample_ids == ("1",)
        with pytest.raises(PreparedDataLoadingError, match="'2'"):
            next(iterator)
        assert not loader.counters()["thread_alive"]
        assert loader.counters()["prefetch_reserved_bytes"] == 0


@pytest.mark.parametrize("interrupt", [False, True])
def test_abandoned_iterator_and_keyboard_interrupt_close_reader(cache, interrupt):
    loader = build_prepared_dataloader(cache[0], "train", **loader_options(cache[2]))
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            with loader:
                for value in loader:
                    raise KeyboardInterrupt
    else:
        iterator = iter(loader)
        next(iterator)
        with pytest.raises(RuntimeError, match="current prepared iterator"):
            iter(loader)
        del iterator
        gc.collect()
        assert loader.counters()["reader_closed"]
        assert len(list(loader)) == 5
        loader.close()
    assert not loader.counters()["thread_alive"]
    assert loader.counters()["prefetch_reserved_bytes"] == 0


@pytest.mark.parametrize("scoring", ["linear_sigmoid", "mlp"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_training_steps_match_synchronous_reader(cache, scoring, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    trajectories = []
    for prefetch in (False, True):
        torch.manual_seed(543)
        layer = Layer(1, [{"tree_type": "max-tree", "attributes": [
            morphology.AttributeType.AREA, morphology.AttributeType.GRAY_LEVEL_HEIGHT],
            "scoring": {"kind": scoring}}], scale_mode="none", device=device)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
        states = []
        with build_prepared_dataloader(cache[0], "train", shuffle=True,
                generator=torch.Generator().manual_seed(432), **loader_options(cache[2], prefetch=prefetch)) as loader:
            for _ in range(2):
                for batch, target in loader:
                    optimizer.zero_grad(set_to_none=True)
                    output = layer(batch)
                    loss = ((output / 255 - target.to(device)) ** 2).mean()
                    loss.backward()
                    optimizer.step()
                    states.append((batch.sample_ids, loss.detach().cpu().clone(),
                        [p.detach().cpu().clone() for p in layer.parameters()],
                        [(s["step"].cpu().clone(), s["exp_avg"].cpu().clone(), s["exp_avg_sq"].cpu().clone())
                         for s in optimizer.state.values()], torch.get_rng_state()))
        trajectories.append(states)
    tolerance = REFERENCE["cpu_mps_tolerance" if device == "mps" else "same_device_tolerance"]
    for left, right in zip(*trajectories):
        assert left[0] == right[0] and torch.equal(left[-1], right[-1])
        torch.testing.assert_close(left[1:4], right[1:4], **tolerance)


@pytest.mark.parametrize("options", [
    {"prefetch": True}, {"max_prefetch_bytes": -1}, {"max_prefetch_tasks": 0}, {"max_batch_bytes": 0},
    {"read_workspace_bytes": -1}, {"source_memory_bytes": -1}, {"source_thread_safe": 1},
    {"close_timeout": float("inf")}, {"close_timeout": -1},
    {"batch_size": None}, {"batch_size": 0},
])
def test_invalid_loader_options(cache, options):
    with pytest.raises(ValueError):
        build_prepared_dataloader(cache[0], "train", **options)
