import gc
import json
import os
import pickle
from pathlib import Path
import time
import weakref

import pytest
import torch
from torch.utils.data import Subset, TensorDataset

from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer, load_checkpoint
from mtlearn.layers.cfp import CFPPreprocessor, build_prepared_dataloader, PreparedDataLoadingError
from mtlearn.layers.cfp.preparation._shared_batch import SharedBatchLease, write_shared_batch, read_shared_batch
from mtlearn.layers.cfp.preparation import _process_reader
from test_cfp_prepared_dataloader import cache, loader_options, FIXTURES, REFERENCE

pytestmark = pytest.mark.integration


class Source:
    def __init__(self, count=5, delay_index=-1, fail_index=-1, crash_index=-1, marker=None):
        image = torch.arange(42, dtype=torch.uint8).reshape(1, 6, 7)
        images = torch.stack([image] * count)
        self.dataset = TensorDataset(images, images.float() / 255)
        self.delay_index, self.fail_index, self.crash_index = delay_index, fail_index, crash_index
        self.marker = marker

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.marker:
            with open(self.marker, "a") as stream:
                stream.write(f"{os.getpid()} {index}\n")
        if index == self.crash_index:
            os._exit(23)
        if index == self.delay_index:
            time.sleep(1 if self.marker is None else 30)
        if index == self.fail_index:
            raise ValueError("injected source failure")
        return self.dataset[index]


def source_factory(**kwargs):
    return Source(**kwargs)


def baseline_source(path, selection=None):
    saved = torch.load(path, weights_only=True)
    source = TensorDataset(saved["images"][3:], saved["targets"][3:])
    return source if selection is None else Subset(source, selection)


def options(**kwargs):
    return loader_options(source_factory=source_factory, num_workers=2, worker_memory_bytes=256*1024**2,
                          max_prefetch_bytes=4*1024**2, max_batch_bytes=4*1024**2) | kwargs


def test_shared_storage_transport_preserves_views_and_lifetime():
    lease = SharedBatchLease(4096)
    reference = weakref.ref(lease)
    value = torch.arange(30, dtype=torch.int64).reshape(5, 6)
    packet = write_shared_batch((value, value[:, 1:], value.to(torch.uint32)), lease.memory)
    assert len(packet["storages"]) == 2
    first, second, unsigned = read_shared_batch(packet, lease, None, ())
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    assert unsigned.dtype == torch.uint32
    torch.testing.assert_close(unsigned, value.to(torch.uint32), rtol=0, atol=0)
    lease.unlink()
    view = second[1:]
    del first, second, unsigned, lease
    gc.collect()
    assert reference() is not None and torch.equal(view, value[1:, 1:])
    del view
    assert reference() is None


def test_inference_releases_shared_batches_without_cyclic_gc(cache):
    enabled = gc.isenabled()
    gc.disable()
    try:
        with build_prepared_dataloader(cache[0], "train", **options()) as loader:
            for _ in range(3):
                for batch, target in loader:
                    with torch.no_grad():
                        output = cache[1](batch)
                    del batch, target, output
                info = loader.counters()
                assert info["active_bytes"] == info["shared_pending_bytes"] == info["shared_active_bytes"] == 0
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("mmap", [False, True])
def test_processes_order_multiple_epochs_short_batch_and_global_limits(cache, workers, mmap):
    with build_prepared_dataloader(cache[0], "train", batch_size=2, sampler=[3, 1, 4, 0, 2], mmap=mmap,
            max_ram_bytes=10001, **options(num_workers=workers)) as loader:
        for _ in range(2):
            ids = []
            for batch, target in loader:
                ids.extend(batch.sample_ids)
                torch.testing.assert_close(target, torch.stack([cache[2][int(i)][1] for i in batch.sample_ids]))
            assert ids == ["3", "1", "4", "0", "2"]
            assert batch.shape[0] == 1
            info = loader.counters()
            assert info["processes_alive"] == 0
            assert sum(info["worker_ram_limits"]) == 10001
            assert info["worker_base_reserved_bytes"] == workers * 256*1024**2
            assert info["peak_prefetch_reserved_bytes"] <= 4*1024**2
            assert info["peak_prefetch_tasks"] <= 2
            assert info["shared_pending_bytes"] == info["prefetch_reserved_bytes"] == 0
        del batch, target
        gc.collect()
        assert loader.counters()["shared_active_bytes"] == 0


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_process_outputs_gradients_lifetime_match_p0(tmp_path, workers, device, name):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    baseline = torch.load(FIXTURES / "baseline.pt", weights_only=True)
    case = baseline["cases"][name]
    layer, _ = load_checkpoint(FIXTURES / case["checkpoint"],
        lambda configs: Layer.from_config(configs[""], device=device), device=device)
    source = baseline_source(FIXTURES / "baseline.pt", [1, 0, 1])
    CFPPreprocessor.from_layer(layer).prepare_or_reuse(source, path=tmp_path, manifest="test",
        source_version="v1", preprocessing_version="v1", split="evaluation", mode="prepare_missing", max_disk_bytes=1024**2)
    with build_prepared_dataloader(tmp_path, "test", batch_size=2, sampler=[1, 0],
            **options(num_workers=workers, source_factory=baseline_source,
                source_factory_kwargs={"path": str(FIXTURES / "baseline.pt"), "selection": [1, 0, 1]})) as loader:
        iterator = iter(loader)
        batch, target = next(iterator)
        output = layer(batch)
        reference = weakref.ref(batch)
        assert iterator.close()
        del batch
        assert reference() is not None and loader.counters()["shared_active_bytes"] > 0
        loss = ((output / 255 - target.to(device)) ** 2).mean()
        loss.backward(retain_graph=True)
        for parameter in layer.parameters():
            parameter.grad = None
        loss.backward()
        assert reference() is None
        tolerance = REFERENCE["cpu_mps_tolerance" if device == "mps" else "same_device_tolerance"]
        torch.testing.assert_close(output.cpu(), case["output"], **tolerance)
        torch.testing.assert_close(loss.cpu(), case["loss"], **tolerance)
        for key, parameter in layer.named_parameters():
            torch.testing.assert_close(parameter.grad.cpu(), case["gradients"][key], **tolerance)


def test_slow_consumer_out_of_order_completion_and_reservations(cache):
    with build_prepared_dataloader(cache[0], "train", **options(source_factory_kwargs={"delay_index": 0})) as loader:
        iterator = iter(loader)
        iterator._fill()
        completion = []
        deadline = time.monotonic() + 20
        while len(iterator.state.results) < 2 and time.monotonic() < deadline:
            iterator.processes.poll(0.05)
            for key in iterator.state.results:
                if key not in completion:
                    completion.append(key)
        assert completion == [1, 0]
        info = loader.counters()
        assert info["queued_batches"] == 2 and info["building_batches"] == 0
        assert info["shared_pending_bytes"] > 0 and info["shared_pending_bytes"] < info["prefetch_reserved_bytes"]
        assert [batch.sample_ids[0] for batch, target in iterator] == ["0", "1", "2", "3", "4"]


@pytest.mark.parametrize("kind", ["error", "crash"])
def test_worker_failure_keeps_prefix_and_closes_resources(cache, kind):
    with build_prepared_dataloader(cache[0], "train", **options(source_factory_kwargs={
            "fail_index" if kind == "error" else "crash_index": 2})) as loader:
        iterator = iter(loader)
        assert next(iterator)[0].sample_ids == ("0",)
        assert next(iterator)[0].sample_ids == ("1",)
        with pytest.raises(PreparedDataLoadingError):
            next(iterator)
        assert loader.counters()["processes_alive"] == 0
        assert loader.counters()["shared_pending_bytes"] == loader.counters()["prefetch_reserved_bytes"] == 0


def test_nonimportable_factory_falls_back_without_spawning(cache):
    with build_prepared_dataloader(cache[0], "train", **options(source_factory=lambda: cache[2])) as loader:
        assert loader.num_workers == 0 and not loader.prefetch
        assert loader.prefetch_reason == "source_factory_not_serializable"
        assert len(list(loader)) == 5


def test_closed_iterator_releases_factory_source_in_synchronous_fallback(cache):
    references = []
    def factory():
        source = source_factory()
        references.append(weakref.ref(source))
        return source
    with build_prepared_dataloader(cache[0], "train", **options(source_factory=factory)) as loader:
        iterator = iter(loader)
        next(iterator)
        assert references[0]() is not None
        assert iterator.close()
        assert references[0]() is None


def test_source_object_without_factory_is_not_copied(cache):
    with build_prepared_dataloader(cache[0], "train", **options(source=cache[2], source_factory=None)) as loader:
        assert loader.num_workers == 0 and loader.prefetch_reason == "source_factory_required"
        assert len(list(loader)) == 5


def test_parent_source_is_not_read_or_serialized_when_factory_is_available(cache):
    class ParentSource:
        def __len__(self):
            return 5
        def __getitem__(self, index):
            raise AssertionError("Parent source must not be read")
        def __reduce__(self):
            raise AssertionError("Parent source must not be serialized")
    with build_prepared_dataloader(cache[0], "train", **options(source=ParentSource())) as loader:
        assert len(list(loader)) == 5


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("scoring", ["linear_sigmoid", "mlp"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_process_training_matches_synchronous_order_rng_and_optimizer(cache, workers, scoring, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    from mtlearn import morphology
    captures = []
    for count in (0, workers):
        torch.manual_seed(543)
        layer = Layer(1, [{"tree_type": "max-tree", "attributes": [
            morphology.AttributeType.AREA, morphology.AttributeType.GRAY_LEVEL_HEIGHT],
            "scoring": {"kind": scoring}}], scale_mode="none", device=device)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
        states = []
        with build_prepared_dataloader(cache[0], "train", shuffle=True,
                generator=torch.Generator().manual_seed(432), **options(num_workers=count, prefetch=False)) as loader:
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
        captures.append(states)
    tolerance = REFERENCE["cpu_mps_tolerance" if device == "mps" else "same_device_tolerance"]
    for first, second in zip(*captures):
        assert first[0] == second[0] and torch.equal(first[-1], second[-1])
        torch.testing.assert_close(first[1:4], second[1:4], **tolerance)


def test_oversized_process_batch_runs_synchronously(cache):
    with build_prepared_dataloader(cache[0], "train", **options(max_prefetch_bytes=1)) as loader:
        assert len(list(loader)) == 5
        assert loader.counters()["sync_fallback_batches"] == 5
        assert loader.counters()["peak_prefetch_reserved_bytes"] == 0


def test_decode_failure_releases_shared_memory(cache, monkeypatch):
    original = _process_reader.read_shared_batch
    def fail(packet, lease, preprocessor, ids):
        if ids == ("1",):
            packet["used_bytes"] = lease.size + 1
        return original(packet, lease, preprocessor, ids)
    monkeypatch.setattr(_process_reader, "read_shared_batch", fail)
    with build_prepared_dataloader(cache[0], "train", **options()) as loader:
        iterator = iter(loader)
        assert next(iterator)[0].sample_ids == ("0",)
        with pytest.raises(PreparedDataLoadingError, match="shared batch size"):
            next(iterator)
        assert loader.counters()["shared_pending_bytes"] == 0
        assert loader.counters()["processes_alive"] == 0


def test_ipc_deserialization_error_has_context_and_closes_workers(cache, monkeypatch):
    original = _process_reader.receive
    def invalid(connection):
        message = original(connection)
        if message["kind"] == "result":
            raise pickle.UnpicklingError("injected malformed message")
        return message
    monkeypatch.setattr(_process_reader, "receive", invalid)
    with build_prepared_dataloader(cache[0], "train", **options(num_workers=1)) as loader:
        with pytest.raises(PreparedDataLoadingError, match="'0'.*UnpicklingError"):
            next(iter(loader))
        assert loader.counters()["processes_alive"] == loader.counters()["shared_pending_bytes"] == 0


def test_close_stuck_reader_keeps_active_backward_buffers(cache, tmp_path):
    marker = tmp_path / "reads.txt"
    loader = build_prepared_dataloader(cache[0], "train", close_timeout=0.1,
        **options(source_factory_kwargs={"delay_index": 1, "marker": str(marker)}))
    with loader:
        iterator = iter(loader)
        batch, target = next(iterator)
        output = cache[1](batch)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not any(line.endswith(" 1") for line in marker.read_text().splitlines()):
            time.sleep(0.01)
        assert any(line.endswith(" 1") for line in marker.read_text().splitlines())
        reference = weakref.ref(batch)
        assert iterator.close()
        assert loader.counters()["processes_alive"] == 0
        assert loader.counters()["forced_worker_stops"] >= 1
        del batch
        output.sum().backward()
        assert reference() is None
    del output, target
    gc.collect()
    assert loader.counters()["shared_active_bytes"] == 0


@pytest.mark.parametrize("bad", [{"num_workers": 3}, {"num_workers": -1}, {"worker_threads": 0},
    {"worker_memory_bytes": -1}, {"worker_memory_bytes": None}, {"source_factory": 1},
    {"source_factory_kwargs": []}])
def test_invalid_process_options(cache, bad):
    with pytest.raises(ValueError):
        build_prepared_dataloader(cache[0], "train", **options(**bad))
