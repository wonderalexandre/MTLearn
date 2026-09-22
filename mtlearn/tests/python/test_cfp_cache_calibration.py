import copy
import hashlib
import json
from pathlib import Path
import random
import threading
import weakref
from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mtlearn.layers.cfp import CFPPreprocessor, DiskStore, calibrate_disk_cache
from mtlearn.layers.cfp.preparation import calibration as calibration_module
from mtlearn.layers.cfp.preparation import _calibration_measurement as measurement
from mtlearn.layers.cfp.preparation import _read_diagnostics as diagnostics
from test_cfp_prepared_dataloader import cache

pytestmark = pytest.mark.integration


def options(**kwargs):
    return dict(max_samples=3, max_seconds=60, max_memory_bytes=1024**3,
                max_cache_read_bytes=32*1024**2, read_workspace_bytes=4096) | kwargs


def forbidden(*args, **kwargs):
    raise AssertionError("Calibration cannot prepare, update statistics, or write payloads")


def snapshot(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file() and p.name not in {'.writer.lock', 'manifest.sqlite3-shm', 'manifest.sqlite3-wal'}}


def test_read_only_report_equivalence_rng_and_release(cache, monkeypatch):
    path, layer, _ = cache
    before = snapshot(path)
    torch.manual_seed(501)
    random.seed(32)
    np.random.seed(75)
    rng = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
    parameters = copy.deepcopy(layer.state_dict())
    monkeypatch.setattr(CFPPreprocessor, 'prepare_u8', forbidden)
    monkeypatch.setattr(DiskStore, 'statistics', forbidden)
    monkeypatch.setattr(DiskStore, 'put', forbidden)
    batches = []
    original = measurement.batch_digest
    def tracked(batch, check):
        batches.append(weakref.ref(batch))
        return original(batch, check)
    monkeypatch.setattr(measurement, 'batch_digest', tracked)
    events = []
    report = calibrate_disk_cache(path, 'train', progress=events.append,
        **options(modes=('buffered', 'thread', 'ram'), ram_cache_bytes=1024**2))
    assert report['status'] == 'complete'
    assert len(events) == 4 * 4 * 2
    assert len(report['samples']) == 3 and report['metadata_samples_inspected'] == 5
    assert report['reserved_cache_read_bytes'] <= 32*1024**2
    assert report['physical_disk_read_bytes'] is None and report['device_available_memory_bytes'] is None
    assert report['recommendation']['scope'] == 'prepared_payload_read_and_touch'
    expected = None
    for candidate in report['candidates']:
        assert candidate['status'] == 'complete'
        assert candidate['wall_seconds']['count'] == 3
        for trial in candidate['trials']:
            assert trial['equivalent'] and trial['reader_closed']
            expected = trial['signatures'] if expected is None else expected
            assert trial['signatures'] == expected
            assert trial['touched_tensor_bytes'] > 0
            counters = trial['counters']
            assert counters['checksum_bytes'] == counters['logical_load_bytes'] > 0
            assert counters['active_bytes'] == counters['queued_bytes'] == counters['retained_bytes'] == 0
    assert not any(reference() for reference in batches)
    assert snapshot(path) == before
    assert torch.equal(torch.get_rng_state(), rng[0]) and random.getstate() == rng[1]
    np.testing.assert_equal(np.random.get_state(), rng[2])
    for name, value in layer.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
        else:
            assert value == parameters[name]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('workers', [1, 2])
def test_process_metrics_equivalence_and_cleanup(cache, workers):
    path, _, _ = cache
    report = calibrate_disk_cache(path, 'train', **options(modes=(f'process{workers}',),
        worker_memory_bytes=256*1024**2, repeats=3, warmup_rounds=0))
    assert report['status'] == 'complete'
    candidate = report['candidates'][1]
    assert candidate['status'] == 'complete'
    for trial in candidate['trials']:
        counters = trial['counters']
        assert trial['equivalent']
        assert counters['processes_alive'] == counters['shared_pending_bytes'] == counters['shared_active_bytes'] == 0
        assert len(counters['worker_pids']) == workers
        assert 1 <= len(counters['worker_memory_observations']) <= workers
        assert counters['checksum_bytes'] == counters['logical_load_bytes'] > 0
        for observation in counters['worker_memory_observations']:
            assert observation['pid'] in counters['worker_pids']
            assert observation['cpu_seconds'] > 0
            for metric in ('rss_bytes', 'lifetime_peak_rss_bytes'):
                if observation[metric] is not None:
                    assert observation[metric] > 0


@pytest.mark.parametrize('kwargs', [dict(max_samples=0), dict(max_samples=65), dict(max_samples=True),
    dict(max_seconds=0), dict(max_seconds=float('nan')), dict(max_memory_bytes=0), dict(max_cache_read_bytes=0),
    dict(repeats=0), dict(repeats=11), dict(warmup_rounds=3), dict(read_workspace_bytes=-1),
    dict(metadata_samples=2), dict(sample_indices=iter([0])), dict(sample_indices=[]),
    dict(sample_indices=[0, 1, 2, 3]), dict(sample_indices=range(10**12)), dict(sample_indices=[-1]), dict(sample_indices=[5]),
    dict(modes=('unknown',)), dict(modes=('thread', 'thread')), dict(modes=('ram',)),
    dict(modes=('process1',)), dict(worker_memory_bytes=0), dict(max_rss_bytes=0),
    dict(progress=False), dict(cancel=False)])
def test_invalid_input_before_payload(cache, monkeypatch, kwargs):
    monkeypatch.setattr(DiskStore, '_load_file', forbidden)
    with pytest.raises((ValueError, TypeError)):
        calibrate_disk_cache(cache[0], 'train', **options(**kwargs))


@pytest.mark.parametrize('kwargs,reason', [(dict(max_memory_bytes=1), 'baseline_memory_budget'),
    (dict(max_cache_read_bytes=1), 'cache_read_budget'), (dict(max_seconds=1), 'time_limit')])
def test_limits_prevent_payload_read(cache, monkeypatch, kwargs, reason):
    if reason == 'time_limit':
        monkeypatch.setattr(calibration_module, 'time', SimpleNamespace(monotonic=count().__next__))
    monkeypatch.setattr(DiskStore, '_load_file', forbidden)
    report = calibrate_disk_cache(cache[0], 'train', **options(**kwargs))
    assert report['status'] == 'stopped' and report['reason'] == reason
    assert not report['recommendation']['validated']


def test_cancel_during_payload_touch_closes_reader(cache, monkeypatch):
    event = threading.Event()
    original = measurement.batch_digest
    def cancel(batch, check):
        event.set()
        return original(batch, check)
    monkeypatch.setattr(measurement, 'batch_digest', cancel)
    report = calibrate_disk_cache(cache[0], 'train', cancel=event, **options())
    assert report['status'] == 'stopped' and report['reason'] == 'cancelled'
    trial = report['candidates'][0]['trials'][0]
    assert trial['reader_closed'] and trial['counters']['active_bytes'] == 0


def test_explicit_order_duplicates_allow_real_ram_hits(cache):
    report = calibrate_disk_cache(cache[0], 'train', **options(sample_indices=[4, 1, 4],
        modes=('ram',), ram_cache_bytes=1024**2, repeats=1, warmup_rounds=0))
    assert [s['position'] for s in report['samples']] == [4, 1, 4]
    assert report['candidates'][1]['trials'][0]['counters']['ram_hits'] >= 2
    assert report['recommendation']['mode'] == 'synchronous' and not report['recommendation']['validated']


def test_payload_mismatch_rejects_recommendation(cache, monkeypatch):
    original = calibration_module.measure
    calls = 0
    def altered(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            result['signatures'][0] = 'different'
        return result
    monkeypatch.setattr(calibration_module, 'measure', altered)
    report = calibrate_disk_cache(cache[0], 'train', **options(modes=('buffered',)))
    assert report['reason'] == 'payload_equivalence_failed'
    assert not report['recommendation']['validated']


def test_metadata_selection_includes_size_quantiles(cache, monkeypatch):
    original = calibration_module.describe
    def sizes(store, dataset, indices, *args):
        result = original(store, dataset, indices, *args)
        result['payload_bytes'] += (indices[0] + 1) * 1024
        return result
    monkeypatch.setattr(calibration_module, 'describe', sizes)
    report = calibrate_disk_cache(cache[0], 'train', **options(modes=(), max_cache_read_bytes=1))
    assert [s['position'] for s in report['samples']] == [0, 2, 4]


def test_optional_memory_dependency_and_rss_guard(cache, monkeypatch):
    monkeypatch.setattr(diagnostics, 'psutil', None)
    assert diagnostics.available_memory() is None and diagnostics.process_memory()['rss_bytes'] is None
    report = calibrate_disk_cache(cache[0], 'train', **options(modes=('buffered',), max_rss_bytes=1024**3))
    assert report['reason'] == 'rss_unavailable'
    assert not report['recommendation']['validated']


@pytest.mark.parametrize('rss_limit', [1, 2])
def test_observed_rss_limit_stops_before_load(cache, monkeypatch, rss_limit):
    monkeypatch.setattr(measurement, 'process_memory', lambda: {'rss_bytes': 3})
    monkeypatch.setattr(DiskStore, '_load_file', forbidden)
    report = calibrate_disk_cache(cache[0], 'train', **options(max_rss_bytes=rss_limit))
    assert report['reason'] == 'rss_limit'


def test_unavailable_host_capacity_keeps_sync_recommendation(cache, monkeypatch):
    monkeypatch.setattr(calibration_module, 'available_memory', lambda: None)
    report = calibrate_disk_cache(cache[0], 'train', **options(modes=('buffered',)))
    assert report['status'] == 'complete'
    assert report['recommendation']['mode'] == 'synchronous'
    assert not report['candidates'][1]['memory_available']


def test_low_host_capacity_skips_before_load(cache, monkeypatch):
    monkeypatch.setattr(calibration_module, 'available_memory', lambda: 1)
    monkeypatch.setattr(DiskStore, '_load_file', forbidden)
    report = calibrate_disk_cache(cache[0], 'train', **options())
    assert report['reason'] == 'baseline_available_memory_budget'


@pytest.mark.parametrize('candidate,complete,expected', [([1., 1.1, 1.2], True, 'thread'),
    ([1., 1.1, 3.], True, 'synchronous'), ([1., 1.1, 1.2], False, 'synchronous')])
def test_recommendation_requires_separated_repetitions(candidate, complete, expected):
    candidates = [{'mode': mode, 'options': {}, 'memory_available': True, 'status': 'complete',
                   'wall_seconds': measurement.distribution(values)}
                  for mode, values in [('synchronous', [2., 2.1, 2.2]), ('thread', candidate)]]
    assert measurement.recommend(candidates, 3, complete)['mode'] == expected


def test_progress_callback_failure_leaves_no_reader(cache):
    def failed(event):
        if event['phase'] == 'trial_complete':
            raise RuntimeError('callback stopped')
    with pytest.raises(RuntimeError, match='callback stopped'):
        calibrate_disk_cache(cache[0], 'train', progress=failed, **options())
    assert not any(t.name == 'mtlearn-cfp-reader' for t in threading.enumerate())


def test_logical_load_counter_does_not_count_ram_hits(cache):
    with DiskStore(cache[0], readonly=True, max_ram_bytes=1024**2) as store:
        key = store.inspect_manifest('train', limit=1)['samples'][0]['entries'][0]['entry_key']
        store.get(key)
        first = store.counters()
        store.get(key)
        second = store.counters()
        assert first['logical_load_bytes'] == first['checksum_bytes'] > 0
        assert second['logical_load_bytes'] == first['logical_load_bytes']
        assert second['ram_hits'] > first['ram_hits']
