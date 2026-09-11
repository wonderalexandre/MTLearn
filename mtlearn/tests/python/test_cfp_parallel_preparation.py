"""P4: true spawn processes, bounded inputs, single-writer publication and retry."""
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer, load_checkpoint
from mtlearn.layers.cfp import CFPPreprocessor, DiskStore, MemoryStore, PreparedDataset, collate_prepared
from mtlearn.layers.cfp.preparation import _parallel_preparation as parallel
from mtlearn._native import load_bindings

from test_cfp_disk_cache import FIXTURES, MANIFEST, ROOT, QUOTA, model, prepare, assert_stats, baseline

pytestmark = pytest.mark.integration


@pytest.mark.parametrize('name', MANIFEST['cases'])
def test_spawn_matches_frozen_stats_outputs_gradients(baseline, tmp_path, name):
    case = baseline['cases'][name]
    layer, _ = load_checkpoint(FIXTURES / case['checkpoint'], lambda configs: Layer.from_config(configs['']))
    prep = CFPPreprocessor.from_layer(layer)
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        result = prepare(prep, baseline['images'][:3], store, num_workers=2, max_in_flight=2,
                         collect_stats=layer.get_statistics_contract())
        assert_stats(result.statistics.statistics, case['stats'])
        layer.set_stats(result.statistics)
        prepare(prep, TensorDataset(baseline['images'][3:], baseline['targets'][3:]), store,
                manifest='test', split='evaluation', num_workers=2)
        batch, targets = collate_prepared([PreparedDataset(store, 'test',
            TensorDataset(baseline['images'][3:], baseline['targets'][3:]))[i] for i in range(2)])
        output = layer(batch)
        loss = ((output / 255 - targets) ** 2).mean()
        loss.backward()
        torch.testing.assert_close(output, case['output'], **MANIFEST['same_device_tolerance'])
        torch.testing.assert_close(loss, case['loss'], **MANIFEST['same_device_tolerance'])
        for key, parameter in layer.named_parameters():
            torch.testing.assert_close(parameter.grad, case['gradients'][key], **MANIFEST['same_device_tolerance'])
        assert store.info()['writes'] == 5 and store.info()['retained_bytes'] == 0
        assert not list((tmp_path / 'entries').glob('*.tmp'))


def test_nonserializable_source_bounded_queue_dedup_and_resume(baseline, tmp_path, monkeypatch):
    prep = CFPPreprocessor.from_layer(model())
    count, loads = 17, []
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        class LocalSource:
            def __len__(self): return count
            def __getstate__(self): raise AssertionError('Do not pickle notebook sources')
            def __getitem__(self, index):
                committed = store._db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
                assert len(loads) - committed < 3
                loads.append(index)
                return baseline['images'][index % 3], lambda: 'nonserializable target'
        result = prepare(prep, LocalSource(), store, num_workers=2, max_in_flight=3,
                         sample_ids=lambda i: f'logical-{i}', collect_stats=True)
        assert result.sample_count == count and loads == list(range(count))
        assert store.info()['writes'] == 3
        assert store._db.execute('SELECT COUNT(*) FROM contributions').fetchone()[0] == count
        expected = prep.prepare([baseline['images'][i % 3] for i in range(count)], collect_stats=True)
        assert_stats(result.statistics.statistics, expected.statistics.statistics)
        def forbid(*args, **kwargs): raise AssertionError('Warm resume must not spawn/rebuild')
        monkeypatch.setattr(parallel, '_Workers', forbid)
        resumed = prepare(prep, [baseline['images'][i % 3] for i in range(count)], store, num_workers=2,
                          sample_ids=lambda i: f'logical-{i}', collect_stats=True)
        assert resumed.statistics_id == result.statistics_id
        assert store.info()['writes'] == 3


def test_cancel_active_tasks_preserves_published_content(baseline, tmp_path):
    prep = CFPPreprocessor.from_layer(model())
    method = mp.get_start_method(allow_none=True)
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        cancelled = prepare(prep, baseline['images'][:3], store, num_workers=2,
            cancel=lambda: store.info()['writes'] >= 1)
        assert cancelled.status == 'cancelled' and cancelled.statistics is None
        assert not list((tmp_path / 'entries').glob('*.tmp'))
        assert not mp.active_children()
        result = prepare(prep, baseline['images'][:3], store, num_workers=2, collect_stats=True)
        assert result.status == 'complete' and store.info()['writes'] == 3
        assert store._db.execute('SELECT COUNT(*) FROM contributions').fetchone()[0] == 3
    # multiprocessing may lazily select the application's default while
    # constructing synchronization primitives; never force/change its method.
    if method is not None: assert mp.get_start_method() == method


def test_parallel_quota_is_shared_and_resumable(baseline, tmp_path):
    prep = CFPPreprocessor.from_layer(model())
    with DiskStore(tmp_path / 'size', max_disk_bytes=QUOTA) as reference_store:
        prepare(prep, baseline['images'][:1], reference_store)
        one_size = reference_store.info()['disk_bytes']
    with DiskStore(tmp_path / 'quota', max_disk_bytes=int(one_size * 1.1)) as store:
        with pytest.raises(OSError, match='quota exceeded'):
            prepare(prep, baseline['images'][:3], store, num_workers=2, max_retries=2)
        assert store.info()['disk_bytes'] <= store.max_disk_bytes
        assert not list(store._files.glob('*.tmp'))
        store.max_disk_bytes = QUOTA
        result = prepare(prep, baseline['images'][:3], store, num_workers=2, collect_stats=True)
        assert result.status == 'complete' and store.info()['writes'] == 3
        assert store._db.execute('SELECT COUNT(*) FROM contributions').fetchone()[0] == 3


@pytest.mark.parametrize('options', [dict(num_workers=-1), dict(num_workers=True), dict(num_workers=1.2),
    dict(num_workers=2, max_in_flight=0), dict(num_workers=2, max_in_flight=True),
    dict(worker_threads=0), dict(worker_threads=True), dict(max_sample_bytes=0), dict(max_retries=-1),
    dict(max_retries=True), dict(max_retries=1), dict(max_in_flight=1)])
def test_invalid_parallel_configuration_does_not_begin_manifest(baseline, tmp_path, options):
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        with pytest.raises(ValueError): prepare(CFPPreprocessor.from_layer(model()), baseline['images'], store, **options)
        assert store._db.execute('SELECT COUNT(*) FROM manifests').fetchone()[0] == 0


@pytest.mark.parametrize('store', [None, MemoryStore(0)])
def test_parallel_requires_explicit_disk_store(baseline, store):
    with pytest.raises(ValueError, match='DiskStore'):
        CFPPreprocessor.from_layer(model()).prepare(baseline['images'], store=store, num_workers=1)


def test_per_image_budget_and_nested_loader_rejected(baseline, tmp_path, monkeypatch):
    prep = CFPPreprocessor.from_layer(model())
    with DiskStore(tmp_path, max_disk_bytes=QUOTA) as store:
        with pytest.raises(ValueError, match='max_sample_bytes'):
            prepare(prep, baseline['images'], store, num_workers=1, max_sample_bytes=1)
        monkeypatch.setattr(torch.utils.data, 'get_worker_info', lambda: object())
        with pytest.raises(RuntimeError, match='DataLoader'):
            prepare(prep, baseline['images'], store, num_workers=1)


# Fault injection lives in an importable external driver, never in the library.
DRIVER = '''
import sys, os, json, time
from pathlib import Path
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != '_mtlearn_editable']
sys.path[:0] = [str(Path({root!r})/'mtlearn/python'), {native_dir!r}]
import torch, cv2
from mtlearn.layers.cfp import CFPPreprocessor, DiskStore
from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer
from mtlearn.layers.cfp.preparation import _parallel_preparation as parallel
root = Path({output!r})
original_initialize = parallel._initialize_worker
original_write = parallel._write_entry
fault = {fault!r}
def initialize(config, threads):
    def forbid(*a, **k): raise AssertionError('Workers must not initialize accelerators')
    torch.cuda.init = forbid
    torch.mps.synchronize = forbid
    prep = original_initialize(config, threads)
    assert torch.get_num_threads() == threads
    assert torch.get_num_interop_threads() == 1
    assert 1 <= cv2.getNumThreads() <= threads
    assert os.environ['OMP_NUM_THREADS'] == str(threads)
    assert not torch.cuda.is_initialized()
    (root / ('worker-' + str(os.getpid()))).write_text(json.dumps(config))
    return prep
parallel._initialize_worker = initialize
def write(prep, image, entry, budget, limit):
    if fault == 'order' and image[0,0] == 0: time.sleep(1)
    marker = root / 'fault-fired'
    if not marker.exists():
        marker.touch()
        if fault == 'exit': os._exit(43)
        if fault == 'retry': raise ValueError('transient diagnostic failure')
        if fault == 'corrupt':
            message = original_write(prep, image, entry, budget, limit)
            with open(message['temporary'], 'r+b') as handle: handle.write(b'bad!')
            return message
    return original_write(prep, image, entry, budget, limit)
parallel._write_entry = write
commit_order = []
original_bind = parallel._bind_sample
def bind(store, manifest, index, *args):
    original_bind(store, manifest, index, *args)
    commit_order.append(index)
parallel._bind_sample = bind
if __name__ == '__main__':
    torch.set_num_threads(1)
    layer = Layer(1, [{{'tree_type':'max-tree', 'attributes':[morphology.AttributeType.AREA, morphology.AttributeType.GRAY_LEVEL_HEIGHT]}}])
    prep = CFPPreprocessor.from_layer(layer)
    source = [torch.arange(42).reshape(1,6,7).float()/42, torch.ones(1,6,7)]
    options = dict(manifest='train',source_version='v1',preprocessing_version='v1',collect_stats=True,num_workers=(2 if fault == 'order' else 1),worker_threads=2,max_retries=1)
    with DiskStore(root / 'cache', max_disk_bytes=1024**2) as store:
        if fault in ('exit', 'corrupt'):
            try: prep.prepare(source, store=store, **options)
            except (RuntimeError, ValueError): pass
            else: raise AssertionError('Fault must surface')
        result = prep.prepare(source, store=store, **options)
        assert result.status == 'complete'
        if fault == 'order':
            assert commit_order == [1, 0], commit_order
            expected = prep.prepare(source, collect_stats=True)
            for key, values in expected.statistics.statistics.items():
                for name, value in values.items():
                    assert torch.equal(result.statistics.statistics[key][name], value)
        assert store.info()['writes'] == 2
        assert store._db.execute('SELECT COUNT(*) FROM contributions').fetchone()[0] == 2
        assert not list(store._files.glob('*.tmp'))
    print('P4_FAULT_PASS')
'''


@pytest.mark.parametrize('fault', ['retry', 'exit', 'corrupt', 'order'])
def test_actual_worker_failure_retry_and_cpu_limits(tmp_path, fault):
    driver = tmp_path / 'driver.py'
    driver.write_text(DRIVER.format(root=str(ROOT), native_dir=str(Path(load_bindings().__file__).resolve().parent), output=str(tmp_path), fault=fault))
    result = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'P4_FAULT_PASS' in result.stdout
    for path in tmp_path.glob('worker-*'):
        config = json.loads(path.read_text())
        assert set(config) == {'format_version','implementation','attribute_dtype','trees'}


def test_multichannel_multitree_float64_and_ordered_statistics(baseline, tmp_path):
    from mtlearn import morphology
    layer = Layer(2, [{'tree_type': tree, 'attributes': [morphology.AttributeType.AREA]}
                       for tree in ('max-tree', 'min-tree')], attribute_dtype=np.float64)
    prep = CFPPreprocessor.from_layer(layer)
    images = torch.cat([baseline['images'], 1 - baseline['images']], dim=1)
    with DiskStore(tmp_path / 'sequential', max_disk_bytes=QUOTA) as sequential:
        expected = prepare(prep, images, sequential, collect_stats=True)
        expected_entries = {r['key']: r['summary'] for r in sequential._db.execute('SELECT * FROM entries')}
        with DiskStore(tmp_path / 'parallel', max_disk_bytes=QUOTA) as store:
            actual = prepare(prep, images, store, num_workers=2, max_in_flight=1, collect_stats=True)
            assert actual.statistics_id == expected.statistics_id
            assert_stats(actual.statistics.statistics, expected.statistics.statistics)
            assert {r['key']: r['summary'] for r in store._db.execute('SELECT * FROM entries')} == expected_entries
            assert store._db.execute('SELECT COUNT(*) FROM contributions').fetchone()[0] == 20
            for key in expected_entries:
                first, second = sequential.get(key), store.get(key)
                for name, tensor in first.info.items():
                    if torch.is_tensor(tensor): assert torch.equal(tensor, second.info[name])
                for name, tensor in first.raw_attributes.items():
                    assert tensor.dtype == torch.float64 and torch.equal(tensor, second.raw_attributes[name])


def test_coordinator_death_fences_orphan_worker_before_recovery(tmp_path):
    import time
    driver = tmp_path / 'orphan.py'
    code = DRIVER.format(root=str(ROOT), native_dir=str(Path(load_bindings().__file__).resolve().parent), output=str(tmp_path), fault='none')
    code = code[:code.index("if __name__ == '__main__':")]
    code += '''
blocked_base = original_write
def blocked_write(*args):
    (root / 'writing').write_text(str(os.getpid()))
    while not (root / 'release').exists(): time.sleep(0.05)
    return blocked_base(*args)
parallel._write_entry = blocked_write
if __name__ == '__main__':
    layer = Layer(1, [{'tree_type':'max-tree', 'attributes':[morphology.AttributeType.AREA]}])
    with DiskStore(root / 'cache', max_disk_bytes=1024**2) as store:
        CFPPreprocessor.from_layer(layer).prepare([torch.ones(1,6,7)],store=store,
            manifest='train',source_version='v1',preprocessing_version='v1',num_workers=1)
'''
    driver.write_text(code)
    with (tmp_path / 'log').open('w') as log:
        process = subprocess.Popen([sys.executable, str(driver)], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while not (tmp_path / 'writing').exists() and time.monotonic() < deadline:
                assert process.poll() is None
                time.sleep(0.05)
            assert (tmp_path / 'writing').exists(), (tmp_path / 'log').read_text()
            process.kill()
            process.wait(timeout=5)
            with pytest.raises(RuntimeError, match='worker is still active'):
                DiskStore(tmp_path / 'cache', max_disk_bytes=QUOTA)
        finally:
            (tmp_path / 'release').touch()
            if process.poll() is None:
                process.kill()
                process.wait()
        deadline = time.monotonic() + 20
        while True:
            try:
                reopened = DiskStore(tmp_path / 'cache', max_disk_bytes=QUOTA)
                break
            except RuntimeError:
                if time.monotonic() >= deadline: raise
                time.sleep(0.05)
        with reopened:
            assert not list(reopened._files.glob('*.tmp'))
            from mtlearn import morphology
            layer = Layer(1, [{'tree_type':'max-tree','attributes':[morphology.AttributeType.AREA]}])
            result = CFPPreprocessor.from_layer(layer).prepare([torch.ones(1,6,7)], store=reopened,
                manifest='train',source_version='v1',preprocessing_version='v1',num_workers=1)
            assert result.status == 'complete' and reopened.info()['writes'] == 1
