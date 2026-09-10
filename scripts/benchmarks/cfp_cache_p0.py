"""Freeze pre-refactoring CFP references and measure isolated real-image workloads.

This is an external diagnostic driver, not part of the mtlearn runtime API.
Run with the interpreter used to build the native extension. No dataset writes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def runtime(build_dir):
    # An editable installation on this machine redirects mtlearn to another tree.
    # Change only this diagnostic process; leave the environment and package intact.
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != '_mtlearn_editable']
    build = (ROOT / build_dir).resolve()
    sys.path[:0] = [str(ROOT / 'mtlearn/python'), str(build / 'mtlearn/bindings')]
    import torch
    import cv2
    import numpy as np
    import mtlearn
    import _mtlearn
    from mtlearn import morphology
    from mtlearn.layers import ConnectedFilterPreprocessingLayer
    assert Path(mtlearn.__file__).resolve().is_relative_to(ROOT / 'mtlearn/python')
    assert Path(_mtlearn.__file__).resolve().is_relative_to(build)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    globals().update(torch=torch, cv2=cv2, np=np, mtlearn=mtlearn,
                     morphology=morphology, Layer=ConnectedFilterPreprocessingLayer)
    return build, Path(_mtlearn.__file__).resolve()


def provenance(build, extension):
    import psutil
    paths = sorted((ROOT / 'mtlearn/python/mtlearn').rglob('*.py'))
    paths += sorted((ROOT / 'mtlearn/src').rglob('*.hpp'))
    paths += sorted((ROOT / 'mtlearn/src').rglob('*.cpp'))
    paths += sorted((ROOT / 'mtlearn/bindings').rglob('*.hpp'))
    paths += sorted((ROOT / 'mtlearn/bindings').rglob('*.cpp'))
    flags = {}
    for path in sorted(build.rglob('flags.make')):
        flags[str(path.relative_to(build))] = path.read_text()
    return {
        'recorded_at': dt.datetime.now().astimezone().isoformat(),
        'git_head': command('git', 'rev-parse', 'HEAD'),
        'git_status': command('git', 'status', '--short'),
        'submodules': command('git', 'submodule', 'status'),
        'python': sys.executable, 'python_version': sys.version,
        'mtlearn_path': str(Path(mtlearn.__file__).resolve()),
        'extension_path': str(extension), 'extension_sha256': digest(extension),
        'torch': str(torch.__version__), 'opencv': cv2.__version__, 'numpy': np.__version__,
        'mps_available': torch.backends.mps.is_available(),
        'cuda_available': torch.cuda.is_available(),
        'machine': platform.platform(),
        'cpu': command('sysctl', '-n', 'machdep.cpu.brand_string'),
        'logical_cpus': psutil.cpu_count(), 'memory': psutil.virtual_memory()._asdict(),
        'swap': psutil.swap_memory()._asdict(),
        'disk': psutil.disk_usage('/Volumes/SSD')._asdict(),
        'compiler': command('c++', '--version'), 'cmake_flags': flags,
        'cmake_cache': (build / 'CMakeCache.txt').read_text(),
        'seed': 42, 'torch_threads': torch.get_num_threads(),
        'thread_environment': {key: os.environ.get(key) for key in
                               ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
        'source_sha256': {str(path.relative_to(ROOT)): digest(path) for path in paths},
    }


def layer(scoring, scale='dataset_clipped_zscore01', tree='max-tree', all_attrs=False,
          device='cpu'):
    attributes = ((morphology.AttributeGroup.ALL,) if all_attrs else
                  (morphology.AttributeType.AREA, morphology.AttributeType.GRAY_LEVEL_HEIGHT))
    return Layer(in_channels=1, filter_specs=[{
        'name': 'baseline', 'tree_type': tree, 'attributes': attributes,
        'score_sharpness': 1.0,
        'scoring': ({'kind': 'linear_sigmoid'} if scoring == 'linear' else
                    {'kind': 'mlp', 'hidden_units': [8], 'activation': 'tanh'}),
    }], device=device, scale_mode=scale, clamp=12.0)


def freeze(args, build, extension):
    from torch.utils.data import DataLoader, TensorDataset
    from mtlearn.layers import save_checkpoint
    destination = Path(args.output)
    if (destination / 'baseline.pt').exists():
        raise FileExistsError('Frozen references are immutable; choose another directory.')
    destination.mkdir(parents=True, exist_ok=True)
    palette = torch.tensor([0, 8, 16, 32, 64, 96, 128, 192, 255], dtype=torch.float32)
    index = torch.randint(len(palette), (5, 1, 6, 7), generator=torch.Generator().manual_seed(42))
    images = palette[index] / 255.0
    targets = (images > 0.3).float()
    cases = {}
    matrix = [(s, mode, 'max-tree', mode == 'dataset_clipped_zscore01')
              for s in ['linear', 'mlp']
              for mode in ['none', 'dataset_minmax01', 'dataset_zscore', 'dataset_clipped_zscore01']]
    matrix += [(s, 'dataset_clipped_zscore01', tree, False)
               for s in ['linear', 'mlp'] for tree in ['min-tree', 'tree-of-shapes']]
    for scoring, scale, tree, all_attrs in matrix:
        name = f'{tree}_{scoring}_{scale}'
        model = layer(scoring, scale, tree, all_attrs)
        train = DataLoader(TensorDataset(images[:3], targets[:3]), batch_size=2, shuffle=False)
        model.build_dataloader_cached(train)
        with torch.no_grad():
            for number, parameter in enumerate(model.parameters()):
                parameter.copy_(torch.linspace(-0.12, 0.12, parameter.numel()).reshape_as(parameter)
                                + 0.01 * (number + 1))
        checkpoint = destination / f'{name}.pt'
        save_checkpoint(checkpoint, model)
        model.save_stats(destination / f'{name}.stats.pt')
        model.zero_grad(set_to_none=True)
        output = model(images[3:])
        loss = ((output / 255.0 - targets[3:]) ** 2).mean()
        loss.backward()
        gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
        assert torch.isfinite(output).all()
        assert all(torch.isfinite(g).all() for g in gradients.values())
        assert sum(g.abs().sum().item() for g in gradients.values()) > 0
        cases[name] = {
            'config': model.get_config(), 'contracts': model.get_contracts(),
            'stats': model.get_extra_state()['ds_stats'],
            'output': output.detach(), 'loss': loss.detach(), 'gradients': gradients,
            'checkpoint': checkpoint.name,
        }
        print('FROZEN', name, flush=True)
    torch.save({'format_version': 1, 'seed': 42, 'images': images, 'targets': targets,
                'train_indices': [0, 1, 2], 'test_indices': [3, 4], 'cases': cases},
               destination / 'baseline.pt')
    write_json(destination / 'manifest.json', {
        'created_at': dt.datetime.now().astimezone().isoformat(),
        'producer': 'scripts/benchmarks/cfp_cache_p0.py freeze',
        'source_revision': command('git', 'rev-parse', 'HEAD'),
        'native_sha256': digest(extension),
        'files': {p.name: digest(p) for p in sorted(destination.glob('*.pt'))},
        'cases': list(cases), 'reference_dtype': 'float32',
        'same_device_tolerance': {'rtol': 1e-5, 'atol': 1e-6},
        'cpu_mps_tolerance': {'rtol': 1e-4, 'atol': 1e-5},
    })
    write_json(args.provenance, provenance(build, extension))


def tensor_bytes(value):
    storages = {}
    def visit(item):
        if torch.is_tensor(item):
            storage = item.untyped_storage()
            storages[(str(item.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(item, dict):
            for v in item.values(): visit(v)
        elif isinstance(item, (tuple, list)):
            for v in item: visit(v)
    visit(value)
    return sum(storages.values())


def map_tensors(value, device):
    if torch.is_tensor(value): return value.to(device)
    if isinstance(value, dict): return {k: map_tensors(v, device) for k, v in value.items()}
    return value


class Timings:
    def __init__(self, output):
        self.output = Path(output)
        self.rows = []
        self.output.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def stage(self, name, device='cpu'):
        import psutil
        process = psutil.Process()
        def synchronize():
            if device == 'mps': torch.mps.synchronize()
        synchronize()
        begin = time.perf_counter()
        write_json(self.output / 'current_stage.json', {'stage': name, 'time': time.time()})
        try:
            yield
        finally:
            synchronize()
            row = {'stage': name, 'seconds': time.perf_counter() - begin,
                   'rss_end_bytes': process.memory_info().rss,
                   'available_end_bytes': psutil.virtual_memory().available}
            if device == 'mps':
                row.update(mps_active_bytes=torch.mps.current_allocated_memory(),
                           mps_driver_bytes=torch.mps.driver_allocated_memory())
            self.rows.append(row)
            write_json(self.output / 'timings.json', self.rows)
            print(json.dumps(row), flush=True)

    def wrap(self, name, function):
        def wrapped(*a, **kw):
            with self.stage(name): return function(*a, **kw)
        return wrapped


def measure(args, build, extension):
    from torch.utils.data import DataLoader, TensorDataset
    root = Path(args.dataset)
    output_dir = Path(args.output)
    timing = Timings(output_dir)
    input_path = root / 'enhancement' / f'{args.image_id}.png'
    target_path = root / 'frag_component' / f'{args.image_id:04d}.png'
    write_json(output_dir / 'environment.json', provenance(build, extension))
    with timing.stage('read_and_decode'):
        image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        target = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
        assert image is not None and target is not None
        assert image.shape == target.shape and image.dtype == target.dtype == np.uint8
        assert image.ndim == target.ndim == 2
    with timing.stage('input_and_mask_conversion'):
        x = torch.from_numpy(image.astype(np.float32) / 255.0)[None, None]
        y = torch.from_numpy((target > 0).astype(np.float32))[None, None]
    del image, target
    model = layer('linear', all_attrs=True)
    provider = model._tree_payload_provider
    normalizer = model._attribute_normalizer
    originals = (provider.build_tree, provider.compute_tree_info,
                 morphology.compute_attributes, normalizer.update, normalizer.normalize)
    provider.build_tree = timing.wrap('tree_construction', originals[0])
    provider.compute_tree_info = timing.wrap('tree_tensors', originals[1])
    morphology.compute_attributes = timing.wrap('attribute_extraction', originals[2])
    normalizer.update = timing.wrap('statistics_update', originals[3])
    normalizer.normalize = timing.wrap('normalization', originals[4])
    with timing.stage('legacy_cpu_preparation_total'):
        model.build_dataloader_cached(DataLoader(TensorDataset(x, y), batch_size=1))
    (provider.build_tree, provider.compute_tree_info, morphology.compute_attributes,
     normalizer.update, normalizer.normalize) = originals
    key = model.filter_specs[0].tree_key
    payload = model._tree_payload_cache.get('0_0', key)
    raw = {'info': dict(payload['info']),
           'base_attrs': {attr.name: value for attr, value in payload['base_attrs'].items()}}
    raw['info']['tree_type'] = str(getattr(raw['info']['tree_type'], 'value', raw['info']['tree_type']))
    for values in payload['base_attrs'].values(): assert torch.isfinite(values).all()
    for values in payload['norm_attrs'].values(): assert torch.isfinite(values).all()
    result = {
        'image_id': args.image_id, 'input_sha256': digest(input_path),
        'target_sha256': digest(target_path), 'shape': list(x.shape), 'device': args.device,
        'attributes': [a.name for a in model.filter_specs[0].attributes],
        'num_nodes': payload['info']['tpre'].numel(), 'pixels': x.numel(),
        'raw_payload_bytes': tensor_bytes(raw), 'legacy_payload_bytes': tensor_bytes(payload),
        'topology_bytes': tensor_bytes(payload['info']),
        'raw_attributes_bytes': tensor_bytes(payload['base_attrs']),
        'normalized_attributes_bytes': tensor_bytes(payload['norm_attrs']),
        'sample_count_in_cache': model.cached_sample_count(),
    }
    assert result['sample_count_in_cache'] == 1
    write_json(output_dir / 'result.json', result)
    # Probe on the external SSD in an ignored build directory, without keeping a cache.
    with tempfile.TemporaryDirectory(prefix='payload-probe-', dir=build) as tmp:
        path = Path(tmp) / 'raw.pt'
        with timing.stage('serialize_raw_to_ssd'):
            torch.save(raw, path)
        result['serialized_raw_bytes'] = path.stat().st_size
        with timing.stage('read_raw_from_ssd_warm_mmap'):
            restored = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            # Touch the pixel map and all attributes; opening an mmap alone is insufficient.
            assert torch.equal(restored['info']['node_of_pixel'], raw['info']['node_of_pixel'])
            assert all(torch.equal(restored['base_attrs'][a], value)
                       for a, value in raw['base_attrs'].items())
        del restored
    write_json(output_dir / 'result.json', result)
    result['models'] = {}
    positive_weight = (y.numel() - y.sum().item()) / y.sum().item()
    for scoring in ['linear', 'mlp']:
        if scoring == 'linear' and args.device == 'cpu':
            current = model
            current_payload = payload
        else:
            current = layer(scoring, all_attrs=True, device=args.device)
            current._ds_stats = model._ds_stats
            current._stats_epoch = model._stats_epoch
            current.freeze_ds_stats()
            with timing.stage(f'{scoring}_transfer_payload', args.device):
                current_payload = map_tensors(payload, args.device)
            current._tree_payload_cache.set('0_0', key, current_payload)
            current._tree_payload_cache.set_epoch('0_0', current._stats_epoch)
        current.init_identity(p0=0.995)
        current.zero_grad(set_to_none=True)
        with timing.stage(f'{scoring}_forward', args.device):
            response = current((x, torch.tensor([0])))
        with timing.stage(f'{scoring}_segmentation_loss', args.device):
            prediction = response / 255.0
            low, high = prediction.amin(), prediction.amax()
            prediction = (prediction - low) / (high - low).clamp_min(1e-6)
            mask = y.to(args.device)
            dice = 1 - (2 * (prediction * mask).sum() + 1e-6) / (prediction.sum() + mask.sum() + 1e-6)
            bce = torch.nn.functional.binary_cross_entropy(prediction.clamp(1e-4, 1-1e-4), mask, reduction='none')
            loss = 0.5 * dice + 0.5 * (bce * torch.where(mask >= 0.5, positive_weight, 1.0)).mean()
        with timing.stage(f'{scoring}_backward', args.device):
            loss.backward()
        gradients = [p.grad for p in current.parameters() if p.grad is not None]
        assert torch.isfinite(response).all() and torch.isfinite(loss)
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        result['models'][scoring] = {
            'loss': float(loss.detach().cpu()),
            'gradient_l2': float(sum(g.detach().float().square().sum().cpu() for g in gradients).sqrt()),
            'response_sum': float(response.detach().double().cpu().sum()) if args.device == 'cpu'
                            else float(response.detach().cpu().double().sum()),
            'parameters': sum(p.numel() for p in current.parameters()),
        }
        write_json(output_dir / 'result.json', result)
        del gradients, response, prediction, mask, bce, dice, loss
        if current is not model:
            del current, current_payload
        model.zero_grad(set_to_none=True)
        gc.collect()
        if args.device == 'mps': torch.mps.empty_cache()
    result['status'] = 'passed'
    result['timings'] = timing.rows
    write_json(output_dir / 'result.json', result)
    print('MEASUREMENT_PASS', args.image_id, args.device, flush=True)


def supervise(args):
    import psutil
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    child_args = [sys.executable, str(Path(__file__).resolve()), '--build-dir', args.build_dir,
                  'measure', '--dataset', args.dataset, '--image-id', str(args.image_id),
                  '--device', args.device, '--output', str(output)]
    records = []
    reason = None
    started = time.monotonic()
    with (output / 'worker.log').open('w') as log:
        child = subprocess.Popen(child_args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                rss = sum(p.memory_info().rss for p in [process] + process.children(recursive=True))
            except psutil.NoSuchProcess:
                break
            virtual = psutil.virtual_memory()
            stage_file = output / 'current_stage.json'
            try: stage = json.loads(stage_file.read_text())['stage']
            except (FileNotFoundError, json.JSONDecodeError): stage = 'startup'
            records.append({'elapsed_s': time.monotonic() - started, 'rss_bytes': rss,
                            'available_bytes': virtual.available, 'swap_used_bytes': psutil.swap_memory().used,
                            'stage': stage})
            if rss > args.max_rss_gib * GIB:
                reason = f'RSS above {args.max_rss_gib} GiB'
            elif virtual.available < args.min_available_gib * GIB:
                reason = f'System available memory below {args.min_available_gib} GiB'
            elif time.monotonic() - started > args.timeout:
                reason = f'Timeout after {args.timeout} seconds'
            if reason:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill()
                break
            time.sleep(0.1)
        code = child.wait()
    summary = {'command': child_args, 'returncode': code, 'stop_reason': reason,
               'elapsed_s': time.monotonic() - started,
               'rss_peak_bytes': max((r['rss_bytes'] for r in records), default=0),
               'available_min_bytes': min((r['available_bytes'] for r in records), default=0),
               'limits': {'max_rss_gib': args.max_rss_gib, 'min_available_gib': args.min_available_gib,
                          'timeout_s': args.timeout}, 'sampling_interval_s': 0.1}
    write_json(output / 'monitor.json', {'summary': summary, 'samples': records})
    print(json.dumps(summary, indent=2), flush=True)
    if code != 0:
        print((output / 'worker.log').read_text()[-8000:], flush=True)
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', default='build/cfp-cache-p0-release')
    sub = parser.add_subparsers(dest='mode', required=True)
    freeze_parser = sub.add_parser('freeze')
    freeze_parser.add_argument('--output', required=True)
    freeze_parser.add_argument('--provenance', required=True)
    for name in ['measure', 'supervise']:
        p = sub.add_parser(name)
        p.add_argument('--dataset', default='/Volumes/SSD/GitHub/Dennis/dataset_patches/205_SA_L3D14M3')
        p.add_argument('--image-id', type=int, required=True)
        p.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
        p.add_argument('--output', required=True)
        if name == 'supervise':
            p.add_argument('--max-rss-gib', type=float, default=4.0)
            p.add_argument('--min-available-gib', type=float, default=1.0)
            p.add_argument('--timeout', type=float, default=600)
    test_parser = sub.add_parser('test')
    test_parser.add_argument('pytest_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == 'supervise': return supervise(args)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    build, extension = runtime(args.build_dir)
    if args.mode == 'freeze': freeze(args, build, extension)
    elif args.mode == 'measure': measure(args, build, extension)
    elif args.mode == 'test':
        import pytest
        pytest_args = args.pytest_args[1:] if args.pytest_args[:1] == ['--'] else args.pytest_args
        raise SystemExit(pytest.main(pytest_args))


if __name__ == '__main__':
    main()
