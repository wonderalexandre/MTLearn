"""External P2 measurement: byte-bounded CPU preparation shared by two scorers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import weakref

import cfp_cache_p0 as reference


def worker(args):
    build, extension = reference.runtime(args.build_dir)
    torch, cv2, np = reference.torch, reference.cv2, reference.np
    from mtlearn.layers.cfp import CFPPreprocessor, MemoryStore, StatisticsSnapshot
    from mtlearn.layers.cfp.normalization import AttributeNormalizer
    output = Path(args.output)
    reference.write_json(output / 'environment.json', reference.provenance(build, extension))
    timing = reference.Timings(output)
    # Reuse the frozen train-only streaming fit measured in P1. No fitting in
    # this experiment, and no per-image preparations loaded from that report.
    source = reference.ROOT / 'docs/cfp-cache/p1/runs/fit_cpu_3_validated/result.json'
    fitted = json.loads(source.read_text())
    stats = {key: {name: torch.tensor(value, dtype=torch.int64 if name == 'count' else torch.float64)
                   for name, value in values.items()} for key, values in fitted['statistics'].items()}
    models = {kind: reference.layer(kind, all_attrs=True, device=args.device) for kind in ('linear', 'mlp')}
    snapshot = StatisticsSnapshot(models['linear']._statistics_contract(), stats, sample_count=3)
    for layer in models.values():
        layer.set_stats(snapshot)
    preprocessor = CFPPreprocessor.from_layer(models['linear'])
    store = MemoryStore(256 * 1024**2)
    counts = {'prepare': 0, 'statistics_updates': 0}
    original_prepare = CFPPreprocessor.prepare_u8
    def prepare(self, *a, **kw):
        counts['prepare'] += 1
        return original_prepare(self, *a, **kw)
    def forbidden_update(*a, **kw):
        counts['statistics_updates'] += 1
        raise AssertionError('Training must not update statistics')
    CFPPreprocessor.prepare_u8 = prepare
    AttributeNormalizer.update = forbidden_update
    AttributeNormalizer.summarize = forbidden_update
    AttributeNormalizer.merge = forbidden_update
    image_ids = [int(i) for i in args.image_ids.split(',')]
    rows, previous = [], None
    for index, ident in enumerate(image_ids):
        with timing.stage(f'read_image_{ident}'):
            image = cv2.imread(str(Path(args.dataset) / 'enhancement' / f'{ident}.png'), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(Path(args.dataset) / 'frag_component' / f'{ident:04d}.png'), cv2.IMREAD_UNCHANGED)
            assert image.shape == mask.shape == (2748, 2748)
            images = torch.from_numpy(image)[None, None]
            targets = torch.from_numpy((mask > 0).astype(np.float32))[None, None].to(args.device)
        if previous is not None:
            assert previous() is None
        with timing.stage(f'prepare_cold_{ident}'):
            batch = preprocessor.prepare_batch(images, store=store, sample_ids=(str(ident),))
        previous = weakref.ref(batch)
        morphology = next(iter(batch.samples[0][0].values()))
        row = {'image_id': ident, 'nodes': morphology.num_nodes, 'raw_bytes': morphology.nbytes,
               'attributes': len(morphology.raw_attributes), 'shape': list(morphology.image_shape)}
        with timing.stage(f'prepare_hit_{ident}'):
            warm = preprocessor.prepare_batch(images, store=store, sample_ids=(f'reordered-{ident}',))
        assert morphology.info['tpre'].data_ptr() == next(iter(warm.samples[0][0].values())).info['tpre'].data_ptr()
        del warm, morphology
        row['store_before_training'] = store.info()
        assert row['store_before_training']['retained_bytes'] <= store.max_bytes
        assert row['store_before_training']['active_entries'] == 1
        row['models'] = {}
        for kind, layer in models.items():
            layer.zero_grad(set_to_none=True)
            with timing.stage(f'{kind}_forward_{ident}', args.device):
                response = layer.forward_prepared(batch)
            with timing.stage(f'{kind}_loss_{ident}', args.device):
                loss = ((response / 255.0 - targets) ** 2).mean()
            if index == len(image_ids) - 1 and kind == 'mlp':
                # No caller/store owner survives this point; autograd must keep
                # the data it needs and release the active handle after backward.
                store.clear()
                del batch
                row['store_evicted_before_backward'] = store.info()
                assert previous() is not None and store.info()['active_bytes'] > 0
            with timing.stage(f'{kind}_backward_{ident}', args.device):
                loss.backward()
            gradients = [p.grad for p in layer.parameters() if p.grad is not None]
            assert torch.isfinite(response).all() and torch.isfinite(loss)
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            row['models'][kind] = {'loss': loss.item(),
                'gradient_l2': sum(g.float().square().sum().item() for g in gradients)**0.5}
            del response, loss, gradients
            layer.zero_grad(set_to_none=True)
            assert layer.cached_sample_count() == 0
        if index != len(image_ids) - 1:
            del batch
        assert previous() is None and store.info()['active_bytes'] == 0
        row['store_after_training'] = store.info()
        rows.append(row)
        del images, targets, image, mask
    assert counts == {'prepare': len(image_ids), 'statistics_updates': 0}
    assert store.info()['hits'] == len(image_ids)
    assert store.info()['misses'] == len(image_ids)
    assert store.info()['evictions'] == len(image_ids) - 1
    assert store.info()['live_bytes'] == 0
    result = {'status': 'passed', 'device': args.device, 'cache_max_bytes': store.max_bytes,
        'statistics_source': str(source.relative_to(reference.ROOT)), 'statistics_source_sha256': reference.digest(source),
        'counts': counts, 'images': rows, 'final_store': store.info(), 'timings': timing.rows,
        'loss_definition': 'mean((response / 255 - binary_mask) ** 2)',
        'note': 'Resource and integration probe; not model quality evaluation or a full training epoch.'}
    reference.write_json(output / 'result.json', result)
    print('P2_PREPARED_RUNTIME_PASS', flush=True)


def supervise(args):
    import psutil
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--build-dir', args.build_dir,
               '--dataset', args.dataset, '--output', str(output), '--device', args.device, '--image-ids', args.image_ids]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    samples, reason = [], None
    start = time.monotonic()
    with (output / 'worker.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=reference.ROOT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                rss = sum(p.memory_info().rss for p in [process] + process.children(recursive=True))
            except psutil.NoSuchProcess:
                break
            available = psutil.virtual_memory().available
            try:
                stage = json.loads((output / 'current_stage.json').read_text())['stage']
            except (FileNotFoundError, json.JSONDecodeError):
                stage = 'startup'
            samples.append({'elapsed_s': time.monotonic() - start, 'rss_bytes': rss,
                            'available_bytes': available, 'stage': stage, 'swap_used_bytes': psutil.swap_memory().used})
            if rss > args.max_rss_gib * 1024**3:
                reason = 'RSS limit exceeded'
            elif available < args.min_available_gib * 1024**3:
                reason = 'Available-memory floor reached'
            elif time.monotonic() - start > args.timeout:
                reason = 'Time limit exceeded'
            if reason:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
            time.sleep(0.1)
        code = child.wait()
    summary = {'command': command, 'returncode': code, 'stop_reason': reason,
               'elapsed_s': time.monotonic() - start,
               'rss_peak_bytes': max((s['rss_bytes'] for s in samples), default=0),
               'available_min_bytes': min((s['available_bytes'] for s in samples), default=0),
               'sampling_interval_s': 0.1,
               'limits': {'max_rss_gib': args.max_rss_gib, 'min_available_gib': args.min_available_gib, 'timeout_s': args.timeout}}
    reference.write_json(output / 'monitor.json', {'summary': summary, 'samples': samples})
    print(json.dumps(summary, indent=2), flush=True)
    if code:
        print((output / 'worker.log').read_text()[-6000:])
        raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--build-dir', default='build/cfp-cache-p0-release')
    parser.add_argument('--dataset', default='/Volumes/SSD/GitHub/Dennis/dataset_patches/205_SA_L3D14M3')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--image-ids', default='0,1000,2040')
    parser.add_argument('--max-rss-gib', type=float, default=4.0)
    parser.add_argument('--min-available-gib', type=float, default=1.0)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    worker(args) if args.worker else supervise(args)


if __name__ == '__main__':
    main()
