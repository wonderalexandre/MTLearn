"""Isolated P1 streaming-fit measurement; no full dataset cache or dataset writes."""
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
    from mtlearn.layers.cfp import CFPPreprocessor
    from mtlearn.layers.cfp.normalization import AttributeNormalizer
    output = Path(args.output)
    reference.write_json(output / 'environment.json', reference.provenance(build, extension))
    timing = reference.Timings(output)
    image_ids = [0, 1000, 2040]
    rows, previous = [], []
    counts = {'summarize': 0, 'merge': 0}

    class Images(torch.utils.data.Dataset):
        def __len__(self):
            return len(image_ids)

        def __getitem__(self, index):
            ident = image_ids[index]
            with timing.stage(f'read_image_{ident}'):
                image = cv2.imread(str(Path(args.dataset) / 'enhancement' / f'{ident}.png'), cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(str(Path(args.dataset) / 'frag_component' / f'{ident:04d}.png'), cv2.IMREAD_UNCHANGED)
                assert image.shape == mask.shape == (2748, 2748)
                x = torch.from_numpy(image.astype(np.float32) / 255)[None]
                y = torch.from_numpy((mask > 0).astype(np.float32))[None]
                return x, y

    original_prepare = CFPPreprocessor.prepare_u8
    original_summarize, original_merge = AttributeNormalizer.summarize, AttributeNormalizer.merge

    def prepare(self, *a, **kw):
        assert all(r() is None for r in previous), 'Previous preparation still retained'
        ident = image_ids[len(rows)]
        with timing.stage(f'prepare_image_{ident}'):
            prepared = original_prepare(self, *a, **kw)
        rows.append({'image_id': ident, 'shape': list(prepared.image_shape),
                     'nodes': prepared.num_nodes, 'raw_bytes': prepared.nbytes,
                     'attributes': len(prepared.raw_attributes)})
        previous[:] = [weakref.ref(prepared), weakref.ref(next(iter(prepared.raw_attributes.values())))]
        return prepared

    def summarize(self, *a, **kw):
        counts['summarize'] += 1
        with timing.stage('summarize_attribute'):
            return original_summarize(self, *a, **kw)

    def merge(self, *a, **kw):
        counts['merge'] += 1
        with timing.stage('merge_summary'):
            return original_merge(self, *a, **kw)

    layer = reference.layer('linear', all_attrs=True)
    CFPPreprocessor.prepare_u8 = prepare
    AttributeNormalizer.summarize, AttributeNormalizer.merge = summarize, merge
    try:
        with timing.stage('fit_stats_total'):
            snapshot = layer.fit_stats(torch.utils.data.DataLoader(Images(), batch_size=1, num_workers=0))
    finally:
        CFPPreprocessor.prepare_u8 = original_prepare
        AttributeNormalizer.summarize, AttributeNormalizer.merge = original_summarize, original_merge
    assert all(r() is None for r in previous)
    assert layer.cached_sample_count() == 0 and snapshot.sample_count == 3
    stats = snapshot.statistics
    assert all(torch.isfinite(v) for values in stats.values() for v in values.values())
    expected_nodes = sum(row['nodes'] for row in rows)
    assert all(int(values['count']) == expected_nodes for values in stats.values())
    result = {'status': 'passed', 'sample_count': snapshot.sample_count,
              'cached_sample_count': layer.cached_sample_count(), 'all_preparations_released': True,
              'statistics_bytes': reference.tensor_bytes(stats), 'counts': counts, 'images': rows,
              'statistics': {key: {name: value.item() for name, value in values.items()} for key, values in stats.items()},
              'timings': timing.rows}
    reference.write_json(output / 'result.json', result)
    print('P1_STREAMING_FIT_PASS', flush=True)


def supervise(args):
    import psutil
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--build-dir', args.build_dir,
               '--dataset', args.dataset, '--output', str(output)]
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
                            'available_bytes': available, 'stage': stage,
                            'swap_used_bytes': psutil.swap_memory().used})
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
    parser.add_argument('--max-rss-gib', type=float, default=4.0)
    parser.add_argument('--min-available-gib', type=float, default=1.0)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    worker(args) if args.worker else supervise(args)


if __name__ == '__main__':
    main()
