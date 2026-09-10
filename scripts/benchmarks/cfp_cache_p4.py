"""External P4 CPU-process diagnostics; no profiling hooks in the library."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# A local editable installation redirects to another checkout. This change is
# limited to this diagnostic process and the spawned imports of this driver.
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != '_mtlearn_editable']
import cfp_cache_p0 as reference


def install_worker_diagnostics(output):
    import torch
    import cv2
    import mtlearn
    import _mtlearn
    from mtlearn.layers.cfp.preparation import _parallel_preparation as parallel
    original_init, original_write = parallel._initialize_worker, parallel._write_entry
    output = Path(output)

    def initialize(config, threads):
        prep = original_init(config, threads)
        assert Path(mtlearn.__file__).resolve().is_relative_to(reference.ROOT / 'mtlearn/python')
        assert Path(_mtlearn.__file__).resolve().is_relative_to(reference.ROOT / 'build/cfp-cache-p0-release')
        reference.write_json(output / f'cpu-worker-{os.getpid()}.json', {
            'pid': os.getpid(), 'mtlearn_path': mtlearn.__file__, 'extension_path': _mtlearn.__file__,
            'extension_sha256': reference.digest(_mtlearn.__file__), 'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(), 'opencv_threads': cv2.getNumThreads(),
            'cuda_initialized': torch.cuda.is_initialized(), 'config': config})
        return prep

    def write(prep, image, entry, budget, limit):
        start = time.monotonic()
        result = original_write(prep, image, entry, budget, limit)
        with (output / f'cpu-worker-{os.getpid()}.jsonl').open('a') as handle:
            handle.write(json.dumps({'entry_key': entry['key'], 'elapsed_s': time.monotonic()-start,
                                     'size_bytes': result['size_bytes'], 'pixels': int(image.size)}) + '\n')
        return result
    parallel._initialize_worker, parallel._write_entry = initialize, write


if __name__ == '__mp_main__' and os.environ.get('CFP_P4_DIAGNOSTIC_DIR'):
    install_worker_diagnostics(os.environ['CFP_P4_DIAGNOSTIC_DIR'])


def coordinator(args):
    build, extension = reference.runtime(args.build_dir)
    torch, cv2 = reference.torch, reference.cv2
    cv2.setNumThreads(0)
    from mtlearn.layers.cfp import CFPPreprocessor, DiskStore
    from mtlearn.layers.cfp.preparation import _parallel_preparation as parallel
    output = Path(args.output)
    reference.write_json(output / 'provenance.json', reference.provenance(build, extension))
    timing = reference.Timings(output)
    layer = reference.layer('linear', all_attrs=True)
    prep = CFPPreprocessor.from_layer(layer)
    ids = [0, 1000, 2040]
    reads = []

    class Source:
        def __len__(self): return len(ids)
        def __getstate__(self): raise AssertionError('The source must remain in the coordinator')
        def __getitem__(self, index):
            reads.append(ids[index])
            path = Path(args.dataset) / 'enhancement' / f'{ids[index]}.png'
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None: raise FileNotFoundError(path)
            assert image.shape == (2748, 2748)
            return image

    options = dict(manifest='train',source_version='205-SA-L3D14M3-ids-0-1000-2040-v1',
        preprocessing_version='original-u8-v1',split='train',sample_ids=[str(i) for i in ids],
        collect_stats=layer.get_statistics_contract(),num_workers=args.num_workers,
        max_in_flight=args.num_workers,worker_threads=1,max_sample_bytes=8*1024**2)
    commits = []
    original_publish = parallel._publish
    def publish(store, task, message):
        original_publish(store, task, message)
        commits.append(task['sample_id'])
    parallel._publish = publish
    with DiskStore(args.cache_dir, max_disk_bytes=1024**3) as store:
        assert store.info()['disk_entries'] == 0, 'Use a fresh cache for a cold comparison'
        with timing.stage('cold_prepare'):
            result = prep.prepare(Source(), store=store, **options)
        assert result.status == 'complete' and result.sample_count == 3
        expected = json.loads((reference.ROOT / 'docs/cfp-cache/p1/runs/fit_cpu_3_validated/result.json').read_text())
        for key, values in expected['statistics'].items():
            for name, value in values.items():
                actual = result.statistics.statistics[key][name]
                torch.testing.assert_close(actual, torch.tensor(value, dtype=actual.dtype), rtol=1e-10, atol=1e-12)
        # Compare the stored digest of every validated raw tensor with the final
        # P3 run. Metadata reads use mmap and do not hold two full images in RAM.
        with DiskStore(args.reference_cache, readonly=True) as baseline:
            for row in store._db.execute('SELECT * FROM entries ORDER BY key'):
                previous = baseline._db.execute('SELECT * FROM entries WHERE key=?', (row['key'],)).fetchone()
                assert previous and row['summary'] == previous['summary']
                actual = torch.load(store._entry_path(row['key']),map_location='cpu',weights_only=True,mmap=True)
                expected_data = torch.load(baseline._entry_path(row['key']),map_location='cpu',weights_only=True,mmap=True)
                assert actual['tensor_sha256'] == expected_data['tensor_sha256']
                del actual, expected_data
        def forbid(*a, **kw): raise AssertionError('A warm resume must not spawn workers')
        parallel._Workers = forbid
        with timing.stage('warm_resume'):
            resumed = prep.prepare(Source(), store=store, **options)
        assert resumed.statistics_id == result.statistics_id
        info = store.info()
        assert info['writes'] == 3 and info['disk_bytes'] <= 1024**3
        assert info['retained_bytes'] == 0 and info['active_bytes'] == 0
        contributions = store._db.execute('SELECT COUNT(*) FROM contributions').fetchone()[0]
        assert contributions == 3
        reference.write_json(output / 'result.json', {
            'status': result.status, 'num_workers': args.num_workers, 'max_in_flight': args.num_workers,
            'max_sample_bytes': 8*1024**2, 'worker_threads': 1, 'sample_ids': ids,
            'statistics_match_p1': True, 'all_raw_tensors_match_p3': True,
            'statistics_bytes': reference.tensor_bytes(result.statistics.statistics),
            'statistics_id': result.statistics_id, 'contributions': contributions,
            'publication_order': commits, 'source_reads': reads, 'cache': info, 'timings': timing.rows,
            'note': 'Three original images, preparation and warm resume only; not a training epoch.'})
    print('P4_PARALLEL_PASS', flush=True)


def supervise(args):
    import psutil
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--coordinator', '--num-workers', str(args.num_workers),
        '--output', str(output.resolve()), '--cache-dir', args.cache_dir, '--reference-cache', args.reference_cache,
        '--build-dir',args.build_dir,'--dataset',args.dataset]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               CFP_P4_DIAGNOSTIC_DIR=str(output.resolve()))
    samples, reason, descendants = [], None, {}
    start = time.monotonic()
    with (output / 'worker.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=reference.ROOT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try: processes = [process] + process.children(recursive=True)
            except psutil.NoSuchProcess: break
            rows = []
            for proc in processes:
                try:
                    rows.append({'pid':proc.pid, 'rss_bytes':proc.memory_info().rss})
                    if proc.pid != child.pid: descendants[proc.pid] = proc
                except psutil.NoSuchProcess: pass
            available = psutil.virtual_memory().available
            rss = sum(row['rss_bytes'] for row in rows)
            try: stage = json.loads((output / 'current_stage.json').read_text())['stage']
            except (FileNotFoundError, json.JSONDecodeError): stage = 'startup'
            samples.append({'elapsed_s':time.monotonic()-start,'rss_bytes':rss,'available_bytes':available,
                'processes':rows,'stage':stage,'swap_used_bytes':psutil.swap_memory().used})
            if rss > args.max_rss_gib*1024**3: reason = 'Aggregate RSS limit exceeded'
            elif available < args.min_available_gib*1024**3: reason = 'Available-memory floor reached'
            elif time.monotonic()-start > args.timeout: reason = 'Time limit exceeded'
            if reason:
                for proc in reversed(processes):
                    try: proc.terminate()
                    except psutil.NoSuchProcess: pass
                _, alive = psutil.wait_procs(processes, timeout=5)
                for proc in alive:
                    try: proc.kill()
                    except psutil.NoSuchProcess: pass
                break
            time.sleep(0.1)
        code = child.wait()
    summary = {'command':command,'returncode':code,'stop_reason':reason,'elapsed_s':time.monotonic()-start,
        'rss_peak_bytes':max((s['rss_bytes'] for s in samples),default=0),
        'available_min_bytes':min((s['available_bytes'] for s in samples),default=0),'sampling_interval_s':0.1,
        'limits':{'max_rss_gib':args.max_rss_gib,'min_available_gib':args.min_available_gib,'timeout_s':args.timeout},
        'note':'RSS is summed over coordinator, spawned workers and resource tracker; shared pages may be counted twice.'}
    reference.write_json(output/'monitor.json', {'summary':summary,'samples':samples})
    print(json.dumps(summary,indent=2),flush=True)
    if code or reason:
        print((output/'worker.log').read_text()[-6000:])
        raise SystemExit(code or 1)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='test':
        reference.runtime('build/cfp-cache-p0-release')
        import pytest
        args = sys.argv[2:]
        if args[:1] == ['--']: args = args[1:]
        raise SystemExit(pytest.main(args))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coordinator',action='store_true')
    parser.add_argument('--num-workers',type=int,choices=[1,2],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--reference-cache',default='build/cfp-cache-p3-validated')
    parser.add_argument('--build-dir',default='build/cfp-cache-p0-release')
    parser.add_argument('--dataset',default='/Volumes/SSD/GitHub/Dennis/dataset_patches/205_SA_L3D14M3')
    parser.add_argument('--max-rss-gib',type=float,default=4)
    parser.add_argument('--min-available-gib',type=float,default=1)
    parser.add_argument('--timeout',type=float,default=600)
    args=parser.parse_args()
    coordinator(args) if args.coordinator else supervise(args)


if __name__=='__main__': main()
