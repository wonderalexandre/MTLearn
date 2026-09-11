"""Execute the P5 notebook on a small original-resolution subset under supervision."""
import argparse
import ast
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cfp_cache_p0 as reference

NOTEBOOK = reference.ROOT / 'notebooks/experiments/CFP_linear_vs_mlp_scoring_205_SA_L3D14M3_segmentation.ipynb'


def replace_assignments(source, changes):
    lines = source.splitlines(keepends=True)
    edits = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in changes:
            name = node.targets[0].id
            edits.append((node.lineno-1,node.end_lineno,f'{name} = {changes.pop(name)}\n'))
    if changes: raise ValueError(f'Missing parameter assignments: {changes}')
    for begin,end,text in reversed(edits): lines[begin:end] = [text]
    return ''.join(lines)


def execute(args):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    output = Path(args.output).resolve()
    # Restrict the editable-install workaround to the diagnostic kernel and its
    # children. The canonical notebook and installed environment are unchanged.
    bootstrap = output / 'kernel_bootstrap'
    bootstrap.mkdir()
    native = (reference.ROOT / args.build_dir / 'mtlearn/bindings').resolve()
    (bootstrap/'sitecustomize.py').write_text(
        "import sys\n"
        "sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != '_mtlearn_editable']\n"
        f"sys.path[:0] = {[str(reference.ROOT/'mtlearn/python'),str(native)]!r}\n")
    env = dict(os.environ, PYTHONPATH=str(bootstrap), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    notebook = nbformat.read(NOTEBOOK,as_version=4)
    overrides = {'MAX_SAMPLES':str(args.samples),'NUM_EPOCHS':'1','REPORT_EPOCHS':'{1}',
                 'CACHE_DIR':f'Path({str(Path(args.cache_dir).resolve())!r})',
                 'CACHE_MAX_DISK_BYTES':str(1024**3),'CACHE_MAX_RAM_BYTES':'0',
                 'PREPARATION_WORKERS':'1','PREPARATION_MAX_IN_FLIGHT':'1'}
    notebook.cells[7].source = replace_assignments(notebook.cells[7].source,dict(overrides))
    notebook.cells[5].source = replace_assignments(notebook.cells[5].source,{'device':repr(args.device)})
    prefix = nbformat.v4.new_code_cell(f'''
import sys
from pathlib import Path
sys.path.insert(0, {str(reference.ROOT/'scripts/benchmarks')!r})
import cfp_cache_p0 as diagnostic_reference
build, extension = diagnostic_reference.runtime({args.build_dir!r})
import cv2
cv2.setNumThreads(0)
P5_OUTPUT = Path({str(output)!r})
diagnostic_reference.write_json(P5_OUTPUT / "provenance.json", diagnostic_reference.provenance(build, extension))
''')
    guard = nbformat.v4.new_code_cell('''
# Diagnostic assertions only; this cell is absent from the canonical notebook.
from mtlearn.layers.cfp.normalization import AttributeNormalizer
p5_forbidden_calls = {"morphology": 0, "stats_update": 0, "stats_merge": 0, "stats_summary": 0}
def p5_forbid(name):
    def forbidden(*args, **kwargs):
        p5_forbidden_calls[name] += 1
        raise AssertionError(f"Unexpected {name} during training/evaluation")
    return forbidden
CFPPreprocessor.prepare_u8 = p5_forbid("morphology")
AttributeNormalizer.update = p5_forbid("stats_update")
AttributeNormalizer.merge = p5_forbid("stats_merge")
AttributeNormalizer.summarize = p5_forbid("stats_summary")
def p5_parameter_hash(layer):
    digest = hashlib.sha256()
    for name, parameter in layer.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
p5_initial_hashes = {name: p5_parameter_hash(layer) for name, layer in cfp_layers.items()}
p5_gradient_norms = []
p5_clip = torch.nn.utils.clip_grad_norm_
def p5_check_gradients(parameters, *args, **kwargs):
    parameters = list(parameters)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    norm = p5_clip(parameters, *args, **kwargs)
    p5_gradient_norms.append(float(norm.detach().cpu()))
    return norm
torch.nn.utils.clip_grad_norm_ = p5_check_gradients
''')
    original_cells = list(notebook.cells)
    notebook.cells = [prefix]
    for index,cell in enumerate(original_cells):
        notebook.cells.append(cell)
        if index == 11: notebook.cells.append(guard)
    # Add final evidence before the reader-closing cell.
    evidence = nbformat.v4.new_code_cell('''
import json
assert not any(p5_forbidden_calls.values()), p5_forbidden_calls
assert train_preparation.statistics.sample_count == len(trainset)
assert len(trainset) + len(testset) == MAX_SAMPLES
assert len(p5_gradient_norms) == 2 * len(prepared_loaders["train"])
assert all(math.isfinite(value) and value > 0 for value in p5_gradient_norms)
assert NUM_ROWS is None and NUM_COLS is None and BATCH_SIZE == 1
assert all(layer.cached_sample_count() == 0 for layer in cfp_layers.values())
for model_name, layer in cfp_layers.items():
    assert p5_initial_hashes[model_name] != p5_parameter_hash(layer)
    for key, fields in train_preparation.statistics.statistics.items():
        for name, value in fields.items():
            torch.testing.assert_close(layer.get_extra_state()["ds_stats"][key][name].cpu(), value)
    assert int((trajectory["model"] == model_name).sum()) == 2
cache = cache_store.info()
assert cache["retained_bytes"] == 0 and cache["active_bytes"] == 0, cache
assert cache["writes"] == 0
assert not plt.get_fignums()
trajectory.to_csv(P5_OUTPUT / "trajectory.csv", index=False)
split_manifest.to_csv(P5_OUTPUT / "split.csv", index=False)
summary = {
    "device": device, "sample_count": len(dataset), "train_ids": train_ids, "test_ids": test_ids,
    "loaded_size": [loaded_rows, loaded_cols], "epochs_per_model": NUM_EPOCHS,
    "gradient_norms_before_clip": p5_gradient_norms, "forbidden_calls": p5_forbidden_calls,
    "foreground_fraction": FOREGROUND_FRACTION, "positive_bce_weight": POSITIVE_BCE_WEIGHT,
    "preparation": preparation_info, "reader": cache, "selected_thresholds": selected_thresholds,
    "statistics_id": train_preparation.statistics_id, "manifest_names": manifest_names,
    "initial_parameter_sha256": p5_initial_hashes,
    "final_parameter_sha256": {name: p5_parameter_hash(layer) for name, layer in cfp_layers.items()},
    "config": config, "trajectory": trajectory.where(trajectory.notna(), None).to_dict(orient="records"),
    "note": "Integration test with three original-size images and one epoch per scorer; not a model-quality conclusion."
}
# Missing train_epoch_loss at epoch 0 is represented as null in JSON.
for row in summary["trajectory"]:
    for key, value in row.items():
        if isinstance(value, float) and math.isnan(value): row[key] = None
diagnostic_reference.write_json(P5_OUTPUT / "result.json", summary)
print("P5_NOTEBOOK_PASS")
''')
    notebook.cells.insert(len(notebook.cells)-1,evidence)
    notebook.metadata['cfp_p5_validation']={'source':str(NOTEBOOK),'source_sha256':reference.digest(NOTEBOOK),
        'overrides':overrides,'device':args.device,'original_resolution':True}
    for cell in notebook.cells:
        if cell.cell_type == 'code': cell.outputs=[];cell.execution_count=None
    nbformat.validate(notebook)
    nbformat.write(notebook,output/'validation-input.ipynb')
    timings=[]
    starts={}
    def start(cell,cell_index):
        starts[cell_index]=time.monotonic()
        reference.write_json(output/'current_stage.json',{'cell_index':cell_index,'stage':cell.source.splitlines()[0] if cell.source else 'empty'})
    def finished(cell,cell_index,execute_reply):
        timings.append({'cell_index':cell_index,'seconds':time.monotonic()-starts[cell_index],
                        'status':execute_reply['content']['status']})
        reference.write_json(output/'timings.json',timings)
        print(json.dumps(timings[-1]),flush=True)
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(notebook,km=manager,timeout=600,allow_errors=False,record_timing=True,
        resources={'metadata':{'path':str(reference.ROOT)}},on_cell_start=start,on_cell_executed=finished)
    try:
        client.execute(env=env,cwd=str(reference.ROOT),cleanup_kc=True)
    finally:
        nbformat.write(notebook,output/'executed.ipynb')
    figures=output/'figures';figures.mkdir()
    for index,cell in enumerate(notebook.cells):
        for number,cell_output in enumerate(cell.get('outputs',[])):
            png=cell_output.get('data',{}).get('image/png')
            if png: (figures/f'cell-{index}-{number}.png').write_bytes(base64.b64decode(png))
    errors=[o for c in notebook.cells for o in c.get('outputs',[]) if o.output_type=='error']
    assert not errors
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type=='code')
    assert (output/'result.json').exists()
    print('P5_EXECUTION_PASS',flush=True)


def supervise(args):
    import psutil
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),'--execute','--output',str(output),
             '--cache-dir',args.cache_dir,'--device',args.device,'--samples',str(args.samples),'--build-dir',args.build_dir]
    samples=[];reason=None;start=time.monotonic()
    with (output/'worker.log').open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=reference.ROOT)
        process=psutil.Process(child.pid)
        while child.poll() is None:
            try: processes=[process]+process.children(recursive=True)
            except psutil.NoSuchProcess: break
            rows=[]
            for proc in processes:
                try: rows.append({'pid':proc.pid,'rss_bytes':proc.memory_info().rss})
                except psutil.NoSuchProcess: pass
            available=psutil.virtual_memory().available;rss=sum(p['rss_bytes'] for p in rows)
            try: stage=json.loads((output/'current_stage.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError): stage={'stage':'startup'}
            samples.append({'elapsed_s':time.monotonic()-start,'rss_bytes':rss,'available_bytes':available,
                            'processes':rows,**stage,'swap_used_bytes':psutil.swap_memory().used})
            if rss>args.max_rss_gib*1024**3: reason='Aggregate RSS limit exceeded'
            elif available<args.min_available_gib*1024**3: reason='Available-memory floor reached'
            elif time.monotonic()-start>args.timeout: reason='Timeout'
            if reason:
                for proc in reversed(processes):
                    try: proc.terminate()
                    except psutil.NoSuchProcess: pass
                _,alive=psutil.wait_procs(processes,timeout=5)
                for proc in alive:
                    try: proc.kill()
                    except psutil.NoSuchProcess: pass
                break
            time.sleep(.1)
        code=child.wait()
    result={'command':command,'returncode':code,'stop_reason':reason,'elapsed_s':time.monotonic()-start,
        'rss_peak_bytes':max((s['rss_bytes'] for s in samples),default=0),
        'available_min_bytes':min((s['available_bytes'] for s in samples),default=0),
        'limits':{'max_rss_gib':args.max_rss_gib,'min_available_gib':args.min_available_gib,'timeout_s':args.timeout},
        'sampling_interval_s':.1,'note':'RSS summed across execution driver, Jupyter kernel and descendants; does not measure all MPS driver allocations.'}
    reference.write_json(output/'monitor.json',{'summary':result,'samples':samples})
    print(json.dumps(result,indent=2),flush=True)
    if code or reason:
        print((output/'worker.log').read_text()[-8000:]);raise SystemExit(code or 1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',required=True)
    parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--samples',default=3,type=int)
    parser.add_argument('--device',default='mps',choices=['mps','cpu'])
    parser.add_argument('--build-dir',default='build/cfp-cache-p0-release')
    parser.add_argument('--max-rss-gib',default=4.,type=float)
    parser.add_argument('--min-available-gib',default=1.,type=float)
    parser.add_argument('--timeout',default=900,type=float)
    args=parser.parse_args()
    execute(args) if args.execute else supervise(args)


if __name__=='__main__': main()
