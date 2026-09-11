"""External SSD preparation/resume/reopen probe on original-resolution images."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cfp_cache_p0 as reference


def worker(args):
    build, extension = reference.runtime(args.build_dir)
    torch, cv2, np = reference.torch, reference.cv2, reference.np
    from mtlearn.layers.cfp import CFPPreprocessor, DiskStore, PreparedDataset, collate_prepared
    from mtlearn.layers.cfp.storage import _disk_format
    from mtlearn.layers.cfp.normalization import AttributeNormalizer
    output = Path(args.output)
    reference.write_json(output/'environment.json',reference.provenance(build,extension))
    timing=reference.Timings(output)
    image_ids=[0,1000,2040]
    current=[None]
    counts={'prepare':0,'summaries':0,'model_statistics_updates':0}
    rows=[]
    original_prepare=CFPPreprocessor.prepare_u8
    original_summary=_disk_format.summarize
    def prepare(self,*a,**kw):
        if args.phase=='consume':
            raise AssertionError('Fresh reader must not reconstruct morphology')
        counts['prepare']+=1
        with timing.stage(f'prepare_{current[0]}'):
            result=original_prepare(self,*a,**kw)
        rows.append({'image_id':current[0],'raw_bytes':result.nbytes,'nodes':result.num_nodes})
        return result
    def summarize(prepared):
        if args.phase=='consume':
            raise AssertionError('Fresh reader must not summarize raw attributes')
        counts['summaries']+=1
        with timing.stage(f'summarize_{current[0]}'):
            return original_summary(prepared)
    def forbidden_update(*a,**kw):
        counts['model_statistics_updates']+=1
        raise AssertionError('Preparation/read must not update model normalization')
    CFPPreprocessor.prepare_u8=prepare
    _disk_format.summarize=summarize
    AttributeNormalizer.update=forbidden_update
    class Images(torch.utils.data.Dataset):
        def __len__(self): return len(image_ids)
        def __getitem__(self,index):
            ident=image_ids[index];current[0]=ident
            with timing.stage(f'read_image_{ident}'):
                image=cv2.imread(str(Path(args.dataset)/'enhancement'/f'{ident}.png'),cv2.IMREAD_UNCHANGED)
                mask=cv2.imread(str(Path(args.dataset)/'frag_component'/f'{ident:04d}.png'),cv2.IMREAD_UNCHANGED)
                assert image.shape==mask.shape==(2748,2748)
                return torch.from_numpy(image)[None],torch.from_numpy((mask>0).astype(np.float32))[None]
    source=Images()
    models={kind:reference.layer(kind,all_attrs=True,device=args.device) for kind in ('linear','mlp')}
    preprocessor=CFPPreprocessor.from_layer(models['linear'])
    contract=models['linear'].get_statistics_contract()
    readonly=args.phase=='consume'
    if args.phase=='first' and Path(args.cache_dir).exists():
        raise FileExistsError('The first pass requires a new cache directory')
    with DiskStore(args.cache_dir,max_disk_bytes=None if readonly else 1024**3,
                   max_ram_bytes=0,readonly=readonly) as store:
        before=store.info()
        if args.phase in ('first','resume'):
            cancelled=[0]
            def cancel():
                cancelled[0]+=1
                return cancelled[0]>1
            with timing.stage(f'{args.phase}_pass'):
                result=preprocessor.prepare(source,store=store,manifest='train',source_version='205-SA-L3D14M3-ids-0-1000-2040-v1',
                    preprocessing_version='original-u8-v1',split='train',sample_ids=[str(i) for i in image_ids],
                    collect_stats=contract,cancel=cancel if args.phase=='first' else None)
            status=result.status
            snapshot=result.statistics
            assert counts['prepare']==(1 if args.phase=='first' else 2)
            assert counts['summaries']==counts['prepare']
            assert status==('cancelled' if args.phase=='first' else 'complete')
        else:
            with timing.stage('statistics_from_summaries'):
                snapshot=store.statistics('train',contract)
            status='complete'
        if snapshot is not None:
            fitted=json.loads((reference.ROOT/'docs/cfp-cache/p1/runs/fit_cpu_3_validated/result.json').read_text())
            actual=snapshot.statistics
            assert snapshot.sample_count==3
            for key,values in fitted['statistics'].items():
                for name,value in values.items():
                    expected=torch.tensor(value,dtype=actual[key][name].dtype)
                    torch.testing.assert_close(actual[key][name],expected,rtol=1e-10,atol=1e-12)
            for layer in models.values(): layer.set_stats(snapshot)
        if args.phase=='consume':
            prepared=PreparedDataset(store,'train',source)
            for index in range(args.consume_count):
                ident=image_ids[index]
                with timing.stage(f'load_prepared_{ident}'):
                    batch,target=prepared[index]
                raw=next(iter(batch.samples[0][0].values()))
                row={'image_id':ident,'raw_bytes':raw.nbytes,'nodes':raw.num_nodes,
                     'active_before_forward':store.info(),'models':{}}
                del raw
                target=target.unsqueeze(0).to(args.device)
                for kind,layer in models.items():
                    with timing.stage(f'{kind}_forward_{ident}',args.device):
                        response=layer(batch)
                    with timing.stage(f'{kind}_backward_{ident}',args.device):
                        loss=((response/255-target)**2).mean()
                        loss.backward()
                    gradients=[p.grad for p in layer.parameters() if p.grad is not None]
                    assert torch.isfinite(response).all() and torch.isfinite(loss)
                    assert gradients and all(torch.isfinite(g).all() for g in gradients)
                    row['models'][kind]={'loss':loss.item(),'gradient_l2':sum(g.float().square().sum().item() for g in gradients)**0.5}
                    del response,loss,gradients
                    layer.zero_grad(set_to_none=True)
                    assert layer.cached_sample_count()==0
                del batch,target
                assert store.info()['active_bytes']==0
                rows.append(row)
            assert counts=={'prepare':0,'summaries':0,'model_statistics_updates':0}
        after=store.info()
        assert after['retained_bytes']==0 and after['active_bytes']==0
        assert after['disk_bytes']<=1024**3
        result={'status':status,'phase':args.phase,'device':args.device,'counts':counts,
                'cache_dir':str(Path(args.cache_dir).resolve()),'before':before,'after':after,'images':rows,
                'contributions':store._db.execute('SELECT COUNT(*) FROM contributions').fetchone()[0],
                'statistics_match_p1':snapshot is not None,'sample_count':snapshot.sample_count if snapshot else None,
                'statistics_bytes':reference.tensor_bytes(snapshot.statistics) if snapshot else 0,'timings':timing.rows,
                'note':'Integration/resource probe with MSE, not a full epoch or model-quality evaluation.'}
        assert result['contributions']==(1 if args.phase=='first' else 3)
        reference.write_json(output/'result.json',result)
    print('P3_SSD_PASS',flush=True)


def supervise(args):
    import psutil
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--build-dir',args.build_dir,
             '--dataset',args.dataset,'--output',str(output),'--cache-dir',args.cache_dir,
             '--phase',args.phase,'--device',args.device,'--consume-count',str(args.consume_count)]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    samples=[];reason=None;start=time.monotonic()
    with (output/'worker.log').open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=reference.ROOT)
        process=psutil.Process(child.pid)
        while child.poll() is None:
            try: rss=sum(p.memory_info().rss for p in [process]+process.children(recursive=True))
            except psutil.NoSuchProcess: break
            available=psutil.virtual_memory().available
            try: stage=json.loads((output/'current_stage.json').read_text())['stage']
            except (FileNotFoundError,json.JSONDecodeError): stage='startup'
            samples.append({'elapsed_s':time.monotonic()-start,'rss_bytes':rss,'available_bytes':available,
                            'stage':stage,'swap_used_bytes':psutil.swap_memory().used})
            if rss>args.max_rss_gib*1024**3: reason='RSS limit exceeded'
            elif available<args.min_available_gib*1024**3: reason='Available-memory floor reached'
            elif time.monotonic()-start>args.timeout: reason='Time limit exceeded'
            if reason:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill()
                break
            time.sleep(0.1)
        code=child.wait()
    summary={'command':command,'returncode':code,'stop_reason':reason,'elapsed_s':time.monotonic()-start,
             'rss_peak_bytes':max((s['rss_bytes'] for s in samples),default=0),
             'available_min_bytes':min((s['available_bytes'] for s in samples),default=0),'sampling_interval_s':0.1,
             'limits':{'max_rss_gib':args.max_rss_gib,'min_available_gib':args.min_available_gib,'timeout_s':args.timeout}}
    reference.write_json(output/'monitor.json',{'summary':summary,'samples':samples})
    print(json.dumps(summary,indent=2),flush=True)
    if code:
        print((output/'worker.log').read_text()[-6000:]);raise SystemExit(code)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--build-dir',default='build/cfp-cache-p0-release')
    parser.add_argument('--dataset',default='/Volumes/SSD/GitHub/Dennis/dataset_patches/205_SA_L3D14M3')
    parser.add_argument('--output',required=True)
    parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--phase',required=True,choices=['first','resume','consume'])
    parser.add_argument('--device',default='cpu',choices=['cpu','mps'])
    parser.add_argument('--consume-count',default=3,type=int,choices=[1,2,3])
    parser.add_argument('--max-rss-gib',default=4.0,type=float)
    parser.add_argument('--min-available-gib',default=1.0,type=float)
    parser.add_argument('--timeout',default=600,type=float)
    args=parser.parse_args()
    worker(args) if args.worker else supervise(args)


if __name__=='__main__': main()
