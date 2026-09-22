import dataclasses
import json
import threading

import pytest
import torch

from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer
from mtlearn.layers.cfp import (CFPPreprocessor, DatasetSource, DatasetCacheConfig, DiskStore,
    PreparedDataLoadingError, prepare_dataset_cache, open_dataset_cache)
from test_cfp_disk_preparation_flow import experiment, assert_stats
from test_cfp_dataset_source import first_two, source_cache
from test_reusable_paired_dataset import pair_folders


@pytest.fixture
def experiment_cache(experiment,tmp_path):
    preprocessor,layer,data=experiment
    source=DatasetSource(data,extract_sample=first_two,sample_ids=['a','b','c','d'])
    config=DatasetCacheConfig.from_preprocessor(preprocessor,path=tmp_path/'cache',manifest='train',
        source_version='v1',preprocessing_version='u8')
    result=prepare_dataset_cache(config,source,max_disk_bytes=4*1024**2,collect_stats=layer.get_statistics_contract())
    assert result.status=='complete'
    return config,source,layer,result


def forbidden(*args,**kwargs):
    raise AssertionError('Must not prepare or write')


def test_open_is_readonly_no_trees_writes_or_statistics_fit(experiment_cache,monkeypatch):
    config,source,layer,result=experiment_cache
    monkeypatch.setattr(CFPPreprocessor,'prepare_u8',forbidden)
    monkeypatch.setattr(DiskStore,'get_or_prepare',forbidden)
    monkeypatch.setattr(DiskStore,'statistics',forbidden)
    with open_dataset_cache(config,source,layer=layer,statistics=result.statistics) as loader:
        report=loader.opening_report
        assert report['readonly'] and report['prepared_entries']==report['repaired_entries']==0
        assert report['reused_entries']==4
        assert not loader.prefetch and loader.num_workers==0
        for batch,target in loader:
            assert layer(batch).shape==target.shape
    assert_stats(layer.get_stats(),result.statistics)
    assert DatasetCacheConfig(**json.loads(json.dumps(config.get_config())))==config
    assert DatasetCacheConfig.from_manifest(config.path,'train')==config


def test_attribute_subset_keeps_original_manifest(experiment_cache):
    config,source,_,_=experiment_cache
    layer=Layer(1,[{'tree_type':'max-tree','attributes':[morphology.AttributeType.AREA]}],scale_mode='none')
    with open_dataset_cache(config,source,layer=layer) as loader:
        batch,_=next(iter(loader))
        assert layer(batch).shape==(1,1,6,7)
    assert DatasetCacheConfig.from_manifest(config.path,'train').preparation==config.preparation
    smaller=dataclasses.replace(config,preparation=CFPPreprocessor.from_layer(layer).get_config())
    with pytest.raises(ValueError,match='config changed'):
        open_dataset_cache(smaller,source)
    bad=Layer(1,[{'tree_type':'max-tree','attributes':[morphology.AttributeType.VOLUME]}],scale_mode='none')
    with pytest.raises(ValueError,match='attributes absent'):
        open_dataset_cache(config,source,layer=bad)


@pytest.mark.parametrize('damage',['missing','corrupt'])
def test_corruption_requires_explicit_repair(experiment_cache,damage):
    from pathlib import Path
    config,source,layer,result=experiment_cache
    with DiskStore(config.path,readonly=True) as store:
        sample=store.inspect_manifest('train',limit=1)['samples'][0]
    path=Path(config.path)/'entries'/(sample['entries'][0]['entry_key']+'.pt')
    if damage=='missing':
        path.unlink()
    else:
        with path.open('r+b') as stream:
            stream.write(b'bad!')
    with pytest.raises((RuntimeError,ValueError,PreparedDataLoadingError),match='a|entry|complete'):
        with open_dataset_cache(config,source) as loader:
            next(iter(loader))
    repaired=prepare_dataset_cache(config,source,max_disk_bytes=4*1024**2,collect_stats=layer.get_statistics_contract())
    assert repaired.repaired_entries==1 and repaired.reused_entries==3
    assert_stats(repaired.statistics,result.statistics)


def test_cancel_resume_statistics_and_quota_are_explicit(experiment,tmp_path):
    preprocessor,layer,data=experiment
    source=DatasetSource(data,extract_sample=first_two,sample_ids=['a','b','c','d'])
    config=DatasetCacheConfig.from_preprocessor(preprocessor,path=tmp_path/'cache',manifest='train',
        source_version='v1',preprocessing_version='u8')
    with pytest.raises(TypeError,match='max_disk_bytes'):
        prepare_dataset_cache(config,source)
    cancel=threading.Event()
    def progress(event):
        if event['phase']=='prepare' and event['sample_count']==2:
            cancel.set()
    partial=prepare_dataset_cache(config,source,max_disk_bytes=4*1024**2,cancel=cancel,progress=progress,
        collect_stats=layer.get_statistics_contract())
    assert partial.status=='cancelled' and partial.sample_count==2
    with pytest.raises((ValueError,RuntimeError)):
        open_dataset_cache(config,source)
    result=prepare_dataset_cache(config,source,max_disk_bytes=4*1024**2,collect_stats=layer.get_statistics_contract())
    expected=preprocessor.prepare(source,collect_stats=layer.get_statistics_contract())
    assert result.reused_entries==result.prepared_entries==2
    assert_stats(result.statistics,expected.statistics)


def test_changed_pixels_and_wrong_ids_fail_before_pairing(experiment_cache):
    config,source,_,_=experiment_cache
    wrong=DatasetSource(source.dataset,extract_sample=first_two,sample_ids=['bad','b','c','d'])
    with pytest.raises(ValueError,match='position 0'):
        open_dataset_cache(config,wrong)
    source.dataset.tensors[0][0,0,0,0]^=255
    with open_dataset_cache(config,source) as loader:
        with pytest.raises(PreparedDataLoadingError,match="samples.*a.*pixels"):
            next(iter(loader))


@pytest.mark.parametrize('workers',[0,1,2])
def test_composed_source_recreation_shuffle_and_repetitions(source_cache,workers):
    source,path=source_cache
    config=DatasetCacheConfig.from_manifest(path,'train')
    options={} if not workers else dict(num_workers=workers,worker_memory_bytes=128*1024**2,
        max_prefetch_bytes=8*1024**2,max_batch_bytes=4*1024**2,source_memory_bytes=1024,read_workspace_bytes=1024)
    orders=[]
    for _ in range(2):
        with open_dataset_cache(config,source,shuffle=True,generator=torch.Generator().manual_seed(9),**options) as loader:
            assert loader.num_workers==workers
            orders.append([batch.sample_ids[0] for batch,_ in loader])
        assert loader.counters().get('processes_alive',0)==0
    assert orders[0]==orders[1]
    with open_dataset_cache(config,source,sampler=[2,0,2],**options) as loader:
        assert [batch.sample_ids[0] for batch,_ in loader]==[source.sample_ids[i] for i in (2,0,2)]


def test_variable_resolution_is_not_resized_or_padded(experiment,tmp_path):
    preprocessor,_,_=experiment
    data=[(torch.full((1,h,w),i*53,dtype=torch.uint8),torch.zeros((1,h,w)))
          for i,(h,w) in enumerate([(4,5),(6,7)])]
    source=DatasetSource(data,extract_sample=first_two,sample_ids=['small','large'])
    config=DatasetCacheConfig.from_preprocessor(preprocessor,path=tmp_path/'variable',manifest='train',
        source_version='v1',preprocessing_version='u8')
    assert prepare_dataset_cache(config,source,max_disk_bytes=4*1024**2).status=='complete'
    with open_dataset_cache(config,source) as loader:
        assert [batch.shape for batch,_ in loader]==[(1,1,4,5),(1,1,6,7)]
    with open_dataset_cache(config,source,batch_size=2) as loader:
        with pytest.raises(PreparedDataLoadingError,match='batch_size=1'):
            next(iter(loader))
