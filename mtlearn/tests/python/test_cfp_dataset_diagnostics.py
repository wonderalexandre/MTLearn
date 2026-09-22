import copy
import json
import time

import pytest
import torch

from mtlearn.layers.cfp import (DatasetReaderConfig, DiskStore, calibrate_disk_cache,
    inspect_dataset_cache, open_dataset_cache)
from test_cfp_dataset_cache import experiment_cache
from test_cfp_disk_preparation_flow import experiment, assert_stats


def test_metadata_payload_and_source_validation_are_distinct(experiment_cache,monkeypatch):
    config,source,_,_=experiment_cache
    original=DiskStore.get
    def forbidden(*args,**kwargs):
        raise AssertionError('Metadata cannot read payloads')
    monkeypatch.setattr(DiskStore,'get',forbidden)
    metadata=inspect_dataset_cache(config,limit=2)
    assert not metadata['payloads_checked'] and not metadata['source_pixels_checked']
    monkeypatch.setattr(DiskStore,'get',original)
    payload=inspect_dataset_cache(config,validation='payload',limit=2)
    assert payload['integrity']['checked_samples']==2 and not payload['integrity']['fully_validated']
    assert not payload['source_pixels_checked']
    verified=inspect_dataset_cache(config,validation='source',source=source,limit=4)
    assert verified['source']['fully_validated']
    source.dataset.tensors[0][0,0,0,0]^=255
    verified=inspect_dataset_cache(config,validation='source',source=source,limit=1)
    assert verified['source']['status']=='invalid'
    assert verified['source']['errors'][0]['sample_id']=='a'
    json.dumps(verified)


def test_explicit_reader_resources_and_snapshot_reuse(experiment_cache):
    config,source,layer,prepared=experiment_cache
    reader=DatasetReaderConfig(dict(prefetch=True,source_thread_safe=True,source_memory_bytes=1024,
        max_prefetch_bytes=1024**2,max_batch_bytes=1024**2,read_workspace_bytes=1024,
        collect_read_metrics=True))
    assert DatasetReaderConfig(json.loads(json.dumps(reader.get_config())))==reader
    with open_dataset_cache(config,source,layer=layer,statistics=prepared.statistics,reader_config=reader) as loader:
        for batch,target in loader:
            layer(batch)
        assert loader.counters()['queued_bytes']==0
        assert loader.counters()['retained_bytes']==0
        assert loader.counters()['peak_prefetch_reserved_bytes']<=1024**2
    assert_stats(layer.get_stats(),prepared.statistics)
    with pytest.raises(ValueError,match='only once'):
        open_dataset_cache(config,source,reader_config=reader,prefetch=True)
    with pytest.raises(ValueError):
        open_dataset_cache(config,source,reader_config=DatasetReaderConfig({'max_batch_bytes':1}))
    with pytest.raises(ValueError,match='Unsupported'):
        DatasetReaderConfig({'unknown':1})


def test_calibration_is_bounded_and_distinct_from_complete_cycle(experiment_cache):
    config,source,layer,prepared=experiment_cache
    report=calibrate_disk_cache(config.path,config.manifest,max_samples=2,max_seconds=10,
        max_memory_bytes=64*1024**2,max_cache_read_bytes=4*1024**2,
        read_workspace_bytes=1024,modes=('thread',),repeats=2,warmup_rounds=0)
    assert report['status']=='complete'
    assert not report['source_and_targets_read'] and not report['model_executed']
    assert report['reserved_cache_read_bytes']<=4*1024**2
    state=copy.deepcopy(layer.state_dict())
    started=time.perf_counter()
    with open_dataset_cache(config,source,layer=layer,statistics=prepared.statistics,sampler=[0,1],
                            collect_read_metrics=True) as loader:
        optimizer=torch.optim.Adam(layer.parameters(),lr=.02)
        for batch,target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=((layer(batch)/255-target)**2).mean()
            loss.backward()
            optimizer.step()
    assert time.perf_counter()-started>0
    assert_stats(layer.get_stats(),prepared.statistics)
