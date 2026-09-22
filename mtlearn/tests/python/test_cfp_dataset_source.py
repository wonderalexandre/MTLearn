import json

import pytest
import torch
from torch.utils.data import Subset

from mtlearn import morphology
from mtlearn.datasets import PairedImageDataset
from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer
from mtlearn.layers.cfp import CFPPreprocessor, DatasetSource, DiskStore, build_prepared_dataloader, paired_source_from_config
from test_reusable_paired_dataset import pair_folders


def first_two(sample):
    return sample[0], sample[1]


@pytest.fixture
def source_cache(pair_folders, tmp_path):
    dataset = PairedImageDataset.from_folders(pair_folders/'inputs',pair_folders/'targets',
        ordering='numeric', strict_grayscale_uint8=True,binary_target=True)
    source = DatasetSource.from_paired_dataset(Subset(Subset(dataset,[3,1,0,2]),[2,0,1,3]))
    layer = Layer(1,[{'name':'area','tree_type':'max-tree','attributes':[morphology.AttributeType.AREA]}], device='cpu')
    preprocessor = CFPPreprocessor.from_layer(layer)
    path=tmp_path/'cache'
    result = preprocessor.prepare_or_reuse(source,path=path,manifest='train',source_version='v1',
        preprocessing_version='u8',sample_ids=source.sample_ids,mode='prepare_missing',max_disk_bytes=8*1024**2)
    assert result.status=='complete'
    return source,path


def test_explicit_extraction_and_ids_are_separate():
    with pytest.raises(TypeError):
        DatasetSource([(torch.zeros(1,2,2),torch.ones(1,2,2),'id')])
    source = DatasetSource([(torch.zeros(1,2,2),torch.ones(1,2,2),'ignored')],
        extract_sample=first_two, sample_ids=['explicit'])
    assert source.sample_ids==('explicit',) and len(source[0])==2


def test_nested_subsets_repetition_and_paginated_alignment(source_cache):
    source,path=source_cache
    assert source.sample_ids==('1','20','2','10')
    with DiskStore(path,readonly=True) as store:
        assert source.validate_manifest(store,'train',page_size=1)['checked_samples']==4
        wrong=DatasetSource(source.dataset,extract_sample=first_two,sample_ids=['wrong',*source.sample_ids[1:]])
        with pytest.raises(ValueError,match="position 0.*'1'.*'wrong'"):
            wrong.validate_manifest(store,'train',page_size=2)
    repeated=DatasetSource.from_paired_dataset(Subset(source.dataset,[0,0,2]))
    assert repeated.sample_ids==('1','1','2')
    config=json.loads(json.dumps(source.get_config()))
    rebuilt=paired_source_from_config(config=config)
    assert rebuilt.sample_ids==source.sample_ids
    assert rebuilt.target_fingerprint()==source.target_fingerprint()


@pytest.mark.parametrize('workers',[0,1,2])
def test_reordered_parallel_readers_and_early_close(source_cache,workers):
    source,path=source_cache
    options={} if not workers else dict(num_workers=workers,source_factory=paired_source_from_config,
        source_factory_kwargs={'config':source.get_config()},worker_memory_bytes=128*1024**2,
        max_batch_bytes=4*1024**2,source_memory_bytes=1024,read_workspace_bytes=1024,max_prefetch_bytes=8*1024**2)
    sampler=[2,0,0,3,1]
    with build_prepared_dataloader(path,'train',source=source,sampler=sampler,**options) as loader:
        assert loader.num_workers==workers
        for (batch,target),position in zip(loader,sampler):
            assert batch.sample_ids==(source.sample_ids[position],)
            torch.testing.assert_close(target[0],source[position][1],rtol=0,atol=0)
        iterator=iter(loader)
        next(iterator)
        assert iterator.close()
    assert not loader.counters().get('processes_alive',0)


def test_changed_mask_reuses_pixels_and_changes_provenance(source_cache):
    import cv2
    import numpy as np
    source,path=source_cache
    before=source.target_fingerprint()
    dataset=source.dataset.dataset.dataset
    assert cv2.imwrite(dataset.records[0].target_path,np.zeros((5,6),np.uint8))
    assert source.target_fingerprint()!=before
    with build_prepared_dataloader(path,'train',source=source,sampler=[0]) as loader:
        batch,target=next(iter(loader))
        assert batch.sample_ids==('1',) and torch.count_nonzero(target)==0


def test_large_subset_factory_stays_within_existing_metadata_budget(pair_folders):
    from mtlearn.layers.cfp.preparation._reader_process import factory_descriptor
    dataset=PairedImageDataset.from_folders(pair_folders/'inputs',pair_folders/'targets',ordering='numeric')
    source=DatasetSource.from_paired_dataset(Subset(Subset(dataset,[3,0,2,1]),[0,2,1,3]*375))
    descriptor=factory_descriptor(paired_source_from_config,{'config':source.get_config()})
    assert len(descriptor[2])<64*1024
    recreated=paired_source_from_config(config=json.loads(json.dumps(source.get_config())))
    assert recreated.sample_ids==source.sample_ids
    (pair_folders/'targets/1.png').unlink()
    with pytest.raises(ValueError,match='Unmatched|sample IDs changed'):
        paired_source_from_config(config=source.get_config())


def test_legacy_cfp_import_does_not_load_image_dataset_dependencies():
    import os
    import subprocess
    import sys
    subprocess.run([sys.executable,'-S','-c',
        "import json, sys; sys.path[:] = json.loads(sys.argv[1]); "
        "import mtlearn.layers.cfp; assert 'cv2' not in sys.modules; assert 'pandas' not in sys.modules",
        json.dumps(sys.path)], check=True,env=os.environ.copy())
