import json

import cv2
import numpy as np
import pytest
import torch

from mtlearn.datasets import PairedImageDataset

from _paired_dataset_reference import EnhancementFragmentDataset


@pytest.fixture
def pair_folders(tmp_path):
    for folder in ('inputs', 'targets'):
        (tmp_path/folder).mkdir()
    for i in (10, 2, 1, 20):
        image = (np.arange(30).reshape(5, 6)*7+i).astype(np.uint8)
        mask = ((image > 100) * (1 if i%2 else 255)).astype(np.uint8)
        assert cv2.imwrite(str(tmp_path/'inputs'/f'{i:02}.png'), image)
        assert cv2.imwrite(str(tmp_path/'targets'/f'{i}.png'), mask)
    return tmp_path


@pytest.mark.parametrize('resize', [None, (3, 4)])
@pytest.mark.parametrize('invert', [False, True])
def test_matches_extracted_reference_without_running_notebook(pair_folders, resize, invert):
    original = EnhancementFragmentDataset(pair_folders, *(resize or (None,None)), invert_input=invert)
    dataset = PairedImageDataset.from_folders(pair_folders/'inputs', pair_folders/'targets',
        ordering='numeric', strict_grayscale_uint8=True, binary_target=True, invert_in=invert,
        num_rows=None if resize is None else resize[0], num_cols=None if resize is None else resize[1])
    assert isinstance(dataset, PairedImageDataset)
    assert dataset.sample_ids == ('1', '2', '10', '20')
    for i in range(len(original)):
        for a,b in zip(dataset[i], original[i]):
            if isinstance(a, str):
                assert a == b
            else:
                torch.testing.assert_close(a,b,rtol=0,atol=0)
    reconstructed = type(dataset).from_config(json.loads(json.dumps(dataset.get_config())))
    assert reconstructed.sample_ids == dataset.sample_ids
    assert dataset.train_test_split(test_size=1)[0].indices == [3, 0, 2]
    assert not hasattr(dataset, '_cache')


def test_metadata_without_decoding_and_explicit_order(pair_folders, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('decoded during metadata access')
    monkeypatch.setattr(cv2, 'imread', forbidden)
    dataset = PairedImageDataset.from_pairs([
        ('chosen-2',pair_folders/'inputs/02.png',pair_folders/'targets/2.png'),
        ('chosen-1',pair_folders/'inputs/01.png',pair_folders/'targets/1.png')])
    assert dataset.sample_ids == ('chosen-2','chosen-1')
    assert dataset.records[0].sample_id == 'chosen-2'


def test_numeric_collision_and_unmatched_policy(pair_folders):
    (pair_folders/'inputs/1.png').write_bytes((pair_folders/'inputs/01.png').read_bytes())
    with pytest.raises(ValueError,match='Duplicate sample ID'):
        PairedImageDataset.from_folders(pair_folders/'inputs',pair_folders/'targets',ordering='numeric')
    (pair_folders/'inputs/1.png').unlink()
    (pair_folders/'targets/2.png').unlink()
    with pytest.raises(ValueError,match='Unmatched'):
        PairedImageDataset.from_folders(pair_folders/'inputs',pair_folders/'targets',ordering='numeric')
    with pytest.warns(UserWarning,match='Unmatched'):
        dataset = PairedImageDataset.from_folders(pair_folders/'inputs',pair_folders/'targets',ordering='numeric',unmatched='warn')
    assert dataset.missing_target_ids == ('2',)
    with pytest.raises(ValueError,match='Duplicate sample ID'):
        PairedImageDataset.from_pairs([('01',*dataset.pairs[0]),('1',*dataset.pairs[1])],ordering='numeric')


@pytest.mark.parametrize('bad', ['rgb','uint16','nonbinary','shape'])
def test_strict_errors_include_sample(pair_folders, bad):
    dataset = PairedImageDataset.from_folders(pair_folders/'inputs',pair_folders/'targets',
        ordering='numeric',strict_grayscale_uint8=True,binary_target=True,num_rows=2,num_cols=2)
    image = {'rgb':np.zeros((5,6,3),np.uint8), 'uint16':np.zeros((5,6),np.uint16),
             'nonbinary':np.full((5,6),2,np.uint8), 'shape':np.zeros((2,2),np.uint8)}[bad]
    assert cv2.imwrite(str(pair_folders/'targets/1.png'), image)
    with pytest.raises(ValueError,match="Sample '1'"):
        dataset[0]


def test_textual_order_and_explicit_duplicates(pair_folders):
    dataset=PairedImageDataset.from_pairs([
        ('2',pair_folders/'inputs/02.png',pair_folders/'targets/2.png'),
        ('10',pair_folders/'inputs/10.png',pair_folders/'targets/10.png')],ordering='textual')
    assert dataset.sample_ids==('10','2')
    with pytest.raises(ValueError,match='Duplicate'):
        PairedImageDataset.from_pairs([('2',*dataset.pairs[0]),('2',*dataset.pairs[1])])
