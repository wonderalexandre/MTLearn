import random

import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from mtlearn.datasets import split_ids_by_group, subsets_from_ids, validate_split_ids


class MetadataOnly:
    sample_ids = ('01', '1', 'β', 'four', 'five')

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        raise AssertionError('Identity operations must not read samples')


def test_split_order_identity_and_report():
    dataset = MetadataOnly()
    groups = {'01': 'a', '1': 'a', 'β': 'b', 'four': 'c', 'five': 'c'}
    splits = {'evaluation': ['β'], 'learning': ['1', '01']}
    report = validate_split_ids(dataset.sample_ids, splits, groups)
    assert report.sample_counts == {'evaluation': 1, 'learning': 2}
    assert report.group_counts == {'evaluation': 1, 'learning': 1}
    assert report.unassigned_ids == ('four', 'five')
    assert report.checks['complete_groups']
    subsets = subsets_from_ids(dataset, splits, groups=groups)
    assert list(subsets) == ['evaluation', 'learning']
    assert subsets['learning'].dataset is dataset
    assert subsets['learning'].indices == [1, 0]
    exported = report.to_dict()
    exported['sample_counts']['learning'] = 0
    assert report.sample_counts['learning'] == 2
    with pytest.raises(TypeError):
        report.sample_counts['learning'] = 0


@pytest.mark.parametrize('ids,splits,groups,options,match', [
    (['a', 'a'], {'train': ['a']}, None, {}, 'duplicate'),
    (['a', ''], {'train': ['a']}, None, {}, 'nonempty'),
    (['a', 1], {'train': ['a']}, None, {}, 'string'),
    (['a'], {'train': ['a', 'a']}, None, {}, 'duplicate'),
    (['a'], {'train': ['a'], 'test': ['a']}, None, {}, 'splits'),
    (['a'], {'train': ['unknown']}, None, {}, 'unknown'),
    (['a'], {'': ['a']}, None, {}, 'keys'),
    (['a'], {'train': []}, None, {}, 'empty'),
    (['a'], {}, None, {'allow_empty': True}, 'At least'),
    (['a'], {'train': []}, None, {'allow_empty': True}, 'At least'),
    (['a', 'b'], {'train': ['a']}, None, {'require_complete': True}, 'Unassigned'),
    (['a'], {'train': ['a']}, {}, {}, 'missing'),
    (['a'], {'train': ['a']}, {'a': ''}, {}, 'nonempty'),
    (['a'], {'train': ['a']}, {'a': 1}, {}, 'nonempty'),
    (['a'], {'train': ['a']}, {'a': 'g', 'b': 'g'}, {}, 'unknown'),
    (['a', 'b'], {'train': ['a'], 'test': ['b']}, {'a': 'g', 'b': 'g'}, {}, 'Group'),
    (['a', 'b'], {'train': ['a']}, {'a': 'g', 'b': 'g'}, {}, 'incomplete'),
    (['a'], {'train': ['a']}, None, {'require_complete': 1}, 'boolean'),
    (['a'], {'train': 'a'}, None, {}, 'ordered'),
    ({'a', 'b'}, {'train': ['a']}, None, {}, 'ordered'),
])
def test_invalid_splits(ids, splits, groups, options, match):
    with pytest.raises((TypeError, ValueError), match=match):
        validate_split_ids(ids, splits, groups, **options)


def test_partial_groups_and_empty_split_are_explicit():
    report = validate_split_ids(['a', 'b'], {'train': ['a'], 'empty': []}, {'a': 'g', 'b': 'g'},
                                require_complete_groups=False, allow_empty=True)
    assert report.unassigned_ids == ('b',)
    assert report.group_counts['empty'] == 0
    assert not report.checks['complete_groups']


def test_nested_subsets_and_explicit_metadata():
    base = MetadataOnly()
    dataset = Subset(Subset(base, [4, 0, 2, 1]), [3, 0, 1])
    result = subsets_from_ids(dataset, {'train': ['01', 'five'], 'test': ['1']}, require_complete=True)
    assert result['train'].dataset is dataset
    assert result['train'].indices == [2, 1]
    repeated = Subset(base, [0, 0])
    with pytest.raises(ValueError, match='duplicate'):
        subsets_from_ids(repeated, {'train': ['01']})
    class NoMetadata:
        def __len__(self): return 2
        def __getitem__(self, index): raise AssertionError('No reading')
    with pytest.raises(ValueError, match='explicitly'):
        subsets_from_ids(NoMetadata(), {'train': ['a']})
    assert subsets_from_ids(NoMetadata(), {'train': ['b']}, sample_ids=['a', 'b'])['train'].indices == [1]
    with pytest.raises(ValueError, match='length'):
        subsets_from_ids(NoMetadata(), {'train': ['a']}, sample_ids=['a'])


def test_groups_match_historical_permutation_without_changing_rng():
    ids = ('z2', 'x1', 'z1', 'y1', 'x2', 'w1')
    groups = {sample_id: sample_id[0] for sample_id in ids}
    counts = {'test': 1, 'train': 2, 'val': 1}
    expected, start = {}, 0
    names = sorted(set(groups.values()))
    order = np.random.RandomState(17).permutation(len(names))
    for split, count in counts.items():
        expected[split] = [sample_id for position in order[start:start + count]
                           for sample_id in ids if groups[sample_id] == names[position]]
        start += count
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert split_ids_by_group(ids, groups, counts, seed=17) == expected
    assert split_ids_by_group(ids, groups, counts, seed=17) == expected
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0] and actual_numpy[2:] == numpy_state[2:]
    np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    validate_split_ids(ids, expected, groups, require_complete=True)


@pytest.mark.parametrize('counts,seed', [({'train': True}, 0), ({'train': 0}, 0),
    ({'train': 1.0}, 0), ({'train': 2}, 0), ({'train': 1}, True), ({'train': 1}, -1), ({'train': 1}, None)])
def test_invalid_group_counts_and_seeds(counts, seed):
    with pytest.raises(ValueError):
        split_ids_by_group(['a'], {'a': 'g'}, counts, seed)
