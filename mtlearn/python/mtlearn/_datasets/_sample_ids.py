"""Resolve opaque sample IDs from metadata without reading image tensors."""

from collections.abc import Mapping, Set
import operator

from torch.utils.data import Subset


def _ids(values, name, *, unique=True, nonempty=False):
    if isinstance(values, (str, bytes, Mapping, Set)):
        raise TypeError(f'{name} must be an ordered iterable of IDs')
    values = tuple(values)
    seen = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f'{name} must contain nonempty string IDs: {value!r}')
        if unique and value in seen:
            raise ValueError(f'{name} contains duplicate ID {value!r}')
        seen.add(value)
    if nonempty and not values:
        raise ValueError(f'{name} must not be empty')
    return values


def _resolve_sample_ids(dataset, sample_ids=None):
    if sample_ids is not None:
        values = _ids(sample_ids, 'sample_ids', unique=False)
    elif hasattr(dataset, 'sample_ids'):
        values = _ids(dataset.sample_ids, 'dataset.sample_ids', unique=False)
    elif isinstance(dataset, Subset):
        base = _resolve_sample_ids(dataset.dataset)
        values = tuple(base[operator.index(index)] for index in dataset.indices)
    else:
        raise ValueError('Provide sample_ids explicitly for a dataset without identity metadata')
    if len(values) != len(dataset):
        raise ValueError('sample_ids length must match the dataset')
    return values
