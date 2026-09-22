import hashlib
import json
import operator

import torch
from torch.utils.data import Dataset, Subset


def paired_sample(sample):
    if not isinstance(sample, (tuple, list)) or len(sample) != 3:
        raise ValueError('A paired dataset sample must contain (input, target, sample_id).')
    return sample[0], sample[1]


def _paired_ids(dataset):
    from ....datasets import PairedImageDataset
    if isinstance(dataset, Subset):
        ids = _paired_ids(dataset.dataset)
        return tuple(ids[operator.index(index)] for index in dataset.indices)
    if not isinstance(dataset, PairedImageDataset):
        raise TypeError('from_paired_dataset requires PairedImageDataset or its Subset.')
    if hasattr(dataset, 'sample_ids'):
        return tuple(dataset.sample_ids)
    from pathlib import Path
    return tuple(Path(pair[0]).name for pair in dataset.pairs)


def _paired_config(dataset):
    from ....datasets import PairedImageDataset
    if isinstance(dataset, Subset):
        indices = [operator.index(index) for index in dataset.indices]
        base = dataset.dataset
        while isinstance(base, Subset):
            indices = [operator.index(base.indices[index]) for index in indices]
            base = base.dataset
        return {'subset': _paired_config(base), 'indices': ','.join(str(index) for index in indices)}
    if not isinstance(dataset, PairedImageDataset) or not hasattr(dataset, 'get_config'):
        raise TypeError('Recreating a source requires an explicit paired dataset or its Subset.')
    return {'pairs': dataset.get_config()}


def _paired_from_config(config):
    if set(config) == {'subset', 'indices'}:
        indices = [int(index) for index in config['indices'].split(',')] if config['indices'] else []
        return Subset(_paired_from_config(config['subset']), indices)
    if set(config) != {'pairs'}:
        raise ValueError('Invalid paired source configuration.')
    from ...._datasets._explicit_pairs import ExplicitPairedImageDataset
    return ExplicitPairedImageDataset.from_config(config['pairs'])


class DatasetSource(Dataset):
    def __init__(self, dataset, *, extract_sample, sample_ids):
        if not callable(extract_sample):
            raise TypeError('extract_sample must explicitly return (input, target).')
        if not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'):
            raise TypeError('dataset must be a finite map-style dataset.')
        self.dataset, self.extract_sample = dataset, extract_sample
        self.sample_ids = tuple(sample_ids(index) for index in range(len(dataset))) if callable(sample_ids) else tuple(sample_ids)
        if len(self.sample_ids) != len(dataset) or any(not isinstance(i,str) or not i for i in self.sample_ids):
            raise ValueError('sample_ids must contain one nonempty string per source position.')
        self._paired = False

    @classmethod
    def from_paired_dataset(cls, dataset):
        source = cls(dataset, extract_sample=paired_sample, sample_ids=_paired_ids(dataset))
        source._paired = True
        return source

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        index = operator.index(index)
        sample_id = self.sample_ids[index]
        try:
            sample = self.dataset[index]
            value = self.extract_sample(sample)
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError('extract_sample must return exactly (input, target).')
            if self._paired and sample[2] != sample_id:
                raise ValueError(f'Paired dataset now reports ID {sample[2]!r}.')
            return value[0], value[1]
        except Exception as exc:
            raise ValueError(f'Sample {sample_id!r} at source position {index}: {exc}') from exc

    def validate_manifest(self, store, manifest, *, page_size=128):
        if type(page_size) is not int or not 1 <= page_size <= 1000:
            raise ValueError('page_size must be an integer between 1 and 1000.')
        offset = checked = 0
        expected = None
        while True:
            page = store.inspect_manifest(manifest, offset=offset, limit=page_size)
            header = {key: page[key] for key in ('state','sample_count','bound_samples','config',
                'source_version','preprocessing_version','split')}
            if expected is not None and header != expected:
                raise ValueError(f'Manifest {manifest!r} changed during source alignment.')
            expected = header
            if page['sample_count'] != len(self) or page['bound_samples'] != len(self):
                raise ValueError(f'Manifest {manifest!r}: source length/positions do not match.')
            for sample in page['samples']:
                position = sample['position']
                if position != checked or position >= len(self):
                    raise ValueError(f'Manifest {manifest!r}: unexpected source position {position}.')
                if sample['sample_id'] != self.sample_ids[position]:
                    raise ValueError(f'Manifest {manifest!r}, position {position}: sample ID '
                        f'{sample["sample_id"]!r} differs from source {self.sample_ids[position]!r}.')
                checked += 1
            offset = page['next_offset']
            if offset is None:
                break
        if checked != len(self):
            raise ValueError(f'Manifest {manifest!r}: incomplete source alignment.')
        return {'manifest': manifest, 'checked_samples': checked, 'validation': 'ids_and_positions'}

    def target_fingerprint(self):
        digest = hashlib.sha256()
        for index, sample_id in enumerate(self.sample_ids):
            _, target = self[index]
            if not torch.is_tensor(target):
                raise TypeError(f'Sample {sample_id!r}: target_fingerprint requires tensor targets.')
            target = target.detach().cpu().contiguous()
            header = json.dumps([sample_id, str(target.dtype), list(target.shape)], separators=(',',':')).encode()
            digest.update(len(header).to_bytes(8, 'big'))
            digest.update(header)
            data = target.reshape(-1).view(torch.uint8).numpy()
            digest.update(data.nbytes.to_bytes(8, 'big'))
            digest.update(memoryview(data))
        return digest.hexdigest()

    def get_config(self):
        if not self._paired:
            raise TypeError('Use an importable source factory for a custom dataset/extractor.')
        return _paired_config(self.dataset)


def paired_source_from_config(*, config):
    return DatasetSource.from_paired_dataset(_paired_from_config(config))
