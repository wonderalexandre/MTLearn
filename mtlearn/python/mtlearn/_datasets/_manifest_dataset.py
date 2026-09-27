"""Reconstruct fixed paired datasets from compact manifest references."""

from pathlib import Path

import torch

from ._dataset_contracts import _json_copy, _preprocessing, _reader_options, _roots


def manifest_dataset(path, *, roots, split=None, expected_fingerprint=None):
    from ._explicit_pairs import ExplicitPairedImageDataset
    from ._split_manifest import SplitManifest
    path = Path(path).expanduser().resolve()
    manifest = SplitManifest.load(path)
    fingerprint = manifest.fingerprint()
    if expected_fingerprint is not None and expected_fingerprint != fingerprint:
        raise ValueError('Dataset manifest changed since source configuration')
    roots = _roots(roots)
    options = manifest.to_dict()['reader']
    options['dtype'] = getattr(torch, options['dtype'])
    dataset = ExplicitPairedImageDataset(manifest.pairs(roots=roots, split=split), **options)
    dataset._manifest_config = dict(path=str(path), roots={name: str(root) for name, root in roots.items()},
                                    split=split, expected_fingerprint=fingerprint)
    dataset._manifest_preprocessing = manifest.preprocessing
    dataset._manifest_records = dataset.records
    return dataset


def manifest_config(dataset):
    if dataset.records != dataset._manifest_records or dataset.sample_ids != tuple(p.sample_id for p in dataset.records):
        raise ValueError('Manifest dataset records changed since reconstruction')
    if _preprocessing(_reader_options(dataset)) != dataset._manifest_preprocessing:
        raise ValueError('Manifest dataset preprocessing changed since reconstruction')
    return {'manifest': _json_copy(dataset._manifest_config)}
