import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from mtlearn.datasets import PairedImageDataset
from mtlearn.layers.cfp import DatasetSource, paired_source_from_config


def save_image(path, array):
    path = Path(path)
    valid, encoded = cv2.imencode(path.suffix.lower(), array)
    assert valid
    path.write_bytes(encoded.tobytes())


def make_pairs(path):
    for sample_id in ('1', '01', 'β'):
        save_image(str(path / f'img_{sample_id}_in.PNG'), np.arange(24, dtype=np.uint8).reshape(4, 6))
        save_image(str(path / f'mask_{sample_id}_target.pgm'), np.full((4, 6), 255, dtype=np.uint8))
    return dict(root_dir=path, prefix_in='img_', prefix_target='mask_', extensions=('PNG', '.pgm'))


def test_suffixes_preserve_ids_without_decoding_and_roundtrip(tmp_path, monkeypatch):
    options = make_pairs(tmp_path)
    with monkeypatch.context() as context:
        context.setattr(cv2, 'imread', lambda *_: pytest.fail('Scanner must not decode'))
        dataset = PairedImageDataset.from_suffixes(**options, invert_target=True)
    assert dataset.sample_ids == ('01', '1', 'β')
    assert dataset.missing_input_ids == dataset.missing_target_ids == ()
    source = DatasetSource.from_paired_dataset(dataset)
    config = json.loads(json.dumps(source.get_config()))
    assert 'suffixes' in config['pairs'] and 'pairs' not in config['pairs']
    restored = paired_source_from_config(config=config)
    assert restored.sample_ids == source.sample_ids
    for index in range(3):
        actual, target = restored[index]
        expected, expected_target = source[index]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(target, expected_target, rtol=0, atol=0)
    save_image(str(tmp_path / 'img_new_in.png'), np.zeros((4, 6), dtype=np.uint8))
    save_image(str(tmp_path / 'mask_new_target.pgm'), np.zeros((4, 6), dtype=np.uint8))
    with pytest.raises(ValueError, match='IDs changed'):
        paired_source_from_config(config=config)


@pytest.mark.parametrize('unmatched', ['error', 'warn', 'ignore'])
def test_unmatched_policy_never_resolves_duplicates(tmp_path, unmatched):
    options = make_pairs(tmp_path)
    (tmp_path / 'mask_1_target.pgm').unlink()
    if unmatched == 'error':
        with pytest.raises(ValueError, match='missing_target_ids'):
            PairedImageDataset.from_suffixes(**options)
    else:
        if unmatched == 'warn':
            with pytest.warns(UserWarning, match='Unmatched'):
                dataset = PairedImageDataset.from_suffixes(**options, unmatched=unmatched)
        else:
            dataset = PairedImageDataset.from_suffixes(**options, unmatched=unmatched)
        assert dataset.missing_target_ids == ('1',)
    save_image(str(tmp_path / 'img_01_in.pgm'), np.zeros((4, 6), dtype=np.uint8))
    with pytest.raises(ValueError, match='Duplicate input'):
        PairedImageDataset.from_suffixes(**options, unmatched=unmatched)


@pytest.mark.parametrize('options', [dict(suffix_in=''), dict(suffix_target='_in'),
    dict(prefix_in=1), dict(suffix_in='../in'), dict(ordering='numeric'), dict(extensions='.png'),
    dict(extensions=[]), dict(unmatched='first')])
def test_invalid_scanner_options(tmp_path, options):
    with pytest.raises((ValueError, TypeError)):
        PairedImageDataset.from_suffixes(tmp_path, **options)


def test_ambiguous_file_and_empty_id(tmp_path):
    save_image(str(tmp_path / 'x_target.png'), np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match='both'):
        PairedImageDataset.from_suffixes(tmp_path, suffix_in='target', suffix_target='_target')
    save_image(str(tmp_path / '_in.png'), np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match='Empty'):
        PairedImageDataset.from_suffixes(tmp_path)


def test_ignored_patterns_and_no_pairs(tmp_path):
    (tmp_path / 'README.txt').write_text('not an image')
    save_image(str(tmp_path / 'unrelated.png'), np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match='No matched'):
        PairedImageDataset.from_suffixes(tmp_path)
