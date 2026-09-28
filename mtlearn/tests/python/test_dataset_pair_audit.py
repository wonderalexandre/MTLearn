import gc
import weakref
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from mtlearn.datasets import PairedImageDataset, SplitManifest, audit_image_pairs


def save_image(path, array):
    path = Path(path)
    valid, encoded = cv2.imencode(path.suffix.lower(), array)
    assert valid
    path.write_bytes(encoded.tobytes())


@pytest.fixture
def audited_pairs(tmp_path):
    for sample_id in ('01', '1', 'β'):
        image = np.arange(24, dtype=np.uint8).reshape(4, 6)
        target = np.where(image > 8, 255, 0).astype(np.uint8)
        save_image(str(tmp_path / f'{sample_id}_in.png'), image)
        save_image(str(tmp_path / f'{sample_id}_target.png'), target)
    dataset = PairedImageDataset.from_suffixes(tmp_path)
    roots = {'input': tmp_path, 'target': tmp_path}
    return dataset, roots


def test_native_metadata_and_effective_reader_are_distinct(tmp_path):
    image = np.array([[0, 256], [512, 65535]], dtype=np.uint16)
    save_image(str(tmp_path / 'a_in.png'), image)
    save_image(str(tmp_path / 'a_target.png'), image)
    dataset = PairedImageDataset.from_suffixes(tmp_path, scale_in=False)
    audit = audit_image_pairs(dataset, {'input': tmp_path, 'target': tmp_path})
    assert audit.sources['a']['input']['dtype'] == 'uint16'
    assert audit.sources['a']['input']['shape'] == [2, 2]
    assert audit.preprocessing['input']['read_mode'] == 'grayscale_uint8'
    assert audit.preprocessing['input']['dtype'] == 'float32'
    torch.testing.assert_close(dataset[0][0], torch.tensor([[[0., 1.], [2., 255.]]]))


def test_callback_metadata_defensive_copies_and_one_pair_memory(audited_pairs, monkeypatch):
    dataset, roots = audited_pairs
    from mtlearn._datasets import _pair_audit
    read_image_file = _pair_audit._read_image_file
    references, live = [], []
    def read(*args):
        gc.collect()
        live.append(sum(ref() is not None for ref in references))
        array = read_image_file(*args)
        references.append(weakref.ref(array))
        return array
    metadata = {'mask_values': [0, 255], 'original_zero_fraction': 0.375}
    def validate(record, image, target):
        assert not image.flags.writeable and not target.flags.writeable
        assert record.sample_id in dataset.sample_ids
        return metadata
    monkeypatch.setattr(_pair_audit, '_read_image_file', read)
    audit = audit_image_pairs(dataset, roots, validate)
    assert live == [0, 1] * 3
    assert not any(ref() is not None for ref in references)
    metadata['mask_values'].append(12)
    assert audit.sources['01']['metadata']['mask_values'] == [0, 255]
    exported = audit.to_dict()
    exported['sources']['01']['metadata']['mask_values'].append(33)
    assert audit.sources['01']['metadata']['mask_values'] == [0, 255]


@pytest.mark.parametrize('metadata', [[], {'bad': float('nan')}, {'bad': np.int64(1)}, {1: 'bad'}])
def test_invalid_callback_metadata(audited_pairs, metadata):
    dataset, roots = audited_pairs
    with pytest.raises((TypeError, ValueError)):
        audit_image_pairs(dataset, roots, lambda *_: metadata)


def test_callback_failure_has_sample_id(audited_pairs):
    dataset, roots = audited_pairs
    def reject(*_):
        raise ValueError('Nonbinary mask')
    with pytest.raises(ValueError, match="01.*Nonbinary"):
        audit_image_pairs(dataset, roots, reject)


def test_audit_rejects_mutation_during_or_after_inspection(audited_pairs):
    dataset, roots = audited_pairs
    audit = audit_image_pairs(dataset, roots)
    filename = dataset.records[0].target_path
    save_image(filename, np.full((4, 6), 255, dtype=np.uint8))
    with pytest.raises(RuntimeError, match='changed'):
        SplitManifest.from_audit(audit, {'train': list(dataset.sample_ids)})
    def mutate(record, *_):
        save_image(record.input_path, np.zeros((4, 6), dtype=np.uint8))
    with pytest.raises(RuntimeError, match='changed'):
        audit_image_pairs(dataset, roots, mutate)


def test_invalid_native_shape_and_unreadable_file(audited_pairs):
    dataset, roots = audited_pairs
    save_image(dataset.records[0].target_path, np.zeros((2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match='01.*spatial'):
        audit_image_pairs(dataset, roots)
    from pathlib import Path
    Path(dataset.records[0].input_path).write_bytes(b'not an image')
    with pytest.raises(ValueError, match='01.*input'):
        audit_image_pairs(dataset, roots)


def test_audit_rejects_files_outside_declared_root(audited_pairs, tmp_path):
    dataset, roots = audited_pairs
    (tmp_path / 'other').mkdir()
    with pytest.raises(ValueError, match='outside'):
        audit_image_pairs(dataset, dict(roots, input=tmp_path / 'other'))


def test_audit_rejects_inconsistent_resize_configuration(audited_pairs):
    dataset, roots = audited_pairs
    dataset.resize_shape = (2, 3)
    with pytest.raises(ValueError, match='resize_shape'):
        audit_image_pairs(dataset, roots)


def test_effective_output_dtype_tracks_scaling_and_binary_targets(audited_pairs):
    dataset, roots = audited_pairs
    dataset.dtype = torch.uint8
    contract = dataset.get_preprocessing_contract()
    assert contract['input']['dtype'] == 'uint8'
    assert contract['input']['output_dtype'] == str(dataset[0][0].dtype).removeprefix('torch.')
    manifest = SplitManifest.from_audit(audit_image_pairs(dataset, roots),
                                      {'train': list(dataset.sample_ids)})
    default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert dataset.get_preprocessing_contract()['input']['output_dtype'] == 'float64'
        with pytest.raises(ValueError, match='input'):
            manifest.validate_preprocessing(dataset)
    finally:
        torch.set_default_dtype(default)
    dataset.binary_target = True
    assert dataset.get_preprocessing_contract()['target']['output_dtype'] == 'uint8'
    assert dataset[0][1].dtype == torch.uint8


def test_cpu_reader_contract_does_not_require_the_default_device(audited_pairs, tmp_path):
    dataset, roots = audited_pairs
    expected = dataset.get_preprocessing_contract()
    with torch.device('cuda:999'):
        assert dataset[0][0].device.type == 'cpu'
        assert dataset.get_preprocessing_contract() == expected
        audit = audit_image_pairs(dataset, roots)
        manifest = SplitManifest.from_audit(audit, {'train': list(dataset.sample_ids)})
        path = tmp_path / 'selection.json'
        manifest.save(path)
        restored = SplitManifest.load(path)
        reopened = PairedImageDataset.from_manifest(path, roots=roots, split='train')
        assert reopened[0][0].device.type == 'cpu'
        assert restored.validate_preprocessing(reopened)['valid']


def test_unicode_paths_decode_when_opencv_cannot_open_the_filename(audited_pairs, monkeypatch):
    from mtlearn._datasets import _image_ops
    dataset, roots = audited_pairs
    monkeypatch.setattr(_image_ops, '_needs_unicode_file_decode',
                        lambda filename: not filename.isascii())
    ordinary_imread = cv2.imread
    def ascii_only(path, flags):
        if not str(path).isascii():
            raise AssertionError('Native path-based decoding cannot open this filename')
        return ordinary_imread(path, flags)
    monkeypatch.setattr(cv2, 'imread', ascii_only)
    assert dataset[2][2] == 'β'
    audit = audit_image_pairs(dataset, roots)
    assert audit.sample_ids == ('01', '1', 'β')
    assert audit.sources['β']['input']['shape'] == [4, 6]
