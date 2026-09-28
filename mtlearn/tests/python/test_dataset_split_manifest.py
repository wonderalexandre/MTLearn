import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from mtlearn import morphology
from mtlearn.datasets import PairedImageDataset, SplitManifest, audit_image_pairs, subsets_from_ids
from mtlearn.layers import ConnectedFilterPreprocessingLayer
from mtlearn.layers.cfp import (CFPPreprocessor, DatasetCacheConfig, DatasetSource,
    open_dataset_cache, paired_source_from_config, prepare_dataset_cache)
from mtlearn.layers.cfp.preparation._reader_process import factory_descriptor
from test_dataset_pair_audit import audited_pairs, save_image


@pytest.fixture
def manifest(audited_pairs):
    dataset, roots = audited_pairs
    audit = audit_image_pairs(dataset, roots)
    return SplitManifest.from_audit(audit, {'train': ['1', '01'], 'test': ['β']},
        groups={'01': 'a', '1': 'a', 'β': 'b'}, provenance={'seed': 42})


def test_manifest_roundtrip_and_relocation_are_stable(manifest, audited_pairs, tmp_path, monkeypatch):
    dataset, roots = audited_pairs
    path = tmp_path / 'selection.json'
    manifest.save(path)
    with monkeypatch.context() as context:
        context.setattr(cv2, 'imread', lambda *_: pytest.fail('Manifest operations do not decode'))
        restored = SplitManifest.load(path)
        assert restored.to_dict() == manifest.to_dict()
        assert restored.fingerprints() == manifest.fingerprints()
        assert restored.validate_files(roots=roots)['checked_files'] == 6
    moved = tmp_path / 'moved'
    moved.mkdir()
    for record in dataset.records:
        for filename in (record.input_path, record.target_path):
            shutil.copy2(filename, moved / Path(filename).name)
    moved_roots = {'input': moved, 'target': moved}
    assert restored.validate_files(roots=moved_roots)['valid']
    reopened = PairedImageDataset.from_manifest(path, roots=moved_roots, split='train')
    assert reopened.sample_ids == ('1', '01')
    assert reopened.get_preprocessing_contract() == manifest.preprocessing
    path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=True, sort_keys=False))
    assert SplitManifest.load(path).fingerprints() == manifest.fingerprints()
    data = manifest.to_dict()
    data['provenance'] = {'timestamp': 'later'}
    assert SplitManifest(data).fingerprints() == manifest.fingerprints()
    data['sources']['01']['input']['path'] = 'renamed.png'
    assert SplitManifest(data).fingerprint('input') == manifest.fingerprint('input')


def test_defensive_data_and_effective_preprocessing(manifest, audited_pairs):
    dataset, _ = audited_pairs
    data = manifest.to_dict()
    data['splits']['train'].reverse()
    assert manifest.splits['train'] == ['1', '01']
    reordered = SplitManifest(data)
    assert reordered.fingerprint('splits') != manifest.fingerprint('splits')
    assert reordered.fingerprint('input') == manifest.fingerprint('input')
    assert manifest.validate_preprocessing(dataset)['valid']
    assert manifest.validate_preprocessing(Subset(dataset, [0]))['valid']
    dataset.invert_target = True
    with pytest.raises(ValueError, match='target'):
        manifest.validate_preprocessing(dataset)
    data = manifest.to_dict()
    data['preprocessing']['target']['invert'] = True
    with pytest.raises(ValueError, match='effective reader'):
        SplitManifest(data)


@pytest.mark.parametrize('mutation', ['format', 'version', 'bool_version', 'hash', 'extra', 'missing',
    'groups', 'empty_preprocessing', 'coerced_preprocessing', 'bad_shape', 'bad_dtype', 'boolean_policy', 'reader_bool', 'nan'])
def test_malformed_schema_is_rejected(manifest, mutation):
    data = manifest.to_dict()
    if mutation == 'format': data['format'] = 'other'
    if mutation == 'version': data['schema_version'] = 2
    if mutation == 'bool_version': data['schema_version'] = True
    if mutation == 'hash': data['sources']['01']['input']['sha256'] = 'abc'
    if mutation == 'extra': data['extra'] = 1
    if mutation == 'missing': del data['sources']['01']
    if mutation == 'groups': data['groups'] = {}
    if mutation == 'empty_preprocessing': data['preprocessing'] = {'input': {}, 'target': {}}
    if mutation == 'coerced_preprocessing': data['preprocessing']['input']['invert'] = 0
    if mutation == 'bad_shape': data['sources']['01']['input']['shape'] = [True, 6]
    if mutation == 'bad_dtype': data['sources']['01']['input']['dtype'] = 'object'
    if mutation == 'boolean_policy': data['policy']['allow_empty'] = 0
    if mutation == 'reader_bool': data['reader']['invert_target'] = 1
    if mutation == 'nan': data['provenance'] = {'value': float('nan')}
    with pytest.raises((TypeError, ValueError)):
        SplitManifest(data)


@pytest.mark.parametrize('path', ['/absolute.png', '../outside.png', 'x/../outside.png',
    'C:/data/a.png', r'x\a.png', './a.png', 'x//a.png', '', 'nul\x00.png'])
def test_portable_paths_reject_nonrelative_and_noncanonical_values(manifest, path):
    data = manifest.to_dict()
    data['sources']['01']['input']['path'] = path
    with pytest.raises(ValueError, match='POSIX'):
        SplitManifest(data)


def test_unknown_roots_and_external_symlinks(manifest, audited_pairs, tmp_path):
    _, roots = audited_pairs
    with pytest.raises(ValueError, match='Unknown root'):
        manifest.validate_files(roots={'unknown': tmp_path})
    external = tmp_path / 'external'
    external.mkdir()
    save_image(str(external / 'image.png'), np.zeros((4, 6), dtype=np.uint8))
    inside = tmp_path / 'inside'
    inside.mkdir()
    try:
        (inside / 'link.png').symlink_to(external / 'image.png')
    except OSError:
        pytest.skip('Symlinks unavailable')
    data = manifest.to_dict()
    data['sources']['01']['input']['path'] = 'link.png'
    changed = SplitManifest(data)
    report = changed.validate_files(roots=dict(roots, input=inside), raise_on_error=False)
    assert any(issue['sample_id'] == '01' and 'escapes' in issue['error'] for issue in report['issues'])


def test_duplicate_json_keys_and_atomic_save(manifest, tmp_path, monkeypatch):
    path = tmp_path / 'selection.json'
    manifest.save(path)
    old = path.read_bytes()
    from mtlearn._datasets import _split_manifest
    def fail(*_):
        raise OSError('replace failed')
    with monkeypatch.context() as context:
        context.setattr(_split_manifest.os, 'replace', fail)
        with pytest.raises(OSError, match='replace failed'):
            manifest.save(path)
    assert path.read_bytes() == old
    assert not list(tmp_path.glob('.selection.json.*.tmp'))
    path.write_text('{"format": "a", "format": "b"}')
    with pytest.raises(ValueError, match='Duplicate JSON'):
        SplitManifest.load(path)


def test_file_and_selection_fingerprints_have_independent_scopes(manifest, audited_pairs):
    dataset, roots = audited_pairs
    before_train = manifest.fingerprints(split='train')
    save_image(dataset.records[2].input_path, np.full((4, 6), 77, dtype=np.uint8))
    changed = SplitManifest.from_audit(audit_image_pairs(dataset, roots), manifest.splits, groups=manifest.groups)
    assert changed.fingerprint('input') != manifest.fingerprint('input')
    assert changed.fingerprints(split='train') == before_train
    assert changed.fingerprint('target') == manifest.fingerprint('target')
    assert manifest.validate_files(roots=roots, split='train')['valid']
    report = manifest.validate_files(roots=roots, raise_on_error=False)
    assert [(issue['sample_id'], issue['role']) for issue in report['issues']] == [('β', 'input')]
    with pytest.raises(ValueError, match='β.*input'):
        manifest.validate_files(roots=roots)
    dataset.invert_target = True
    target_changed = SplitManifest.from_audit(audit_image_pairs(dataset, roots), manifest.splits, groups=manifest.groups)
    assert target_changed.fingerprint('input') == changed.fingerprint('input')
    assert target_changed.fingerprint('preprocessing_input') == changed.fingerprint('preprocessing_input')
    assert target_changed.fingerprint('preprocessing_target') != changed.fingerprint('preprocessing_target')


def test_manifest_source_freezes_selection_and_detects_configuration_drift(manifest, audited_pairs, tmp_path):
    dataset, roots = audited_pairs
    path = tmp_path / 'selection.json'
    manifest.save(path)
    restored = PairedImageDataset.from_manifest(path, roots=roots)
    source = DatasetSource.from_paired_dataset(Subset(Subset(restored, [2, 0, 1]), [2, 1, 2]))
    config = source.get_config()
    save_image(str(tmp_path / 'new_in.png'), np.zeros((4, 6), dtype=np.uint8))
    save_image(str(tmp_path / 'new_target.png'), np.zeros((4, 6), dtype=np.uint8))
    recreated = paired_source_from_config(config=config)
    assert recreated.sample_ids == ('1', '01', '1')
    for index in range(len(source)):
        for actual, expected in zip(recreated[index], source[index]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored.invert_target = True
    with pytest.raises(ValueError, match='preprocessing changed'):
        source.get_config()
    data = manifest.to_dict()
    data['splits']['train'].reverse()
    SplitManifest(data).save(path)
    with pytest.raises(ValueError, match='manifest changed'):
        paired_source_from_config(config=config)


@pytest.mark.parametrize('workers', [0, 1, 2])
def test_manifest_configuration_stays_compact_and_workers_are_active(audited_pairs, tmp_path, workers):
    dataset, roots = audited_pairs
    pair = dataset.records[0]
    many = PairedImageDataset.from_pairs([(f'{index:04d}', pair.input_path, pair.target_path) for index in range(120)])
    manifest = SplitManifest.from_audit(audit_image_pairs(many, roots), {'train': list(many.sample_ids)})
    path = tmp_path / 'many.json'
    manifest.save(path)
    source = DatasetSource.from_paired_dataset(PairedImageDataset.from_manifest(path, roots=roots, split='train'))
    descriptor = factory_descriptor(paired_source_from_config, {'config': source.get_config()})
    assert len(descriptor[2]) < 4096
    layer = ConnectedFilterPreprocessingLayer(1, [{'tree_type': 'max-tree',
        'attributes': morphology.AttributeType.AREA}], scale_mode='dataset_clipped_zscore01')
    cache = DatasetCacheConfig.from_preprocessor(CFPPreprocessor.from_layer(layer), path=tmp_path / 'cache',
        manifest='train', source_version=manifest.fingerprint('input', split='train'),
        preprocessing_version=manifest.fingerprint('preprocessing_input'))
    result = prepare_dataset_cache(cache, source, max_disk_bytes=4 * 1024**2,
                                   collect_stats=layer.get_statistics_contract())
    assert result.status == 'complete'
    options = {} if not workers else dict(num_workers=workers, worker_memory_bytes=128 * 1024**2,
        max_prefetch_bytes=8 * 1024**2, max_batch_bytes=1024**2, source_memory_bytes=1024,
        read_workspace_bytes=1024)
    with open_dataset_cache(cache, source, layer=layer, statistics=result.statistics,
                            sampler=[119, 0, 119], **options) as loader:
        assert loader.num_workers == workers
        assert loader.prefetch == bool(workers)
        observed = []
        for batch, target in loader:
            observed.extend(batch.sample_ids)
            assert layer.forward_prepared(batch).shape == target.shape
        assert observed == ['0119', '0000', '0119']
    assert loader.counters().get('processes_alive', 0) == 0


def test_split_mapping_order_survives_save_and_load(audited_pairs, tmp_path):
    dataset, roots = audited_pairs
    manifest = SplitManifest.from_audit(audit_image_pairs(dataset, roots),
        {'validation': ['β'], 'train': ['1', '01'], 'empty': []}, allow_empty=True)
    path = tmp_path / 'ordered.json'
    manifest.save(path)
    assert list(SplitManifest.load(path).splits) == ['validation', 'train', 'empty']


def test_preprocessing_comparison_rejects_coerced_types(manifest, audited_pairs):
    dataset, roots = audited_pairs
    contract = manifest.preprocessing
    contract['target']['schema_version'] = True
    with pytest.raises(ValueError, match='target'):
        manifest.validate_preprocessing(contract)
    with pytest.raises(ValueError, match='audited reader'):
        SplitManifest.from_audit(audit_image_pairs(dataset, roots), manifest.splits,
                                preprocessing=contract)


def test_manifest_cache_invalidation_separates_targets_training_and_pixels(audited_pairs, tmp_path):
    from test_cfp_disk_preparation_flow import assert_stats
    dataset, roots = audited_pairs
    splits = {'train': ['1', '01'], 'test': ['β']}
    layer = ConnectedFilterPreprocessingLayer(1, [{'tree_type': 'max-tree',
        'attributes': morphology.AttributeType.AREA}], scale_mode='dataset_clipped_zscore01')
    preprocessor = CFPPreprocessor.from_layer(layer)
    def prepare(name, selected=splits):
        manifest = SplitManifest.from_audit(audit_image_pairs(dataset, roots), selected)
        path = tmp_path / 'current.json'
        manifest.save(path)
        source = DatasetSource.from_paired_dataset(PairedImageDataset.from_manifest(
            path, roots=roots, split='train'))
        config = DatasetCacheConfig.from_preprocessor(preprocessor, path=tmp_path / 'cache',
            manifest=name, source_version=manifest.fingerprint('input', split='train'),
            preprocessing_version=manifest.fingerprint('preprocessing_input'))
        result = prepare_dataset_cache(config, source, max_disk_bytes=4 * 1024**2,
                                       collect_stats=layer.get_statistics_contract())
        assert result.status == 'complete'
        return manifest, source, config, result
    first, _, _, baseline = prepare('train')
    save_image(dataset.records[0].target_path, np.zeros((4, 6), dtype=np.uint8))
    masks, source, config, result = prepare('train')
    assert masks.fingerprint('target') != first.fingerprint('target')
    assert result.prepared_entries == 0
    assert_stats(result.statistics, baseline.statistics)
    with open_dataset_cache(config, source) as loader:
        batches = list(loader)
        assert batches[1][0].sample_ids == ('01',)
        assert torch.count_nonzero(batches[1][1]) == 0
    dataset.invert_target = True
    inverted, source, config, result = prepare('train')
    assert inverted.fingerprint('preprocessing_target') != masks.fingerprint('preprocessing_target')
    assert result.prepared_entries == 0
    with open_dataset_cache(config, source) as loader:
        assert torch.all(list(loader)[1][1] == 1)
    save_image(dataset.records[2].input_path, np.full((4, 6), 77, dtype=np.uint8))
    evaluation, _, _, result = prepare('train')
    assert evaluation.fingerprints(split='train') == inverted.fingerprints(split='train')
    assert result.prepared_entries == 0
    assert_stats(result.statistics, baseline.statistics)
    reordered, source, config, result = prepare('reordered', {'train': ['01', '1'], 'test': ['β']})
    assert reordered.fingerprint('splits') != evaluation.fingerprint('splits')
    assert result.prepared_entries == 0
    with open_dataset_cache(config, source) as loader:
        assert [batch.sample_ids for batch, _ in loader] == [('01',), ('1',)]
    save_image(dataset.records[0].input_path, np.full((4, 6), 160, dtype=np.uint8))
    pixels, _, _, result = prepare('changed-input')
    assert pixels.fingerprint('input', split='train') != first.fingerprint('input', split='train')
    assert result.prepared_entries > 0
    dataset.invert_in = True
    preprocessing, _, _, result = prepare('changed-preprocessing')
    assert preprocessing.fingerprint('preprocessing_input') != pixels.fingerprint('preprocessing_input')
    assert result.prepared_entries > 0
