from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time

from ..storage import DiskStore
from .cfp_preprocessor import CFPPreprocessor
from .dataset_source import DatasetSource, paired_source_from_config
from .prepared_dataloader import build_prepared_dataloader


@dataclass(frozen=True)
class DatasetCacheConfig:
    path: str
    manifest: str
    source_version: str
    preprocessing_version: str
    preparation: dict
    split: str = 'train'

    def __post_init__(self):
        for name in ('manifest', 'source_version', 'preprocessing_version', 'split'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'{name} must be a nonempty string.')
        object.__setattr__(self, 'path', str(Path(self.path).expanduser().resolve()))
        preparation = json.loads(json.dumps(self.preparation, allow_nan=False))
        CFPPreprocessor.from_config(preparation)
        object.__setattr__(self, 'preparation', preparation)

    @classmethod
    def from_preprocessor(cls, preprocessor, **kwargs):
        return cls(preparation=preprocessor.get_config(), **kwargs)

    @classmethod
    def from_manifest(cls, path, manifest):
        with DiskStore(path, readonly=True, max_ram_bytes=0) as store:
            header = store.inspect_manifest(manifest, limit=1)
        return cls(path=path, manifest=manifest, source_version=header['source_version'],
            preprocessing_version=header['preprocessing_version'], split=header['split'],
            preparation=header['config'])

    def get_config(self):
        return asdict(self)

    def preparation_options(self):
        return {key: getattr(self,key) for key in ('path','manifest','source_version','preprocessing_version','split')}


def _source(source):
    if not isinstance(source, DatasetSource):
        raise TypeError('source must be a DatasetSource with explicit extraction and sample IDs.')
    if len(set(source.sample_ids)) != len(source):
        raise ValueError('Persistent manifests require unique sample IDs; repeat positions with a reader sampler.')
    return source


def prepare_dataset_cache(config, source, *, max_disk_bytes, **options):
    source = _source(source)
    preprocessor = CFPPreprocessor.from_config(config.preparation)
    return preprocessor.prepare_or_reuse(source, **config.preparation_options(),
        sample_ids=source.sample_ids, mode='prepare_missing', max_disk_bytes=max_disk_bytes, **options)


def _compatible_layer(preprocessor, layer):
    requested = CFPPreprocessor.from_layer(layer)
    if requested.attribute_dtype != preprocessor.attribute_dtype:
        raise ValueError('Model attribute dtype differs from the original preparation contract.')
    for key, tree in requested.tree_specs.items():
        if key not in preprocessor.tree_specs or tree != preprocessor.tree_specs[key]:
            raise ValueError(f'Model tree {key!r} is absent from the original preparation contract.')
        missing = set(requested.features[key].attributes) - set(preprocessor.features[key].attributes)
        if missing:
            raise ValueError(f'Model attributes absent from tree {key!r}: {sorted(a.name for a in missing)}')


def open_dataset_cache(config, source, *, layer=None, statistics=None, reader_config=None, **reader_options):
    source = _source(source)
    if reader_config is not None:
        from .dataset_diagnostics import DatasetReaderConfig
        if not isinstance(reader_config, DatasetReaderConfig):
            raise TypeError('reader_config must be a DatasetReaderConfig.')
        configured = reader_config.get_config()
        if set(configured) & set(reader_options):
            raise ValueError('Reader options must be configured only once.')
        reader_options = dict(configured, **reader_options)
    started = time.perf_counter()
    preprocessor = CFPPreprocessor.from_config(config.preparation)
    if layer is not None:
        _compatible_layer(preprocessor, layer)
    if statistics is not None and layer is None:
        raise ValueError('statistics requires the consuming layer.')
    if statistics is not None and statistics.contract != layer.get_statistics_contract():
        raise ValueError('Statistics snapshot is incompatible with the consuming layer.')
    if 'source_factory' in reader_options or 'source_factory_kwargs' in reader_options:
        raise ValueError('Source factories are derived from DatasetSource.get_config in this flow.')
    with DiskStore(config.path, readonly=True, max_ram_bytes=0) as store:
        alignment = source.validate_manifest(store, config.manifest)
    aligned = time.perf_counter()
    reuse = preprocessor.prepare_or_reuse(**config.preparation_options(), sample_ids=source.sample_ids, mode='reuse')
    if reuse.status != 'complete':
        raise RuntimeError(f'Manifest {config.manifest!r}: reuse did not complete: {reuse.status}.')
    inspected = time.perf_counter()
    if reader_options.get('num_workers', 0):
        reader_options = dict(reader_options, source_factory=paired_source_from_config,
                              source_factory_kwargs={'config': source.get_config()})
    loader = build_prepared_dataloader(config.path, config.manifest, source=source, **reader_options)
    try:
        if statistics is not None:
            layer.set_stats(statistics)
    except BaseException:
        loader.close()
        raise
    loader.opening_report = {'manifest': config.manifest, 'source_alignment': alignment,
        'reused_entries': reuse.reused_entries, 'prepared_entries': reuse.prepared_entries,
        'repaired_entries': reuse.repaired_entries, 'readonly': True,
        'payload_validation': 'on_access', 'source_pixel_validation': 'on_access',
        'statistics': 'provided_snapshot' if statistics is not None else 'unchanged',
        'seconds': {'alignment': aligned-started, 'reuse_metadata': inspected-aligned,
                    'open': time.perf_counter()-inspected}, 'resources': reuse.resources}
    return loader
