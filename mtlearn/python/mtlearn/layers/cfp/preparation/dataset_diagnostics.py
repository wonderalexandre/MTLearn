from dataclasses import dataclass
import inspect
import json
import time

from ..storage import DiskStore
from .dataset_source import DatasetSource
from .prepared_dataset import PreparedDataset
from .prepared_dataloader import PreparedDataLoader


@dataclass(frozen=True)
class DatasetReaderConfig:
    options: dict

    def __post_init__(self):
        options = json.loads(json.dumps(self.options, allow_nan=False))
        if not isinstance(options, dict):
            raise TypeError('Reader options must be a dictionary.')
        allowed = set(inspect.signature(PreparedDataLoader).parameters) - {
            'path','manifest','source','source_factory','source_factory_kwargs','sampler','generator'}
        unknown = set(options)-allowed
        if unknown:
            raise ValueError(f'Unsupported reader configuration: {sorted(unknown)}')
        object.__setattr__(self, 'options', options)

    def get_config(self):
        return json.loads(json.dumps(self.options))


def inspect_dataset_cache(config, *, validation='metadata', source=None, offset=0, limit=100,
                          progress=None, cancel=None):
    if validation not in ('metadata','payload','source'):
        raise ValueError("validation must be 'metadata', 'payload', or 'source'.")
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError('offset must be nonnegative and limit must be between 1 and 1000.')
    if progress is not None and not callable(progress):
        raise TypeError('progress must be callable or None.')
    if cancel is not None and not callable(cancel) and not callable(getattr(cancel,'is_set',None)):
        raise TypeError('cancel must be callable, an event, or None.')
    if validation == 'source' and not isinstance(source, DatasetSource):
        raise TypeError('Source validation requires DatasetSource.')
    started = time.perf_counter()
    with DiskStore(config.path, readonly=True, max_ram_bytes=0) as store:
        page = store.inspect_manifest(config.manifest,offset=offset,limit=limit)
        expected = {'config':config.preparation,'source_version':config.source_version,
                    'preprocessing_version':config.preprocessing_version,'split':config.split}
        if any(page[key]!=value for key,value in expected.items()):
            raise ValueError(f'Manifest {config.manifest!r} differs from its original preparation contract.')
        report = {'validation':validation,'metadata':page,'readonly':True,
                  'payloads_checked':False,'source_pixels_checked':False}
        if validation == 'payload':
            report['integrity'] = store.validate_manifest(config.manifest,offset=offset,limit=limit,
                                                          progress=progress,cancel=cancel)
            report['payloads_checked'] = report['integrity']['checked_entries'] > 0
        elif validation == 'source':
            source.validate_manifest(store, config.manifest)
            dataset = PreparedDataset(store, config.manifest, source=source)
            checked, validated, errors, status = 0, 0, [], 'valid'
            for index in range(offset,min(len(dataset),offset+limit)):
                if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
                    status = 'cancelled'
                    break
                try:
                    item = dataset[index]
                    del item
                    validated += 1
                except Exception as exc:
                    errors.append({'position':index,'sample_id':source.sample_ids[index],'message':str(exc)})
                checked += 1
                if progress is not None:
                    progress({'validation':'source','checked_samples':checked,'error_count':len(errors)})
            report['source'] = {'status':'invalid' if errors else status,'checked_samples':checked, 'validated_samples':validated,
                'errors':errors,'fully_validated':not errors and status=='valid' and offset==0 and checked==len(dataset)}
            report['source_pixels_checked'] = checked > 0
            report['payloads_checked'] = validated > 0
        report['seconds'] = time.perf_counter()-started
        return report
