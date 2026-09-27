"""Canonical JSON, effective reader contracts and portable file references."""

import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath

import torch


READER_FIELDS = frozenset(('strict_grayscale_uint8', 'binary_target', 'num_rows', 'num_cols',
    'grayscale_in', 'grayscale_target', 'invert_in', 'invert_target', 'dtype', 'scale_in', 'scale_out'))


def _json_copy(value):
    def check(item):
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError('JSON mapping keys must be strings')
            for child in item.values():
                check(child)
        elif type(item) is list:
            for child in item:
                check(child)
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError('JSON numbers must be finite')
        elif item is not None and type(item) not in (str, bool, int):
            raise TypeError(f'Unsupported JSON value: {type(item).__name__}')
    check(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _canonical(value):
    return json.dumps(_json_copy(value), sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def _fingerprint(domain, value):
    return hashlib.sha256(_canonical([domain, value]).encode('utf-8')).hexdigest()


def _load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key {key!r}')
            result[key] = value
        return result
    with Path(path).open(encoding='utf-8') as stream:
        return _json_copy(json.load(stream, object_pairs_hook=unique))


def _validate_reader(options):
    options = _json_copy(options)
    if not isinstance(options, dict) or set(options) != READER_FIELDS:
        raise ValueError('Reader configuration must contain exactly the supported reading options')
    for name in READER_FIELDS - {'dtype', 'num_rows', 'num_cols'}:
        if type(options[name]) is not bool:
            raise ValueError(f'Reader {name} must be a boolean')
    rows, cols = options['num_rows'], options['num_cols']
    if (rows is None) != (cols is None) or any(value is not None and (type(value) is not int or value <= 0)
                                            for value in (rows, cols)):
        raise ValueError('Reader dimensions must both be positive integers or None')
    dtype = options['dtype']
    if not isinstance(dtype, str) or not isinstance(getattr(torch, dtype, None), torch.dtype):
        raise ValueError('Reader dtype must name a torch dtype')
    if options['strict_grayscale_uint8'] and not (options['grayscale_in'] and options['grayscale_target']):
        raise ValueError('Strict reading requires grayscale input and target')
    if options['binary_target'] and (not options['grayscale_target'] or options['invert_target']):
        raise ValueError('Binary targets require grayscale without target inversion')
    return options


def _reader_options(dataset):
    from ._explicit_pairs import ExplicitPairedImageDataset
    if not isinstance(dataset, ExplicitPairedImageDataset):
        raise TypeError('Use an explicit PairedImageDataset factory to describe the reader')
    options = {name: getattr(dataset, name) for name in READER_FIELDS}
    options['dtype'] = str(options['dtype']).removeprefix('torch.')
    shape = None if options['num_rows'] is None else (options['num_rows'], options['num_cols'])
    if dataset.resize_shape != shape:
        raise ValueError('Reader resize_shape differs from configured dimensions')
    return _validate_reader(options)


def _preprocessing(options):
    options = _validate_reader(options)
    result = {}
    for role, suffix, scale in (('input', 'in', 'scale_in'), ('target', 'target', 'scale_out')):
        binary = role == 'target' and options['binary_target']
        unchanged = options['strict_grayscale_uint8'] or binary
        gray = options[f'grayscale_{suffix}']
        scaled = options[scale] and not binary
        dtype = getattr(torch, options['dtype'])
        output_dtype = torch.result_type(torch.empty((), dtype=dtype, device='cpu'), 255.0) if scaled else dtype
        result[role] = dict(
            format='mtlearn.image-reader', schema_version=1,
            decoder='opencv', read_mode='unchanged_grayscale_uint8' if unchanged else
                ('grayscale_uint8' if gray else 'color_uint8'),
            color_order='GRAY' if gray else 'RGB',
            alpha_policy='reject' if unchanged else 'drop',
            orientation='ignore_exif' if unchanged else 'opencv_default',
            invert=options[f'invert_{suffix}'],
            resize=None if options['num_rows'] is None else [options['num_rows'], options['num_cols']],
            interpolation='area' if role == 'input' else 'nearest',
            binary='positive_to_one' if binary else 'none',
            dtype=options['dtype'], output_dtype=str(output_dtype).removeprefix('torch.'),
            scale='divide_by_255' if scaled else 'none',
        )
    return result


def _portable_path(value):
    if (not isinstance(value, str) or not value or '\\' in value or '\x00' in value
            or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(part in ('', '.', '..') for part in value.split('/'))):
        raise ValueError(f'Expected a relative POSIX path within its root: {value!r}')
    return value


def _roots(roots):
    if not isinstance(roots, dict) or not roots or any(not isinstance(key, str) or not key for key in roots):
        raise ValueError('roots must map nonempty names to directories')
    result = {name: Path(path).expanduser().resolve() for name, path in roots.items()}
    for path in result.values():
        if not path.is_dir():
            raise FileNotFoundError(f'Root directory not found: {path}')
    return result


def _resolve_file(reference, roots):
    name = reference['root']
    if name not in roots:
        raise ValueError(f'Unknown root {name!r}')
    root = roots[name]
    path = (root / _portable_path(reference['path'])).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'File escapes root {name!r}: {reference["path"]}') from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _signature(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _hash_file(path):
    before = _signature(path)
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if _signature(path) != before:
        raise RuntimeError(f'File changed while hashing: {path}')
    return digest.hexdigest()
