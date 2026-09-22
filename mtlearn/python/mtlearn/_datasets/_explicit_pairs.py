from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import warnings

import cv2
import numpy as np
import torch

from ._paired_image import PairedImageDataset
from ._image_ops import (
    DEFAULT_IMAGE_EXTENSIONS, normalize_extensions, normalize_resize_shape,
    read_image, resize_image, to_channel_first_tensor, scale_tensor,
)


@dataclass(frozen=True)
class ImagePair:
    sample_id: str
    input_path: str
    target_path: str


def _sample_id(value, ordering):
    if not isinstance(value, str) or not value:
        raise ValueError('sample_id must be a nonempty string.')
    if ordering == 'numeric':
        if not value.isdecimal():
            raise ValueError(f'Expected a numeric sample ID: {value!r}')
        return str(int(value))
    return value


def folder_pairs(input_dir, target_dir, *, ordering, unmatched, extensions):
    if ordering not in ('textual', 'numeric'):
        raise ValueError("ordering must be 'textual' or 'numeric'.")
    if unmatched not in ('error', 'warn', 'ignore'):
        raise ValueError("unmatched must be 'error', 'warn', or 'ignore'.")
    extensions = normalize_extensions(extensions)
    indexed = []
    for folder in (input_dir, target_dir):
        folder = Path(folder).expanduser()
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        paths = {}
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                sample_id = _sample_id(path.stem, ordering)
                if sample_id in paths:
                    raise ValueError(f'Duplicate sample ID {sample_id!r}: {paths[sample_id]} and {path}')
                paths[sample_id] = str(path.resolve())
        indexed.append(paths)
    inputs, targets = indexed
    missing = {'missing_target_ids': sorted(inputs.keys()-targets.keys()),
               'missing_input_ids': sorted(targets.keys()-inputs.keys())}
    if any(missing.values()):
        message = f'Unmatched image pairs: {missing}'
        if unmatched == 'error':
            raise ValueError(message)
        if unmatched == 'warn':
            warnings.warn(message, UserWarning, stacklevel=3)
    ids = sorted(inputs.keys() & targets.keys(), key=int if ordering == 'numeric' else None)
    return [(sample_id, inputs[sample_id], targets[sample_id]) for sample_id in ids], missing


class ExplicitPairedImageDataset(PairedImageDataset):
    def __init__(self, pairs, *, ordering='provided', strict_grayscale_uint8=False,
                 binary_target=False, num_rows=None, num_cols=None, grayscale_in=True,
                 grayscale_target=True, invert_in=False, invert_target=False,
                 dtype=torch.float32, scale_in=True, scale_out=True):
        if ordering not in ('provided', 'textual', 'numeric'):
            raise ValueError("ordering must be 'provided', 'textual', or 'numeric'.")
        if strict_grayscale_uint8 and (not grayscale_in or not grayscale_target):
            raise ValueError('strict_grayscale_uint8 requires grayscale input and target.')
        if binary_target and (not grayscale_target or invert_target):
            raise ValueError('binary_target requires grayscale_target=True and invert_target=False.')
        records, seen = [], set()
        for pair in pairs:
            if isinstance(pair, ImagePair):
                pair = (pair.sample_id, pair.input_path, pair.target_path)
            if not isinstance(pair, (tuple, list)) or len(pair) != 3:
                raise ValueError('Each pair must contain (sample_id, input_path, target_path).')
            sample_id = _sample_id(pair[0], ordering)
            if sample_id in seen:
                raise ValueError(f'Duplicate sample ID {sample_id!r}.')
            seen.add(sample_id)
            paths = [str(Path(path).expanduser().resolve()) for path in pair[1:]]
            for path in paths:
                if not Path(path).is_file():
                    raise FileNotFoundError(f'Sample {sample_id!r}: {path}')
            records.append(ImagePair(sample_id, *paths))
        if not records:
            raise ValueError('At least one matched image pair is required.')
        if ordering != 'provided':
            records.sort(key=lambda pair: int(pair.sample_id) if ordering == 'numeric' else pair.sample_id)
        self.records = tuple(records)
        self.sample_ids = tuple(pair.sample_id for pair in records)
        self.pairs = tuple((pair.input_path, pair.target_path) for pair in records)
        self.resize_shape = normalize_resize_shape(num_rows, num_cols)
        self.num_rows = None if self.resize_shape is None else self.resize_shape[0]
        self.num_cols = None if self.resize_shape is None else self.resize_shape[1]
        self.strict_grayscale_uint8 = bool(strict_grayscale_uint8)
        self.binary_target = bool(binary_target)
        self.grayscale_in, self.grayscale_target = bool(grayscale_in), bool(grayscale_target)
        self.invert_in, self.invert_target = bool(invert_in), bool(invert_target)
        self.dtype, self.scale_in, self.scale_out = dtype, bool(scale_in), bool(scale_out)
        self.missing_input_ids = self.missing_target_ids = ()

    def _read(self, path, grayscale, sample_id, *, target=False):
        if not self.strict_grayscale_uint8 and not (self.binary_target and target):
            return read_image(path, grayscale=grayscale)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f'Sample {sample_id!r}: could not read {path}')
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError(f'Sample {sample_id!r}: expected grayscale uint8 at {path}; got {image.shape}, {image.dtype}')
        return image

    def __getitem__(self, index):
        pair = self.records[index]
        image = self._read(pair.input_path, self.grayscale_in, pair.sample_id)
        target = self._read(pair.target_path, self.grayscale_target, pair.sample_id, target=True)
        if image.shape[:2] != target.shape[:2]:
            raise ValueError(f'Sample {pair.sample_id!r}: input/target shape mismatch: {image.shape}, {target.shape}')
        if self.binary_target:
            values = set(np.unique(target).tolist())
            if not (values <= {0, 1} or values <= {0, 255}):
                raise ValueError(f'Sample {pair.sample_id!r}: expected a binary mask, got {values}')
        if self.invert_in:
            image = 255-image
        if self.invert_target:
            target = 255-target
        image = resize_image(image, self.resize_shape, interpolation=cv2.INTER_AREA)
        target = resize_image(target, self.resize_shape, interpolation=cv2.INTER_NEAREST)
        x = scale_tensor(to_channel_first_tensor(image, dtype=self.dtype), enabled=self.scale_in)
        y = (to_channel_first_tensor((target > 0).astype(np.uint8), dtype=self.dtype) if self.binary_target else
             scale_tensor(to_channel_first_tensor(target, dtype=self.dtype), enabled=self.scale_out))
        return x, y, pair.sample_id

    def get_config(self):
        config = {'pairs': [(p.sample_id, p.input_path, p.target_path) for p in self.records],
                'strict_grayscale_uint8': self.strict_grayscale_uint8, 'binary_target': self.binary_target,
                'num_rows': self.num_rows, 'num_cols': self.num_cols, 'grayscale_in': self.grayscale_in,
                'grayscale_target': self.grayscale_target, 'invert_in': self.invert_in,
                'invert_target': self.invert_target, 'dtype': str(self.dtype).removeprefix('torch.'),
                'scale_in': self.scale_in, 'scale_out': self.scale_out}

        if hasattr(self, '_folder_config'):
            config.pop('pairs')
            config['folders'] = dict(self._folder_config)
            config['sample_ids_sha256'] = hashlib.sha256(json.dumps(self.sample_ids).encode()).hexdigest()
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config['dtype'] = getattr(torch, config['dtype'])
        if 'folders' in config:
            folders = config.pop('folders')
            digest = config.pop('sample_ids_sha256')
            dataset = PairedImageDataset.from_folders(**folders, **config)
            if hashlib.sha256(json.dumps(dataset.sample_ids).encode()).hexdigest() != digest:
                raise ValueError('Paired folder sample IDs changed since source configuration.')
            return dataset
        return cls(**config)
