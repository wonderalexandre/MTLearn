from pathlib import Path
from types import SimpleNamespace
import warnings

import cv2
import numpy as np
import torch

pd = SimpleNamespace(DataFrame=lambda rows: rows)
INPUT_SUBDIR = "inputs"
TARGET_SUBDIR = "targets"


class EnhancementFragmentDataset(torch.utils.data.Dataset):

    def __init__(self, root, rows=None, cols=None, *, invert_input=False, cache_images=False, max_samples=None, seed=42):
        self.root = Path(root).expanduser()
        if (rows is None) != (cols is None):
            raise ValueError('NUM_ROWS and NUM_COLS must both be None or both be positive.')
        self.rows = None if rows is None else int(rows)
        self.cols = None if cols is None else int(cols)
        if self.rows is not None and (self.rows <= 0 or self.cols <= 0):
            raise ValueError('NUM_ROWS and NUM_COLS must be positive when resizing.')
        self.invert_input = bool(invert_input)
        self.cache_images = bool(cache_images)
        self._cache = {}
        inputs = self._index_folder(self.root / INPUT_SUBDIR)
        targets = self._index_folder(self.root / TARGET_SUBDIR)
        self.missing_target_ids = sorted(inputs.keys() - targets.keys())
        self.missing_input_ids = sorted(targets.keys() - inputs.keys())
        for label, missing in [('Inputs without masks', self.missing_target_ids), ('Masks without inputs', self.missing_input_ids)]:
            if missing:
                warnings.warn(f'{label}: {missing}. Excluding these IDs.', stacklevel=2)
        paired_ids = sorted(inputs.keys() & targets.keys())
        self.available_pairs = len(paired_ids)
        if max_samples is not None:
            if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 2:
                raise ValueError('MAX_SAMPLES must be None or an integer >= 2.')
            if max_samples < len(paired_ids):
                selected = np.random.RandomState(seed).choice(len(paired_ids), size=max_samples, replace=False)
                paired_ids = [paired_ids[int(i)] for i in sorted(selected)]
        if len(paired_ids) < 2:
            raise ValueError(f'Need at least two matched pairs in {self.root}.')
        self.pairs = [(sample_id, inputs[sample_id], targets[sample_id]) for sample_id in paired_ids]
        self.manifest = pd.DataFrame([{'id': sample_id, 'input': image_path.name, 'target': mask_path.name} for sample_id, image_path, mask_path in self.pairs])
        print(f'Inputs: {len(inputs)} | Masks: {len(targets)} | Matched: {self.available_pairs} | Selected: {len(self)}')

    @staticmethod
    def _index_folder(folder):
        if not folder.is_dir():
            raise FileNotFoundError(f'Dataset folder not found: {folder}')
        indexed = {}
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() != '.png':
                continue
            if not path.stem.isdecimal():
                raise ValueError(f'Expected a numeric PNG filename: {path}')
            sample_id = int(path.stem)
            if sample_id in indexed:
                raise ValueError(f'Duplicate numeric ID {sample_id}: {indexed[sample_id]} and {path}')
            indexed[sample_id] = path
        return indexed

    @staticmethod
    def _read_grayscale(path):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f'Could not read PNG: {path}')
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError(f'Expected a grayscale uint8 PNG: {path}; got shape={image.shape}, dtype={image.dtype}')
        return image

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        index = int(index)
        if index in self._cache:
            return self._cache[index]
        sample_id, image_path, mask_path = self.pairs[index]
        image = self._read_grayscale(image_path)
        mask = self._read_grayscale(mask_path)
        if image.shape != mask.shape:
            raise ValueError(f'Shape mismatch for ID {sample_id}: input={image.shape}, mask={mask.shape}')
        values = set(np.unique(mask).tolist())
        if not (values <= {0, 255} or values <= {0, 1}):
            raise ValueError(f'Expected a binary mask for ID {sample_id}; got {values}')
        if self.invert_input:
            image = 255 - image
        if self.rows is not None:
            image = cv2.resize(image, (self.cols, self.rows), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.cols, self.rows), interpolation=cv2.INTER_NEAREST)
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        target_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        sample = (image_tensor, target_tensor, str(sample_id))
        if self.cache_images:
            self._cache[index] = sample
        return sample
