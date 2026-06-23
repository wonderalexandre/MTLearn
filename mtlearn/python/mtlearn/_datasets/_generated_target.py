"""Datasets that generate supervised image targets on demand."""

from __future__ import annotations

import glob
import os
from collections.abc import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ._image_ops import (
    DEFAULT_IMAGE_EXTENSIONS,
    basename,
    invert_image,
    normalize_extensions,
    normalize_resize_shape,
    read_image,
    resize_image,
    scale_tensor,
    to_channel_first_tensor,
    validate_spatial_shape,
)
from ._split import _split_indices


class GeneratedTargetImageDataset(Dataset):
    """Read source images and generate each target with ``target_fn``.

    Samples are returned as ``(input_tensor, target_tensor, filename)`` with
    channel-first tensors. Resize is optional; when configured, the source image
    is resized before calling ``target_fn``. The generated target must match the
    post-resize source spatial shape.
    """

    def __init__(
        self,
        root_dir: str,
        target_fn: Callable[[np.ndarray], np.ndarray],
        *,
        num_rows: int | None = None,
        num_cols: int | None = None,
        grayscale: bool = True,
        invert_in: bool = False,
        extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
        dtype: torch.dtype = torch.float32,
        scale_in: bool = True,
        scale_out: bool = True,
        suffix_in: str = "",
    ):
        """Create a dataset that generates targets from source images."""

        super().__init__()
        if not callable(target_fn):
            raise TypeError("target_fn must be callable")

        self.root_dir = os.fspath(root_dir)
        self.resize_shape = normalize_resize_shape(num_rows, num_cols)
        self.num_rows = None if self.resize_shape is None else self.resize_shape[0]
        self.num_cols = None if self.resize_shape is None else self.resize_shape[1]
        self.target_fn = target_fn
        self.grayscale = bool(grayscale)
        self.invert_in = bool(invert_in)
        self.extensions = normalize_extensions(extensions)
        self.dtype = dtype
        self.scale_in = bool(scale_in)
        self.scale_out = bool(scale_out)
        self.suffix_in = str(suffix_in)

        self.paths = self._scan_images()
        if not self.paths:
            raise FileNotFoundError(
                f"No images found in {self.root_dir} "
                f"with suffix {self.suffix_in!r} and extensions {self.extensions}."
            )

    def __len__(self):
        """Return the number of source images selected by the scanner."""

        return len(self.paths)

    def __getitem__(self, idx: int):
        """Return ``(input_tensor, target_tensor, filename)`` for one image."""

        path = self.paths[idx]

        image_in = read_image(path, grayscale=self.grayscale)
        if self.invert_in:
            image_in = invert_image(image_in)

        image_in = resize_image(
            image_in,
            self.resize_shape,
            interpolation=cv2.INTER_AREA,
        )
        image_target = self._target_image(path, image_in)

        tensor_in = to_channel_first_tensor(image_in, dtype=self.dtype)
        tensor_target = to_channel_first_tensor(image_target, dtype=self.dtype)

        tensor_in = scale_tensor(tensor_in, enabled=self.scale_in)
        tensor_target = scale_tensor(tensor_target, enabled=self.scale_out)

        return tensor_in, tensor_target, basename(path)

    def _scan_images(self) -> list[str]:
        """Collect source image paths matching the extension and suffix rules."""

        paths = []
        for extension in self.extensions:
            pattern = os.path.join(self.root_dir, f"*{self.suffix_in}{extension}")
            paths.extend(glob.glob(pattern))
        paths.sort()
        return paths

    def _target_image(self, path: str, image_in: np.ndarray) -> np.ndarray:
        """Run ``target_fn`` and validate its image-like return value."""

        image_target = self.target_fn(np.array(image_in, copy=True))
        if not isinstance(image_target, np.ndarray):
            raise TypeError(
                "target_fn must return a numpy.ndarray; "
                f"got {type(image_target).__name__} for {path}"
            )
        if image_target.ndim not in (2, 3):
            raise ValueError(
                "target_fn must return a 2D grayscale or 3D channel-last image; "
                f"got shape={image_target.shape} for {path}"
            )
        validate_spatial_shape(
            image_target,
            image_in.shape[:2],
            name="target_fn output",
            path=path,
        )
        return image_target

    def train_test_split(self, test_size=0.25, shuffle=True, random_state=42):
        """Return ``(train_subset, test_subset)`` using stable dataset indices."""

        train_idx, test_idx = _split_indices(
            len(self),
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
        )
        return (
            torch.utils.data.Subset(self, train_idx.tolist()),
            torch.utils.data.Subset(self, test_idx.tolist()),
        )
