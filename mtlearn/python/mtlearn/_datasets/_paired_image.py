"""Paired image dataset implementation for mtlearn experiments."""

from __future__ import annotations

import glob
import os

import cv2
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


class PairedImageDataset(Dataset):
    """Read matched input/target image pairs from one directory.

    Samples are returned as ``(input_tensor, target_tensor, filename)`` with
    channel-first tensors. Resize is optional; when configured, both input and
    target images are resized to ``(num_rows, num_cols)``. Without resize, the
    input and target spatial shapes must already match.
    """

    def __init__(
        self,
        root_dir: str,
        num_rows: int | None = None,
        num_cols: int | None = None,
        *,
        grayscale_in: bool = True,
        grayscale_target: bool = True,
        invert_in: bool = False,
        invert_target: bool = False,
        extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
        dtype: torch.dtype = torch.float32,
        scale_in: bool = True,
        scale_out: bool = True,
        prefix_in: str = "",
        prefix_target: str = "",
        suffix_in: str = "_in",
        suffix_target: str = "_target",
    ):
        """Create a dataset from image pairs stored in one directory."""

        super().__init__()
        self.root_dir = os.fspath(root_dir)
        self.resize_shape = normalize_resize_shape(num_rows, num_cols)
        self.num_rows = None if self.resize_shape is None else self.resize_shape[0]
        self.num_cols = None if self.resize_shape is None else self.resize_shape[1]
        self.grayscale_in = bool(grayscale_in)
        self.grayscale_target = bool(grayscale_target)
        self.invert_in = bool(invert_in)
        self.invert_target = bool(invert_target)
        self.extensions = normalize_extensions(extensions)
        self.dtype = dtype
        self.scale_in = bool(scale_in)
        self.scale_out = bool(scale_out)
        self.prefix_in = str(prefix_in)
        self.prefix_target = str(prefix_target)
        self.suffix_in = str(suffix_in)
        self.suffix_target = str(suffix_target)

        self.pairs = self._scan_pairs()
        if not self.pairs:
            raise RuntimeError(
                f"No input/target image pairs found in {self.root_dir} "
                f"with suffixes {self.suffix_in!r}/{self.suffix_target!r} "
                f"and extensions {self.extensions}."
            )

    def __len__(self):
        """Return the number of matched input/target pairs."""

        return len(self.pairs)

    def __getitem__(self, idx: int):
        """Return one matched image pair as ``(input, target, filename)``."""

        input_path, target_path = self.pairs[idx]

        image_in = read_image(input_path, grayscale=self.grayscale_in)
        image_target = read_image(target_path, grayscale=self.grayscale_target)

        if self.invert_in:
            image_in = invert_image(image_in)
        if self.invert_target:
            image_target = invert_image(image_target)

        image_in = resize_image(
            image_in,
            self.resize_shape,
            interpolation=cv2.INTER_AREA,
        )
        image_target = resize_image(
            image_target,
            self.resize_shape,
            interpolation=cv2.INTER_NEAREST,
        )
        validate_spatial_shape(
            image_target,
            image_in.shape[:2],
            name="target image",
            path=target_path,
        )

        tensor_in = to_channel_first_tensor(image_in, dtype=self.dtype)
        tensor_target = to_channel_first_tensor(image_target, dtype=self.dtype)

        tensor_in = scale_tensor(tensor_in, enabled=self.scale_in)
        tensor_target = scale_tensor(tensor_target, enabled=self.scale_out)

        return tensor_in, tensor_target, basename(input_path)

    def _scan_pairs(self) -> list[tuple[str, str]]:
        """Find input files and match each one to the first existing target."""

        pairs = []
        suffix_in_len = len(self.suffix_in)
        for extension in self.extensions:
            pattern = os.path.join(
                self.root_dir,
                f"{self.prefix_in}*{self.suffix_in}{extension}",
            )
            for input_path in glob.glob(pattern):
                base = basename(input_path)
                if not base.startswith(self.prefix_in) or not base.endswith(
                    self.suffix_in + extension
                ):
                    continue
                stem = base[len(self.prefix_in) : -suffix_in_len - len(extension)]
                target_path = None
                for target_extension in self.extensions:
                    candidate = f"{self.prefix_target}{stem}{self.suffix_target}{target_extension}"
                    candidate_path = os.path.join(self.root_dir, candidate)
                    if os.path.exists(candidate_path):
                        target_path = candidate_path
                        break
                if target_path is not None:
                    pairs.append((input_path, target_path))

        pairs.sort(key=lambda pair: pair[0])
        return pairs

    def train_test_split(self, test_size=0.25, shuffle=True, random_state=42):
        """Return ``(train_subset, test_subset)`` using stable pair indices."""

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
