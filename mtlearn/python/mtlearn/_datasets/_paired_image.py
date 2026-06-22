"""Paired image dataset implementation for mtlearn experiments.

This module contains the concrete dataset used by the current training
notebooks: inputs and targets are stored as separate image files in the same
directory and are matched by configurable prefix/suffix naming rules. Samples
are returned as ``(input_tensor, target_tensor, filename)`` with channel-first
image tensors, making them compatible with ordinary PyTorch training loops and
the CFP cached-dataloader path.
"""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ._split import _split_indices


class PairedImageDataset(Dataset):
    """
    Read image pairs named like ``01_in.jpg`` and ``01_target.jpg``.

    Parameters
    ----------
    root_dir:
        Directory containing the images.
    numRows, numCols:
        Desired image size. Resize is applied only when both are defined.
    grayscale_in, grayscale_target:
        If true, load the corresponding image as one grayscale channel.
    invert_in, invert_target:
        If true, apply ``255 - image`` before normalization.
    extensions:
        Supported file extensions.
    dtype:
        Output tensor dtype.
    scale_in, scale_out:
        If true, normalize the corresponding tensor to ``[0, 1]``.
    prefix_in, prefix_target, suffix_in, suffix_target:
        Naming rules used to match pairs.
    """

    def __init__(
        self,
        root_dir: str,
        numRows: int | None = None,
        numCols: int | None = None,
        *,
        grayscale_in: bool = True,
        grayscale_target: bool = True,
        invert_in: bool = False,
        invert_target: bool = False,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
        dtype: torch.dtype = torch.float32,
        scale_in: bool = True,
        scale_out: bool = True,
        prefix_in: str = "",
        prefix_target: str = "",
        suffix_in: str = "_in",
        suffix_target: str = "_target",
    ):
        """Create a dataset from image pairs stored in one directory.

        Args:
            root_dir: Directory containing input and target files.
            numRows: Optional output height. Resize is used only when both
                ``numRows`` and ``numCols`` are provided.
            numCols: Optional output width. Resize is used only when both
                ``numRows`` and ``numCols`` are provided.
            grayscale_in: If true, read inputs as single-channel grayscale.
            grayscale_target: If true, read targets as single-channel grayscale.
            invert_in: If true, invert input intensities before scaling.
            invert_target: If true, invert target intensities before scaling.
            extensions: File extensions considered for both inputs and targets.
            dtype: Torch dtype used for returned tensors.
            scale_in: If true, divide input tensors by 255.
            scale_out: If true, divide target tensors by 255.
            prefix_in: Optional prefix required for input filenames.
            prefix_target: Optional prefix used when looking up target names.
            suffix_in: Suffix identifying input files.
            suffix_target: Suffix identifying target files.
        """

        super().__init__()
        self.root_dir = root_dir
        self.numRows = numRows
        self.numCols = numCols
        self.grayscale_in = bool(grayscale_in)
        self.grayscale_target = bool(grayscale_target)
        self.invert_in = bool(invert_in)
        self.invert_target = bool(invert_target)
        self.extensions = tuple(e.lower() for e in extensions)
        self.dtype = dtype
        self.scale_in = bool(scale_in)
        self.scale_out = bool(scale_out)
        self.prefix_in = prefix_in
        self.prefix_target = prefix_target
        self.suffix_in = suffix_in
        self.suffix_target = suffix_target

        if (self.numRows is None) ^ (self.numCols is None):
            print(
                "[PairedImageDataset] Warning: numRows and numCols must both be "
                "None or both be defined. Resize will be ignored because only "
                "one was provided."
            )
            self.numRows = None
            self.numCols = None

        self.pairs = self._scan_pairs()
        if not self.pairs:
            raise RuntimeError(
                f"No *_in / *_target image pairs found in {root_dir} "
                f"with extensions {self.extensions}."
            )

    def __len__(self):
        """Return the number of matched input/target pairs."""

        return len(self.pairs)

    def __getitem__(self, idx: int):
        """Return one matched image pair as ``(input, target, filename)``."""

        input_path, target_path = self.pairs[idx]

        image_in = self._read_image(input_path, self.grayscale_in)
        image_target = self._read_image(target_path, self.grayscale_target)

        if self.invert_in:
            image_in = self._invert(image_in)
        if self.invert_target:
            image_target = self._invert(image_target)

        if self.numRows is not None and self.numCols is not None:
            image_in = cv2.resize(
                image_in,
                (self.numCols, self.numRows),
                interpolation=cv2.INTER_AREA,
            )
            image_target = cv2.resize(
                image_target,
                (self.numCols, self.numRows),
                interpolation=cv2.INTER_NEAREST,
            )

        tensor_in = self._to_tensor(image_in, self.grayscale_in)
        tensor_target = self._to_tensor(image_target, self.grayscale_target)

        if self.scale_in:
            tensor_in = tensor_in / 255.0
        if self.scale_out:
            tensor_target = tensor_target / 255.0

        return tensor_in, tensor_target, os.path.basename(input_path)

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
                base = os.path.basename(input_path)
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

    def _read_image(self, path: str, grayscale_flag: bool) -> np.ndarray:
        """Read an image with OpenCV and return grayscale or RGB data."""

        if grayscale_flag:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Failed to read grayscale image: {path}")
            return image

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read color image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _invert(img: np.ndarray) -> np.ndarray:
        """Return a uint8-compatible image negative."""

        if img.dtype != np.uint8:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
            return 255 - img_u8
        return 255 - img

    def _to_tensor(self, img: np.ndarray, grayscale_flag: bool) -> torch.Tensor:
        """Convert an ndarray to a ``(C, H, W)`` tensor."""

        if grayscale_flag:
            tensor = torch.from_numpy(img).unsqueeze(0)
        else:
            tensor = torch.from_numpy(img).permute(2, 0, 1)

        return tensor.to(self.dtype)

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
