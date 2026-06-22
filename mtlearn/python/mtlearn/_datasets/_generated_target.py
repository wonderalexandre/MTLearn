"""Datasets that generate supervised image targets on demand.

This module contains ``GeneratedTargetImageDataset``, a PyTorch dataset for
experiments where the desired target is a deterministic function of the input
image instead of a second file on disk. The dataset reads source images from a
directory, applies the common mtlearn preprocessing steps (optional grayscale
loading, optional inversion, resize, tensor conversion, and scaling), calls a
user-provided target function, and returns samples shaped like
``(input_tensor, target_tensor, filename)``.

The returned input tensor is intended to satisfy the CFP cached-dataloader
contract: channel-first ``(C, H, W)`` image data in either uint8-like intensity
space or normalized ``[0, 1]`` floating-point space. This makes the dataset
usable with ``ConnectedFilterPreprocessingLayer.build_dataloader_cached(...)``.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ._split import _split_indices


class GeneratedTargetImageDataset(Dataset):
    """
    Read images from a directory and generate each target with a callable.

    ``target_fn`` receives the input image after reading, optional inversion, and
    resize. It must return an image with the requested spatial dimensions.
    """

    def __init__(
        self,
        root_dir: str,
        numRows: int,
        numCols: int,
        target_fn: Callable[[np.ndarray], np.ndarray],
        *,
        grayscale: bool = True,
        invert_in: bool = False,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
        dtype: torch.dtype = torch.float32,
        scale_in: bool = True,
        scale_out: bool = True,
        suffix_in: str = "",
    ):
        """Create a dataset that generates targets from source images.

        Args:
            root_dir: Directory containing source images.
            numRows: Output image height after preprocessing.
            numCols: Output image width after preprocessing.
            target_fn: Callable receiving the preprocessed input image as a
                ``numpy.ndarray`` and returning the desired target image.
            grayscale: If true, read source images as single-channel grayscale.
                Otherwise, read color images and convert OpenCV BGR to RGB.
            invert_in: If true, apply ``255 - image`` to the source image before
                resize and target generation.
            extensions: File extensions considered when scanning ``root_dir``.
            dtype: Torch dtype used for both returned tensors.
            scale_in: If true, divide the input tensor by 255.
            scale_out: If true, divide the target tensor by 255.
            suffix_in: Optional filename suffix used to select source images,
                such as ``"_in"`` for files named ``sample_in.png``.
        """

        super().__init__()
        if not callable(target_fn):
            raise TypeError("target_fn must be callable")

        self.root_dir = os.fspath(root_dir)
        self.numRows = int(numRows)
        self.numCols = int(numCols)
        if self.numRows <= 0 or self.numCols <= 0:
            raise ValueError("numRows and numCols must be positive")

        self.target_fn = target_fn
        self.grayscale = bool(grayscale)
        self.invert_in = bool(invert_in)
        self.extensions = tuple(extension.lower() for extension in extensions)
        self.dtype = dtype
        self.scale_in = bool(scale_in)
        self.scale_out = bool(scale_out)
        self.suffix_in = str(suffix_in)

        self.paths = self._scan_images()
        if not self.paths:
            raise FileNotFoundError(
                f"No images found in {self.root_dir} "
                f"with extensions {self.extensions}."
            )

    def __len__(self):
        """Return the number of source images selected by the scanner."""

        return len(self.paths)

    def __getitem__(self, idx: int):
        """Return ``(input_tensor, target_tensor, filename)`` for one image.

        The input image is read, optionally inverted, resized, converted to a
        tensor, and optionally scaled. The target image is generated from the
        preprocessed input image and must match the configured spatial shape.
        """

        path = self.paths[idx]

        image_in = self._read_image(path)
        if self.invert_in:
            image_in = self._invert(image_in)

        image_in = cv2.resize(
            image_in,
            (self.numCols, self.numRows),
            interpolation=cv2.INTER_AREA,
        )
        image_target = self._target_image(path, image_in)

        tensor_in = self._to_tensor(image_in)
        tensor_target = self._to_tensor(image_target)

        if self.scale_in:
            tensor_in = tensor_in / 255.0
        if self.scale_out:
            tensor_target = tensor_target / 255.0

        return tensor_in, tensor_target, os.path.basename(path)

    def _scan_images(self) -> list[str]:
        """Collect source image paths matching the extension and suffix rules."""

        paths = []
        for extension in self.extensions:
            # ``suffix_in`` lets users keep generated/target files in the same
            # directory without accidentally treating them as source images.
            pattern = os.path.join(self.root_dir, f"*{self.suffix_in}{extension}")
            paths.extend(glob.glob(pattern))
        paths.sort()
        return paths

    def _read_image(self, path: str) -> np.ndarray:
        """Read one image from disk using the configured color mode."""

        if self.grayscale:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Failed to read grayscale image: {path}")
            return image

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read color image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _target_image(self, path: str, image_in: np.ndarray) -> np.ndarray:
        """Run ``target_fn`` and validate its image-like return value."""

        # Target functions may do in-place operations, so pass a copy and keep
        # the input tensor generation independent from target generation.
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
        if image_target.shape[:2] != (self.numRows, self.numCols):
            raise ValueError(
                "target_fn must return an image with shape "
                f"({self.numRows}, {self.numCols}) or "
                f"({self.numRows}, {self.numCols}, C); got shape={image_target.shape} "
                f"for {path}"
            )
        return image_target

    @staticmethod
    def _invert(img: np.ndarray) -> np.ndarray:
        """Return a uint8-compatible image negative."""

        if img.dtype != np.uint8:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
            return 255 - img_u8
        return 255 - img

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert a grayscale or channel-last image to ``(C, H, W)`` tensor."""

        img = np.ascontiguousarray(img)
        if img.ndim == 2:
            tensor = torch.from_numpy(img).unsqueeze(0)
        else:
            tensor = torch.from_numpy(img).permute(2, 0, 1)
        return tensor.to(self.dtype)

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
