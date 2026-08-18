"""Shared image preprocessing helpers for mtlearn datasets."""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import torch

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def normalize_extensions(extensions: tuple[str, ...]) -> tuple[str, ...]:
    """Return lowercase extensions with leading dots."""

    normalized = []
    for extension in extensions:
        ext = str(extension).lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext)
    return tuple(normalized)


def normalize_resize_shape(
    num_rows: int | None,
    num_cols: int | None,
) -> tuple[int, int] | None:
    """Validate optional resize dimensions and return ``(rows, cols)``."""

    if (num_rows is None) ^ (num_cols is None):
        raise ValueError("num_rows and num_cols must both be None or both be defined.")
    if num_rows is None:
        return None

    rows = int(num_rows)
    cols = int(num_cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("num_rows and num_cols must be positive.")
    return rows, cols


def read_image(path: str, *, grayscale: bool) -> np.ndarray:
    """Read one image from disk as grayscale or RGB."""

    if grayscale:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read grayscale image: {path}")
        return image

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read color image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def invert_image(image: np.ndarray) -> np.ndarray:
    """Return a uint8-compatible image negative."""

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return 255 - image


def resize_image(
    image: np.ndarray,
    resize_shape: tuple[int, int] | None,
    *,
    interpolation: int,
) -> np.ndarray:
    """Resize an image to ``resize_shape`` when a target shape is configured."""

    if resize_shape is None:
        return image
    rows, cols = resize_shape
    return cv2.resize(image, (cols, rows), interpolation=interpolation)


def validate_spatial_shape(
    image: np.ndarray,
    expected_shape: tuple[int, int],
    *,
    name: str,
    path: str,
) -> None:
    """Validate the leading image dimensions."""

    if image.shape[:2] != expected_shape:
        raise ValueError(
            f"{name} must have spatial shape {expected_shape}; "
            f"got {image.shape[:2]} for {path}"
        )


def to_channel_first_tensor(image: np.ndarray, *, dtype: torch.dtype) -> torch.Tensor:
    """Convert a grayscale or channel-last image to ``(C, H, W)``."""

    image = np.ascontiguousarray(image)
    if image.ndim == 2:
        tensor = torch.from_numpy(image).unsqueeze(0)
    elif image.ndim == 3:
        tensor = torch.from_numpy(image).permute(2, 0, 1)
    else:
        raise ValueError(f"expected a 2D or 3D image array, got shape={image.shape}")
    return tensor.to(dtype)


def scale_tensor(tensor: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    """Scale image tensors from uint8 intensity space to ``[0, 1]``."""

    if enabled:
        return tensor / 255.0
    return tensor


def basename(path: Any) -> str:
    """Return the filesystem basename for a path-like object."""

    return os.path.basename(os.fspath(path))
