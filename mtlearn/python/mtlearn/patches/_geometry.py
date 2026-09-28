"""Patch extraction and its summing adjoint on BCHW tensors."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch
import torch.nn.functional as F


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Patch:
    """Describe a square supervised core and its surrounding image context.

    Args:
        row: Zero-based row of the core's upper-left pixel.
        col: Zero-based column of the core's upper-left pixel.
        size: Positive side length of the core, in pixels.
        halo: Nonnegative context width on each side of the core, in pixels.

    All fields are integers excluding booleans. The core must fit inside the
    image when used for extraction, reconstruction or training. Context outside
    the image is filled with zeros. Instances are immutable.
    """

    row: int
    col: int
    size: int
    halo: int = 0

    def __post_init__(self):
        for name, minimum in (("row", 0), ("col", 0), ("size", 1), ("halo", 0)):
            _integer(getattr(self, name), name, minimum)

    @property
    def target(self) -> tuple[slice, slice]:
        """Return the row and column slices of the core in the global image."""
        return (
            slice(self.row, self.row + self.size),
            slice(self.col, self.col + self.size),
        )


def patch_grid(height: int, width: int, size: int, stride: int, halo: int = 0) -> list[Patch]:
    """Cover an image with square cores in row-major order.

    Args:
        height: Positive image height, in pixels.
        width: Positive image width, in pixels.
        size: Core side length, no greater than either image dimension.
        stride: Distance between consecutive core starts, with
            ``0 < stride <= size``.
        halo: Nonnegative context width on each side of a core.

    Returns:
        A list of patches covering every image pixel. The final start on each
        axis is added when the stride does not reach the boundary exactly.
        Overlapping cores retain row-major order.

    Raises:
        ValueError: An argument is not an integer excluding booleans, or the
            dimensions, stride or halo violate the geometry constraints.
    """
    for name, value in (("height", height), ("width", width), ("size", size), ("stride", stride)):
        _integer(value, name, 1)
    _integer(halo, "halo", 0)
    if size > min(height, width):
        raise ValueError("size must not exceed the image height or width")
    if stride > size:
        raise ValueError("stride must not exceed size; larger strides leave gaps")

    def starts(length):
        positions = list(range(0, length - size + 1, stride))
        if positions[-1] != length - size:
            positions.append(length - size)
        return positions

    return [Patch(row, col, size, halo) for row in starts(height) for col in starts(width)]


def _validate_image(image, name="image", *, floating=False):
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if image.layout != torch.strided or image.ndim != 4 or any(n <= 0 for n in image.shape):
        raise ValueError(f"{name} must be a nonempty strided BCHW tensor")
    if floating and image.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must have dtype float32 or float64")


def _validate_patch(patch, height, width):
    if not isinstance(patch, Patch):
        raise TypeError("patch must be a Patch")
    patch.__post_init__()
    if patch.row + patch.size > height or patch.col + patch.size > width:
        raise ValueError("patch core must lie within the image")


def _validate_patches(patches, height, width):
    patches = tuple(patches)
    if not patches:
        raise ValueError("patches must not be empty")
    for index, patch in enumerate(patches):
        try:
            _validate_patch(patch, height, width)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"patch {index} geometry: {exc}") from exc
    return patches


def input_patch(
    image: torch.Tensor, patch: Patch,
) -> tuple[torch.Tensor, tuple[slice, slice], tuple[slice, slice]]:
    """Extract a core with halo, filling context outside the image with zeros.

    Args:
        image: Nonempty strided tensor shaped ``(B, C, H, W)``.
        patch: Core geometry and context width. The core must fit in the image.

    Returns:
        ``(local_input, source_slices, local_slices)``. The local tensor has
        spatial shape ``(size + 2 * halo, size + 2 * halo)``. The two pairs of
        slices identify matching valid regions for :func:`add_patch_gradient`.
        Batch, channels, dtype and device are preserved. Extraction remains
        differentiable for floating-point input.
    """
    _validate_image(image)
    height, width = image.shape[-2:]
    _validate_patch(patch, height, width)
    top, left = patch.row - patch.halo, patch.col - patch.halo
    bottom = patch.row + patch.size + patch.halo
    right = patch.col + patch.size + patch.halo
    row_start, col_start = max(0, top), max(0, left)
    row_stop, col_stop = min(height, bottom), min(width, right)
    region = image[..., row_start:row_stop, col_start:col_stop]
    padded = F.pad(region, (col_start - left, right - col_stop, row_start - top, bottom - row_stop))
    source = (slice(row_start, row_stop), slice(col_start, col_stop))
    local = (slice(row_start - top, row_stop - top), slice(col_start - left, col_stop - left))
    return padded, source, local


def supervised_logits(logits: torch.Tensor, patch: Patch) -> torch.Tensor:
    """Remove the halo from a model output at the full patch resolution.

    Args:
        logits: Nonempty float32 or float64 BCHW tensor with spatial shape
            ``(size + 2 * halo, size + 2 * halo)``.
        patch: Geometry used to produce the model input.

    Returns:
        A differentiable view of the supervised ``size`` by ``size`` core.
    """
    _validate_image(logits, "logits", floating=True)
    if not isinstance(patch, Patch):
        raise TypeError("patch must be a Patch")
    patch.__post_init__()
    side = patch.size + 2 * patch.halo
    if logits.shape[-2:] != (side, side):
        raise ValueError(f"logits spatial shape must be {(side, side)} including the halo")
    return logits[..., patch.halo:patch.halo + patch.size, patch.halo:patch.halo + patch.size]


def _validate_output(logits, patch, batch_size):
    core = supervised_logits(logits, patch)
    if logits.shape[0] != batch_size:
        raise ValueError("model output must preserve the input batch size")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("model output must be finite")
    return core


def _slice_shape(slices, shape, name):
    if not isinstance(slices, (tuple, list)) or len(slices) != 2:
        raise ValueError(f"{name} must contain two spatial slices")
    lengths = []
    for item, limit in zip(slices, shape):
        if not isinstance(item, slice):
            raise ValueError(f"{name} must contain slices")
        _integer(item.start, f"{name} slice start", 0)
        _integer(item.stop, f"{name} slice stop", 1)
        if item.step is not None:
            _integer(item.step, f"{name} slice step", 1)
            if item.step != 1:
                raise ValueError(f"{name} slice step must be one")
        if item.start >= item.stop or item.stop > limit:
            raise ValueError(f"{name} slices must be nonempty and within the tensor")
        lengths.append(item.stop - item.start)
    return tuple(lengths)


def add_patch_gradient(
    global_gradient: torch.Tensor,
    patch_gradient: torch.Tensor,
    source: tuple[slice, slice],
    local: tuple[slice, slice],
) -> None:
    """Add a detached local gradient in place, discarding only external padding.

    Args:
        global_gradient: Detached float32 or float64 BCHW accumulator, modified
            in place.
        patch_gradient: Detached BCHW gradient of a local input including halo.
        source: Row and column slices into the global accumulator, as returned
            by :func:`input_patch`.
        local: Corresponding row and column slices into the local gradient.

    Both tensors must have matching batch, channels, dtype and device. Spatial
    slices must be explicit, bounded and equally sized, with unit steps.
    Overlapping contributions are summed without coverage normalization.
    This is the adjoint of extraction, including the valid halo; it is distinct
    from averaging overlapping core logits during reconstruction.
    """
    _validate_image(global_gradient, "global_gradient", floating=True)
    _validate_image(patch_gradient, "patch_gradient", floating=True)
    if global_gradient.requires_grad or patch_gradient.requires_grad:
        raise ValueError("gradient tensors must be detached; only first-order accumulation is supported")
    if (global_gradient.shape[:2] != patch_gradient.shape[:2]
            or global_gradient.dtype != patch_gradient.dtype
            or global_gradient.device != patch_gradient.device):
        raise ValueError("gradient tensors must have matching batch, channels, dtype and device")
    if (_slice_shape(source, global_gradient.shape[-2:], "source")
            != _slice_shape(local, patch_gradient.shape[-2:], "local")):
        raise ValueError("source and local slices must have matching shapes")
    global_gradient[..., source[0], source[1]].add_(patch_gradient[..., local[0], local[1]])
