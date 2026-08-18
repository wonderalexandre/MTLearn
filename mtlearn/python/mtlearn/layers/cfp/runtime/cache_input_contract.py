"""Input contract checks for CFP cached DataLoaders.

The CFP cached-dataloader path precomputes morphology trees and attributes from
``dataset[idx][0]`` before training. A tensor can have the right shape while
still being scientifically invalid for this operation: normalized z-score data,
negative values, NaNs, or ambiguous floating-point scales can alter the
gray-level order used by max-trees, min-trees, and trees of shapes.

This module centralizes the runtime checks for that contract. The checks are
strict enough to fail early for common preprocessing mistakes while still
accepting the image representations used by mtlearn examples: uint8 images,
integer intensities in ``[0, 255]``, normalized floats in ``[0, 1]``, and direct
float intensities in ``[0, 255]`` when the scale is not ambiguous.
"""

from __future__ import annotations

import torch


class CFPCacheInputError(ValueError):
    """Raised when a DataLoader batch cannot safely populate CFP caches."""


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
for _dtype_name in ("uint16", "uint32", "uint64"):
    if hasattr(torch, _dtype_name):
        _INTEGER_DTYPES.add(getattr(torch, _dtype_name))


def validate_cfp_cache_batch_x(
    x,
    *,
    expected_channels: int | None = None,
    sample_indices=None,
) -> None:
    """Validate that a batch can be converted to uint8 morphology images.

    ``build_dataloader_cached(...)`` constructs morphology trees from
    ``dataset[idx][0]``. The batch must therefore be a stable image tensor shaped
    ``(B, C, H, W)`` with finite, non-negative intensities that map to uint8
    without silently changing the intended gray-level ordering.
    """

    context = _sample_context(sample_indices)
    if not isinstance(x, torch.Tensor):
        raise CFPCacheInputError(
            "CFP cached DataLoader requires dataset[idx][0] to collate into a "
            f"torch.Tensor shaped (B, C, H, W){context}; got {type(x).__name__}."
        )

    if x.dim() != 4:
        raise CFPCacheInputError(
            "CFP cached DataLoader requires dataset[idx][0] to collate into "
            f"(B, C, H, W){context}; got shape={tuple(x.shape)}."
        )

    batch_size, channels, height, width = x.shape
    if expected_channels is not None and channels != int(expected_channels):
        raise CFPCacheInputError(
            "CFP cached DataLoader input channel mismatch"
            f"{context}: expected C={int(expected_channels)}, got C={channels}."
        )
    if height <= 0 or width <= 0:
        raise CFPCacheInputError(
            "CFP cached DataLoader requires non-empty image dimensions"
            f"{context}; got shape={tuple(x.shape)}."
        )
    if batch_size == 0:
        return

    if x.dtype == torch.bool:
        return
    if torch.is_complex(x):
        raise CFPCacheInputError(
            "CFP cached DataLoader requires real image intensities"
            f"{context}; got dtype={x.dtype}."
        )
    if not torch.is_floating_point(x) and x.dtype not in _INTEGER_DTYPES:
        raise CFPCacheInputError(
            "CFP cached DataLoader requires uint8, integer, or floating-point "
            f"image intensities{context}; got dtype={x.dtype}."
        )

    values = x.detach()
    if torch.is_floating_point(values) and not bool(torch.isfinite(values).all()):
        raise CFPCacheInputError(
            "CFP cached DataLoader requires finite image intensities"
            f"{context}; got NaN or inf values."
        )

    min_value = float(values.min())
    max_value = float(values.max())
    if min_value < 0.0:
        raise CFPCacheInputError(
            "CFP cached DataLoader requires non-negative image intensities"
            f"{context}; got min={min_value:g}."
        )
    if max_value > 255.0:
        raise CFPCacheInputError(
            "CFP cached DataLoader requires image intensities in uint8, [0, 1], "
            f"or [0, 255]{context}; got max={max_value:g}."
        )

    if torch.is_floating_point(values):
        normalized_upper = 1.0 + 1e-6
        conversion_branch_upper = 1.5
        if max_value <= normalized_upper or max_value > conversion_branch_upper:
            return
        # ``to_numpy_u8`` treats max <= 1.5 as normalized image data. Values in
        # (1, 1.5] are therefore ambiguous and could silently alter altitudes.
        raise CFPCacheInputError(
            "CFP cached DataLoader received an ambiguous floating-point image "
            f"scale{context}: max={max_value:g}. Use [0, 1] normalized floats, "
            "[0, 255] intensity floats with max > 1.5, or uint8."
        )


def _sample_context(sample_indices) -> str:
    """Format dataset indices for inclusion in validation error messages."""

    if sample_indices is None:
        return ""
    try:
        indices = sample_indices.detach().to("cpu").flatten()
    except AttributeError:
        return ""
    if indices.numel() == 0:
        return " at empty dataset-index batch"
    preview_values = [str(int(value)) for value in indices[:6].tolist()]
    suffix = ", ..." if indices.numel() > 6 else ""
    return f" at dataset indices [{', '.join(preview_values)}{suffix}]"
