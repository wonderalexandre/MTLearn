"""Uniform averaging of overlapping core logits without gradient tracking."""

from __future__ import annotations

import torch

from ._geometry import _validate_image, _validate_output, _validate_patches, input_patch


@torch.no_grad()
def reconstruct_logits(model, image: torch.Tensor, patches) -> torch.Tensor:
    """Reconstruct BCHW logits by averaging all overlapping supervised cores.

    Args:
        model: Callable receiving one BCHW patch with halo and returning a
            float32 or float64 tensor with the same batch and spatial shape.
        image: Nonempty float32 or float64 BCHW tensor to partition.
        patches: Nonempty iterable of patches whose cores cover every image
            pixel. All cores must fit inside the image.

    Returns:
        Detached BCHW logits, averaging contributions at each covered pixel.
        Output channels, dtype and device are inferred from the first model
        output and must remain consistent across patches.

    The model runs once per patch in the supplied order. Its train/eval mode is
    preserved; call ``model.eval()`` explicitly for evaluation. No sigmoid,
    softmax, threshold or response scaling is applied. This reconstructs patch
    predictions and does not guarantee equivalence to a full-image forward.

    Invalid geometry or incomplete coverage fails before any model call.
    Model-output failures report the patch index; model buffers are not rolled
    back if a call changes them.
    """
    _validate_image(image, floating=True)
    if not callable(model):
        raise TypeError("model must be callable")
    patches = _validate_patches(patches, *image.shape[-2:])
    coverage = torch.zeros(image.shape[-2:], dtype=torch.int64, device="cpu")
    for patch in patches:
        coverage[patch.target].add_(1)
    if (coverage == 0).any():
        raise ValueError("patches do not cover the entire image")
    logits_sum = None
    for index, patch in enumerate(patches):
        try:
            local_input, _, _ = input_patch(image, patch)
            logits = model(local_input)
            core = _validate_output(logits, patch, image.shape[0])
            if logits_sum is None:
                logits_sum = logits.new_zeros((image.shape[0], logits.shape[1], *image.shape[-2:]))
                coverage = coverage.to(device=logits.device)
            elif (logits.shape[1] != logits_sum.shape[1] or logits.dtype != logits_sum.dtype
                  or logits.device != logits_sum.device):
                raise ValueError("model outputs must keep the same channels, dtype and device across patches")
            logits_sum[..., patch.target[0], patch.target[1]].add_(core)
        except (TypeError, ValueError, RuntimeError, FloatingPointError) as exc:
            raise type(exc)(f"patch {index} reconstruction: {exc}") from exc
    logits_sum.div_(coverage)
    if not torch.isfinite(logits_sum).all():
        raise FloatingPointError("reconstructed logits must be finite")
    return logits_sum
