"""Sequential patch backward followed by one composed global backward."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from numbers import Real

import torch

from ..patches._geometry import (
    _validate_image,
    _validate_output,
    _validate_patches,
    add_patch_gradient,
    input_patch,
)
from ._patch_diagnostics import _PatchInstrumentation, _preserve_response_grad


@dataclass
class PatchBackwardResult:
    """Report detached loss values and instrumentation for one backward call.

    Attributes:
        main_loss: Arithmetic mean of the per-patch main losses.
        aux_loss: Unweighted global auxiliary loss, or None if no callback ran.
        aux_weight: Nonnegative multiplier applied to the auxiliary loss.
        weighted_aux_loss: Product of ``aux_weight`` and ``aux_loss``, or zero
            when no auxiliary callback was supplied.
        parameter_loss: Already weighted parameter penalty, or zero if absent.
        total_loss: Sum of main, weighted auxiliary and parameter losses.
        patches: Number of patches consumed in the supplied order.
        diagnostics: Optional gradient norms as Python floats. Keys identify
            response or producer contributions; this is not an optimizer state.
        timings: Optional elapsed seconds by stage, including explicit device
            synchronization. Unmeasured stages are absent.

    Loss fields are Python scalars and retain no autograd graph. The detached
    response gradient is returned separately by the training function.
    """

    main_loss: float = 0.0
    aux_loss: float | None = None
    aux_weight: float = 0.0
    weighted_aux_loss: float = 0.0
    parameter_loss: float = 0.0
    total_loss: float = 0.0
    patches: int = 0
    diagnostics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


@contextmanager
def _stage(name):
    try:
        yield
    except (TypeError, ValueError, RuntimeError, FloatingPointError) as exc:
        raise type(exc)(f"{name}: {exc}") from exc


def _scalar_loss(loss, name, device, *, differentiable=True):
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.is_floating_point():
        raise ValueError(f"{name} must return a floating scalar tensor")
    if loss.device != device:
        raise ValueError(f"{name} must be on the response device")
    if not torch.isfinite(loss):
        raise FloatingPointError(f"{name} must be finite")
    if differentiable and not loss.requires_grad:
        raise ValueError(f"{name} must be differentiable; use a graph-connected zero for a zero contribution")


def _validate_inputs(response, model, target, patches, main_loss, pixel_weights,
                     auxiliary_loss, auxiliary_weight, parameter_loss, strategy,
                     diagnostic_parameters, diagnostics, timing):
    _validate_image(response, "response", floating=True)
    _validate_image(target, "target")
    if target.is_complex() or target.requires_grad:
        raise ValueError("target must be real and must not require gradients")
    if (target.shape[0] != response.shape[0] or target.shape[-2:] != response.shape[-2:]
            or target.device != response.device):
        raise ValueError("target and response must have matching batch, spatial shape and device")
    for name, value in (("response", response), ("target", target)):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} must be finite")
    if not callable(model) or not callable(main_loss):
        raise TypeError("model and main_loss must be callable")
    if strategy not in ("local", "global_leaf"):
        raise ValueError("strategy must be 'local' or 'global_leaf'")
    if (isinstance(auxiliary_weight, bool) or not isinstance(auxiliary_weight, Real)
            or not math.isfinite(auxiliary_weight) or auxiliary_weight < 0):
        raise ValueError("auxiliary_weight must be finite and nonnegative")
    if auxiliary_loss is not None and not callable(auxiliary_loss):
        raise TypeError("auxiliary_loss must be callable or None")
    if auxiliary_weight > 0 and (auxiliary_loss is None or not response.requires_grad):
        raise ValueError("positive auxiliary_weight requires auxiliary_loss and a differentiable response")
    if not isinstance(diagnostics, bool) or not isinstance(timing, bool):
        raise TypeError("diagnostics and timing must be booleans")
    if pixel_weights is not None:
        _validate_image(pixel_weights, "pixel_weights")
        if pixel_weights.requires_grad or pixel_weights.is_complex():
            raise ValueError("pixel_weights must be real and must not require gradients")
        if pixel_weights.shape != target.shape or pixel_weights.device != target.device:
            raise ValueError("pixel_weights must match the target shape and device")
        if not torch.isfinite(pixel_weights).all() or (pixel_weights < 0).any():
            raise ValueError("pixel_weights must be finite and nonnegative")
    if parameter_loss is not None:
        _scalar_loss(parameter_loss, "parameter_loss", response.device)
        if response.requires_grad:
            with _preserve_response_grad(response):
                dependency = torch.autograd.grad(parameter_loss, response, retain_graph=True, allow_unused=True)[0]
            if dependency is not None:
                raise ValueError("parameter_loss must not depend on response")
    parameters = []
    for parameter in diagnostic_parameters:
        if not isinstance(parameter, torch.Tensor) or not parameter.is_leaf:
            raise ValueError("diagnostic_parameters must contain leaf tensors")
        if parameter.requires_grad and all(parameter is not previous for previous in parameters):
            parameters.append(parameter)
    return _validate_patches(patches, *response.shape[-2:]), tuple(parameters)


def _backward_patch(response, global_leaf, model, target, pixel_weights, patch, count,
                    main_loss, instrumentation, index):
    with _stage(f"patch {index} extraction"):
        local_input, source, local = input_patch(global_leaf if global_leaf is not None else response, patch)
        local_leaf = None
        if global_leaf is None:
            local_input = local_input.detach()
            if response.requires_grad:
                local_leaf = local_input.requires_grad_(True)
                # Keep the gradient owner a leaf while allowing in-place input operations.
                local_input = local_leaf.clone()
    with _stage(f"patch {index} model output/main_loss"):
        with instrumentation.measure("model_forward_loss_seconds"):
            logits = model(local_input)
            core = _validate_output(logits, patch, response.shape[0])
            if logits.device != response.device:
                raise ValueError("model output must be on the response device")
            rows, cols = patch.target
            truth = target[..., rows, cols]
            weights = None if pixel_weights is None else pixel_weights[..., rows, cols]
            loss = main_loss(core, truth, weights)
            _scalar_loss(loss, "main_loss", response.device)
            loss = loss / count
    with _stage(f"patch {index} backward"):
        with instrumentation.measure("model_backward_seconds"):
            loss.backward()
        gradient = None if local_leaf is None else local_leaf.grad
        checked_gradient = gradient if global_leaf is None else global_leaf.grad
        if checked_gradient is not None and not torch.isfinite(checked_gradient).all():
            raise FloatingPointError("response gradient must be finite")
    return loss.detach(), gradient, source, local


def backward_from_global_response(
    response, model, target, patches, *, main_loss, pixel_weights=None,
    auxiliary_loss=None, auxiliary_weight=0.0, parameter_loss=None,
    strategy="local", diagnostic_parameters=(), diagnostics=False, timing=False,
) -> tuple[PatchBackwardResult, torch.Tensor | None]:
    """Accumulate the mean patch loss and propagate its gradient to a producer.

    Args:
        response: Caller-produced float32 or float64 BCHW tensor. Its producer
            graph must remain available until this function finishes.
        model: Consumer called sequentially on each patch with halo. It must
            return float32 or float64 BCHW logits on the response device,
            preserving batch size and patch resolution, including the halo.
        target: Real BCHW tensor on the response device with the same batch and
            spatial shape. Channel compatibility belongs to ``main_loss``.
            Targets are constants and must not require gradients.
        patches: Nonempty iterable of patches whose cores fit in the response.
            Partial spatial coverage is allowed.
        main_loss: Callback ``main_loss(logits, target, weights)`` returning one
            finite differentiable scalar on the response device. Arguments are
            cropped to the supervised core. Do not divide by the patch count;
            this function applies that factor once. The callback must use its
            arguments without capturing the global response or another graph
            reused across patches.
        pixel_weights: Optional finite, nonnegative constant tensor matching
            the target shape and device. Passed to ``main_loss`` after cropping;
            weighting and pixel reduction belong to the callback.
        auxiliary_loss: Optional callback ``auxiliary_loss(response, target)``
            depending only on the response and constants. Evaluated once even
            at weight zero, for reporting.
        auxiliary_weight: Finite nonnegative multiplier. A positive weight
            requires a differentiable response and an auxiliary loss connected
            to that response.
        parameter_loss: Optional already weighted differentiable scalar on the
            response device, independent of the response. It contributes once
            to parameter gradients. Its graph may share upstream producer
            operations with the response.
        strategy: ``"local"`` accumulates the extraction adjoint explicitly;
            ``"global_leaf"`` uses autograd on a detached global leaf. Both
            support in-place consumer input operations.
        diagnostic_parameters: Producer leaf tensors to measure when
            diagnostics are enabled. This selection does not restrict which
            parameters receive gradients.
        diagnostics: Enable additional vector-Jacobian products and gradient
            norms. Assumes backward operations and hooks have no random or
            other stateful effects.
        timing: Enable stage timings with explicit device synchronization.
            Validation and scalar reporting can still synchronize when false.

    Returns:
        ``(result, response_gradient)``. The result contains detached Python
        loss scalars and optional instrumentation. The detached gradient sums
        main and weighted auxiliary contributions and excludes parameter_loss.
        It is None if the response does not require gradients. A disconnected
        differentiable response returns zeros, but receives no backward solely
        to materialize zero parameter gradients. A parameter penalty may still
        update its own parameters.

    Gradients can reach the valid halo outside a supervised core. Overlapping
    contributions are summed without coverage normalization. The caller owns
    train/eval modes, gradient zeroing, clipping and the optimizer step.
    Existing gradients accumulate. Producer and consumer must not share
    parameters. Only first-order differentiation is supported; AMP, GradScaler,
    DDP, higher orders and torch.compile are outside this contract.

    Geometry and argument errors are checked before patch backward. Runtime
    failures report the patch or global stage and can leave partial gradients
    or changed model buffers. Clear gradients and recompute the forward before
    retrying; do not step the optimizer after failure. No state rollback occurs.
    """
    patches, parameters = _validate_inputs(
        response, model, target, patches, main_loss, pixel_weights,
        auxiliary_loss, auxiliary_weight, parameter_loss, strategy,
        diagnostic_parameters, diagnostics, timing,
    )
    instrumentation = _PatchInstrumentation(response, parameters, diagnostics, timing)
    gradient = torch.zeros_like(response) if response.requires_grad else None
    global_leaf = response.detach().requires_grad_(True) if response.requires_grad and strategy == "global_leaf" else None
    connected = False
    main_total = None
    for index, patch in enumerate(patches):
        value, patch_gradient, source, local = _backward_patch(
            response, global_leaf, model, target, pixel_weights, patch, len(patches),
            main_loss, instrumentation, index,
        )
        main_total = value if main_total is None else main_total + value
        if patch_gradient is not None:
            with _stage(f"patch {index} gradient accumulation"):
                add_patch_gradient(gradient, patch_gradient, source, local)
            connected = True
        del patch_gradient
    if global_leaf is not None and global_leaf.grad is not None:
        gradient = global_leaf.grad.detach()
        connected = True
    instrumentation.response_contribution("main", gradient if connected else None)
    aux_value = None
    aux_gradient = None
    if auxiliary_loss is not None:
        with _stage("auxiliary_loss"):
            with instrumentation.measure("auxiliary_seconds"):
                aux = auxiliary_loss(response, target)
                _scalar_loss(aux, "auxiliary_loss", response.device, differentiable=auxiliary_weight > 0)
                aux_value = aux.detach()
                if auxiliary_weight > 0:
                    with _preserve_response_grad(response):
                        aux_gradient = torch.autograd.grad(aux, response, allow_unused=True)[0]
                    if aux_gradient is None:
                        raise ValueError("auxiliary_loss must depend on response")
                    aux_gradient = aux_gradient.detach() * auxiliary_weight
                del aux
    instrumentation.response_contribution("aux_weighted", aux_gradient)
    instrumentation.parameter_contribution(parameter_loss)
    if aux_gradient is not None:
        gradient.add_(aux_gradient)
        connected = True
    if gradient is not None and not torch.isfinite(gradient).all():
        raise FloatingPointError("global response gradient must be finite")
    roots, derivatives = [], []
    # An absent path must not turn into zero .grad values that trigger weight decay.
    if connected:
        roots.append(response)
        derivatives.append(gradient)
    if parameter_loss is not None:
        roots.append(parameter_loss)
        derivatives.append(None)
    if roots:
        with _stage("global backward"):
            with instrumentation.measure("producer_backward_seconds"):
                torch.autograd.backward(roots, derivatives)
    result = PatchBackwardResult(
        main_loss=float(main_total), aux_loss=None if aux_value is None else float(aux_value),
        aux_weight=float(auxiliary_weight),
        parameter_loss=0.0 if parameter_loss is None else float(parameter_loss.detach()),
        patches=len(patches), diagnostics=instrumentation.diagnostics, timings=instrumentation.timings,
    )
    result.weighted_aux_loss = result.aux_weight * (result.aux_loss if result.aux_loss is not None else 0.0)
    result.total_loss = result.main_loss + result.weighted_aux_loss + result.parameter_loss
    return result, gradient
