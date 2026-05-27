"""Checkpoint helpers for models that contain CFP preprocessing layers."""

from __future__ import annotations

import inspect
from typing import Any, Callable

import torch

from .ConnectedFilterPreprocessingLayer import ConnectedFilterPreprocessingLayer


def collect_cfp_configs(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    """Return serializable configs for every primary CFP layer in ``model``."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    return {
        name: module.get_config()
        for name, module in model.named_modules()
        if isinstance(module, ConnectedFilterPreprocessingLayer)
    }


def save_checkpoint(
    path,
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Save a PyTorch checkpoint with automatically collected CFP configs.

    The model parameters are saved with the normal PyTorch ``state_dict``.
    Primary CFP layers are discovered by module name and their constructor
    configs are saved separately so a caller can reconstruct the model before
    calling ``load_state_dict``. The payload intentionally contains only model
    weights and CFP configs that define the meaning and shape of CFP-related
    weights.
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "cfp_configs": collect_cfp_configs(model),
    }
    torch.save(payload, path)
    return payload


def _load_torch_checkpoint(path, *, device=None, weights_only: bool = True):
    try:
        return torch.load(path, map_location=device, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=device)


def _build_model_from_factory(factory: Callable[..., torch.nn.Module], cfp_configs: dict[str, dict[str, Any]]):
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        model = factory(cfp_configs)
    else:
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        ]
        if len(required) == 0:
            model = factory()
        elif len(required) == 1 and required[0].kind is not inspect.Parameter.KEYWORD_ONLY:
            model = factory(cfp_configs)
        else:
            raise TypeError("model_factory must accept zero arguments or one positional cfp_configs argument.")

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model_factory must return a torch.nn.Module.")
    return model


def load_checkpoint(
    path,
    model_or_factory: torch.nn.Module | Callable[..., torch.nn.Module],
    *,
    device=None,
    strict: bool = True,
    weights_only: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a checkpoint saved by ``save_checkpoint``.

    ``model_or_factory`` can be either an already constructed module or a
    callable that returns a module. Factories may accept no arguments when the
    model constructor already hard-codes its CFP layers, or one positional
    ``cfp_configs`` argument when they need the saved CFP configs.
    """
    checkpoint = _load_torch_checkpoint(path, device=device, weights_only=weights_only)
    cfp_configs = checkpoint.get("cfp_configs", {})

    if isinstance(model_or_factory, torch.nn.Module):
        model = model_or_factory
    elif callable(model_or_factory):
        model = _build_model_from_factory(model_or_factory, cfp_configs)
    else:
        raise TypeError("model_or_factory must be a torch.nn.Module or callable.")

    if device is not None:
        model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return model, checkpoint


__all__ = [
    "collect_cfp_configs",
    "load_checkpoint",
    "save_checkpoint",
]
