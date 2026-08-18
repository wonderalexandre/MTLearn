"""Layer-owned linear-sigmoid parameter initialization for CFP layers."""

from __future__ import annotations

import math

import torch

from .linear_sigmoid import LinearSigmoidScorer


def _uses_layer_owned_linear_parameters(scoring_model) -> bool:
    return (
        isinstance(scoring_model, LinearSigmoidScorer)
        and not scoring_model.owns_parameters
    )


class LayerOwnedLinearParameterInitializer:
    """Create and initialize layer-owned linear scorer tensors."""

    @staticmethod
    def create_parameter_dicts(filter_specs, *, device) -> tuple[torch.nn.ParameterDict, torch.nn.ParameterDict]:
        """Return ``(_weights, _biases)`` ParameterDicts for layer-owned linear scorers."""
        weights = torch.nn.ParameterDict()
        biases = torch.nn.ParameterDict()
        device = torch.device(device)
        for spec in filter_specs:
            if not _uses_layer_owned_linear_parameters(spec.scoring_model):
                continue
            weight, bias = LayerOwnedLinearParameterInitializer.create_parameters(spec, device=device)
            weights[spec.key] = weight
            biases[spec.key] = bias
        return weights, biases

    @staticmethod
    def create_parameters(spec, *, device) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        """Create one initialized weight vector and scalar bias."""
        k = len(spec.attributes)
        weight = torch.empty(k, dtype=torch.float32, device=device)
        bias = torch.empty(1, dtype=torch.float32, device=device)
        fan_in, fan_out = k, 1
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        torch.nn.init.uniform_(weight, -math.sqrt(3.0) * std, math.sqrt(3.0) * std)
        torch.nn.init.constant_(bias, 0.0)
        return (
            torch.nn.Parameter(weight, requires_grad=True),
            torch.nn.Parameter(bias, requires_grad=True),
        )

    @staticmethod
    def logit(p: float) -> float:
        """Return a numerically clipped logit."""
        return LinearSigmoidScorer.identity_logit(p)

    def init_identity_with_bias(
        self,
        filter_specs,
        weights: torch.nn.ParameterDict,
        biases: torch.nn.ParameterDict,
        *,
        score_sharpness: float,
        p0: float = 0.995,
    ) -> None:
        """Set layer-owned linear scorers close to identity using positive bias."""
        with torch.no_grad():
            for spec in filter_specs:
                if spec.key not in weights:
                    continue
                spec.scoring_model.init_identity(
                    score_sharpness=score_sharpness,
                    p0=p0,
                    weight=weights[spec.key],
                    bias=biases[spec.key],
                )
