"""MLP CFP scoring model."""

from __future__ import annotations

from collections.abc import Sequence
import numbers

import torch

from .base import ScoringModel


class MLPScorer(ScoringModel):
    """Score tree nodes with a small MLP over normalized CFP attributes."""

    _ACTIVATIONS = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "gelu": torch.nn.GELU,
        "sigmoid": torch.nn.Sigmoid,
        "identity": torch.nn.Identity,
    }

    def __init__(
        self,
        num_features: int,
        *,
        hidden_units: Sequence[int] = (16,),
        activation: str = "relu",
        score_sharpness: float = 1.0,
        clamp: tuple[float, float] | None = None,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if int(num_features) < 1:
            raise ValueError("num_features must be positive.")
        self.num_features = int(num_features)
        self.hidden_units = self._normalize_hidden_units(hidden_units)
        self.activation = str(activation).lower()
        if self.activation not in self._ACTIVATIONS:
            supported = ", ".join(sorted(self._ACTIVATIONS))
            raise ValueError(f"unsupported MLP activation: {activation!r}; supported activations: {supported}")
        self.score_sharpness = float(score_sharpness)
        self.clamp = None if clamp is None else (float(clamp[0]), float(clamp[1]))
        if self.clamp is not None and self.clamp[0] >= self.clamp[1]:
            raise ValueError("clamp bounds must satisfy min < max.")

        layers = []
        in_features = self.num_features
        activation_cls = self._ACTIVATIONS[self.activation]
        for hidden in self.hidden_units:
            layers.append(torch.nn.Linear(in_features, hidden, device=device, dtype=dtype))
            layers.append(activation_cls())
            in_features = hidden
        layers.append(torch.nn.Linear(in_features, 1, device=device, dtype=dtype))
        self.network = torch.nn.Sequential(*layers)

    def init_identity(
        self,
        *,
        score_sharpness: float,
        p0: float = 0.995,
        output_weight_scale: float = 1e-3,
    ) -> None:
        """Initialize the MLP so scores start close to ``p0`` for all nodes."""
        sharpness = float(score_sharpness)
        if sharpness == 0.0:
            raise ValueError("identity initialization requires score_sharpness != 0.")
        output_weight_scale = float(output_weight_scale)
        if output_weight_scale < 0.0:
            raise ValueError("output_weight_scale must be non-negative.")

        linear_layers = [module for module in self.network if isinstance(module, torch.nn.Linear)]
        if not linear_layers:
            raise RuntimeError("MLPScorer identity initialization requires at least one Linear layer.")

        with torch.no_grad():
            final_layer = linear_layers[-1]
            torch.nn.init.normal_(final_layer.weight, mean=0.0, std=output_weight_scale)
            final_layer.bias.fill_(self.identity_logit(p0) / sharpness)

    @staticmethod
    def _normalize_hidden_units(hidden_units: Sequence[int]) -> tuple[int, ...]:
        if isinstance(hidden_units, numbers.Integral) and not isinstance(hidden_units, bool):
            hidden_units = (hidden_units,)
        if isinstance(hidden_units, (str, bytes)):
            raise TypeError("hidden_units must be a sequence of positive integers.")
        if not isinstance(hidden_units, Sequence):
            raise TypeError("hidden_units must be a sequence of positive integers.")

        normalized = []
        for units in hidden_units:
            if isinstance(units, bool) or not isinstance(units, numbers.Integral):
                raise TypeError("hidden_units must contain only positive integers.")
            units = int(units)
            if units < 1:
                raise ValueError("hidden_units must contain only positive integers.")
            normalized.append(units)
        return tuple(normalized)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        """Return unscaled node logits."""
        if features.dim() != 2:
            raise ValueError(f"expected features with shape (num_nodes, K), got {tuple(features.shape)}")
        if features.size(1) != self.num_features:
            raise ValueError(f"expected {self.num_features} features, got {features.size(1)}")
        return self.network(features).view(-1)

    def forward(
        self,
        features: torch.Tensor,
        tree_info=None,
        context=None,
        *,
        score_sharpness: float | None = None,
        clamp: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Return sigmoid scores for each tree node."""
        sharpness = self.score_sharpness if score_sharpness is None else float(score_sharpness)
        clamp_bounds = self.clamp if clamp is None else clamp
        scaled = sharpness * self.logits(features)
        if clamp_bounds is not None:
            scaled = torch.clamp(scaled, clamp_bounds[0], clamp_bounds[1])
        return torch.sigmoid(scaled)

    def to_config(self) -> dict:
        """Return a serializable scoring config."""
        return {
            "kind": "mlp",
            "hidden_units": list(self.hidden_units),
            "activation": self.activation,
        }
