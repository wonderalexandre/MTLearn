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
        hidden_channels: Sequence[int] = (16,),
        activation: str = "relu",
        beta_f: float = 1.0,
        clamp: tuple[float, float] | None = None,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if int(num_features) < 1:
            raise ValueError("num_features must be positive.")
        self.num_features = int(num_features)
        self.hidden_channels = self._normalize_hidden_channels(hidden_channels)
        self.activation = str(activation).lower()
        if self.activation not in self._ACTIVATIONS:
            supported = ", ".join(sorted(self._ACTIVATIONS))
            raise ValueError(f"unsupported MLP activation: {activation!r}; supported activations: {supported}")
        self.beta_f = float(beta_f)
        self.clamp = None if clamp is None else (float(clamp[0]), float(clamp[1]))
        if self.clamp is not None and self.clamp[0] >= self.clamp[1]:
            raise ValueError("clamp bounds must satisfy min < max.")

        layers = []
        in_features = self.num_features
        activation_cls = self._ACTIVATIONS[self.activation]
        for hidden in self.hidden_channels:
            layers.append(torch.nn.Linear(in_features, hidden, device=device, dtype=dtype))
            layers.append(activation_cls())
            in_features = hidden
        layers.append(torch.nn.Linear(in_features, 1, device=device, dtype=dtype))
        self.network = torch.nn.Sequential(*layers)

    @staticmethod
    def _normalize_hidden_channels(hidden_channels: Sequence[int]) -> tuple[int, ...]:
        if isinstance(hidden_channels, numbers.Integral) and not isinstance(hidden_channels, bool):
            hidden_channels = (hidden_channels,)
        if isinstance(hidden_channels, (str, bytes)):
            raise TypeError("hidden_channels must be a sequence of positive integers.")
        if not isinstance(hidden_channels, Sequence):
            raise TypeError("hidden_channels must be a sequence of positive integers.")

        normalized = []
        for channel in hidden_channels:
            if isinstance(channel, bool) or not isinstance(channel, numbers.Integral):
                raise TypeError("hidden_channels must contain only positive integers.")
            channel = int(channel)
            if channel < 1:
                raise ValueError("hidden_channels must contain only positive integers.")
            normalized.append(channel)
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
        beta_f: float | None = None,
        clamp: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Return sigmoid scores for each tree node."""
        beta = self.beta_f if beta_f is None else float(beta_f)
        clamp_bounds = self.clamp if clamp is None else clamp
        scaled = beta * self.logits(features)
        if clamp_bounds is not None:
            scaled = torch.clamp(scaled, clamp_bounds[0], clamp_bounds[1])
        return torch.sigmoid(scaled)

    def to_config(self) -> dict:
        """Return a serializable scoring config."""
        return {
            "kind": "mlp",
            "hidden_channels": list(self.hidden_channels),
            "activation": self.activation,
        }
