"""Linear sigmoid CFP scoring model."""

from __future__ import annotations

import math

import torch

from .base import ScoringModel


class LinearSigmoidScorer(ScoringModel):
    """Current CFP gate model as a reusable scoring component."""

    def __init__(
        self,
        num_features: int,
        *,
        beta_f: float = 1.0,
        clamp: tuple[float, float] | None = None,
        device=None,
        dtype: torch.dtype = torch.float32,
        owns_parameters: bool = True,
    ):
        super().__init__()
        if int(num_features) < 1:
            raise ValueError("num_features must be positive.")
        self.num_features = int(num_features)
        self.beta_f = float(beta_f)
        self.clamp = None if clamp is None else (float(clamp[0]), float(clamp[1]))
        if self.clamp is not None and self.clamp[0] >= self.clamp[1]:
            raise ValueError("clamp bounds must satisfy min < max.")
        self.owns_parameters = bool(owns_parameters)

        if self.owns_parameters:
            weight = torch.empty(self.num_features, dtype=dtype, device=device)
            bias = torch.empty(1, dtype=dtype, device=device)
            fan_in, fan_out = self.num_features, 1
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            torch.nn.init.uniform_(weight, -math.sqrt(3.0) * std, math.sqrt(3.0) * std)
            torch.nn.init.constant_(bias, 0.0)
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias)
        else:
            self.weight = None
            self.bias = None

    def logits(self, features: torch.Tensor, *, weight=None, bias=None) -> torch.Tensor:
        """Return unscaled linear node logits."""
        if features.dim() != 2:
            raise ValueError(f"expected features with shape (num_nodes, K), got {tuple(features.shape)}")
        if features.size(1) != self.num_features:
            raise ValueError(f"expected {self.num_features} features, got {features.size(1)}")
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        if weight is None or bias is None:
            raise ValueError("LinearSigmoidScorer requires weight and bias parameters.")
        return features @ weight.view(-1) + bias

    def init_identity(
        self,
        *,
        beta_f: float,
        p0: float = 0.995,
        weight=None,
        bias=None,
    ) -> None:
        """Set linear weights to zero and bias so scores are close to ``p0``."""
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        if weight is None or bias is None:
            raise ValueError("LinearSigmoidScorer identity initialization requires weight and bias parameters.")
        beta = float(beta_f)
        if beta == 0.0:
            raise ValueError("identity initialization requires beta_f != 0.")
        with torch.no_grad():
            weight.zero_()
            bias.fill_(self.identity_logit(p0) / beta)

    def forward(
        self,
        features: torch.Tensor,
        tree_info=None,
        context=None,
        *,
        weight=None,
        bias=None,
        beta_f: float | None = None,
        clamp: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Return sigmoid scores for each tree node."""
        beta = self.beta_f if beta_f is None else float(beta_f)
        clamp_bounds = self.clamp if clamp is None else clamp
        scaled = beta * self.logits(features, weight=weight, bias=bias)
        if clamp_bounds is not None:
            scaled = torch.clamp(scaled, clamp_bounds[0], clamp_bounds[1])
        return torch.sigmoid(scaled)

    def to_config(self) -> dict:
        """Return a serializable scoring config."""
        return {"kind": "linear_sigmoid"}
