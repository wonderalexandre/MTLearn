"""Optional gradient diagnostics and synchronized timing for patch training."""

from __future__ import annotations

from contextlib import contextmanager
import math
from time import perf_counter

import torch


def _norm(tensors):
    return math.sqrt(math.fsum(float(t.detach().norm()) ** 2 for t in tensors if t is not None))


@contextmanager
def _preserve_response_grad(response):
    """Prevent diagnostic VJPs and auxiliary differentiation from accumulating twice."""
    retained = response.is_leaf or response.retains_grad
    previous = response.grad if retained else None
    if retained:
        response.grad = None
    try:
        yield
    finally:
        if retained:
            response.grad = previous


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class _PatchInstrumentation:
    def __init__(self, response, parameters, diagnostics, timing):
        self.response = response
        self.parameters = parameters
        self.diagnostics_enabled = diagnostics
        self.timing_enabled = timing
        self.diagnostics = {}
        self.timings = {}

    @contextmanager
    def measure(self, name):
        if not self.timing_enabled:
            yield
            return
        _synchronize(self.response.device)
        started = perf_counter()
        yield
        _synchronize(self.response.device)
        self.timings[name] = self.timings.get(name, 0.0) + perf_counter() - started

    def response_contribution(self, name, gradient):
        if not self.diagnostics_enabled:
            return
        self.diagnostics[f"response_{name}_grad_norm"] = _norm((gradient,))
        gradients = ()
        if gradient is not None and self.parameters:
            with _preserve_response_grad(self.response):
                gradients = torch.autograd.grad(
                    self.response, self.parameters, grad_outputs=gradient,
                    retain_graph=True, allow_unused=True,
                )
        self.diagnostics[f"producer_{name}_grad_norm"] = _norm(gradients)

    def parameter_contribution(self, loss):
        if not self.diagnostics_enabled:
            return
        gradients = ()
        if loss is not None and self.parameters:
            gradients = torch.autograd.grad(loss, self.parameters, retain_graph=True, allow_unused=True)
        self.diagnostics["producer_parameter_grad_norm"] = _norm(gradients)
