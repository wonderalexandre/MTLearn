"""Tie active CPU preparation ownership to the lifetime of autograd saved data."""
import torch


class _PreparedLifetime(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, batch):
        # Saving a tensor (instead of a ctx Python attribute) releases the owner
        # after ordinary backward, and preserves it for retain_graph=True.
        token = torch.empty(0, device="cpu")
        token._prepared_batch = batch
        ctx.save_for_backward(token)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def keep_prepared_alive(output, batch):
    if torch.is_grad_enabled() and output.requires_grad:
        return _PreparedLifetime.apply(output, batch)
    return output
