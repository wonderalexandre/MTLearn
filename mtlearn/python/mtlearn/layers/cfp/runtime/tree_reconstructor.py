"""Tree-signal reconstruction using compact preorder intervals."""

from __future__ import annotations

import torch


def reconstruct_from_info(node_signal, tpre, tpost, node_of_pixel):
    """Compute ``J.T @ node_signal`` with compact subtree intervals.

    For T nodes, preorder entries lie in [0, T) and exclusive subtree ends
    lie in (0, T]. The T+1 buffer includes the root's final endpoint.
    Native trees with inactive node slots use empty [0, 0) intervals; sizing
    by slot count also accommodates these trees without a device-to-CPU read.
    """
    delta = node_signal.new_zeros(tpre.numel() + 1)
    delta.index_add_(0, tpre, node_signal)
    delta.index_add_(0, tpost, -node_signal)
    y_cumsum = torch.cumsum(delta, dim=0)
    # Keep the saved pixel map compact; signed indices exist only for this operation.
    return y_cumsum[tpre[node_of_pixel.to(torch.int64)]]


def propagate_pixels_to_nodes(grad_output, tpre, tpost, node_of_pixel):
    """Compute ``J @ grad_output`` by summing compact subtree intervals.

    Accumulate proper-part gradients directly in preorder, then subtract
    exclusive prefix sums at each node's subtree endpoints. No traversal
    permutation or conversion from event times to ranks is needed.
    """
    g_pix = grad_output.reshape(-1)
    proper_grad = g_pix.new_zeros(tpre.numel())
    pixel_pre = tpre[node_of_pixel.reshape(-1).to(torch.int64)]
    proper_grad.index_add_(0, pixel_pre, g_pix)
    prefix = torch.cat([proper_grad.new_zeros(1), torch.cumsum(proper_grad, dim=0)])
    return prefix[tpost] - prefix[tpre]


class TreeReconstructionFunction(torch.autograd.Function):
    """Autograd boundary for reconstructing pixels from tree-node signals."""

    @staticmethod
    def forward(ctx, node_signal, tpre, tpost, node_of_pixel, num_rows: int, num_cols: int):
        ctx.save_for_backward(tpre, tpost, node_of_pixel)
        y = reconstruct_from_info(node_signal, tpre, tpost, node_of_pixel)
        return y.reshape(num_rows, num_cols)

    @staticmethod
    def backward(ctx, grad_output):
        tpre, tpost, node_of_pixel = ctx.saved_tensors
        grad_node_signal = propagate_pixels_to_nodes(grad_output, tpre, tpost, node_of_pixel)
        return grad_node_signal, None, None, None, None, None


class TreeReconstructor:
    """Reconstruct pixels from one scalar signal per morphology-tree node."""

    @staticmethod
    def forward_from_info(node_signal, tree_info):
        """Reconstruct a 2D image from precomputed tree tensors."""
        y = reconstruct_from_info(
            node_signal,
            tree_info["tpre"],
            tree_info["tpost"],
            tree_info["node_of_pixel"],
        )
        return y.reshape(tree_info["num_rows"], tree_info["num_cols"])

    @staticmethod
    def apply(node_signal, tree_info):
        """Reconstruct using the differentiable autograd function."""
        return TreeReconstructionFunction.apply(
            node_signal,
            tree_info["tpre"],
            tree_info["tpost"],
            tree_info["node_of_pixel"],
            tree_info["num_rows"],
            tree_info["num_cols"],
        )


__all__ = [
    "TreeReconstructionFunction",
    "TreeReconstructor",
    "propagate_pixels_to_nodes",
    "reconstruct_from_info",
]
