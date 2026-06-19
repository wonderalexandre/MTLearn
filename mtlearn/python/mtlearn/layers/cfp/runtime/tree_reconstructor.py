"""Tree-signal reconstruction helpers."""

from __future__ import annotations

import torch


def reconstruct_from_info(node_signal, tpre, tpost, node_of_pixel, parent, order_forward=None):
    """Reconstruct pixels from one scalar signal per tree node.

    The operation is equivalent to multiplying by the transpose of the dense
    node-to-pixel Jacobian, but uses entry/exit times and a prefix scan.
    """
    max_t = int(tpost.max().item()) + 1
    delta = torch.zeros(max_t, device=node_signal.device, dtype=node_signal.dtype)
    delta.index_add_(0, tpre, node_signal)
    delta.index_add_(0, tpost, -node_signal)
    y_cumsum = torch.cumsum(delta, dim=0)
    return y_cumsum[tpre[node_of_pixel]]


def propagate_pixels_to_nodes(grad_output, tpre, tpost, parent, node_of_pixel, order_pre=None):
    """Propagate pixel gradients back to tree nodes without a dense matrix."""
    g_pix = grad_output.reshape(-1)
    num_nodes = tpre.numel()
    base = torch.zeros(num_nodes, dtype=g_pix.dtype, device=g_pix.device)
    base.index_add_(0, node_of_pixel.reshape(-1), g_pix)

    if order_pre is None:
        order_pre = torch.argsort(tpre)
    pre_rank = torch.empty_like(order_pre)
    pre_rank[order_pre] = torch.arange(num_nodes, device=order_pre.device)

    base_sorted = base[order_pre]
    pref = torch.cumsum(base_sorted, dim=0)
    pref0 = torch.cat([pref.new_zeros(1), pref], dim=0)

    max_time = int(torch.max(tpost).item()) + 1
    counts = torch.bincount(tpre, minlength=max_time)
    cum = torch.cumsum(counts, dim=0)
    time_to_rank = torch.cat([cum.new_zeros(1), cum[:-1]], dim=0)

    left = pre_rank
    right = time_to_rank[tpost]
    return pref0[right] - pref0[left]


class TreeReconstructionFunction(torch.autograd.Function):
    """Autograd boundary for reconstructing pixels from tree-node signals."""

    @staticmethod
    def forward(
        ctx,
        node_signal,
        tpre,
        tpost,
        parent,
        node_of_pixel,
        numRows: int,
        numCols: int,
        order_forward=None,
        order_backward=None,
    ):
        ctx.save_for_backward(tpre, tpost, parent, node_of_pixel)
        ctx.order_backward = order_backward
        y = reconstruct_from_info(node_signal, tpre, tpost, node_of_pixel, parent, order_forward)
        return y.reshape(numRows, numCols)

    @staticmethod
    def backward(ctx, grad_output):
        tpre, tpost, parent, node_of_pixel = ctx.saved_tensors
        grad_node_signal = propagate_pixels_to_nodes(
            grad_output.reshape(-1),
            tpre,
            tpost,
            parent,
            node_of_pixel,
            ctx.order_backward,
        )
        return (
            grad_node_signal,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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
            tree_info["parent"],
            tree_info.get("order_forward"),
        )
        return y.reshape(tree_info["numRows"], tree_info["numCols"])

    @staticmethod
    def apply(node_signal, tree_info):
        """Reconstruct using the differentiable autograd function."""
        return TreeReconstructionFunction.apply(
            node_signal,
            tree_info["tpre"],
            tree_info["tpost"],
            tree_info["parent"],
            tree_info["node_of_pixel"],
            tree_info["numRows"],
            tree_info["numCols"],
            tree_info.get("order_forward"),
            tree_info.get("order_backward"),
        )


__all__ = [
    "TreeReconstructionFunction",
    "TreeReconstructor",
    "propagate_pixels_to_nodes",
    "reconstruct_from_info",
]
