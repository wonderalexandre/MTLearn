"""Compatibility autograd function for implicit CFP reconstruction."""

from __future__ import annotations

import torch

from .tree_reconstructor import propagate_pixels_to_nodes, reconstruct_from_info


class ConnectedFilterPreprocessingImplicitJacobianFunction(torch.autograd.Function):
    """Autograd function for CFP with an implicit morphology-tree Jacobian.

    The forward reconstruction is mathematically equivalent to
    ``J.T @ filtered_residues`` where ``J`` is the dense node-to-pixel
    Jacobian, but the implementation uses tree entry/exit times and a prefix
    scan instead of materializing ``J``.
    """

    @staticmethod
    def forward_from_info(filtered_res, tpre, tpost, node_of_pixel, parent, order_forward=None):
        """Reconstruct pixels from filtered residues without a dense Jacobian."""
        return reconstruct_from_info(filtered_res, tpre, tpost, node_of_pixel, parent, order_forward)

    def backward_from_info(grad_output, tpre, tpost, parent, node_of_pixel, order_pre=None):
        """Propagate pixel gradients back to tree nodes without a dense matrix."""
        return propagate_pixels_to_nodes(grad_output, tpre, tpost, parent, node_of_pixel, order_pre)

    @staticmethod
    def forward(
        ctx,
        weight,
        bias,
        residues,
        tpre,
        tpost,
        parent,
        node_of_pixel,
        attrs2d,
        numRows: int,
        numCols: int,
        beta_f: float = 1.0,
        clamp_min=None,
        clamp_max=None,
        order_forward=None,
        order_backward=None,
    ):
        """Apply the connected filter using implicit reconstruction metadata."""
        logits = attrs2d @ weight.view(-1) + bias
        s = beta_f * logits
        if isinstance(clamp_min, bool) and clamp_max is None:
            clamp_min, clamp_max = (-12.0, 12.0) if clamp_min else (None, None)
        if (clamp_min is None) != (clamp_max is None):
            raise ValueError("clamp_min and clamp_max must be provided together.")
        if clamp_min is not None and clamp_max is not None:
            if clamp_min >= clamp_max:
                raise ValueError("clamp_min must be smaller than clamp_max.")
            clamp_mask = (s >= clamp_min) & (s <= clamp_max)
            s = torch.clamp(s, clamp_min, clamp_max)
        else:
            clamp_mask = torch.ones_like(s, dtype=torch.bool)
        sigmoid = torch.sigmoid(s)

        filtered_res = residues * sigmoid
        y = ConnectedFilterPreprocessingImplicitJacobianFunction.forward_from_info(
            filtered_res,
            tpre,
            tpost,
            node_of_pixel,
            parent,
            order_forward,
        )
        y_2d = y.reshape(numRows, numCols)

        ctx.save_for_backward(attrs2d, residues, sigmoid, clamp_mask, tpre, tpost, parent, node_of_pixel)
        ctx.beta_f = beta_f
        ctx.order_backward = order_backward
        return y_2d

    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradients for the learnable criterion parameters."""
        attrs2d, residues, sigmoid, clamp_mask, tpre, tpost, parent, node_of_pixel = ctx.saved_tensors
        beta_f = ctx.beta_f
        order_backward = ctx.order_backward
        grad_output_flat = grad_output.flatten()

        grad_nodes = ConnectedFilterPreprocessingImplicitJacobianFunction.backward_from_info(
            grad_output_flat,
            tpre,
            tpost,
            parent,
            node_of_pixel,
            order_backward,
        )

        d_sigmoid = sigmoid * (1 - sigmoid)
        grad_s = grad_nodes * residues * d_sigmoid * beta_f
        grad_s = torch.where(clamp_mask, grad_s, torch.zeros_like(grad_s))

        dW = attrs2d.T @ grad_s
        dB = grad_s.sum().view(1)

        return (
            dW,
            dB,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
