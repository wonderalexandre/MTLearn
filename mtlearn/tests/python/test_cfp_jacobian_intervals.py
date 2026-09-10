"""Implicit reconstruction against independent, hand-computable tree supports."""
import pytest
import torch

import mtlearn
from mtlearn.layers import ConnectedFilterPreprocessingLayer
from mtlearn.layers.cfp import CFPPreprocessor
from mtlearn.layers.cfp.runtime.tree_reconstructor import (
    TreeReconstructionFunction, propagate_pixels_to_nodes, reconstruct_from_info,
)

pytestmark = pytest.mark.integration


def test_interval_products_and_full_output_gradcheck(monkeypatch):
    # Node 2 is the root, 0 and 1 its children, and 3 a child of 0.
    # IDs deliberately differ from preorder; siblings share compact endpoints.
    pre, post = torch.tensor([1, 3, 0, 2]), torch.tensor([3, 4, 4, 3])
    pix = torch.tensor([3, 0, 1, 2, 3], dtype=torch.uint32)
    jacobian = torch.tensor([[1, 1, 0, 0, 1], [0, 0, 1, 0, 0],
                             [1, 1, 1, 1, 1], [1, 0, 0, 0, 1]], dtype=torch.float64)
    signal = torch.tensor([0.4, -1., 2., 0.6], dtype=torch.float64, requires_grad=True)
    probe = torch.tensor([0.2, -0.5, 1.2, -0.3, 0.7], dtype=torch.float64)

    def forbidden(*args, **kwargs):
        raise AssertionError("Interval products must not sort, convert event ranks, or read device scalars.")
    monkeypatch.setattr(torch, "argsort", forbidden)
    monkeypatch.setattr(torch, "bincount", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    actual = reconstruct_from_info(signal, pre, post, pix)
    adjoint = propagate_pixels_to_nodes(probe, pre, post, pix)
    torch.testing.assert_close(actual, jacobian.T @ signal)
    torch.testing.assert_close(adjoint, jacobian @ probe)
    torch.testing.assert_close(actual @ probe, signal @ adjoint)

    def reconstruction(values):
        return TreeReconstructionFunction.apply(values, pre, post, pix, 1, 5)
    monkeypatch.undo()
    assert torch.autograd.gradcheck(reconstruction, (signal,))


def test_preparation_validation_and_training_do_not_sort(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CFP preparation and training must not sort DFS indices.")
    monkeypatch.setattr(torch, "argsort", forbidden)
    layer = ConnectedFilterPreprocessingLayer(1, [{
        "tree_type": "max-tree", "attributes": [mtlearn.morphology.AttributeType.AREA],
    }], scale_mode="none")
    image = torch.tensor([[2., 2., 0.], [2., 5., 0.], [3., 3., 1.]])
    prepared = CFPPreprocessor.from_layer(layer).prepare_image(image)
    prepared.validate(full=True)
    layer(image[None, None]).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


@pytest.mark.parametrize("tree_type", ["max-tree", "min-tree", "tree-of-shapes"])
@pytest.mark.parametrize("image", [
    [[137]],
    [[1, 2, 3, 4]],
    [[2, 2, 0], [2, 5, 0], [3, 3, 1]],
])
def test_native_compact_intervals_match_subtrees_and_pixel_support(tree_type, image):
    import numpy as np
    tree = mtlearn.morphology.build_tree(np.array(image, dtype=np.uint8), tree_type)
    residues, pre, post, parent, pix = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)
    n = residues.numel()
    assert torch.equal(torch.bincount(pre, minlength=n), torch.ones(n, dtype=torch.int64))
    assert torch.all(pre < post) and int(post.max()) == n
    # Establish descendant membership through parent links, independently of DFS.
    descendants = torch.zeros((n, n), dtype=torch.bool)
    for node in range(n):
        ancestor = node
        for _ in range(n):
            descendants[ancestor, node] = True
            par = int(parent[ancestor])
            if par < 0 or par == ancestor:
                break
            ancestor = par
        else:
            raise AssertionError("Parent links contain a cycle.")
    intervals = (pre[None, :] >= pre[:, None]) & (pre[None, :] < post[:, None])
    assert torch.equal(intervals, descendants)
    assert torch.equal(post - pre, descendants.sum(1))
    jacobian = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_jacobian(tree).to_dense().bool()
    assert torch.equal(descendants[:, pix.long()], jacobian)


def test_prepared_format_rejects_old_events_and_noncompact_bounds():
    from dataclasses import replace
    layer = ConnectedFilterPreprocessingLayer(1, [{
        "tree_type": "max-tree", "attributes": [mtlearn.morphology.AttributeType.AREA],
    }], scale_mode="none")
    prepared = CFPPreprocessor.from_layer(layer).prepare_image(torch.tensor([[1., 2., 3.]]))
    assert prepared.format_version == 4
    assert set(prepared.info) == {
        "residues", "tpre", "tpost", "parent", "node_of_pixel",
        "num_rows", "num_cols", "tree_type",
    }
    for version in (1, 2, 3):
        with pytest.raises(ValueError, match="Unsupported prepared morphology"):
            replace(prepared, format_version=version)
    for field in ("order_forward", "order_backward", "num_times"):
        with pytest.raises(ValueError, match="unexpected tree fields"):
            replace(prepared, info=dict(prepared.info, **{field: 1}))
    pre = prepared.info["tpre"].clone()
    pre[pre == pre.max()] = 0
    bad = replace(prepared, info=dict(prepared.info, tpre=pre))
    with pytest.raises(ValueError, match="distinct nodes"):
        bad.validate(full=True)
    post = prepared.info["tpost"].clone()
    post[0] = prepared.info["tpre"][0]
    bad = replace(prepared, info=dict(prepared.info, tpost=post))
    with pytest.raises(ValueError, match="bounds"):
        bad.validate(full=True)


def test_compact_native_intervals_keep_inactive_node_slots_empty():
    import numpy as np
    tree = mtlearn.morphology.build_tree(np.array([[1, 2, 3, 4]], dtype=np.uint8), "max-tree")
    removed = next(node for node in tree.alive_node_ids if node != tree.root)
    tree.pruneNode(removed)
    residues, pre, post, parent, pix = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)
    assert residues.numel() == tree.numInternalNodeSlots > tree.numNodes
    assert residues[removed] == pre[removed] == post[removed] == 0
    assert int(post.max()) == tree.numNodes
    J = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_jacobian(tree).to_dense().double()
    signal = torch.linspace(-1, 1, residues.numel(), dtype=torch.float64)
    probe = torch.tensor([0.2, -0.7, 0.4, 0.9], dtype=torch.float64)
    torch.testing.assert_close(reconstruct_from_info(signal, pre, post, pix), J.T @ signal)
    actual = propagate_pixels_to_nodes(probe, pre, post, pix)
    torch.testing.assert_close(actual, J @ probe)
    assert actual[removed] == 0
