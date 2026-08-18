import pytest

import mtlearn

if not getattr(mtlearn, "WITH_TORCH", False):
    pytest.skip("build has no LibTorch support", allow_module_level=True)

try:
    import numpy as np
    import torch
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Python dependency unavailable: {exc}", allow_module_level=True)

from mtlearn import morphology
from mtlearn.layers import (
    ConnectedFilterPreprocessingImplicitJacobianFunction,
    ConnectedFilterPreprocessingLayer,
)

pytestmark = pytest.mark.integration


def _small_image_np():
    return np.array(
        [
            [2, 2, 0],
            [2, 5, 0],
            [3, 3, 1],
        ],
        dtype=np.uint8,
    )


def _small_batch_tensor():
    return torch.tensor(
        [
            [
                [[2, 2, 0], [2, 5, 0], [3, 3, 1]],
                [[1, 0, 1], [4, 4, 2], [0, 2, 2]],
            ],
            [
                [[0, 1, 1], [5, 5, 2], [3, 0, 1]],
                [[2, 3, 4], [1, 1, 0], [0, 2, 5]],
            ],
        ],
        dtype=torch.float32,
    )


def _single_area_layer(*, in_channels=1, tree_type="max-tree", tos_interpolation=None):
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=in_channels,
        filter_specs=[
            {
                "tree_type": tree_type,
                "attributes": (morphology.AttributeType.AREA,),
                "tos_interpolation": tos_interpolation,
            }
        ],
        device="cpu",
        scale_mode="none",
        score_sharpness=1.0,
        clamp=None,
    )
    with torch.no_grad():
        for weight in layer._weights.values():
            weight.fill_(0.2)
        for bias in layer._biases.values():
            bias.fill_(-0.1)
    return layer


def _two_group_layer(*, tree_type="max-tree", tos_interpolation=None):
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": tree_type,
                "attributes": (morphology.AttributeType.AREA,),
                "tos_interpolation": tos_interpolation,
            },
            {
                "tree_type": tree_type,
                "attributes": (morphology.AttributeType.GRAY_HEIGHT,),
                "tos_interpolation": tos_interpolation,
            },
        ],
        device="cpu",
        scale_mode="none",
        score_sharpness=1.0,
        clamp=None,
    )
    with torch.no_grad():
        layer._weights["spec_000"].fill_(0.2)
        layer._biases["spec_000"].fill_(-0.1)
        layer._weights["spec_001"].fill_(-0.15)
        layer._biases["spec_001"].fill_(0.05)
    return layer


def test_implicit_metadata_reconstructs_like_explicit_jacobian():
    tree = morphology.create_max_tree(_small_image_np())
    jacobian = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_jacobian(tree).to_dense()
    residues, tpre, tpost, parent, node_of_pixel = (
        mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)
    )
    filtered_residues = residues * torch.linspace(0.1, 0.9, residues.numel())

    explicit = (jacobian.T @ filtered_residues).reshape(tree.numRows, tree.numCols)
    implicit = ConnectedFilterPreprocessingImplicitJacobianFunction.forward_from_info(
        filtered_residues,
        tpre,
        tpost,
        node_of_pixel,
        parent,
    ).reshape(tree.numRows, tree.numCols)

    assert torch.allclose(implicit, explicit)


def test_predict_preserves_training_mode_parameters_and_shape_for_batch_channels():
    layer = _single_area_layer(in_channels=2)
    layer.train()
    x = _small_batch_tensor()
    before = {name: parameter.detach().clone() for name, parameter in layer.named_parameters()}

    y = layer.predict(x, score_sharpness=1.0)

    assert layer.training is True
    assert y.requires_grad is False
    assert y.dtype == torch.float32
    assert y.shape == (2, 2, 3, 3)
    for name, parameter in layer.named_parameters():
        assert parameter.grad is None
        assert torch.equal(parameter.detach(), before[name])


def test_predict_matches_forward_for_single_group_when_beta_matches_layer_beta():
    layer = _single_area_layer(in_channels=1)
    x = torch.as_tensor(_small_image_np(), dtype=torch.float32).reshape(1, 1, 3, 3)

    forward = layer(x)
    predicted = layer.predict(x, score_sharpness=layer.score_sharpness)

    assert torch.allclose(predicted, forward)


def test_predict_matches_forward_for_multiple_groups_with_and_without_cache():
    x = torch.as_tensor(_small_image_np(), dtype=torch.float32).reshape(1, 1, 3, 3)

    uncached = _two_group_layer()
    forward_uncached = uncached(x).detach()
    predicted_uncached = uncached.predict(x, score_sharpness=uncached.score_sharpness)

    assert predicted_uncached.shape == (1, 2, 3, 3)
    assert torch.allclose(predicted_uncached, forward_uncached)

    cached = _two_group_layer()
    indexed_x = (x, torch.tensor([0]))
    forward_cached = cached(indexed_x).detach()
    predicted_cached = cached.predict(indexed_x, score_sharpness=cached.score_sharpness)

    assert predicted_cached.shape == (1, 2, 3, 3)
    assert torch.allclose(predicted_cached, forward_cached)


def test_predict_matches_forward_for_tree_of_shapes_multiple_groups():
    x = torch.as_tensor(_small_image_np(), dtype=torch.float32).reshape(1, 1, 3, 3)
    layer = _two_group_layer(
        tree_type="tree-of-shapes",
        tos_interpolation=morphology.ToSInterpolation.Min8cMax4c,
    )
    indexed_x = (x, torch.tensor([0]))

    forward = layer(indexed_x).detach()
    predicted = layer.predict(indexed_x, score_sharpness=layer.score_sharpness)

    assert predicted.shape == (1, 2, 3, 3)
    assert torch.allclose(predicted, forward)
