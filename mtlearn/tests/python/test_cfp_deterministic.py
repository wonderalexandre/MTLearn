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


def _single_area_layer(*, in_channels=1, tree_type="max-tree", tos_interpolation=None, device="cpu"):
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=in_channels,
        filter_specs=[
            {
                "tree_type": tree_type,
                "attributes": (morphology.AttributeType.AREA,),
                "tos_interpolation": tos_interpolation,
            }
        ],
        device=device,
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
                "attributes": (morphology.AttributeType.GRAY_LEVEL_HEIGHT,),
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

    explicit = (jacobian.T @ filtered_residues).reshape(tree.num_rows, tree.num_columns)
    implicit = ConnectedFilterPreprocessingImplicitJacobianFunction.forward_from_info(
        filtered_residues,
        tpre,
        tpost,
        node_of_pixel,
    ).reshape(tree.num_rows, tree.num_columns)

    assert torch.allclose(implicit, explicit)


@pytest.mark.parametrize(
    "tree_type",
    [
        pytest.param(morphology.TreeType.MAX_TREE, id="max-tree"),
        pytest.param(morphology.TreeType.MIN_TREE, id="min-tree"),
    ],
)
def test_implicit_unfiltered_residues_reconstruct_input(tree_type):
    image = _small_image_np()
    tree = morphology.build_tree(image, tree_type)
    residues, tpre, tpost, parent, node_of_pixel = (
        mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)
    )

    reconstructed = ConnectedFilterPreprocessingImplicitJacobianFunction.forward_from_info(
        residues,
        tpre,
        tpost,
        node_of_pixel,
    ).reshape(tree.num_rows, tree.num_columns)

    assert torch.equal(reconstructed, torch.as_tensor(image, dtype=residues.dtype))


@pytest.mark.parametrize(
    "tree_type",
    [
        pytest.param(morphology.TreeType.MAX_TREE, id="max-tree"),
        pytest.param(morphology.TreeType.MIN_TREE, id="min-tree"),
    ],
)
def test_implicit_forward_and_parameter_gradients_match_explicit_jacobian(tree_type):
    image = _small_image_np()
    tree = morphology.build_tree(image, tree_type)
    jacobian = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_jacobian(tree).to_dense().double()
    residues, tpre, tpost, parent, node_of_pixel = (
        mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)
    )
    residues = residues.double()
    attributes = morphology.compute_attributes(tree, [morphology.AttributeType.AREA])[1]
    attributes = torch.as_tensor(attributes, dtype=torch.float64).reshape(-1, 1)
    score_sharpness = 1.7

    explicit_weight = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
    explicit_bias = torch.tensor([-0.1], dtype=torch.float64, requires_grad=True)
    explicit_scores = torch.sigmoid(score_sharpness * (attributes @ explicit_weight + explicit_bias))
    explicit_output = (jacobian.T @ (residues * explicit_scores)).reshape(tree.num_rows, tree.num_columns)

    implicit_weight = explicit_weight.detach().clone().requires_grad_(True)
    implicit_bias = explicit_bias.detach().clone().requires_grad_(True)
    implicit_output = ConnectedFilterPreprocessingImplicitJacobianFunction.apply(
        implicit_weight,
        implicit_bias,
        residues,
        tpre,
        tpost,
        node_of_pixel,
        attributes,
        tree.num_rows,
        tree.num_columns,
        score_sharpness,
    )

    probe = torch.linspace(-0.5, 0.75, image.size, dtype=torch.float64).reshape(image.shape)
    explicit_grads = torch.autograd.grad((explicit_output * probe).sum(), (explicit_weight, explicit_bias))
    implicit_grads = torch.autograd.grad((implicit_output * probe).sum(), (implicit_weight, implicit_bias))

    assert torch.allclose(implicit_output, explicit_output, rtol=1e-8, atol=1e-10)
    assert torch.allclose(implicit_grads[0], explicit_grads[0], rtol=1e-8, atol=1e-10)
    assert torch.allclose(implicit_grads[1], explicit_grads[1], rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("tree_type", ["max-tree", "min-tree"])
def test_mps_forward_and_parameter_gradients_match_cpu(tree_type):
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    x_cpu = torch.as_tensor(_small_image_np(), dtype=torch.float32).reshape(1, 1, 3, 3)
    probe_cpu = torch.linspace(-0.5, 0.75, x_cpu.numel(), dtype=torch.float32).reshape_as(x_cpu)
    cpu_layer = _single_area_layer(tree_type=tree_type, device="cpu")
    mps_layer = _single_area_layer(tree_type=tree_type, device="mps")

    cpu_output = cpu_layer(x_cpu)
    (cpu_output * probe_cpu).sum().backward()
    mps_output = mps_layer(x_cpu.to("mps"))
    (mps_output * probe_cpu.to("mps")).sum().backward()

    assert mps_output.device.type == "mps"
    assert torch.allclose(mps_output.cpu(), cpu_output, rtol=1e-4, atol=1e-5)
    cpu_parameters = dict(cpu_layer.named_parameters())
    mps_parameters = dict(mps_layer.named_parameters())
    assert cpu_parameters.keys() == mps_parameters.keys()
    for name, cpu_parameter in cpu_parameters.items():
        assert cpu_parameter.grad is not None
        assert mps_parameters[name].grad is not None
        assert torch.isfinite(mps_parameters[name].grad).all()
        assert torch.allclose(
            mps_parameters[name].grad.cpu(),
            cpu_parameter.grad,
            rtol=1e-4,
            atol=1e-5,
        )


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
        tos_interpolation=morphology.ToSInterpolation.MIN8_MAX4,
    )
    indexed_x = (x, torch.tensor([0]))

    forward = layer(indexed_x).detach()
    predicted = layer.predict(indexed_x, score_sharpness=layer.score_sharpness)

    assert predicted.shape == (1, 2, 3, 3)
    assert torch.allclose(predicted, forward)
