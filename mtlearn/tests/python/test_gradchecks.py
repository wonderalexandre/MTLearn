import mtlearn
import pytest

if not getattr(mtlearn, "WITH_TORCH", False):
    pytest.skip("build has no LibTorch support", allow_module_level=True)

try:
    import numpy as np
    import torch
    from torch.autograd import gradcheck
    try:
        from torch.func import functional_call
    except Exception:  # pragma: no cover - compatibility with older PyTorch
        from torch.nn.utils.stateless import functional_call
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Python dependency unavailable: {exc}", allow_module_level=True)

from mtlearn import morphology

pytestmark = [pytest.mark.gradcheck, pytest.mark.slow]


def _small_gradcheck_case(dtype, tree_type="max-tree", tos_interpolation=None):
    image = np.array(
        [
            [2, 2, 0],
            [2, 5, 0],
            [3, 3, 1],
        ],
        dtype=np.uint8,
    )
    tree = morphology.build_tree(
        image,
        tree_type,
        tos_interpolation=tos_interpolation,
    )
    attributes = morphology.compute_attributes(
        tree,
        [morphology.AttributeType.AREA, morphology.AttributeType.COMPACTNESS],
    )[1]
    attributes = (attributes.min() - attributes) / (
        attributes - attributes.max() + 1e-8
    )
    return tree, torch.as_tensor(attributes, dtype=dtype)


def _learnable_parameters(dtype):
    weight = torch.tensor([0.35, -0.2], dtype=dtype, requires_grad=True)
    bias = torch.tensor([0.1], dtype=dtype, requires_grad=True)
    return weight, bias


def _small_layer_input(dtype=torch.float64):
    return torch.tensor(
        [
            [
                [
                    [1, 1, 0, 4, 4],
                    [1, 3, 0, 4, 2],
                    [5, 3, 3, 2, 2],
                    [5, 0, 6, 6, 2],
                    [0, 0, 6, 1, 1],
                ]
            ]
        ],
        dtype=dtype,
    )


def _tos_spec_kwargs(tree_type):
    if tree_type == morphology.TreeType.TREE_OF_SHAPES:
        return {"tos_interpolation": morphology.ToSInterpolation.Min8cMax4c}
    return {}


def test_explicit_jacobian_function_gradcheck():
    tree, attributes = _small_gradcheck_case(torch.float64)
    jacobian = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_jacobian(tree).to(
        dtype=torch.float64
    )
    residues = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_residues(tree).to(
        dtype=torch.float64
    )
    weight, bias = _learnable_parameters(torch.float64)

    def filtered_mean(w, b):
        return mtlearn.layers.ConnectedFilterPreprocessingExplicitJacobianFunction.apply(
            jacobian,
            residues,
            tree.numRows,
            tree.numCols,
            attributes,
            w,
            b,
            1.0,
            False,
        ).mean()

    assert gradcheck(filtered_mean, (weight, bias), eps=1e-6, atol=1e-4)


def test_implicit_jacobian_function_gradcheck():
    tree, attributes = _small_gradcheck_case(torch.float64)
    residues, tpre, tpost, parent, node_of_pixel = (
        mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)
    )
    residues = residues.to(dtype=torch.float64)
    weight, bias = _learnable_parameters(torch.float64)

    def filtered_mean(w, b):
        return mtlearn.layers.ConnectedFilterPreprocessingImplicitJacobianFunction.apply(
            w,
            b,
            residues,
            tpre,
            tpost,
            parent,
            node_of_pixel,
            attributes,
            tree.numRows,
            tree.numCols,
            2.0,
        ).mean()

    assert gradcheck(filtered_mean, (weight, bias), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize(
    ("clamp_min", "clamp_max"),
    [
        pytest.param(-12.0, 12.0, id="symmetric-12"),
        pytest.param(-8.0, 10.0, id="asymmetric"),
        pytest.param(-1.0, 1.0, id="saturating"),
    ],
)
def test_implicit_jacobian_function_gradcheck_with_clamp_bounds(clamp_min, clamp_max):
    tree, attributes = _small_gradcheck_case(torch.float64)
    residues, tpre, tpost, parent, node_of_pixel = (
        mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)
    )
    residues = residues.to(dtype=torch.float64)
    weight = torch.tensor([2.0, -0.5], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)

    def filtered_mean(w, b):
        return mtlearn.layers.ConnectedFilterPreprocessingImplicitJacobianFunction.apply(
            w,
            b,
            residues,
            tpre,
            tpost,
            parent,
            node_of_pixel,
            attributes,
            tree.numRows,
            tree.numCols,
            3.0,
            clamp_min,
            clamp_max,
        ).mean()

    assert gradcheck(filtered_mean, (weight, bias), eps=1e-6, atol=1e-4)


def test_implicit_jacobian_function_clamp_saturates_backward():
    tree, _ = _small_gradcheck_case(torch.float64)
    residues, tpre, tpost, parent, node_of_pixel = (
        mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)
    )
    residues = residues.to(dtype=torch.float64)
    attributes = torch.ones((residues.numel(), 2), dtype=torch.float64)
    weight = torch.tensor([10.0, 10.0], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)

    output = mtlearn.layers.ConnectedFilterPreprocessingImplicitJacobianFunction.apply(
        weight,
        bias,
        residues,
        tpre,
        tpost,
        parent,
        node_of_pixel,
        attributes,
        tree.numRows,
        tree.numCols,
        1.0,
        -1.0,
        1.0,
    )
    output.sum().backward()

    assert torch.equal(weight.grad, torch.zeros_like(weight))
    assert torch.equal(bias.grad, torch.zeros_like(bias))


def test_implicit_jacobian_function_gradcheck_tree_of_shapes():
    tree, attributes = _small_gradcheck_case(
        torch.float64,
        tree_type="tree-of-shapes",
        tos_interpolation="self-dual",
    )
    residues, tpre, tpost, parent, node_of_pixel = (
        mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)
    )
    residues = residues.to(dtype=torch.float64)
    weight, bias = _learnable_parameters(torch.float64)

    def filtered_mean(w, b):
        return mtlearn.layers.ConnectedFilterPreprocessingImplicitJacobianFunction.apply(
            w,
            b,
            residues,
            tpre,
            tpost,
            parent,
            node_of_pixel,
            attributes,
            tree.numRows,
            tree.numCols,
            1.0,
        ).mean()

    assert gradcheck(filtered_mean, (weight, bias), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize(
    "clamp",
    [
        pytest.param(None, id="clamp-none"),
        pytest.param(12, id="clamp-scalar"),
        pytest.param((-8.0, 10.0), id="clamp-pair"),
        pytest.param(1.0, id="clamp-saturating"),
    ],
)
@pytest.mark.parametrize(
    ("tree_type", "valuation"),
    [
        pytest.param(
            morphology.TreeType.MAX_TREE,
            None,
            id="max-filtered-altitude",
        ),
        pytest.param(
            morphology.TreeType.MAX_TREE,
            mtlearn.layers.CFPValuation.ALTITUDE_TOPHAT,
            id="max-tophat-altitude",
        ),
        pytest.param(
            morphology.TreeType.MIN_TREE,
            None,
            id="min-filtered-altitude",
        ),
        pytest.param(
            morphology.TreeType.MIN_TREE,
            mtlearn.layers.CFPValuation.ALTITUDE_TOPHAT,
            id="min-tophat-altitude",
        ),
        pytest.param(
            morphology.TreeType.TREE_OF_SHAPES,
            None,
            id="tos-filtered-altitude",
        ),
        pytest.param(
            morphology.TreeType.TREE_OF_SHAPES,
            mtlearn.layers.CFPValuation.ALTITUDE_TOPHAT,
            id="tos-tophat-altitude",
        ),
        pytest.param(
            morphology.TreeType.MAX_TREE,
            mtlearn.layers.CFPValuation.node_attribute(morphology.AttributeType.AREA),
            id="max-filtered-area-valuation",
        ),
        pytest.param(
            morphology.TreeType.MAX_TREE,
            mtlearn.layers.CFPValuation.node_attribute(morphology.AttributeType.VARIANCE_LEVEL),
            id="max-filtered-variance-valuation",
        ),
    ],
)
def test_filter_specs_layer_gradcheck(tree_type, valuation, clamp):
    spec = {
        "tree_type": tree_type,
        "attributes": (
            morphology.AttributeType.AREA,
            morphology.AttributeType.COMPACTNESS,
        ),
        **_tos_spec_kwargs(tree_type),
    }
    if valuation is not None:
        spec["valuation"] = valuation

    layer = mtlearn.layers.ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[spec],
        device="cpu",
        scale_mode="none",
        beta_f=2.0,
        clamp=clamp,
    ).double()
    image = _small_layer_input(torch.float64)
    weight, bias = _learnable_parameters(torch.float64)

    def filtered_mean(w, b):
        return functional_call(
            layer,
            {
                "_weights.spec_000": w,
                "_biases.spec_000": b,
            },
            (image,),
        ).mean()

    assert gradcheck(filtered_mean, (weight, bias), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize(
    ("tree_type", "top_hat"),
    [
        pytest.param(morphology.TreeType.MAX_TREE, False, id="max-filtered"),
        pytest.param(morphology.TreeType.MAX_TREE, True, id="max-tophat"),
        pytest.param(morphology.TreeType.MIN_TREE, False, id="min-filtered"),
        pytest.param(morphology.TreeType.MIN_TREE, True, id="min-tophat"),
        pytest.param(morphology.TreeType.TREE_OF_SHAPES, False, id="tos-filtered"),
        pytest.param(morphology.TreeType.TREE_OF_SHAPES, True, id="tos-tophat"),
    ],
)
def test_legacy_layer_gradcheck(tree_type, top_hat):
    layer = mtlearn.layers.ConnectedFilterPreprocessingLayerLegacy(
        in_channels=1,
        attributes_spec=[
            (
                morphology.AttributeType.AREA,
                morphology.AttributeType.COMPACTNESS,
            )
        ],
        tree_type=tree_type,
        device="cpu",
        scale_mode="none",
        beta_f=2.0,
        top_hat=top_hat,
        clamp_logits=False,
        **_tos_spec_kwargs(tree_type),
    ).double()
    image = _small_layer_input(torch.float64)
    weight, bias = _learnable_parameters(torch.float64)

    def filtered_mean(w, b):
        return functional_call(
            layer,
            {
                "_weights.AREA+COMPACTNESS": w,
                "_biases.AREA+COMPACTNESS": b,
            },
            (image,),
        ).mean()

    assert gradcheck(filtered_mean, (weight, bias), eps=1e-6, atol=1e-4)


def test_cpu_tree_traversal_function_matches_numeric_gradient():
    tree, attributes = _small_gradcheck_case(torch.float32)
    weight, bias = _learnable_parameters(torch.float32)
    eps = 3e-4
    atol = 1e-3

    def filtered_mean(w, b):
        return mtlearn.layers.ConnectedFilterPreprocessingCPUTreeTraversalFunction.apply(
            tree,
            attributes,
            w,
            b,
            1.0,
            False,
        ).mean()

    output = filtered_mean(weight, bias)
    output.backward()

    numeric_weight = []
    for index in range(weight.numel()):
        weight_increased = weight.detach().clone()
        weight_decreased = weight.detach().clone()
        weight_increased[index] += eps
        weight_decreased[index] -= eps
        numeric_weight.append(
            (
                filtered_mean(weight_increased, bias.detach())
                - filtered_mean(weight_decreased, bias.detach())
            )
            / (2 * eps)
        )

    bias_increased = bias.detach().clone()
    bias_decreased = bias.detach().clone()
    bias_increased[0] += eps
    bias_decreased[0] -= eps
    numeric_bias = (
        filtered_mean(weight.detach(), bias_increased)
        - filtered_mean(weight.detach(), bias_decreased)
    ) / (2 * eps)

    assert torch.allclose(weight.grad[0], numeric_weight[0], atol=atol)
    assert torch.allclose(weight.grad[1], numeric_weight[1], atol=atol)
    assert torch.allclose(bias.grad[0], numeric_bias, atol=atol)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
