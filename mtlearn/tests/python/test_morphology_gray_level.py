import numpy as np
import pytest
import torch

from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer


@pytest.mark.parametrize("tree_type", ["max-tree", "min-tree", "tree-of-shapes"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("output_space", [morphology.NodeIdSpace.MORPHOLOGICAL_TREE,
                                          morphology.NodeIdSpace.HIGRA])
@pytest.mark.parametrize("constant", [False, True])
def test_gray_level_matches_node_valuation(tree_type, dtype, output_space, constant):
    image = np.full((5, 7), 17, dtype=np.uint8)
    if not constant:
        image[1:4, 1:6] = 83
        image[2, 3] = 219
    tree = morphology.build_tree(image, tree_type)
    if output_space == morphology.NodeIdSpace.HIGRA:
        _, expected = tree.export_higra_hierarchy()
    else:
        expected = [tree.node_altitude(node) for node in range(tree.num_internal_node_slots)]
    attribute = morphology.AttributeType.GRAY_LEVEL
    actual = morphology.compute_single_attribute(tree, attribute, output_space=output_space, dtype=dtype)
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(actual, expected)
    for request in [[attribute], [attribute, attribute],
                    [attribute, morphology.AttributeType.AREA],
                    [attribute, morphology.AttributeGroup.GRAY_LEVEL]]:
        layout, values = morphology.compute_attributes(tree, request, output_space=output_space, dtype=dtype)
        assert values.dtype == np.dtype(dtype)
        assert list(layout.values()) == list(range(len(layout)))
        assert list(layout).count("GRAY_LEVEL") == 1
        np.testing.assert_array_equal(values[:, layout["GRAY_LEVEL"]], expected)
        if "AREA" in layout:
            area = morphology.compute_single_attribute(tree, morphology.AttributeType.AREA,
                output_space=output_space, dtype=dtype)
            np.testing.assert_array_equal(values[:, layout["AREA"]], area)


def test_gray_level_catalog_and_groups():
    attribute = morphology.AttributeType.GRAY_LEVEL
    assert morphology.Attribute.Type.GRAY_LEVEL is attribute
    assert morphology.Attribute.GRAY_LEVEL == morphology.AttributeGroup.GRAY_LEVEL
    assert not hasattr(morphology.AttributeType, "ALTITUDE")
    assert morphology.describe_all_attributes()["GRAY_LEVEL"] == morphology.describe_attribute(attribute)
    for group in [morphology.AttributeGroup.ALL, morphology.AttributeGroup.GRAY_LEVEL]:
        assert morphology.expand_attribute_group(group).count(attribute) == 1
    tree = morphology.build_tree(np.array([[10, 30], [10, 50]], dtype=np.uint8), "max-tree")
    values = morphology.compute_single_attribute(tree, attribute)
    means = morphology.compute_single_attribute(tree, morphology.AttributeType.MEAN_GRAY_LEVEL)
    assert values[tree.root] == 10
    assert means[tree.root] == 25
    assert values.dtype == np.float32


def test_gray_level_cfp_roundtrip_and_backward():
    layer = ConnectedFilterPreprocessingLayer(1, [{"tree_type": "max-tree", "attributes": [
        morphology.AttributeType.GRAY_LEVEL, morphology.AttributeType.BOUNDING_BOX_WIDTH]}], scale_mode="none")
    image = torch.tensor([[[[10, 30], [10, 50]]]], dtype=torch.float32) / 255
    output = layer(image)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in layer.parameters())
    restored = ConnectedFilterPreprocessingLayer.from_config(layer.get_config())
    restored.load_state_dict(layer.state_dict())
    torch.testing.assert_close(restored(image), output)
