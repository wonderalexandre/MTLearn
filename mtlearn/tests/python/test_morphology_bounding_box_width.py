import pickle
import re

import numpy as np
import pytest

from mtlearn import morphology


def test_bounding_box_width_uses_canonical_name():
    attribute = morphology.AttributeType.BOUNDING_BOX_WIDTH
    assert not hasattr(morphology.AttributeType, "BOX_WIDTH")
    assert not hasattr(morphology.Attribute, "BOX_WIDTH")
    assert attribute.name == "BOUNDING_BOX_WIDTH"
    assert int(attribute) == 6
    assert pickle.loads(pickle.dumps(attribute)) == attribute
    assert "BOX_WIDTH" not in morphology.AttributeType.__members__
    assert not re.search(r"\bBOX_WIDTH\b", morphology.AttributeType.__doc__)
    descriptions = morphology.describe_all_attributes()
    assert "BOX_WIDTH" not in descriptions
    assert descriptions[attribute.name] == morphology.describe_attribute(attribute)
    attributes = morphology.expand_attribute_group(morphology.AttributeGroup.SHAPE)
    assert attributes.count(attribute) == 1
    assert "BOX_WIDTH" not in {value.name for value in attributes}


@pytest.mark.parametrize("tree_type", ["max-tree", "min-tree"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bounding_box_width_computes_canonical_columns(tree_type, dtype):
    image = np.zeros((5, 7), dtype=np.uint8)
    image[1:4, 1:6] = 200
    if tree_type == "min-tree":
        image = 255 - image
    tree = morphology.build_tree(image, tree_type)
    node = tree.smallest_node(2 * 7 + 3)
    attribute = morphology.AttributeType.BOUNDING_BOX_WIDTH
    expected = morphology.compute(tree, attribute, dtype=dtype)
    assert expected[node] == 5
    for requested in [attribute, "BOUNDING_BOX_WIDTH"]:
        values = morphology.compute(tree, requested, dtype=dtype)
        np.testing.assert_array_equal(values, expected)
        layout, values = morphology.compute(tree, [requested], dtype=dtype)
        assert layout == {"BOUNDING_BOX_WIDTH": 0}
        np.testing.assert_array_equal(values[:, 0], expected)
    layout, values = morphology.compute(
        tree, [morphology.AttributeGroup.SHAPE], dtype=dtype
    )
    assert "BOX_WIDTH" not in layout
    np.testing.assert_array_equal(values[:, layout[attribute.name]], expected)


def test_bounding_box_width_legacy_statistics_are_canonicalized():
    import torch
    from mtlearn.layers.cfp.normalization.stats_serializer import StatsSerializer

    key = "max-tree|None|0|0"
    stats = {"amin": torch.tensor(1.0), "amax": torch.tensor(7.0)}
    restored = StatsSerializer.deserialize({key + "::BOX_WIDTH": stats}, device="cpu")
    assert set(restored) == {key + "::BOUNDING_BOX_WIDTH"}
    for name, value in stats.items():
        torch.testing.assert_close(restored[key + "::BOUNDING_BOX_WIDTH"][name], value)
    with pytest.raises(ValueError, match="duplicate dataset statistics"):
        StatsSerializer.deserialize({key + "::BOX_WIDTH": stats,
                                     key + "::BOUNDING_BOX_WIDTH": stats}, device="cpu")


def test_bounding_box_width_legacy_config_is_canonicalized():
    from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer

    layer = Layer(1, [{"tree_type": "max-tree", "attributes": [morphology.AttributeType.BOUNDING_BOX_WIDTH]}])
    config = layer.get_config()
    config["filter_specs"][0]["attributes"] = ["BOX_WIDTH"]
    restored = Layer.from_config(config)
    assert restored.get_config()["filter_specs"][0]["attributes"] == ["BOUNDING_BOX_WIDTH"]
    assert not hasattr(morphology.AttributeType, "BOX_WIDTH")
