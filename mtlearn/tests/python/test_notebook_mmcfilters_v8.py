import ast
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "notebooks/experiments"
FILTERING = EXPERIMENTS / "CFP_soft_threshold_scoring_screws_filtering.ipynb"
SEGMENTATION = EXPERIMENTS / "CFP_soft_threshold_scoring_screws_segmentation.ipynb"
LINEAR_MLP = EXPERIMENTS / "CFP_linear_vs_mlp_scoring_screws_segmentation.ipynb"
if not all(path.is_file() for path in (FILTERING, SEGMENTATION, LINEAR_MLP)):
    pytest.skip("Notebook sources are unavailable", allow_module_level=True)


def notebook_cells(path):
    if path.is_symlink():
        pytest.skip("Notebook source has a local symlink override")
    return ["".join(cell["source"]) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
            if cell["cell_type"] == "code"]


def function_from_notebook(path, name, **constants):
    namespace = dict(np=np, morphology=morphology, **constants)
    node = next(node for source in notebook_cells(path) if f"def {name}(" in source
                for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def reconstruct_selected_residues(tree, values, threshold):
    return np.array([sum(tree.node_residue(node) for node in tree.ancestors(tree.smallest_node(pixel))
                         if values[node] > threshold) for pixel in range(tree.num_pixels)],
                    dtype=np.int64).reshape(tree.num_rows, tree.num_columns)


@pytest.mark.parametrize("tree_type", ["max-tree", "min-tree", "tree-of-shapes"])
@pytest.mark.parametrize("attribute", [morphology.AttributeType.AREA, morphology.AttributeType.CIRCULARITY])
@pytest.mark.parametrize("threshold", [0.0, 0.6, 1.0])
def test_filtering_notebook_preserves_strict_threshold_and_signed_residues(tree_type, attribute, threshold):
    image = np.array([[20, 20, 20], [20, 200, 20], [20, 20, 20]], dtype=np.uint8)
    tree = morphology.build_tree(image, tree_type)
    values = morphology.compute(tree, attribute)
    factory = function_from_notebook(FILTERING, "make_attribute_filter_target")
    actual = factory(tree_type, attribute, threshold)(image)
    np.testing.assert_array_equal(actual, reconstruct_selected_residues(tree, values, threshold))
    assert actual.dtype == np.int64
    if attribute == morphology.AttributeType.CIRCULARITY and threshold == 1:
        np.testing.assert_array_equal(actual, np.zeros_like(image))


@pytest.mark.parametrize("threshold", [0.0, 0.6, 1.0])
def test_segmentation_notebook_preserves_target_threshold_and_binarization(threshold):
    image = np.array([[20, 20, 20], [20, 200, 20], [20, 20, 20]], dtype=np.uint8)
    attribute = morphology.AttributeType.CIRCULARITY
    tree = morphology.create_max_tree(image)
    values = morphology.compute(tree, attribute)
    target = function_from_notebook(SEGMENTATION, "segmentation_target_fn",
        TARGET_TREE_TYPE="max-tree", TARGET_ATTRIBUTE=attribute, TARGET_THRESHOLD_ORIGINAL=threshold)
    expected = np.where(reconstruct_selected_residues(tree, values, threshold) > 128, 255, 0).astype(np.uint8)
    np.testing.assert_array_equal(target(image), expected)
    assert target(image).dtype == np.uint8


def test_linear_mlp_notebook_preserves_v130_feature_order_and_dimensions():
    fixture = Path(__file__).parent / "fixtures/mmcfilters_v8/v1.3.0-float32-cache.pt.gz"
    with gzip.open(fixture, "rb") as source:
        reference = torch.load(source, weights_only=True, map_location="cpu")
    expected = [name for name, ordinal in sorted(reference["enum_values"].items(), key=lambda item: item[1])]
    assignment = next(node for source in notebook_cells(LINEAR_MLP) if "FILTER_ATTRIBUTES =" in source
                      for node in ast.parse(source).body
                      if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                      and target.id == "FILTER_ATTRIBUTES" for target in node.targets))
    namespace = dict(Type=morphology.AttributeType, morphology=morphology)
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(LINEAR_MLP), "exec"), namespace)
    attributes = namespace["FILTER_ATTRIBUTES"]
    assert len(attributes) == 133
    assert [attribute.name for attribute in attributes] == expected
    for scoring in ({"kind": "linear_sigmoid"}, {"kind": "mlp", "hidden_units": [8], "activation": "tanh"}):
        layer = ConnectedFilterPreprocessingLayer(1, [{"tree_type": "max-tree", "attributes": attributes,
                                                     "scoring": scoring}], scale_mode="none")
        assert layer.out_channels == 1
        assert layer.get_config()["filter_specs"][0]["attributes"] == expected
        weights = list(layer.parameters())[0]
        assert weights.shape[-1] == 133
