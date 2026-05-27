import mtlearn
import os
import pytest
import subprocess
import sys
from pathlib import Path

if not getattr(mtlearn, "WITH_TORCH", False):
    pytest.skip("build has no LibTorch support", allow_module_level=True)

try:
    import numpy as np
    import torch
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Python dependency unavailable: {exc}", allow_module_level=True)

from mtlearn import morphology
from mtlearn.layers._helpers import build_tree

pytestmark = pytest.mark.integration


def test_morphology_facade_uses_native_backend():
    assert morphology.Tree is mtlearn._bindings.WeightedMorphologicalTree
    assert morphology.WeightedTree is mtlearn._bindings.WeightedMorphologicalTree
    assert morphology.WeightedMorphologicalTree is mtlearn._bindings.WeightedMorphologicalTree


def test_top_level_public_modules_are_exported():
    assert "data" in mtlearn.__all__
    assert "datasets" in mtlearn.__all__
    assert "layers" in mtlearn.__all__
    assert "morphology" in mtlearn.__all__
    assert "TreeStats" not in mtlearn.__all__
    assert "make_tree_stats" not in mtlearn.__all__
    assert "make_tree_tensor" not in mtlearn.__all__
    assert "ConnectedFilterByJacobian" not in mtlearn.__all__
    assert "InfoTree" not in mtlearn.__all__
    assert "ConnectedFilterByMorphologicalTree" not in mtlearn.__all__
    assert not hasattr(mtlearn, "TreeStats")
    assert not hasattr(mtlearn, "make_tree_stats")
    assert not hasattr(mtlearn, "make_tree_tensor")
    assert not hasattr(mtlearn, "ConnectedFilterByJacobian")
    assert not hasattr(mtlearn, "InfoTree")
    assert not hasattr(mtlearn, "ConnectedFilterByMorphologicalTree")
    assert not hasattr(mtlearn._bindings, "TreeStats")
    assert not hasattr(mtlearn._bindings, "make_tree_stats")
    assert not hasattr(mtlearn._bindings, "make_tree_tensor")
    assert not hasattr(mtlearn._bindings, "ConnectedFilterByJacobian")
    assert not hasattr(mtlearn._bindings, "InfoTree")
    assert not hasattr(mtlearn._bindings, "ConnectedFilterByMorphologicalTree")
    assert hasattr(mtlearn, "ConnectedFilterPreprocessingTreeTensors")
    assert hasattr(mtlearn, "ConnectedFilterPreprocessingTreeTraversal")
    assert hasattr(mtlearn._bindings, "ConnectedFilterPreprocessingTreeTensors")
    assert hasattr(mtlearn._bindings, "ConnectedFilterPreprocessingTreeTraversal")


def test_top_level_import_does_not_require_sklearn():
    code = """
import importlib.abc
import sys

class BlockSklearn(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sklearn" or fullname.startswith("sklearn."):
            raise ImportError("blocked sklearn")
        return None

sys.meta_path.insert(0, BlockSklearn())
import mtlearn
from mtlearn import datasets
assert mtlearn.__version__
assert datasets.PairedImageDataset
"""
    package_root = str(Path(mtlearn.__file__).resolve().parent.parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        entry for entry in (package_root, env.get("PYTHONPATH", "")) if entry
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_dataset_split_indices_match_expected_sizes():
    from mtlearn.datasets import _split_indices

    train_idx, test_idx = _split_indices(
        10,
        test_size=0.25,
        shuffle=True,
        random_state=42,
    )

    assert len(train_idx) == 7
    assert len(test_idx) == 3
    assert sorted(np.concatenate([train_idx, test_idx]).tolist()) == list(range(10))

    train_idx, test_idx = _split_indices(10, test_size=3, shuffle=False)

    assert train_idx.tolist() == list(range(7))
    assert test_idx.tolist() == [7, 8, 9]


def test_attribute_filter_dataset_accepts_tree_of_shapes_options(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from mtlearn.datasets import AttributeFilterDataset

    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), img)

    dataset = AttributeFilterDataset(
        root=str(tmp_path),
        tree_type="tos",
        attributes=[morphology.AttributeType.AREA],
        thresholds={"AREA": 0.0},
        top_hat=True,
        tos_interpolation="min4c-max8c",
    )

    img_in, img_out, name = dataset[0]

    assert dataset.tree_type == "tree-of-shapes"
    assert dataset.tos_interpolation == morphology.ToSInterpolation.Min4cMax8c
    assert name == "sample.png"
    assert img_in.shape == (1, 2, 2)
    assert img_out.shape == (1, 2, 2)
    assert img_in.dtype == torch.float32
    assert img_out.dtype == torch.float32


def test_build_tree_returns_weighted_tree_for_supported_types():
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    for tree_type in ("max-tree", "min-tree", "tos", "tree-of-shapes"):
        tree = build_tree(img, tree_type)

        assert morphology.is_tree(tree)
        assert tree.getRoot() >= 0
        assert tree.getProperPartOwner(0) >= 0
        assert not hasattr(tree, "getSmallestComponent")


def test_morphology_facade_computes_attributes_and_filters():
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    tree = morphology.create_max_tree(img)

    assert morphology.AttributeType is morphology.Attribute.Type
    assert morphology.AttributeGroup is morphology.Attribute.Group
    assert hasattr(morphology.AttributeGroup, "SHAPE")
    assert hasattr(morphology.AttributeGroup, "BOUNDARY")
    assert not hasattr(morphology.AttributeGroup, "GEOMETRIC")
    assert morphology.TreeType.MAX_TREE.value == "max-tree"
    assert morphology.normalize_tree_type(morphology.TreeType.TREE_OF_SHAPES) == "tree-of-shapes"
    assert morphology.AttributeType.ALTITUDE == morphology.AttributeType.LEVEL
    assert hasattr(morphology.AttributeType, "CONTOUR_PERIMETER")
    boundary_attributes = morphology.expand_attribute_group(morphology.AttributeGroup.BOUNDARY)
    assert morphology.AttributeType.BITQUADS_AREA in boundary_attributes
    assert morphology.AttributeType.CONTOUR_PERIMETER in boundary_attributes
    assert morphology.AttributeType.MAX_DIST not in boundary_attributes

    attr_index, attr_values = morphology.compute_attributes(
        tree,
        [morphology.AttributeType.AREA, morphology.AttributeGroup.TREE_TOPOLOGY],
    )
    single_attr = morphology.Attribute.computeSingleAttribute(
        tree,
        morphology.AttributeType.AREA,
        morphology.NodeIdSpace.MORPHOLOGICAL_TREE,
    )
    attribute_filter = morphology.create_attribute_filter(tree)
    filtered = attribute_filter.filteringSubtractiveRule(
        np.ones(attr_values.shape[0], dtype=bool)
    )

    assert "AREA" in attr_index
    assert len(attr_index) > 1
    assert attr_values.shape[0] == single_attr.shape[0]
    assert filtered.shape == img.shape

    area_description = morphology.Attribute.describe(morphology.AttributeType.AREA)
    all_descriptions = morphology.Attribute.describeAll()
    assert "Area:" in area_description
    assert all_descriptions["AREA"] == area_description
    assert "CIRCULARITY" in all_descriptions
    assert morphology.describe_attribute(morphology.AttributeType.AREA) == area_description
    assert morphology.describe_all_attributes()["AREA"] == area_description


def test_connected_filter_reconstructs_image_when_all_nodes_are_kept():
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    expected = torch.tensor(img, dtype=torch.float32)

    for tree_type in ("max-tree", "min-tree", "tos"):
        tree = build_tree(img, tree_type)
        residues = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_residues(tree)
        sigmoid = torch.ones_like(residues, dtype=torch.float32)

        filtered = mtlearn.ConnectedFilterPreprocessingTreeTraversal.filtering(tree, sigmoid)

        assert filtered.dtype == torch.float32
        assert filtered.shape == expected.shape
        assert torch.allclose(filtered, expected)


def test_connected_filter_preprocessing_tree_tensors_have_consistent_shapes():
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    tree = build_tree(img, "max-tree")

    residues = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_residues(tree)
    info = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)

    assert residues.dtype == torch.float32
    assert len(info) == 5
    assert all(item.shape == residues.shape for item in info[:4])
    assert info[4].shape == (img.size,)
    assert not hasattr(mtlearn.ConnectedFilterPreprocessingTreeTensors, "get_jacobian_dense")
    assert not hasattr(mtlearn.ConnectedFilterPreprocessingTreeTensors, "get_acumulated_gradient")


def test_connected_filter_preprocessing_public_aliases():
    assert mtlearn.layers.CFPLayer is mtlearn.layers.ConnectedFilterPreprocessingLayer
    assert (
        mtlearn.layers.CFPLayerWithExplicitJacobian
        is mtlearn.layers.ConnectedFilterPreprocessingLayerWithExplicitJacobian
    )
    assert (
        mtlearn.layers.CFPExplicitJacobianFunction
        is mtlearn.layers.ConnectedFilterPreprocessingExplicitJacobianFunction
    )
    assert (
        mtlearn.layers.CFPLayerWithCPUTreeTraversal
        is mtlearn.layers.ConnectedFilterPreprocessingLayerWithCPUTreeTraversal
    )
    assert mtlearn.layers.CFPLayer is mtlearn.layers.ConnectedFilterPreprocessingLayer
    assert hasattr(mtlearn.layers, "ConnectedFilterPreprocessingLayerLegacy")
    assert hasattr(mtlearn.layers, "CFPValuation")
    assert hasattr(mtlearn.layers.CFPValuation, "ALTITUDE")
    assert hasattr(mtlearn.layers.CFPValuation, "ALTITUDE_TOPHAT")
    assert hasattr(mtlearn.layers, "collect_cfp_configs")
    assert hasattr(mtlearn.layers, "save_checkpoint")
    assert hasattr(mtlearn.layers, "load_checkpoint")
    assert hasattr(mtlearn.layers, "ConnectedFilterPreprocessingImplicitJacobianFunction")
    assert hasattr(mtlearn.layers, "ConnectedFilterPreprocessingCPUTreeTraversalFunction")
    assert not hasattr(mtlearn.layers, "ConnectedFilterLayerByThresholds")
    assert not hasattr(mtlearn.layers, "ConnectedFilterLayerWithImplicitJacobian")
    assert not hasattr(mtlearn.layers, "ConnectedFilterLayerWithJacobian")
    assert not hasattr(mtlearn.layers, "ConnectedFilterWithJacobianFunction")
    assert not hasattr(mtlearn.layers, "ConnectedFilterLayer")
    assert not hasattr(mtlearn.layers, "ConnectedFilterFunction")
    assert not hasattr(mtlearn.layers, "ConnectedFilterLayerBySingleThreshold")
    assert not hasattr(mtlearn.layers, "ConnectedFilterFunctionBySingleThreshold")
    assert not hasattr(mtlearn.layers, "ConnectedFilterSingleThresholdLayer")
    assert not hasattr(mtlearn.layers, "ConnectedFilterSingleThresholdFunction")
    assert not hasattr(mtlearn.ConnectedFilterPreprocessingTreeTraversal, "gradientsOfThreshold")
    assert not hasattr(mtlearn.ConnectedFilterPreprocessingTreeTraversal, "gradientsOfThresholds")


def test_connected_filter_preprocessing_layer_forward_smoke():
    layer = mtlearn.layers.ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
            }
        ],
        device="cpu",
        scale_mode="none",
    )
    x = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32)

    y = layer(x)

    assert y.dtype == torch.float32
    assert y.shape == x.shape


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
