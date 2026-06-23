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


def test_paired_image_dataset_reads_matching_pairs(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from mtlearn.datasets import PairedImageDataset

    assert PairedImageDataset.__module__ == "mtlearn.datasets"

    image = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    target = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "01_in.png"), image)
    assert cv2.imwrite(str(tmp_path / "01_target.png"), target)
    assert cv2.imwrite(str(tmp_path / "02_in.png"), image)

    dataset = PairedImageDataset(
        str(tmp_path),
        invert_in=True,
        suffix_in="_in",
        suffix_target="_target",
    )

    tensor_in, tensor_target, name = dataset[0]

    assert len(dataset) == 1
    assert name == "01_in.png"
    assert tensor_in.shape == (1, 2, 2)
    assert tensor_target.shape == (1, 2, 2)
    assert tensor_in.dtype == torch.float32
    assert tensor_target.dtype == torch.float32
    expected_in = torch.from_numpy(255 - image).to(torch.float32).unsqueeze(0) / 255.0
    expected_target = torch.from_numpy(target).to(torch.float32).unsqueeze(0) / 255.0
    assert torch.allclose(tensor_in, expected_in)
    assert torch.allclose(tensor_target, expected_target)


def test_paired_image_dataset_uses_snake_case_resize_shape(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from mtlearn.datasets import PairedImageDataset

    image = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    target = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "01_in.png"), image)
    assert cv2.imwrite(str(tmp_path / "01_target.png"), target)

    dataset = PairedImageDataset(
        str(tmp_path),
        num_rows=4,
        num_cols=4,
        suffix_in="_in",
        suffix_target="_target",
    )

    tensor_in, tensor_target, _ = dataset[0]

    assert dataset.num_rows == 4
    assert dataset.num_cols == 4
    assert tensor_in.shape == (1, 4, 4)
    assert tensor_target.shape == (1, 4, 4)


def test_generated_target_image_dataset_applies_callable_without_resize(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from mtlearn.datasets import GeneratedTargetImageDataset

    assert GeneratedTargetImageDataset.__module__ == "mtlearn.datasets"

    image = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    ignored = np.full((2, 2), 255, dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "sample_in.png"), image)
    assert cv2.imwrite(str(tmp_path / "sample_target.png"), ignored)

    dataset = GeneratedTargetImageDataset(
        str(tmp_path),
        target_fn=lambda img: np.where(img > 127, 255, 0).astype(np.uint8),
        suffix_in="_in",
    )

    tensor_in, tensor_target, name = dataset[0]

    assert len(dataset) == 1
    assert name == "sample_in.png"
    assert dataset.num_rows is None
    assert dataset.num_cols is None
    assert tensor_in.shape == (1, 2, 2)
    assert tensor_target.shape == (1, 2, 2)
    assert tensor_in.dtype == torch.float32
    assert tensor_target.dtype == torch.float32
    assert 0.0 <= float(tensor_in.min()) <= float(tensor_in.max()) <= 1.0
    assert set(torch.unique(tensor_target).tolist()).issubset({0.0, 1.0})


def test_generated_target_image_dataset_uses_optional_resize(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from mtlearn.datasets import GeneratedTargetImageDataset

    image = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "sample_in.png"), image)

    dataset = GeneratedTargetImageDataset(
        str(tmp_path),
        target_fn=lambda img: np.where(img > 127, 255, 0).astype(np.uint8),
        num_rows=4,
        num_cols=4,
        suffix_in="_in",
    )

    tensor_in, tensor_target, _ = dataset[0]

    assert dataset.num_rows == 4
    assert dataset.num_cols == 4
    assert tensor_in.shape == (1, 4, 4)
    assert tensor_target.shape == (1, 4, 4)


def test_datasets_reject_partial_resize_shape(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from mtlearn.datasets import GeneratedTargetImageDataset, PairedImageDataset

    image = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "sample_in.png"), image)
    assert cv2.imwrite(str(tmp_path / "sample_target.png"), image)

    with pytest.raises(ValueError, match="num_rows and num_cols"):
        PairedImageDataset(
            str(tmp_path),
            num_rows=4,
            suffix_in="_in",
            suffix_target="_target",
        )

    with pytest.raises(ValueError, match="num_rows and num_cols"):
        GeneratedTargetImageDataset(
            str(tmp_path),
            target_fn=lambda img: img,
            num_cols=4,
            suffix_in="_in",
        )


def test_attribute_filter_dataset_is_not_public():
    import mtlearn.datasets as datasets

    assert "AttributeFilterDataset" not in datasets.__all__
    with pytest.raises(AttributeError):
        getattr(datasets, "AttributeFilterDataset")


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
    single_attr64 = morphology.Attribute.computeSingleAttribute(
        tree,
        morphology.AttributeType.AREA,
        morphology.NodeIdSpace.MORPHOLOGICAL_TREE,
        np.float64,
    )
    attribute_filter = morphology.create_attribute_filter(tree)
    filtered = attribute_filter.filteringSubtractiveRule(
        np.ones(attr_values.shape[0], dtype=bool)
    )

    assert "AREA" in attr_index
    assert len(attr_index) > 1
    assert attr_values.shape[0] == single_attr.shape[0]
    assert single_attr64.dtype == np.float64
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
    from mtlearn.layers.cfp.runtime import (
        ConnectedFilterPreprocessingImplicitJacobianFunction,
        TreeReconstructionFunction,
    )

    assert mtlearn.layers.cfp.ConnectedFilterPreprocessingLayer is mtlearn.layers.ConnectedFilterPreprocessingLayer
    assert not hasattr(mtlearn.layers, "CFPLayer")
    assert not hasattr(mtlearn.layers.cfp, "CFPLayer")
    assert hasattr(mtlearn.layers.cfp, "LinearSigmoidScorer")
    assert not hasattr(mtlearn.layers.cfp, "LayerOwnedLinearParameterInitializer")
    assert not hasattr(mtlearn.layers.cfp, "LegacyLinearParameterInitializer")
    assert hasattr(mtlearn.layers.cfp, "PathScoreMonotonicityRegularizer")
    assert hasattr(mtlearn.layers.cfp, "AttributeOrderScoreMonotonicityRegularizer")
    assert hasattr(mtlearn.layers.cfp, "EdgeScoreMonotonicityRegularizer")
    assert not hasattr(mtlearn.layers.cfp, "AncestorConsistencyRegularizer")
    assert not hasattr(mtlearn.layers.cfp, "AttributeOrderMonotonicityRegularizer")
    assert not hasattr(mtlearn.layers.cfp, "MonotoneScoresRegularizer")
    assert hasattr(mtlearn.layers.cfp, "PreserveRootConstraint")
    assert ConnectedFilterPreprocessingImplicitJacobianFunction is mtlearn.layers.ConnectedFilterPreprocessingImplicitJacobianFunction
    assert TreeReconstructionFunction.__name__ == "TreeReconstructionFunction"
    assert not hasattr(mtlearn.layers.cfp, "ConnectedFilterPreprocessingImplicitJacobianFunction")
    assert not hasattr(mtlearn.layers.cfp, "TreeReconstructionFunction")
    assert not hasattr(mtlearn.layers, "CFPLayerWithExplicitJacobian")
    assert not hasattr(mtlearn.layers, "CFPExplicitJacobianFunction")
    assert not hasattr(mtlearn.layers, "CFPLayerWithCPUTreeTraversal")
    assert not hasattr(mtlearn.layers, "ConnectedFilterPreprocessingLayerLegacy")
    assert not hasattr(mtlearn.layers, "ConnectedFilterPreprocessingLayerWithExplicitJacobian")
    assert not hasattr(mtlearn.layers, "ConnectedFilterPreprocessingExplicitJacobianFunction")
    assert not hasattr(mtlearn.layers, "ConnectedFilterPreprocessingLayerWithCPUTreeTraversal")
    assert not hasattr(mtlearn.layers, "CFPValuation")
    assert hasattr(mtlearn.layers, "collect_cfp_configs")
    assert hasattr(mtlearn.layers, "save_checkpoint")
    assert hasattr(mtlearn.layers, "load_checkpoint")
    assert hasattr(mtlearn.layers, "ConnectedFilterPreprocessingImplicitJacobianFunction")
    assert not hasattr(mtlearn.layers, "ConnectedFilterPreprocessingCPUTreeTraversalFunction")
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
