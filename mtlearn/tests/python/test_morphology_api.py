from pathlib import Path

import pytest

import mtlearn

if not getattr(mtlearn, "WITH_TORCH", False):
    pytest.skip("build has no LibTorch support", allow_module_level=True)

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    pytest.skip(f"NumPy unavailable: {exc}", allow_module_level=True)

from mtlearn import morphology

pytestmark = pytest.mark.integration


def _small_image():
    return np.array([[1, 2], [3, 4]], dtype=np.uint8)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_pgm(path: Path) -> np.ndarray:
    def next_token(handle):
        token = bytearray()
        while True:
            ch = handle.read(1)
            if not ch:
                raise RuntimeError(f"unexpected end of file while reading {path}")
            if ch == b"#":
                handle.readline()
                continue
            if ch.isspace():
                continue
            token.extend(ch)
            break

        while True:
            ch = handle.read(1)
            if not ch or ch.isspace():
                break
            if ch == b"#":
                handle.readline()
                break
            token.extend(ch)
        return bytes(token)

    with path.open("rb") as handle:
        magic = next_token(handle)
        cols = int(next_token(handle))
        rows = int(next_token(handle))
        max_value = int(next_token(handle))

        if magic == b"P5":
            dtype = np.uint8 if max_value < 256 else ">u2"
            expected = rows * cols * np.dtype(dtype).itemsize
            payload = handle.read(expected)
            if len(payload) != expected:
                raise RuntimeError(f"{path.name}: incomplete PGM payload")
            image = np.frombuffer(payload, dtype=dtype).reshape(rows, cols)
        elif magic == b"P2":
            values = []
            try:
                while True:
                    values.append(int(next_token(handle)))
            except RuntimeError:
                pass
            if len(values) != rows * cols:
                raise RuntimeError(f"{path.name}: unexpected PGM sample count")
            image = np.asarray(values, dtype=np.uint16).reshape(rows, cols)
        else:
            raise RuntimeError(f"{path.name}: unsupported PGM magic {magic!r}")

    if max_value != 255:
        image = np.rint(image.astype(np.float64) * (255.0 / float(max_value)))
    return np.ascontiguousarray(image.astype(np.uint8))


def _sampled(image: np.ndarray, limit: int = 32) -> np.ndarray:
    row_step = max(1, image.shape[0] // limit)
    col_step = max(1, image.shape[1] // limit)
    return np.ascontiguousarray(image[::row_step, ::col_step][:limit, :limit])


def _synthetic_attribute_images():
    unit = np.array([[9]], dtype=np.uint8)

    constant = np.full((8, 8), 37, dtype=np.uint8)

    concentric = np.full((11, 11), 20, dtype=np.uint8)
    concentric[2:9, 2:9] = 100
    concentric[4:7, 4:7] = 180

    cross = np.zeros((9, 13), dtype=np.uint8)
    cross[4, 1:12] = 220
    cross[:, 6] = np.maximum(cross[:, 6], 140)

    ring = np.zeros((14, 14), dtype=np.uint8)
    ring[2:12, 2:12] = 180
    ring[5:9, 5:9] = 30

    checker = np.indices((10, 10)).sum(axis=0) % 2
    checker = (checker * 190 + 30).astype(np.uint8)

    return [
        ("unit", unit),
        ("constant", constant),
        ("concentric", concentric),
        ("cross", cross),
        ("ring", ring),
        ("checker", checker),
    ]


def _real_attribute_images():
    data_dir = _repo_root() / "external" / "mmcfilters" / "dat"
    paths = [data_dir / name for name in ("lena.pgm", "brain2.pgm", "wrist.pgm")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        pytest.skip("real attribute fixtures unavailable; initialize the mmcfilters submodule")
    return [(path.stem, _sampled(_read_pgm(path))) for path in paths]


def _non_finite_examples(values: np.ndarray, layout: dict[str, int]) -> str:
    bad = np.argwhere(~np.isfinite(values))
    if bad.size == 0:
        return ""

    inverse_layout = {int(column): str(name) for name, column in layout.items()}
    examples = []
    for row, col in bad[:8]:
        name = inverse_layout.get(int(col), f"column {int(col)}")
        examples.append(f"{name}@row {int(row)}={values[row, col]!r}")
    return ", ".join(examples)


def _assert_all_attributes_finite(image: np.ndarray, label: str):
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous

    for tree_name, tree_factory in (
        ("max-tree", morphology.create_max_tree),
        ("min-tree", morphology.create_min_tree),
    ):
        tree = tree_factory(image)
        for dtype in (np.float32, np.float64):
            layout, values = morphology.compute_attributes(
                tree,
                [morphology.AttributeGroup.ALL],
                dtype=dtype,
            )

            assert values.dtype == np.dtype(dtype), f"{label}: {tree_name}: expected {dtype}"
            assert values.shape == (tree.numInternalNodeSlots, len(layout))
            assert "ECCENTRICITY" in layout
            assert "BITQUADS_CIRCULARITY" in layout
            assert "MAX_DIST" in layout

            examples = _non_finite_examples(values, layout)
            assert not examples, f"{label}: {tree_name}: {np.dtype(dtype).name}: non-finite values: {examples}"


def test_tree_constructors_return_public_facade_type():
    image = _small_image()

    trees = [
        morphology.create_max_tree(image),
        morphology.create_min_tree(image),
        morphology.create_tree_of_shapes(image),
    ]

    for tree in trees:
        assert morphology.is_tree(tree)
        assert isinstance(tree, morphology.WeightedMorphologicalTree)
        assert tree.numRows == image.shape[0]
        assert tree.numCols == image.shape[1]
        assert tree.numNodes > 0
        assert tree.numInternalNodeSlots >= tree.numNodes


def test_tree_of_shapes_facade_accepts_interpolation_options():
    image = _small_image()

    tree = morphology.create_tree_of_shapes(
        image,
        interpolation="min4c-max8c",
        infinity_seed_row=0,
        infinity_seed_col=0,
    )

    assert tree.treeType == 2
    assert tree.hasTreeOfShapesAdjacencyPolicy is True
    assert tree.getTreeOfShapesMinTreeAdjacencyRadius() == 1.0
    assert tree.getTreeOfShapesMaxTreeAdjacencyRadius() == 1.5

    enum_tree = morphology.build_tree(
        image,
        "tree-of-shapes",
        tos_interpolation=morphology.ToSInterpolation.Min8cMax4c,
    )

    assert enum_tree.treeType == 2
    assert enum_tree.getTreeOfShapesMinTreeAdjacencyRadius() == 1.5
    assert enum_tree.getTreeOfShapesMaxTreeAdjacencyRadius() == 1.0


def test_build_tree_rejects_unknown_tree_type():
    with pytest.raises(ValueError, match="unknown tree_type"):
        morphology.build_tree(_small_image(), "not-a-tree")


def test_tree_constructors_reject_non_2d_images():
    image = np.array([1, 2, 3], dtype=np.uint8)

    with pytest.raises(ValueError, match="2D uint8 array"):
        morphology.create_max_tree(image)


def test_compute_attributes_returns_sorted_index_and_expected_shape():
    tree = morphology.create_max_tree(_small_image())

    attr_index, attr_values = morphology.compute_attributes(
        tree,
        [
            morphology.AttributeType.AREA,
            morphology.AttributeType.COMPACTNESS,
            morphology.AttributeGroup.TREE_TOPOLOGY,
        ],
    )

    assert list(attr_index.values()) == sorted(attr_index.values())
    assert attr_values.shape[0] == tree.numInternalNodeSlots
    assert attr_values.shape[1] == len(attr_index)
    assert "AREA" in attr_index
    assert "COMPACTNESS" in attr_index


def test_compute_attributes_accepts_float_dtype():
    tree = morphology.create_max_tree(_small_image())
    attributes = [morphology.AttributeType.AREA, morphology.AttributeType.COMPACTNESS]

    attr_index32, attr_values32 = morphology.compute_attributes(
        tree,
        attributes,
        dtype=np.float32,
    )
    attr_index64, attr_values64 = morphology.compute_attributes(
        tree,
        attributes,
        dtype=np.float64,
    )
    _, attr_values64_from_string = morphology.compute_attributes(
        tree,
        attributes,
        dtype="float64",
    )

    assert attr_index64 == attr_index32
    assert attr_values32.dtype == np.float32
    assert attr_values64.dtype == np.float64
    assert attr_values64_from_string.dtype == np.float64
    assert np.allclose(attr_values32, attr_values64, equal_nan=True)


def test_all_attributes_are_finite_on_synthetic_images():
    for label, image in _synthetic_attribute_images():
        _assert_all_attributes_finite(image, label)


def test_all_attributes_are_finite_on_real_fixture_images():
    for label, image in _real_attribute_images():
        _assert_all_attributes_finite(image, label)


def test_compute_single_attribute_matches_tree_node_space():
    tree = morphology.create_max_tree(_small_image())

    area = morphology.compute_single_attribute(tree, morphology.AttributeType.AREA)

    assert area.shape == (tree.numInternalNodeSlots,)
    assert area.dtype == np.float32


def test_compute_single_attribute_accepts_float_dtype():
    tree = morphology.create_max_tree(_small_image())

    area32 = morphology.compute_single_attribute(
        tree,
        morphology.AttributeType.AREA,
        dtype=np.float32,
    )
    area64 = morphology.compute_single_attribute(
        tree,
        morphology.AttributeType.AREA,
        dtype=np.float64,
    )
    area64_from_python_float = morphology.compute_single_attribute(
        tree,
        morphology.AttributeType.AREA,
        dtype=float,
    )

    assert area32.dtype == np.float32
    assert area64.dtype == np.float64
    assert area64_from_python_float.dtype == np.float64
    assert np.allclose(area32, area64)


def test_compute_attribute_rejects_non_float_dtype():
    tree = morphology.create_max_tree(_small_image())

    with pytest.raises(ValueError, match="dtype must be np.float32 or np.float64"):
        morphology.compute_single_attribute(
            tree,
            morphology.AttributeType.AREA,
            dtype=np.int32,
        )


def test_attribute_descriptions_are_exposed_through_facade():
    area_description = morphology.describe_attribute(morphology.AttributeType.AREA)
    all_descriptions = morphology.describe_all_attributes()

    assert area_description.startswith("Area:")
    assert all_descriptions["AREA"] == area_description
    assert "MAX_DIST" in all_descriptions
    assert "CONTOUR_PERIMETER" in all_descriptions


def test_attribute_filter_validates_node_sized_inputs():
    tree = morphology.create_max_tree(_small_image())
    attribute_filter = morphology.create_attribute_filter(tree)

    with pytest.raises(ValueError, match="criterion must have length"):
        attribute_filter.filteringSubtractiveRule([True])

    with pytest.raises(ValueError, match="attr must have length"):
        attribute_filter.filteringMin(np.ones(1, dtype=np.float32), 1.0)


def test_attribute_filter_accepts_float64_attributes():
    tree = morphology.create_max_tree(_small_image())
    area = morphology.compute_single_attribute(
        tree,
        morphology.AttributeType.AREA,
        dtype=np.float64,
    )
    attribute_filter = morphology.create_attribute_filter(tree)

    filtered_min = attribute_filter.filteringMin(area, float(np.median(area)))
    filtered_max = attribute_filter.filteringMax(area, float(np.median(area)))

    assert filtered_min.shape == _small_image().shape
    assert filtered_max.shape == _small_image().shape
    assert filtered_min.dtype == np.uint8
    assert filtered_max.dtype == np.uint8


def test_attribute_filter_rejects_non_float_attribute_array():
    tree = morphology.create_max_tree(_small_image())
    attribute_filter = morphology.create_attribute_filter(tree)

    with pytest.raises(ValueError, match="attr must be a 1D np.float32 or np.float64 array"):
        attribute_filter.filteringMin(
            np.ones(tree.numInternalNodeSlots, dtype=np.int32),
            1.0,
        )


def test_removed_backend_symbols_are_not_reexported():
    removed_symbols = [
        "TreeStats",
        "make_tree_stats",
        "make_tree_tensor",
        "ConnectedFilterByJacobian",
        "ConnectedFilterByMorphologicalTree",
        "InfoTree",
    ]

    for name in removed_symbols:
        assert not hasattr(mtlearn, name)
        assert not hasattr(mtlearn._bindings, name)
