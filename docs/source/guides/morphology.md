# Morphology User Guide

`mtlearn.morphology` is the stable Python entry point for building
morphological trees, computing node attributes, applying deterministic
attribute filters, and inspecting tree topology. This guide is written for
users who want to run those operations directly, without going through the
learnable CFP layer.

The examples assume that the native `_mtlearn` extension is installed and that
input images are 2D `numpy.uint8` arrays.

```python
import numpy as np
from mtlearn import morphology
```

## A Complete First Example

Start with a small grayscale image, build a max-tree, compute an area
attribute, and apply a deterministic area filter.

```python
image = np.array(
    [
        [0, 0, 10, 10, 10, 0],
        [0, 20, 50, 50, 10, 0],
        [0, 20, 50, 50, 10, 0],
        [0, 0, 10, 10, 10, 0],
    ],
    dtype=np.uint8,
)

tree = morphology.create_max_tree(image)

area = morphology.compute_single_attribute(
    tree,
    morphology.AttributeType.AREA,
)

filters = morphology.create_attribute_filter(tree)
filtered = filters.filtering_by_pruning_min(area, threshold=4.0)

print(filtered.shape, filtered.dtype)
```

`filtered` is a reconstructed NumPy image. Attribute arrays such as `area` are
node-indexed arrays: they are not images and should not be reshaped to image
size.

## Image Requirements

Tree constructors expect a 2D uint8 image. Keep conversion explicit near the
boundary of your application.

```python
image = np.asarray(image, dtype=np.uint8)
if image.ndim != 2:
    raise ValueError("mtlearn.morphology expects one 2D grayscale image")
```

If your data starts as a normalized float image, scale it before tree
construction.

```python
image_u8 = np.clip(image_float * 255.0, 0, 255).astype(np.uint8)
tree = morphology.create_max_tree(image_u8)
```

For batched tensors, build one tree per sample and per channel. Tree
construction is not a vectorized tensor operation.

## Choose a Tree Type

The tree type encodes the image structures you want to organize.

| Tree type | Use when | Constructor |
| --- | --- | --- |
| Max-tree | Bright connected components are foreground. | `create_max_tree(image)` |
| Min-tree | Dark connected components are foreground. | `create_min_tree(image)` |
| Tree of shapes | Bright and dark structures should be handled symmetrically. | `create_tree_of_shapes(image)` |

```python
max_tree = morphology.create_max_tree(image)
min_tree = morphology.create_min_tree(image)
tos_tree = morphology.create_tree_of_shapes(
    image,
    interpolation="self-dual",
)
```

When tree type comes from a configuration file or command-line argument, use
`build_tree`.

```python
tree = morphology.build_tree(image, "max-tree")
tree = morphology.build_tree(image, morphology.TreeType.MIN_TREE)
tree = morphology.build_tree(
    image,
    "tree-of-shapes",
    tos_interpolation=morphology.ToSInterpolation.SELF_DUAL,
)
```

Accepted tree-of-shapes interpolation aliases include `"self-dual"`,
`"min4c-max8c"`, and `"min8c-max4c"`.

## Inspect a Tree

The tree object exposes image dimensions, live node counts, internal node slot
counts, and common topology queries.

```python
print("image:", tree.num_rows, "x", tree.num_columns)
print("live nodes:", tree.num_nodes)
print("node slots:", tree.num_internal_node_slots)
print("root:", tree.root)
print("leaves:", len(tree.leaves))
```

The backend distinguishes live topology nodes from internal node slots:

- `num_nodes` counts currently live topology nodes.
- `num_internal_node_slots` is the size expected by attribute vectors, boolean
  criteria, score vectors, and CFP tree tensors.

Use the slot count when allocating node-indexed arrays.

```python
criterion = [False] * tree.num_internal_node_slots
for node in tree.alive_node_ids:
    criterion[node] = tree.is_leaf(node)
```

This convention matters after pruning or merging, because inactive node slots
may remain allocated even when the live topology changes.

## Navigate Topology

Topology queries use the same relation names as mmcfilters: `children`,
`parent`, `ancestors`, `descendants`, and `subtree_nodes`.

```python
root = tree.root
children = tree.children(root)
descendants = tree.descendants(root)
leaves = tree.leaves

leaf = leaves[0]
parent = tree.parent(leaf)
path_to_root = tree.ancestors(leaf)
leaf_subtree = tree.subtree_nodes(leaf)
```

The proper part of a node contains the pixels assigned directly to it.
`smallest_node` maps a flattened pixel id to the node that owns it.

```python
row, col = 1, 2
pixel_id = row * tree.num_columns + col

owner = tree.smallest_node(pixel_id)
component_mask = tree.reconstruct_node(owner)
```

`component_mask` is a uint8 image mask for the component represented by the
owning node.

## Compute Scalar Attributes

Scalar attributes are members of `morphology.AttributeType`.

```python
area = morphology.compute_single_attribute(
    tree,
    morphology.AttributeType.AREA,
)

gray_level_height = morphology.compute_single_attribute(
    tree,
    morphology.AttributeType.GRAY_LEVEL_HEIGHT,
)
```

Use `compute_attributes` when you need several attributes at once.

```python
attribute_index, values = morphology.compute_attributes(
    tree,
    [
        morphology.AttributeType.AREA,
        morphology.AttributeType.GRAY_LEVEL_HEIGHT,
        morphology.AttributeType.COMPACTNESS,
    ],
)

area_col = attribute_index["AREA"]
compactness_col = attribute_index["COMPACTNESS"]

area = values[:, area_col]
compactness = values[:, compactness_col]
```

The returned `values` matrix has one row per node slot and one column per
computed scalar attribute.

## Use Attribute Groups

Attribute groups are shortcuts for common sets of scalar attributes.

```python
shape_attrs = morphology.expand_attribute_group(
    morphology.AttributeGroup.SHAPE,
)

attribute_index, shape_values = morphology.compute_attributes(
    tree,
    shape_attrs,
)
```

Groups are useful for exploration, but explicit scalar lists are easier to
track in experiments and checkpoints.

```python
attrs_for_experiment = [
    morphology.AttributeType.AREA,
    morphology.AttributeType.GRAY_LEVEL_HEIGHT,
    morphology.AttributeType.COMPACTNESS,
]
```

Use `describe_all_attributes` to discover the public attribute set.

```python
for name, description in morphology.describe_all_attributes().items():
    print(f"{name}: {description}")
```

For a grouped list of public attributes and their intended use, see
{doc}`../concepts/attributes`.

Attribute availability follows the input contracts in the attribute catalog.
Distance-transform attributes support trees of shapes.

## Find Nodes by Attribute

A common workflow is to compute an attribute, restrict to live nodes, and pick a
node for inspection.

```python
area = morphology.compute_single_attribute(tree, morphology.AttributeType.AREA)
alive = np.asarray(tree.alive_node_ids, dtype=np.int64)

largest_live_node = alive[np.argmax(area[alive])]

print("largest node:", largest_live_node)
print("area:", area[largest_live_node])

mask = tree.reconstruct_node(int(largest_live_node))
```

The same pattern works for shape attributes, topology attributes, and
gray-level attributes.

## Deterministic Attribute Filters

`create_attribute_filter(tree)` returns a helper bound to one tree. Its methods
consume node-slot-sized arrays and return reconstructed images.

```python
filters = morphology.create_attribute_filter(tree)
area = morphology.compute_single_attribute(tree, morphology.AttributeType.AREA)

area_opening = filters.filtering_by_pruning_min(area, threshold=16.0)
```

For boolean-rule filters, build one boolean value per node slot.

```python
criterion = (area >= 16.0).tolist()

direct = filters.apply_direct_attribute_filter(criterion)
subtractive = filters.apply_subtractive_attribute_filter(criterion)
```

Keep this assertion in prototypes; it catches most criterion/attribute shape
mistakes.

```python
assert len(criterion) == tree.num_internal_node_slots
```

Soft subtractive filtering expects one finite score in `[0, 1]` per node slot.

```python
scores = (area / area.max()).astype(np.float32)
score_image = filters.apply_soft_subtractive_attribute_filter(scores.tolist())
```

## Common Filtering Recipes

### Remove Small Bright Components

Use a max-tree and an area threshold.

```python
tree = morphology.create_max_tree(image)
area = morphology.compute_single_attribute(tree, morphology.AttributeType.AREA)
filters = morphology.create_attribute_filter(tree)

opened = filters.filtering_by_pruning_min(area, threshold=25.0)
```

### Remove Small Dark Components

Use a min-tree with the same type of area criterion.

```python
tree = morphology.create_min_tree(image)
area = morphology.compute_single_attribute(tree, morphology.AttributeType.AREA)
filters = morphology.create_attribute_filter(tree)

closed_like = filters.filtering_by_pruning_min(area, threshold=25.0)
```

### Inspect Self-Dual Shapes

Use a tree of shapes when you do not want to choose bright or dark polarity.

```python
tree = morphology.create_tree_of_shapes(
    image,
    interpolation=morphology.ToSInterpolation.SELF_DUAL,
)

attrs = [
    morphology.AttributeType.AREA,
    morphology.AttributeType.COMPACTNESS,
]
attribute_index, values = morphology.compute_attributes(tree, attrs)
```

## Mutating Trees

`prune_node` and `merge_node_into_parent` mutate the tree in place. Query
topology-dependent values again after a mutation.

```python
node = int(tree.leaves[0])
tree.prune_node(node)

reconstructed = tree.reconstruct_from_node_altitudes()
alive_after_prune = tree.alive_node_ids
```

Rebuild the tree if you need to preserve the original topology.

```python
original = morphology.create_max_tree(image)
working = morphology.create_max_tree(image)

working.prune_node(int(working.leaves[0]))
```

Use mutating operations for inspection, deterministic prototypes, and backend
experiments. For learnable preprocessing inside PyTorch models, prefer
`ConnectedFilterPreprocessingLayer`, which keeps tree construction and
attribute computation outside autograd and learns only node-wise gates.

## Common Mistakes

- Passing RGB or batched arrays to tree constructors. Build one tree from one
  2D grayscale image.
- Allocating criteria with `num_nodes` instead of `num_internal_node_slots`.
- Treating attribute arrays as images. Attributes are indexed by node slot.
- Reusing topology-dependent node ids after pruning or merging without
  querying the tree again.
- Comparing max-tree and min-tree outputs without accounting for polarity.

## Where to Go Next

- {doc}`getting-started` gives the shortest end-to-end setup.
- {doc}`../concepts/attributes` lists the public scalar attributes and groups.
- {doc}`connected-filter-preprocessing` explains the learnable PyTorch layer.
- {doc}`custom-attributes-filter-specs` explains how morphology attributes are
  used in CFP filter specs.
- {doc}`../api/python/morphology` contains the generated API reference.
