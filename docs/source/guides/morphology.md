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
filtered = filters.filteringMin(area, threshold=4.0)

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
    tos_interpolation=morphology.ToSInterpolation.SelfDual,
)
```

Accepted tree-of-shapes interpolation aliases include `"self-dual"`,
`"min4c-max8c"`, and `"min8c-max4c"`.

## Inspect a Tree

The tree object exposes image dimensions, live node counts, internal node slot
counts, and common topology queries.

```python
print("image:", tree.numRows, "x", tree.numCols)
print("live nodes:", tree.numNodes)
print("node slots:", tree.numInternalNodeSlots)
print("root:", tree.root)
print("leaves:", len(tree.leaf_node_ids))
```

The backend distinguishes live topology nodes from internal node slots:

- `numNodes` counts currently live topology nodes.
- `numInternalNodeSlots` is the size expected by attribute vectors, boolean
  criteria, score vectors, and CFP tree tensors.

Use the slot count when allocating node-indexed arrays.

```python
criterion = [False] * tree.numInternalNodeSlots
for node in tree.alive_node_ids:
    criterion[node] = tree.isLeaf(node)
```

This convention matters after pruning or merging, because inactive node slots
may remain allocated even when the live topology changes.

## Navigate Topology

Common topology methods have legacy camelCase names and Python-friendly
snake_case aliases. Prefer the snake_case aliases in new Python code.

```python
root = tree.root
children = tree.children_of(root)
descendants = tree.descendants_of(root)
leaves = tree.leaf_node_ids

leaf = leaves[0]
parent = tree.parent_of(leaf)
path_to_root = tree.getPathToRootNodes(leaf)
leaf_subtree = tree.node_subtree_of(leaf)
```

Pixel ownership is exposed through proper parts. For ordinary 2D images, a
proper part can be read as a flattened pixel id.

```python
row, col = 1, 2
pixel_id = row * tree.numCols + col

owner = tree.proper_part_owner_of(pixel_id)
component_mask = tree.reconstructNode(owner)
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

gray_height = morphology.compute_single_attribute(
    tree,
    morphology.AttributeType.GRAY_HEIGHT,
)
```

Use `compute_attributes` when you need several attributes at once.

```python
attribute_index, values = morphology.compute_attributes(
    tree,
    [
        morphology.AttributeType.AREA,
        morphology.AttributeType.GRAY_HEIGHT,
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
    morphology.AttributeType.GRAY_HEIGHT,
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

Tree-of-shapes filters cannot compute every scalar attribute currently exposed
by the backend. In particular, attributes that depend on one-sided component
tree geometry may be unavailable for trees of shapes.

## Find Nodes by Attribute

A common workflow is to compute an attribute, restrict to live nodes, and pick a
node for inspection.

```python
area = morphology.compute_single_attribute(tree, morphology.AttributeType.AREA)
alive = np.asarray(tree.alive_node_ids, dtype=np.int64)

largest_live_node = alive[np.argmax(area[alive])]

print("largest node:", largest_live_node)
print("area:", area[largest_live_node])

mask = tree.reconstructNode(int(largest_live_node))
```

The same pattern works for shape attributes, topology attributes, and
gray-level attributes.

## Deterministic Attribute Filters

`create_attribute_filter(tree)` returns a helper bound to one tree. Its methods
consume node-slot-sized arrays and return reconstructed images.

```python
filters = morphology.create_attribute_filter(tree)
area = morphology.compute_single_attribute(tree, morphology.AttributeType.AREA)

area_opening = filters.filteringMin(area, threshold=16.0)
```

For boolean-rule filters, build one boolean value per node slot.

```python
criterion = (area >= 16.0).tolist()

direct = filters.filteringDirectRule(criterion)
subtractive = filters.filteringSubtractiveRule(criterion)
```

Keep this assertion in prototypes; it catches most criterion/attribute shape
mistakes.

```python
assert len(criterion) == tree.numInternalNodeSlots
```

Score-based subtractive filtering expects one float score per node slot.

```python
scores = area.astype(np.float32)
score_image = filters.filteringSubtractiveScoreRule(scores.tolist())
```

## Common Filtering Recipes

### Remove Small Bright Components

Use a max-tree and an area threshold.

```python
tree = morphology.create_max_tree(image)
area = morphology.compute_single_attribute(tree, morphology.AttributeType.AREA)
filters = morphology.create_attribute_filter(tree)

opened = filters.filteringMin(area, threshold=25.0)
```

### Remove Small Dark Components

Use a min-tree with the same type of area criterion.

```python
tree = morphology.create_min_tree(image)
area = morphology.compute_single_attribute(tree, morphology.AttributeType.AREA)
filters = morphology.create_attribute_filter(tree)

closed_like = filters.filteringMin(area, threshold=25.0)
```

### Inspect Self-Dual Shapes

Use a tree of shapes when you do not want to choose bright or dark polarity.

```python
tree = morphology.create_tree_of_shapes(
    image,
    interpolation=morphology.ToSInterpolation.SelfDual,
)

attrs = [
    morphology.AttributeType.AREA,
    morphology.AttributeType.COMPACTNESS,
]
attribute_index, values = morphology.compute_attributes(tree, attrs)
```

## Mutating Trees

`pruneNode` and `mergeNodeIntoParent` mutate the tree in place. Query
topology-dependent values again after a mutation.

```python
node = int(tree.leaf_node_ids[0])
tree.pruneNode(node)

reconstructed = tree.reconstructionImage()
alive_after_prune = tree.alive_node_ids
```

Rebuild the tree if you need to preserve the original topology.

```python
original = morphology.create_max_tree(image)
working = morphology.create_max_tree(image)

working.pruneNode(int(working.leaf_node_ids[0]))
```

Use mutating operations for inspection, deterministic prototypes, and backend
experiments. For learnable preprocessing inside PyTorch models, prefer
`ConnectedFilterPreprocessingLayer`, which keeps tree construction and
attribute computation outside autograd and learns only node-wise gates.

## Common Mistakes

- Passing RGB or batched arrays to tree constructors. Build one tree from one
  2D grayscale image.
- Allocating criteria with `numNodes` instead of `numInternalNodeSlots`.
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
