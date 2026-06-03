# Getting Started

This guide shows the shortest path from an image to a morphology tree,
attributes, and a filtered reconstruction.

## Build a Tree

`mtlearn.morphology` accepts 2D `numpy.uint8` images. If your data is a PyTorch
tensor in `[0, 1]`, convert it before calling the morphology facade.

```python
import numpy as np
from mtlearn import morphology

image = np.array(
    [
        [0, 0, 10, 10],
        [0, 30, 40, 10],
        [0, 30, 40, 10],
        [0, 0, 10, 10],
    ],
    dtype=np.uint8,
)

tree = morphology.create_max_tree(image)

print(tree.numRows, tree.numCols)
print(tree.numNodes, tree.numInternalNodeSlots)
```

Use a max-tree for bright connected components, a min-tree for dark connected
components, and a tree of shapes when the filter should be self-dual.

```python
max_tree = morphology.create_max_tree(image)
min_tree = morphology.create_min_tree(image)
tos = morphology.create_tree_of_shapes(image, interpolation="self-dual")
```

## Inspect Topology

Node arrays returned by mtlearn are usually indexed by backend node slot. The
slot count can be larger than the number of live nodes after edits, so allocate
criteria and scores with `numInternalNodeSlots`.

```python
root = tree.root
children = tree.children_of(root)
alive_nodes = tree.alive_node_ids

print("root:", root)
print("children:", children)
print("alive:", alive_nodes[:5])
```

For a pixel-level lookup, use the flattened pixel id:

```python
row, col = 1, 2
pixel_id = row * tree.numCols + col
owner = tree.proper_part_owner_of(pixel_id)
print("pixel owner:", owner)
```

## Compute Attributes

`compute_attributes` returns `(attribute_index, values)`. The index maps public
attribute names to columns. The value matrix has one row per node slot.

```python
attrs = [
    morphology.AttributeType.AREA,
    morphology.AttributeType.GRAY_HEIGHT,
    morphology.AttributeType.COMPACTNESS,
]

attribute_index, values = morphology.compute_attributes(tree, attrs)

area_col = attribute_index["AREA"]
areas = values[:, area_col]
print(areas.shape)
```

Use `compute_single_attribute` when only one scalar attribute is needed.

```python
area = morphology.compute_single_attribute(
    tree,
    morphology.AttributeType.AREA,
)
```

## Apply a Simple Attribute Filter

Attribute-filter helpers are bound to one tree. Threshold-based methods consume
one value per node slot and return a reconstructed image.

```python
filters = morphology.create_attribute_filter(tree)
area = morphology.compute_single_attribute(tree, morphology.AttributeType.AREA)

filtered = filters.filteringMin(area, threshold=4.0)
print(filtered.dtype, filtered.shape)
```

Boolean criteria must also have one entry per node slot:

```python
criterion = area >= 4.0
filtered = filters.filteringDirectRule(criterion.tolist())
```

## Next Steps

- {doc}`morphology` explains tree types, node ids, and filter rules.
- {doc}`connected-filter-preprocessing` shows how to turn these operations into
  a learnable PyTorch preprocessing layer.
- {doc}`../api/python/morphology` contains the generated Python reference.
