# Attribute Catalog

This catalog lists the scalar attributes exposed through
`mtlearn.morphology.AttributeType` and the groups exposed through
`mtlearn.morphology.AttributeGroup`.

Attributes are computed per morphology-tree node slot. They are not image
arrays. The usual result layout is:

```python
attribute_index, values = morphology.compute_attributes(tree, attributes)
column = attribute_index["AREA"]
area_by_node_slot = values[:, column]
```

Attribute buffers default to `np.float32`. Pass `dtype=np.float64` to
`compute_attributes` or `compute_single_attribute` when double-precision
attribute values are needed. Deterministic attribute filters accept both
`float32` and `float64` attribute vectors.

Use `tree.numInternalNodeSlots` when allocating criteria, scores, or any
node-indexed array that will be passed back to filtering code.

## Terminology

A morphology tree has two kinds of elements:

- **Proper parts** are the image-domain elements. For 2D images, each proper
  part is one pixel identified by its row-major linear index.
- **Internal tree nodes** represent connected components or shapes built from
  those proper parts. These are the rows addressed by attribute vectors.

A **node slot** is the dense integer row used internally for one internal tree
node. Attribute arrays keep one row per slot so the same id can be reused by
filters, scores, CFP parameters, and reconstruction code. In ordinary trees
these slots correspond to live internal nodes; after advanced topology edits,
some slots may be kept for stable indexing.

The **support** of a node is the set of image pixels represented by that node:
the pixels directly owned by the node plus the pixels represented by its
descendant nodes. In a max-tree or min-tree, this support is the connected
component associated with that tree node at its altitude.

## Attribute Groups

Groups are convenience requests. They expand to canonical scalar attributes
before computation or CFP parameter construction. Aliases such as `ALTITUDE`
resolve to their canonical attribute, `LEVEL`, and do not create an additional
column.

| Group | Meaning |
| --- | --- |
| `GRAY_LEVEL` | Altitude and gray-level statistics. |
| `SHAPE` | Size, bounding box, moment, bitquad, contour, and selected shape descriptors. |
| `MOMENTS` | Central moments, Hu invariants, and moment-derived descriptors. |
| `BOUNDARY` | Bitquad and contour descriptors. |
| `TREE_TOPOLOGY` | Parent/child hierarchy descriptors. |
| `ALL` | Every public scalar attribute. |

```python
shape_attrs = morphology.expand_attribute_group(
    morphology.AttributeGroup.SHAPE,
)
```

## Contracts

The catalog uses three contracts:

| Contract | Meaning |
| --- | --- |
| Altitude-aware | Reads node altitude or gray-level values from a weighted tree. |
| Support/geometry | Reads the pixel support of each node, coordinates, contour, or adjacency information. |
| Tree topology | Reads only parent/child relations in the hierarchy. |

Some support/geometry attributes require metadata that is not meaningful for
every tree type. For example, `MAX_DIST` is currently intended for max-trees
and min-trees with adjacency metadata, not generic tree-of-shapes CFP filters.

## Gray-Level Attributes

| Attribute | Groups | Contract | Use |
| --- | --- | --- | --- |
| `LEVEL` | `GRAY_LEVEL` | Altitude-aware | Node altitude. In component trees, this is the gray level at which the component appears. |
| `ALTITUDE` | `GRAY_LEVEL` | Altitude-aware | Python alias for `LEVEL` when exposed by the native enum. |
| `VOLUME` | `GRAY_LEVEL` | Altitude-aware | Gray-level mass accumulated over the node support. Useful for contrast weighted by size. |
| `RELATIVE_VOLUME` | `GRAY_LEVEL` | Altitude-aware | Contrast-like volume relative to parent/child altitude jumps. |
| `GRAY_HEIGHT` | `GRAY_LEVEL` | Altitude-aware | Gray-level span from the node to its most extreme descendant. |
| `MEAN_LEVEL` | `GRAY_LEVEL` | Altitude-aware | Mean gray level over the full support represented by the node. |
| `VARIANCE_LEVEL` | `GRAY_LEVEL` | Altitude-aware | Gray-level variance over the node support. |

Gray-level attributes are good choices when the filter should depend on
contrast, intensity, or altitude differences rather than pure geometry.

## Size and Bounding-Box Attributes

| Attribute | Groups | Contract | Use |
| --- | --- | --- | --- |
| `AREA` | `SHAPE` | Support/geometry | Number of pixels in the full node support, including descendant supports. For image trees, this is the connected-component size. |
| `BOX_WIDTH` | `SHAPE` | Support/geometry | Width of the smallest axis-aligned bounding box around the support. |
| `BOX_HEIGHT` | `SHAPE` | Support/geometry | Height of the smallest axis-aligned bounding box around the support. |
| `DIAGONAL_LENGTH` | `SHAPE` | Support/geometry | Bounding-box diagonal length. |
| `RECTANGULARITY` | `SHAPE` | Support/geometry | How densely the support fills its bounding box. |
| `RATIO_WH` | `SHAPE` | Support/geometry | Bounding-box aspect ratio. |
| `BOX_COL_MIN` | `SHAPE` | Support/geometry | Minimum image column covered by the support. |
| `BOX_COL_MAX` | `SHAPE` | Support/geometry | Maximum image column covered by the support. |
| `BOX_ROW_MIN` | `SHAPE` | Support/geometry | Minimum image row covered by the support. |
| `BOX_ROW_MAX` | `SHAPE` | Support/geometry | Maximum image row covered by the support. |
| `MAX_DIST` | `SHAPE` | Support/geometry | Maximum distance-transform style descriptor from contour information. Prefer max-tree/min-tree use. |

`AREA` is the usual attribute for size filtering. A high `AREA` value means
that the component represented by the node covers many pixels in the original
image domain; it is not the number of children in the tree.

These attributes are the most direct tools for area openings, size priors, and
filters that should keep or remove elongated connected components.

## Moment Attributes

| Attribute | Groups | Contract | Use |
| --- | --- | --- | --- |
| `CENTRAL_MOMENT_20` | `MOMENTS`, `SHAPE` | Support/geometry | Horizontal second-order spread around the support centroid. |
| `CENTRAL_MOMENT_02` | `MOMENTS`, `SHAPE` | Support/geometry | Vertical second-order spread around the support centroid. |
| `CENTRAL_MOMENT_11` | `MOMENTS`, `SHAPE` | Support/geometry | Mixed second-order moment, similar to covariance. |
| `CENTRAL_MOMENT_30` | `MOMENTS`, `SHAPE` | Support/geometry | Horizontal third-order asymmetry. |
| `CENTRAL_MOMENT_03` | `MOMENTS`, `SHAPE` | Support/geometry | Vertical third-order asymmetry. |
| `CENTRAL_MOMENT_21` | `MOMENTS`, `SHAPE` | Support/geometry | Combined horizontal spread and vertical asymmetry. |
| `CENTRAL_MOMENT_12` | `MOMENTS`, `SHAPE` | Support/geometry | Combined vertical spread and horizontal asymmetry. |
| `HU_MOMENT_1` | `MOMENTS`, `SHAPE` | Support/geometry | First Hu invariant. Useful as a compactness-like invariant descriptor. |
| `HU_MOMENT_2` | `MOMENTS`, `SHAPE` | Support/geometry | Second Hu invariant, sensitive to anisotropy. |
| `HU_MOMENT_3` | `MOMENTS`, `SHAPE` | Support/geometry | Third Hu invariant, sensitive to skewness patterns. |
| `HU_MOMENT_4` | `MOMENTS`, `SHAPE` | Support/geometry | Fourth Hu invariant, sensitive to off-axis symmetry. |
| `HU_MOMENT_5` | `MOMENTS`, `SHAPE` | Support/geometry | Fifth Hu invariant, sensitive to complex asymmetry. |
| `HU_MOMENT_6` | `MOMENTS`, `SHAPE` | Support/geometry | Sixth Hu invariant, combining second- and third-order moments. |
| `HU_MOMENT_7` | `MOMENTS`, `SHAPE` | Support/geometry | Seventh Hu invariant, often sensitive to fine asymmetries and reflection. |
| `INERTIA` | `MOMENTS`, `SHAPE` | Support/geometry | Moment-derived dispersion descriptor. |
| `COMPACTNESS` | `MOMENTS`, `SHAPE` | Support/geometry | Compactness of the support relative to its second-order dispersion. |
| `ECCENTRICITY` | `MOMENTS`, `SHAPE` | Support/geometry | Elongation from the second-moment matrix. |
| `LENGTH_MAJOR_AXIS` | `MOMENTS`, `SHAPE` | Support/geometry | Major-axis length proxy from the equivalent ellipse. |
| `LENGTH_MINOR_AXIS` | `MOMENTS`, `SHAPE` | Support/geometry | Minor-axis length proxy from the equivalent ellipse. |
| `AXIS_ORIENTATION` | `MOMENTS`, `SHAPE` | Support/geometry | Principal-axis orientation. |
| `CIRCULARITY` | `MOMENTS`, `SHAPE` | Support/geometry | Isotropy ratio. Values closer to one indicate rounder supports. |

Moment descriptors are useful when area alone is too coarse and the filter
should react to shape geometry.

## Boundary and Bitquad Attributes

| Attribute | Groups | Contract | Use |
| --- | --- | --- | --- |
| `BITQUADS_AREA` | `BOUNDARY`, `SHAPE` | Support/geometry | Sub-pixel area estimator derived from local 2x2 bitquad patterns. |
| `BITQUADS_NUMBER_EULER` | `BOUNDARY`, `SHAPE` | Support/geometry | Euler-characteristic estimate from bitquad counts. |
| `BITQUADS_NUMBER_HOLES` | `BOUNDARY`, `SHAPE` | Support/geometry | Hole-count estimate derived from the bitquad Euler descriptor. |
| `BITQUADS_PERIMETER` | `BOUNDARY`, `SHAPE` | Support/geometry | Discrete perimeter estimate from bitquad patterns. |
| `BITQUADS_PERIMETER_CONTINUOUS` | `BOUNDARY`, `SHAPE` | Support/geometry | Smoothed perimeter estimate from bitquad counts. |
| `BITQUADS_CIRCULARITY` | `BOUNDARY`, `SHAPE` | Support/geometry | Roundness descriptor based on bitquad area and perimeter. |
| `BITQUADS_PERIMETER_AVERAGE` | `BOUNDARY`, `SHAPE` | Support/geometry | Average perimeter proxy. |
| `BITQUADS_LENGTH_AVERAGE` | `BOUNDARY`, `SHAPE` | Support/geometry | Average longitudinal extent proxy. |
| `BITQUADS_WIDTH_AVERAGE` | `BOUNDARY`, `SHAPE` | Support/geometry | Average transverse extent proxy. |
| `CONTOUR_PIXELS` | `BOUNDARY`, `SHAPE` | Support/geometry | Number of support pixels touching the 4-neighbour complement. |
| `CONTOUR_PERIMETER` | `BOUNDARY`, `SHAPE` | Support/geometry | Number of exposed 4-neighbour sides over the support. |
| `CONTOUR_SIDE_NORTH` | `BOUNDARY`, `SHAPE` | Support/geometry | Exposed north-facing sides. |
| `CONTOUR_SIDE_WEST` | `BOUNDARY`, `SHAPE` | Support/geometry | Exposed west-facing sides. |
| `CONTOUR_SIDE_EAST` | `BOUNDARY`, `SHAPE` | Support/geometry | Exposed east-facing sides. |
| `CONTOUR_SIDE_SOUTH` | `BOUNDARY`, `SHAPE` | Support/geometry | Exposed south-facing sides. |

Boundary attributes are useful when perimeter, holes, or contour exposure are
more informative than raw area.

## Tree-Topology Attributes

| Attribute | Groups | Contract | Use |
| --- | --- | --- | --- |
| `HEIGHT_NODE` | `TREE_TOPOLOGY` | Tree topology | Longest child-edge path from the node to any leaf in its subtree. |
| `DEPTH_NODE` | `TREE_TOPOLOGY` | Tree topology | Number of parent-edge steps from the node to the root. |
| `IS_LEAF_NODE` | `TREE_TOPOLOGY` | Tree topology | Boolean scalar encoded as one for leaves and zero otherwise. |
| `IS_ROOT_NODE` | `TREE_TOPOLOGY` | Tree topology | Boolean scalar encoded as one for the root and zero otherwise. |
| `NUM_CHILDREN_NODE` | `TREE_TOPOLOGY` | Tree topology | Number of direct children. |
| `NUM_SIBLINGS_NODE` | `TREE_TOPOLOGY` | Tree topology | Number of other nodes with the same parent. |
| `NUM_DESCENDANTS_NODE` | `TREE_TOPOLOGY` | Tree topology | Number of internal tree nodes below this node. |
| `NUM_LEAF_DESCENDANTS_NODE` | `TREE_TOPOLOGY` | Tree topology | Number of leaf nodes in the node subtree. |
| `LEAF_RATIO_NODE` | `TREE_TOPOLOGY` | Tree topology | Ratio of leaf descendants to subtree size. |
| `BALANCE_NODE` | `TREE_TOPOLOGY` | Tree topology | Difference between deepest and shallowest child-subtree heights. |
| `AVG_CHILD_HEIGHT_NODE` | `TREE_TOPOLOGY` | Tree topology | Average height of direct child subtrees. |

Tree-topology attributes ignore image coordinates and gray levels. They are
useful when the hierarchy shape itself carries the signal.

## Choosing Attributes

Start small:

- use `AREA` for size filtering;
- add `GRAY_HEIGHT` or `VOLUME` when contrast matters;
- add `COMPACTNESS`, `ECCENTRICITY`, or `RATIO_WH` when component shape
  matters;
- add `TREE_TOPOLOGY` attributes when hierarchy position matters;
- use `BOUNDARY` attributes for contour- or hole-sensitive filters.

For CFP layers, each scalar attribute adds one trainable weight to the filter
spec after group expansion.

```python
filter_spec = {
    "name": "area_contrast_shape",
    "tree_type": "max-tree",
    "attributes": [
        morphology.AttributeType.AREA,
        morphology.AttributeType.GRAY_HEIGHT,
        morphology.AttributeType.COMPACTNESS,
    ],
}
```
