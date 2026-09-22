# Attribute Catalog

This catalog lists the scalar attributes exposed through
`mtlearn.morphology.AttributeType` and the groups exposed through
`mtlearn.morphology.AttributeGroup`: 133 scalar attributes and nine groups.
Names, groups, and definitions follow the
[mmcfilters v5.2.0 attribute catalog](https://github.com/wonderalexandre/mmcfilters/blob/v5.2.0/docs/attribute-catalog.md).

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

Use `tree.num_internal_node_slots` when allocating criteria, scores, or any
node-indexed array that will be passed back to filtering code.

## Terminology

A morphology tree represents connected components or shapes by tree nodes.
The **proper part** of a node is the set of image pixels assigned directly to
that node. Each pixel belongs to one proper part and is identified by its
row-major linear index. Attribute vectors are indexed by tree nodes.

A **node slot** is the dense integer row used internally for one internal tree
node. Attribute arrays keep one row per slot so the same id can be reused by
filters, scores, and reconstruction code. In ordinary trees
these slots correspond to live internal nodes; after advanced topology edits,
some slots may be kept for stable indexing.

The **support** of a node is the set of image pixels represented by that node:
the pixels directly owned by the node plus the pixels represented by its
descendant nodes. In a max-tree or min-tree, this support is the connected
component associated with that tree node at its altitude.

## Attribute Groups

Groups are convenience requests. They expand to canonical scalar attributes
before computation or CFP parameter construction. Each scalar is listed once
below, with its group memberships. `ALL` includes every listed scalar.

| Group | Meaning |
| --- | --- |
| `GRAY_LEVEL` | Node valuation, gray-level mass, contrast, height, mean, and variance. |
| `SHAPE` | Size, bounding box, moment, bitquad, contour, and selected shape descriptors. |
| `MOMENTS` | Central moments, Hu invariants, and moment-derived descriptors. |
| `BOUNDARY` | Bitquad and contour descriptors. |
| `TREE_TOPOLOGY` | Parent/child hierarchy descriptors. |
| `DIST_TRANSF` | Descriptors of the approximate contour-distance field. |
| `DIST_TRANSF_EXACT` | Descriptors of the exact Euclidean contour-distance field. |
| `FILLED_SHAPE` | Geometry of the region enclosed by the external boundary, including holes. |
| `ALL` | Every public scalar attribute. |

```python
shape_attributes = morphology.expand_attribute_group(
    morphology.AttributeGroup.SHAPE,
)
```

## Contracts

The catalog uses three contracts:

| Contract | Meaning |
| --- | --- |
| Altitude-aware | Reads node altitude or gray-level values from a valued tree. |
| Topology/support | Uses node support, pixel coordinates, contours, or adjacency information. |
| Tree topology | Reads only parent/child relations in the hierarchy. |

## Gray-Level Attributes

| Attribute | Groups | Contract | Use |
| --- | --- | --- | --- |
| `GRAY_LEVEL` | `GRAY_LEVEL` | Altitude-aware | Node valuation: the gray level associated with the node. |
| `VOLUME` | `GRAY_LEVEL` | Altitude-aware | Sum of altitude-weighted support contributions over the node subtree. |
| `RELATIVE_VOLUME` | `GRAY_LEVEL` | Altitude-aware | Recursive contrast volume `R(n) = area(n) + sum_c [R(c) + area(c) * abs(altitude(c) - altitude(n))]` over the direct children `c`. |
| `GRAY_LEVEL_HEIGHT` | `GRAY_LEVEL` | Altitude-aware | Maximum absolute altitude difference between the node and any node in its subtree. |
| `MEAN_GRAY_LEVEL` | `GRAY_LEVEL` | Altitude-aware | Arithmetic mean of the image values over the full node support: `sum(f(x), x in X) / card(X)`. |
| `GRAY_LEVEL_VARIANCE` | `GRAY_LEVEL` | Altitude-aware | Population variance of the image values over the full node support, with denominator `card(X)`. |

Gray-level attributes are good choices when the filter should depend on
contrast, intensity, or altitude differences rather than pure geometry.

## Size and Bounding-Box Attributes

| Attribute | Groups | Contract | Use |
| --- | --- | --- | --- |
| `AREA` | `SHAPE` | Topology/support | Number of pixels in the full node support. |
| `BOUNDING_BOX_WIDTH` | `SHAPE` | Topology/support | Width, in columns, of the smallest axis-aligned bounding box enclosing the node support. |
| `BOUNDING_BOX_HEIGHT` | `SHAPE` | Topology/support | Height, in rows, of the smallest axis-aligned bounding box enclosing the node support. |
| `DIAGONAL_LENGTH` | `SHAPE` | Topology/support | Euclidean diagonal length of the bounding box, `sqrt(width^2 + height^2)`. |
| `RECTANGULARITY` | `SHAPE` | Topology/support | Ratio `AREA / (BOUNDING_BOX_WIDTH * BOUNDING_BOX_HEIGHT)`. |
| `RATIO_WH` | `SHAPE` | Topology/support | Bounding-box aspect ratio, `max(width, height) / min(width, height)` for non-degenerate boxes. |
| `BOX_COLUMN_MIN` | `SHAPE` | Topology/support | Minimum image column index covered by the node support. |
| `BOX_COLUMN_MAX` | `SHAPE` | Topology/support | Maximum image column index covered by the node support. |
| `BOX_ROW_MIN` | `SHAPE` | Topology/support | Minimum image row index covered by the node support. |
| `BOX_ROW_MAX` | `SHAPE` | Topology/support | Maximum image row index covered by the node support. |

## Distance-Transform Attributes

These attributes describe distances from support pixels to the foreground
4-connected contour. Contour pixels have distance zero. `DIST_TRANSF` uses the
approximate field; `DIST_TRANSF_EXACT` uses exact Euclidean distances.

`MAX_DIST` is measured in pixels and equals the square root of
`MAX_SQUARED_DIST`, which is measured in squared pixels. The same relation
holds for `MAX_DIST_EXACT` and `MAX_SQUARED_DIST_EXACT`.

### Approximate Distances

| Attribute | Groups | Contract | Use |
| --- | --- | --- | --- |
| `MAX_DIST` | `SHAPE`, `DIST_TRANSF` | Topology/support | Maximum of the approximate distance field, in pixels. |
| `MAX_SQUARED_DIST` | `DIST_TRANSF` | Topology/support | Maximum approximate squared contour-distance cost, in squared pixels. |
| `DIST_SQUARED_SUM` | `DIST_TRANSF` | Topology/support | Sum of the approximate squared-distance field over the node support. |
| `DIST_SQUARED_MEAN` | `DIST_TRANSF` | Topology/support | Arithmetic mean of the approximate squared-distance field over the node support. |
| `DIST_RMS` | `DIST_TRANSF` | Topology/support | Square root of the mean approximate squared contour distance, in pixels. |
| `DIST_SQUARED_VARIANCE` | `DIST_TRANSF` | Topology/support | Population variance of the approximate squared-distance samples. |
| `MAX_DIST_CENTER_ROW` | `DIST_TRANSF` | Topology/support | Zero-based row of a support pixel attaining the approximate maximum; ties select the smallest row-major pixel. |
| `MAX_DIST_CENTER_COLUMN` | `DIST_TRANSF` | Topology/support | Zero-based column of the same deterministic approximate maximum-distance center. |
| `MAX_DIST_PLATEAU_AREA` | `DIST_TRANSF` | Topology/support | Number of support pixels attaining the approximate maximum. |
| `MAX_DIST_PLATEAU_CENTROID_ROW` | `DIST_TRANSF` | Topology/support | Arithmetic mean of row coordinates over the approximate maximum plateau. |
| `MAX_DIST_PLATEAU_CENTROID_COLUMN` | `DIST_TRANSF` | Topology/support | Arithmetic mean of column coordinates over the approximate maximum plateau. |
| `DIST_SUM` | `DIST_TRANSF` | Topology/support | Sum of approximate Euclidean contour distances over the node support, in accumulated pixel units. |
| `DIST_MEAN` | `DIST_TRANSF` | Topology/support | Arithmetic mean of Euclidean distances over the approximate field, in pixels. |
| `DIST_VARIANCE` | `DIST_TRANSF` | Topology/support | Population variance of approximate Euclidean contour distances, in squared pixels. |
| `DIST_MEDIAN` | `DIST_TRANSF` | Topology/support | Lower empirical median of the approximate Euclidean-distance field. |
| `DIST_MODE` | `DIST_TRANSF` | Topology/support | Most frequent approximate Euclidean distance; ties select the smallest distance. |
| `DIST_Q25` | `DIST_TRANSF` | Topology/support | Lower empirical 25th percentile of the approximate Euclidean-distance field. |
| `DIST_Q75` | `DIST_TRANSF` | Topology/support | Lower empirical 75th percentile of the approximate Euclidean-distance field. |
| `DIST_Q90` | `DIST_TRANSF` | Topology/support | Lower empirical 90th percentile of the approximate Euclidean-distance field. |
| `DIST_ENTROPY` | `DIST_TRANSF` | Topology/support | Shannon entropy in bits of the normalized approximate squared-distance histogram. |
| `DIST_POSITIVE_AREA` | `DIST_TRANSF` | Topology/support | Number of support pixels whose approximate squared-distance cost is positive. |
| `DIST_LEVEL_COUNT` | `DIST_TRANSF` | Topology/support | Number of represented approximate squared-distance levels, including zero. |
| `DIST_WEIGHTED_CENTROID_ROW` | `DIST_TRANSF` | Topology/support | Zero-based row centroid weighted by approximate Euclidean distance. All-zero fields use the ordinary support centroid. |
| `DIST_WEIGHTED_CENTROID_COLUMN` | `DIST_TRANSF` | Topology/support | Zero-based column centroid weighted by approximate Euclidean distance. All-zero fields use the ordinary support centroid. |
| `DIST_WEIGHTED_CENTRAL_MOMENT_20` | `DIST_TRANSF` | Topology/support | Unnormalized approximate distance-weighted central column moment. |
| `DIST_WEIGHTED_CENTRAL_MOMENT_02` | `DIST_TRANSF` | Topology/support | Unnormalized approximate distance-weighted central row moment. |
| `DIST_WEIGHTED_CENTRAL_MOMENT_11` | `DIST_TRANSF` | Topology/support | Unnormalized approximate distance-weighted mixed row-column moment. |
| `DIST_WEIGHTED_AXIS_ORIENTATION` | `DIST_TRANSF` | Topology/support | Principal axis of the approximate distance-weighted second moments, in degrees; isotropic fields return zero. |
| `DIST_WEIGHTED_ECCENTRICITY` | `DIST_TRANSF` | Topology/support | Major/minor eigenvalue ratio of the approximate distance-weighted second moments, capped at `1e6`. |

### Exact Distances

| Attribute | Groups | Contract | Use |
| --- | --- | --- | --- |
| `MAX_DIST_EXACT` | `SHAPE`, `DIST_TRANSF_EXACT` | Topology/support | Maximum exact Euclidean distance, in pixels, from a support pixel to the foreground 4-connected contour. |
| `MAX_SQUARED_DIST_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Maximum exact squared Euclidean contour distance, in squared pixels. |
| `DIST_SQUARED_SUM_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Sum of the exact squared Euclidean distance to the foreground A4 contour over every support pixel. |
| `DIST_SQUARED_MEAN_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Arithmetic mean of the exact squared contour-distance field over the node support. |
| `DIST_RMS_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Root mean square Euclidean contour distance, `sqrt(mean(d^2))`, in pixel units. |
| `DIST_SQUARED_VARIANCE_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Population variance of the exact squared contour-distance samples, `mean((d^2)^2) - mean(d^2)^2`, in fourth-power pixel units. |
| `MAX_DIST_CENTER_ROW_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Zero-based row of a support pixel attaining `MAX_DIST_EXACT`; ties select the smallest row-major pixel. |
| `MAX_DIST_CENTER_COLUMN_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Zero-based column of the same deterministic exact maximum-distance center. |
| `MAX_DIST_PLATEAU_AREA_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Number of support pixels attaining the exact maximum squared contour distance. |
| `MAX_DIST_PLATEAU_CENTROID_ROW_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Arithmetic mean of the row coordinates over the exact maximum-distance plateau. |
| `MAX_DIST_PLATEAU_CENTROID_COLUMN_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Arithmetic mean of the column coordinates over the exact maximum-distance plateau. |
| `DIST_SUM_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Sum of exact Euclidean contour distances over the support, in accumulated pixel units. |
| `DIST_MEAN_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Arithmetic mean of exact Euclidean contour distances, in pixels. |
| `DIST_VARIANCE_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Population variance of exact Euclidean contour distances, in squared pixels. |
| `DIST_MEDIAN_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Lower empirical median of the exact Euclidean-distance field. |
| `DIST_MODE_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Most frequent exact Euclidean distance; ties select the smallest distance. |
| `DIST_Q25_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Lower empirical 25th percentile of the exact Euclidean-distance field. |
| `DIST_Q75_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Lower empirical 75th percentile of the exact Euclidean-distance field. |
| `DIST_Q90_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Lower empirical 90th percentile of the exact Euclidean-distance field. |
| `DIST_ENTROPY_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Shannon entropy in bits of the normalized exact squared-distance histogram. |
| `DIST_POSITIVE_AREA_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Number of support pixels whose exact squared contour distance is positive. |
| `DIST_LEVEL_COUNT_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Number of represented exact squared-distance levels, including zero. |
| `DIST_WEIGHTED_CENTROID_ROW_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Zero-based row centroid weighted by exact Euclidean distance. All-zero fields use the ordinary support centroid. |
| `DIST_WEIGHTED_CENTROID_COLUMN_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Zero-based column centroid weighted by exact Euclidean distance. All-zero fields use the ordinary support centroid. |
| `DIST_WEIGHTED_CENTRAL_MOMENT_20_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Unnormalized exact distance-weighted central column moment. |
| `DIST_WEIGHTED_CENTRAL_MOMENT_02_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Unnormalized exact distance-weighted central row moment. |
| `DIST_WEIGHTED_CENTRAL_MOMENT_11_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Unnormalized exact distance-weighted mixed row-column moment. |
| `DIST_WEIGHTED_AXIS_ORIENTATION_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Principal axis of the exact distance-weighted second moments, in degrees; isotropic fields return zero. |
| `DIST_WEIGHTED_ECCENTRICITY_EXACT` | `DIST_TRANSF_EXACT` | Topology/support | Major/minor eigenvalue ratio of the exact distance-weighted second moments, capped at `1e6`. |

## Filled-Shape Attributes

These descriptors measure the region enclosed by the unique external boundary,
including its holes. Each node must have exactly one external boundary.
`HOLE_AREA_FRACTION` and `FILLED_CENTROID_DISPLACEMENT_NORMALIZED` compare that
filled region with the original support.

| Attribute | Groups | Contract | Use |
| --- | --- | --- | --- |
| `FILLED_AREA` | `FILLED_SHAPE` | Topology/support | Number of pixels enclosed by the unique external boundary, including holes; requires exactly one external boundary per live node. |
| `FILLED_CENTROID_ROW` | `FILLED_SHAPE` | Topology/support | Zero-based mean row of the filled region pixels. |
| `FILLED_CENTROID_COLUMN` | `FILLED_SHAPE` | Topology/support | Zero-based mean column of the filled region pixels. |
| `FILLED_LENGTH_MAJOR_AXIS` | `FILLED_SHAPE` | Topology/support | Major-axis length proxy of the filled region, in pixels; follows `LENGTH_MAJOR_AXIS`. |
| `FILLED_LENGTH_MINOR_AXIS` | `FILLED_SHAPE` | Topology/support | Minor-axis length proxy of the filled region, in pixels; follows `LENGTH_MINOR_AXIS`. |
| `FILLED_AXIS_ORIENTATION` | `FILLED_SHAPE` | Topology/support | Non-negative principal column-axis orientation in degrees; isotropic regions return `0`. |
| `FILLED_ECCENTRICITY` | `FILLED_SHAPE` | Topology/support | Major/minor eigenvalue ratio of the filled region moments, capped at `1e6`; one pixel returns `1`. |
| `FILLED_INERTIA` | `FILLED_SHAPE` | Topology/support | Sum of the filled region second central moments divided by squared filled area. |
| `HOLE_AREA_FRACTION` | `FILLED_SHAPE` | Topology/support | Fraction of filled area occupied by holes: `1 - AREA / FILLED_AREA`. |
| `FILLED_CENTROID_DISPLACEMENT_NORMALIZED` | `FILLED_SHAPE` | Topology/support | Euclidean distance between support and filled-region centroids divided by the square root of filled area. |
| `FILLED_COMPACTNESS` | `FILLED_SHAPE` | Topology/support | `FILLED_AREA / (2*pi*(mu20 + mu02))`, using filled-region second central moments; zero dispersion returns `0`. |
| `FILLED_CIRCULARITY` | `FILLED_SHAPE` | Topology/support | Minor/major eigenvalue ratio of the filled-region second moments; one pixel returns `1` and a nontrivial line returns `0`. |

## Moment Attributes

| Attribute | Groups | Contract | Use |
| --- | --- | --- | --- |
| `CENTRAL_MOMENT_20` | `MOMENTS`, `SHAPE` | Topology/support | Second-order central moment `mu20` around the support centroid. |
| `CENTRAL_MOMENT_02` | `MOMENTS`, `SHAPE` | Topology/support | Second-order central moment `mu02` around the support centroid. |
| `CENTRAL_MOMENT_11` | `MOMENTS`, `SHAPE` | Topology/support | Mixed second-order central moment `mu11`. |
| `CENTRAL_MOMENT_30` | `MOMENTS`, `SHAPE` | Topology/support | Third-order central moment `mu30`. |
| `CENTRAL_MOMENT_03` | `MOMENTS`, `SHAPE` | Topology/support | Third-order central moment `mu03`. |
| `CENTRAL_MOMENT_21` | `MOMENTS`, `SHAPE` | Topology/support | Mixed third-order central moment `mu21`. |
| `CENTRAL_MOMENT_12` | `MOMENTS`, `SHAPE` | Topology/support | Mixed third-order central moment `mu12`. |
| `HU_MOMENT_1` | `MOMENTS`, `SHAPE` | Topology/support | First Hu moment invariant computed from normalized central moments. |
| `HU_MOMENT_2` | `MOMENTS`, `SHAPE` | Topology/support | Second Hu moment invariant. |
| `HU_MOMENT_3` | `MOMENTS`, `SHAPE` | Topology/support | Third Hu moment invariant. |
| `HU_MOMENT_4` | `MOMENTS`, `SHAPE` | Topology/support | Fourth Hu moment invariant. |
| `HU_MOMENT_5` | `MOMENTS`, `SHAPE` | Topology/support | Fifth Hu moment invariant. |
| `HU_MOMENT_6` | `MOMENTS`, `SHAPE` | Topology/support | Sixth Hu moment invariant. |
| `HU_MOMENT_7` | `MOMENTS`, `SHAPE` | Topology/support | Seventh Hu moment invariant. |
| `INERTIA` | `MOMENTS`, `SHAPE` | Topology/support | Sum of normalized second-order central moments, `mu20 / area^2 + mu02 / area^2`. |
| `COMPACTNESS` | `MOMENTS`, `SHAPE` | Topology/support | Area normalized by second-order dispersion, `(1 / (2*pi)) * area / (mu20 + mu02)` when the denominator is positive. |
| `ECCENTRICITY` | `MOMENTS`, `SHAPE` | Topology/support | Ratio of the largest to smallest eigenvalue of the second-moment matrix. |
| `LENGTH_MAJOR_AXIS` | `MOMENTS`, `SHAPE` | Topology/support | Length proxy for the major axis of the equivalent second-moment ellipse, derived from the largest inertia eigenvalue and area. |
| `LENGTH_MINOR_AXIS` | `MOMENTS`, `SHAPE` | Topology/support | Length proxy for the minor axis of the equivalent second-moment ellipse, derived from the smallest inertia eigenvalue and area. |
| `AXIS_ORIENTATION` | `MOMENTS`, `SHAPE` | Topology/support | Principal-axis orientation in degrees, `0.5 * atan2(2*mu11, mu20 - mu02)`, normalized to a non-negative angle. |
| `CIRCULARITY` | `MOMENTS`, `SHAPE` | Topology/support | Ratio `lambda2 / lambda1` of second-moment eigenvalues. |

Moment descriptors are useful when area alone is too coarse and the filter
should react to shape geometry.

## Boundary and Bitquad Attributes

| Attribute | Groups | Contract | Use |
| --- | --- | --- | --- |
| `BITQUAD_AREA` | `BOUNDARY`, `SHAPE` | Topology/support | Duda-style sub-pixel area estimator derived from aggregated `2x2` bitquad pattern counts. |
| `BITQUAD_NUMBER_EULER` | `BOUNDARY`, `SHAPE` | Topology/support | Euler-characteristic estimate from bitquad counts under the tree connectivity. |
| `BITQUAD_NUMBER_HOLES` | `BOUNDARY`, `SHAPE` | Topology/support | Number of holes inferred from the bitquad Euler characteristic for a single connected support. |
| `BITQUAD_PERIMETER` | `BOUNDARY`, `SHAPE` | Topology/support | Discrete boundary-length estimate from bitquad edge-contributing patterns. |
| `BITQUAD_PERIMETER_CONTINUOUS` | `BOUNDARY`, `SHAPE` | Topology/support | Smoothed continuous perimeter estimate from bitquad counters, using weighted transitions across local `2x2` configurations. |
| `BITQUAD_CIRCULARITY` | `BOUNDARY`, `SHAPE` | Topology/support | Bitquad compactness measure `(4*pi*BITQUAD_AREA) / BITQUAD_PERIMETER_CONTINUOUS^2`. |
| `BITQUAD_PERIMETER_AVERAGE` | `BOUNDARY`, `SHAPE` | Topology/support | Average continuous perimeter per connected component, computed from bitquad perimeter and Euler-count estimates. |
| `BITQUAD_LENGTH_AVERAGE` | `BOUNDARY`, `SHAPE` | Topology/support | Average longitudinal extent proxy, derived as half of the average continuous perimeter. |
| `BITQUAD_WIDTH_AVERAGE` | `BOUNDARY`, `SHAPE` | Topology/support | Average transverse extent proxy, computed as `2 * BITQUAD_AREA / BITQUAD_PERIMETER_CONTINUOUS` with a zero fallback when the continuous perimeter is degenerate. |
| `CONTOUR_PIXELS` | `BOUNDARY`, `SHAPE` | Topology/support | Number of support pixels that touch the 4-neighbor complement by at least one side. |
| `CONTOUR_PERIMETER` | `BOUNDARY`, `SHAPE` | Topology/support | Total number of exposed 4-neighbor sides over the support. |
| `CONTOUR_SIDE_NORTH` | `BOUNDARY`, `SHAPE` | Topology/support | Number of exposed north-facing sides over support pixels. |
| `CONTOUR_SIDE_WEST` | `BOUNDARY`, `SHAPE` | Topology/support | Number of exposed west-facing sides over support pixels. |
| `CONTOUR_SIDE_EAST` | `BOUNDARY`, `SHAPE` | Topology/support | Number of exposed east-facing sides over support pixels. |
| `CONTOUR_SIDE_SOUTH` | `BOUNDARY`, `SHAPE` | Topology/support | Number of exposed south-facing sides over support pixels. |

Boundary attributes are useful when perimeter, holes, or contour exposure are
more informative than raw area.

## Tree-Topology Attributes

| Attribute | Groups | Contract | Use |
| --- | --- | --- | --- |
| `SUBTREE_HEIGHT` | `TREE_TOPOLOGY` | Tree topology | Longest child-edge path from the node to any leaf in its subtree. |
| `DEPTH_NODE` | `TREE_TOPOLOGY` | Tree topology | Number of parent-edge steps from the node to the root. |
| `IS_LEAF_NODE` | `TREE_TOPOLOGY` | Tree topology | Boolean scalar encoded as `1` when the node has no children and `0` otherwise. |
| `IS_ROOT_NODE` | `TREE_TOPOLOGY` | Tree topology | Boolean scalar encoded as `1` for the tree root and `0` for all other nodes. |
| `NUM_CHILDREN_NODE` | `TREE_TOPOLOGY` | Tree topology | Number of direct child nodes. |
| `NUM_SIBLINGS_NODE` | `TREE_TOPOLOGY` | Tree topology | Number of other nodes sharing the same parent. |
| `NUM_DESCENDANTS_NODE` | `TREE_TOPOLOGY` | Tree topology | Number of internal tree nodes strictly below this node in its subtree. |
| `NUM_LEAF_DESCENDANTS_NODE` | `TREE_TOPOLOGY` | Tree topology | Number of leaf nodes in the node subtree. |
| `LEAF_RATIO_NODE` | `TREE_TOPOLOGY` | Tree topology | Ratio of leaf descendants to subtree size, `leaf_descendants / (descendants + 1)`. |
| `BALANCE_NODE` | `TREE_TOPOLOGY` | Tree topology | Difference between maximum and minimum child-subtree heights. |
| `AVG_CHILD_HEIGHT_NODE` | `TREE_TOPOLOGY` | Tree topology | Average height of direct child subtrees. |

Tree-topology attributes ignore image coordinates and gray levels. They are
useful when the hierarchy shape itself carries the signal.

## Choosing Attributes

Start small:

- use `AREA` for size filtering;
- add `GRAY_LEVEL_HEIGHT` or `VOLUME` when contrast matters;
- add `COMPACTNESS`, `ECCENTRICITY`, or `RATIO_WH` when component shape
  matters;
- add `TREE_TOPOLOGY` attributes when hierarchy position matters;
- use `BOUNDARY` attributes for contour- or hole-sensitive filters;
- use `DIST_TRANSF` or `DIST_TRANSF_EXACT` for interior-distance descriptors;
- use `FILLED_SHAPE` to describe enclosed geometry and the effect of holes.

For CFP layers, each scalar attribute becomes an input feature for the scoring
model after group expansion.

```python
filter_spec = {
    "name": "area_contrast_shape",
    "tree_type": "max-tree",
    "attributes": [
        morphology.AttributeType.AREA,
        morphology.AttributeType.GRAY_LEVEL_HEIGHT,
        morphology.AttributeType.COMPACTNESS,
    ],
}
```
