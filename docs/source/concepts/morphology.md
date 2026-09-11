# Morphology Concepts

This page defines the terms used by the public `mtlearn` API and the CFP
layer documentation.

## Morphological Trees

The library represents image hierarchies as rooted trees of connected subsets,
referred to here as *morphological trees*. These trees serve as image models and
share the same topology model. The currently supported tree types are *max-tree*,
*min-tree*, and *tree of shapes*.

## Max-Tree and Min-Tree

A max-tree organizes connected components of upper level sets of a gray-scale
image. It is useful when bright components are the primary structures of
interest. A min-tree organizes connected components of lower level sets and is
useful when dark components are the primary structures of interest.

Both tree types are component trees. Their construction accepts an adjacency
radius, exposed as `radius` in the native tree constructors.

## Tree of Shapes

A tree of shapes is self-dual: it represents bright and dark structures in one
hierarchy instead of choosing max-tree or min-tree polarity. The current API
exposes interpolation choices with `ToSInterpolation` and string aliases such
as `"self-dual"`, `"min4c-max8c"`, and `"min8c-max4c"`.

## Node Ids and Node Slots

`NodeId` values identify nodes in the morphology backend. Some arrays are
indexed by live tree nodes, while attribute and CFP arrays usually use the
backend node-slot domain. When allocating a node-indexed criterion, attribute
vector, or CFP tensor, use `num_internal_node_slots` rather than `num_nodes`.

The distinction matters after pruning or merging: a tree can have inactive
slots even when the live node count changes.

## Proper Parts

The proper part of a node is the set of pixels in its support that do not
belong to the support of any child. Each pixel belongs to one proper part.
`smallest_node(pixel_id)` returns the node whose proper part contains that
pixel.

## Scalar Attributes and Groups

A scalar attribute is one node-level measurement such as `AREA`,
`GRAY_LEVEL_HEIGHT`, `COMPACTNESS`, or `CONTOUR_PERIMETER`.

An attribute group is a named bundle of scalar attributes, such as
`AttributeGroup.SHAPE` or `AttributeGroup.TREE_TOPOLOGY`. Groups are expanded
before attribute computation or CFP parameter construction.

See {doc}`attributes` for the user-facing attribute catalog.
