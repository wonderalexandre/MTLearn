# Morphology Concepts

This page defines the terms used by the public `mtlearn` API and the CFP
layer documentation.

## Morphological Trees

A morphological tree is a hierarchy of connected components derived from an
image. Each node represents a component or shape, and parent-child relations
encode inclusion in the hierarchy. `mtlearn` currently builds max-trees,
min-trees, and trees of shapes through `mtlearn.morphology`.

## Max-Tree and Min-Tree

A max-tree organizes upper level sets of a gray-scale image. It is useful when
bright components are the primary structures of interest. A min-tree organizes
lower level sets and is useful when dark components are the primary
structures.

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
vector, or CFP tensor, use `numInternalNodeSlots` rather than `numNodes`.

The distinction matters after pruning or merging: a tree can have inactive
slots even when the live node count changes.

## Proper Parts

Proper parts are pixel-level ownership elements used by the backend topology.
For ordinary 2D image use, they can be understood as flattened image pixels
owned by tree nodes. Methods such as `proper_part_owner_of(pixel_id)` expose
this mapping for inspection and debugging.

## Scalar Attributes and Groups

A scalar attribute is one node-level measurement such as `AREA`,
`GRAY_HEIGHT`, `COMPACTNESS`, or `CONTOUR_PERIMETER`.

An attribute group is a named bundle of scalar attributes, such as
`AttributeGroup.SHAPE` or `AttributeGroup.TREE_TOPOLOGY`. Groups are expanded
before attribute computation or CFP parameter construction.

For tree-of-shapes CFP filters, the current backend omits unsupported scalar
members such as explicit `MAX_DIST` when expanding broad groups.

See {doc}`attributes` for the user-facing attribute catalog.

## Dataset Statistics and Caching

The primary CFP layer can cache tree metadata and raw attributes by stable
dataset index. `build_dataloader_cached` wraps a DataLoader so each sample
carries its index, then precomputes tree payloads and dataset-level
normalization statistics. Once the prepass finishes, statistics are frozen and
cached normalized attributes are refreshed.
