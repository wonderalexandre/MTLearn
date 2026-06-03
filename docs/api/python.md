# Python API Map

Python users should prefer the high-level package modules over native binding
classes.

## `mtlearn.morphology`

Stable facade for tree construction, attribute computation, attribute group
expansion, and attribute filters:

- `create_max_tree(image)`
- `create_min_tree(image)`
- `create_tree_of_shapes(image, ...)`
- `build_tree(image, tree_type, ...)`
- `compute_attributes(tree, attributes)`
- `compute_single_attribute(tree, attribute)`
- `describe_attribute(attribute)`
- `describe_all_attributes()`
- `expand_attribute_group(group)`
- `create_attribute_filter(tree)`

The facade accepts NumPy `uint8` 2D images for tree construction and returns
native tree handles that should be treated as `mtlearn.morphology.WeightedTree`
objects.

## `mtlearn.layers`

The primary trainable layer is
`mtlearn.layers.ConnectedFilterPreprocessingLayer`.

It is configured by `filter_specs`; each spec defines one output per input
channel, including tree type, scoring attributes, optional tree-of-shapes
interpolation settings, and optional `CFPValuation`.

Reference and compatibility layers remain importable but are not the preferred
entry point for new experiments.

## Internal Modules

`mtlearn._native`, `mtlearn._backends`, and native pybind classes are
implementation details. They can appear in generated help output because the
Python facade delegates to them, but user code should depend on the public
facade names above.
