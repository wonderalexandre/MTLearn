# Custom Attributes and Filter Specs

This guide focuses on choosing attributes, grouping them into filter specs, and
keeping CFP configurations reproducible.

## Attribute Selection

Start with the smallest attribute set that expresses the prior you need.
For the full attribute list, see {doc}`../concepts/attributes`.

| Intent | Candidate attributes |
| --- | --- |
| Remove small components | `AREA`, `VOLUME` |
| Prefer contrast in the tree | `GRAY_LEVEL_HEIGHT`, `RELATIVE_VOLUME` |
| Prefer elongated or compact shapes | `COMPACTNESS`, `ECCENTRICITY`, `RATIO_WH` |
| Use bounding-box geometry | `BOX_WIDTH`, `BOUNDING_BOX_HEIGHT`, `RECTANGULARITY` |
| Use tree position | `DEPTH_NODE`, `SUBTREE_HEIGHT`, `NUM_CHILDREN_NODE` |
| Use contours | `CONTOUR_PERIMETER`, `CONTOUR_PIXELS` |

Example:

```python
from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer

shape_and_contrast = [
    morphology.AttributeType.AREA,
    morphology.AttributeType.GRAY_LEVEL_HEIGHT,
    morphology.AttributeType.COMPACTNESS,
]
```

## Attribute Groups

Groups expand to scalar attributes before CFP parameters are created.

```python
shape_spec = {
    "name": "shape",
    "tree_type": "max-tree",
    "attributes": morphology.AttributeGroup.SHAPE,
}

mixed_spec = {
    "name": "shape_plus_depth",
    "tree_type": "max-tree",
    "attributes": [
        morphology.AttributeGroup.SHAPE,
        morphology.AttributeType.DEPTH_NODE,
    ],
}
```

For the default linear sigmoid model, a spec has one trainable weight per
scalar attribute after expansion, plus one bias. For an MLP, the parameter
count also depends on its hidden layers.

```python
layer = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    filter_specs=[mixed_spec],
    scale_mode="none",
)

parameter_contract = layer.get_parameter_contract()
print(parameter_contract["weights"]["shape_plus_depth"])
```

## Naming Specs

Use explicit names. They become keys in parameter dictionaries, exported
metadata, checkpoint contracts, and inspection output.

```python
filter_specs = [
    {
        "name": "max_area_contrast",
        "tree_type": "max-tree",
        "attributes": [
            morphology.AttributeType.AREA,
            morphology.AttributeType.GRAY_LEVEL_HEIGHT,
        ],
    },
    {
        "name": "min_area",
        "tree_type": "min-tree",
        "attributes": morphology.AttributeType.AREA,
    },
]
```

Avoid relying on generated names such as `spec_000` in long-running
experiments because reordering specs changes the meaning of saved weights.

## Tree-of-Shapes Specs

Tree-of-shapes specs are useful when the same filter should respond to bright
and dark structures. Pick interpolation explicitly when reproducibility across
experiments matters.

```python
tos_spec = {
    "name": "tos_shape",
    "tree_type": "tree-of-shapes",
    "tos_interpolation": morphology.ToSInterpolation.SELF_DUAL,
    "attributes": [
        morphology.AttributeType.AREA,
        morphology.AttributeType.COMPACTNESS,
    ],
}
```

Groups expand to their complete scalar attribute sets for trees of shapes.
Distance-transform attributes, including `MAX_DIST`, are supported.

## Config Round Trip

Use `get_config` and `from_config` to store the layer architecture separately
from trainable weights.

```python
config = layer.get_config()
restored = ConnectedFilterPreprocessingLayer.from_config(config)
```

This is the same config shape used by checkpoint helpers. It records tree type,
attributes, scoring model, score sharpness, constraints, regularizers,
normalization mode, clamp bounds, and clipped z-score settings.

## Practical Checklist

- Use a small number of specs first; each spec multiplies output channels.
- Name every spec before training.
- Keep max-tree and min-tree specs separate when polarity matters.
- Use tree-of-shapes when polarity should not matter.
- For statistical normalization, fit or load training-set statistics before training.
- Inspect one sample before large experiments to confirm attributes and scores.
