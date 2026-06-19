# Custom Attributes and Filter Specs

This guide focuses on choosing attributes, grouping them into filter specs, and
keeping CFP configurations reproducible.

## Attribute Selection

Start with the smallest attribute set that expresses the prior you need.
For the full attribute list, see {doc}`../concepts/attributes`.

| Intent | Candidate attributes |
| --- | --- |
| Remove small components | `AREA`, `VOLUME` |
| Prefer contrast in the tree | `GRAY_HEIGHT`, `RELATIVE_VOLUME` |
| Prefer elongated or compact shapes | `COMPACTNESS`, `ECCENTRICITY`, `RATIO_WH` |
| Use bounding-box geometry | `BOX_WIDTH`, `BOX_HEIGHT`, `RECTANGULARITY` |
| Use tree position | `DEPTH_NODE`, `HEIGHT_NODE`, `NUM_CHILDREN_NODE` |
| Use contours | `CONTOUR_PERIMETER`, `CONTOUR_PIXELS` |

Example:

```python
from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer

shape_and_contrast = [
    morphology.AttributeType.AREA,
    morphology.AttributeType.GRAY_HEIGHT,
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

The number of trainable weights for a spec equals the number of scalar
attributes after expansion.

```python
layer = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    filter_specs=[mixed_spec],
    scale_mode="minmax01",
)

weights, biases = layer.get_params()
print(weights["shape_plus_depth"].shape)
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
            morphology.AttributeType.GRAY_HEIGHT,
        ],
    },
    {
        "name": "min_area_tophat",
        "tree_type": "min-tree",
        "attributes": morphology.AttributeType.AREA,
    },
]
```

Avoid relying on generated names such as `filter_0` in long-running
experiments because reordering specs changes the meaning of saved weights.

## Tree-of-Shapes Specs

Tree-of-shapes specs are useful when the same filter should respond to bright
and dark structures. Pick interpolation explicitly when reproducibility across
experiments matters.

```python
tos_spec = {
    "name": "tos_shape",
    "tree_type": "tree-of-shapes",
    "tos_interpolation": morphology.ToSInterpolation.SelfDual,
    "attributes": [
        morphology.AttributeType.AREA,
        morphology.AttributeType.COMPACTNESS,
    ],
}
```

The current CFP validation rejects scalar attributes that are undefined for
tree-of-shapes. Broad groups may be expanded with unsupported members removed
when the backend exposes a safe group-level fallback.

## Config Round Trip

Use `get_config` and `from_config` to store the layer architecture separately
from trainable weights.

```python
config = layer.get_config()
restored = ConnectedFilterPreprocessingLayer.from_config(config)
```

This is the same config shape used by checkpoint helpers. It records tree type,
attributes, scoring model, valuation projection, constraints, regularizers,
normalization mode, clamp bounds, and hybrid normalization constants.

## Practical Checklist

- Use a small number of specs first; each spec multiplies output channels.
- Name every spec before training.
- Keep max-tree and min-tree specs separate when polarity matters.
- Use tree-of-shapes when polarity should not matter.
- For `"hybrid"` normalization, build or load dataset stats before training.
- Inspect one sample before large experiments to confirm attributes and gates.
