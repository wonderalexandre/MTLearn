# Connected Filter Preprocessing

`ConnectedFilterPreprocessingLayer` turns morphology-tree attribute filtering
into a trainable PyTorch module. It builds a tree for each sample/channel,
computes node attributes outside autograd, learns node-wise sigmoid gates, and
reconstructs one output image per input channel and filter spec.

## Minimal Layer

Each filter spec defines a tree type and scoring attributes. The public
examples below use the default reconstructed signal.

```python
import torch
from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer

layer = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    filter_specs=[
        {
            "name": "area_opening",
            "tree_type": morphology.TreeType.MAX_TREE,
            "attributes": [
                morphology.AttributeType.AREA,
                morphology.AttributeType.GRAY_HEIGHT,
            ],
        },
    ],
    scale_mode="minmax01",
)

x = torch.rand(4, 1, 32, 32)
y = layer(x)
assert y.shape == (4, 1, 32, 32)
```

With `N` specs and `C` input channels, output channels are ordered by input
channel first, then spec index. The shape is `(B, C * N, H, W)`.

## Filter Specs

A spec has these user-facing fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | No | Stable key for weights, biases, exported params, and checkpoints. |
| `tree_type` | Yes | `"max-tree"`, `"min-tree"`, `"tree-of-shapes"`, or `TreeType`. |
| `attributes` | Yes | One scalar attribute, one group, or a list/tuple of scalar attributes. |
| `tos_interpolation` | No | Per-spec tree-of-shapes interpolation override. |
| `tos_infinity_seed_row` | No | Per-spec tree-of-shapes infinity seed row. |
| `tos_infinity_seed_col` | No | Per-spec tree-of-shapes infinity seed column. |

Multiple specs can share the same tree. mtlearn caches tree metadata per tree
key and only computes distinct trees once per sample/channel/cache key.

```python
filter_specs = [
    {
        "name": "bright_shape",
        "tree_type": "max-tree",
        "attributes": morphology.AttributeGroup.SHAPE,
    },
    {
        "name": "dark_topology",
        "tree_type": "min-tree",
        "attributes": morphology.AttributeGroup.TREE_TOPOLOGY,
    },
]
```

## Normalization and Caching

The default `scale_mode` is `"hybrid"`. It uses dataset-level z-score
statistics, clips values to `[-hybrid_k, hybrid_k]`, and rescales them into a
positive interval controlled by `hybrid_floor_a`.

For `"hybrid"`, call `build_dataloader_cached` or `load_stats` before normal
forward passes.

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=16, shuffle=False)
cached_loader = layer.build_dataloader_cached(loader)

for (x, idx), target in cached_loader:
    y = layer((x, idx))
```

For quick experiments that should not require a stats prepass, use
`scale_mode="minmax01"`, `"zscore_tree"`, or `"none"`.

```python
debug_layer = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    filter_specs=filter_specs,
    scale_mode="minmax01",
)
```

## Initialization

Two helpers initialize filters close to identity.

```python
layer.init_identity_with_bias(p0=0.995)

# Alternative for hybrid-normalized attributes with positive floor.
layer.init_identity_bias_zero(p0=0.99)
```

Use an identity-like initialization when CFP is placed before a pretrained or
sensitive downstream network. Use random initialization when the preprocessing
block is meant to discover strong filtering behavior from scratch.

## Inference

`predict` temporarily switches to evaluation mode, runs without gradients, and
uses a caller-provided sigmoid gain. A large `beta_f` makes gates closer to
hard decisions.

```python
with torch.no_grad():
    y_soft = layer(x)
    y_hard = layer.predict(x, beta_f=1000.0)
```

## Inspect One Sample

Use `inspect_training_sample` to debug attributes, normalized attributes,
tree payloads, and current trainable parameters.

```python
report = layer.inspect_training_sample(x[0], channel=0, idx=0)

for name, spec_report in report["specs"].items():
    print(name)
    print(spec_report["attributes"])
    print(spec_report["weight"])
    print(spec_report["bias"])
```

## Save Stats and Params

Dataset statistics are separate from ordinary model weights.

```python
layer.save_stats("cfp-stats.pt")
layer.load_stats("cfp-stats.pt")

weights, biases = layer.get_params()
layer.export_params("cfp-params.pt")
```

For full model checkpoints, use the helpers documented in
{doc}`pytorch-integration`.
