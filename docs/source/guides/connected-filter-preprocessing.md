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
from torch.utils.data import DataLoader, TensorDataset

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
)

x = torch.rand(4, 1, 32, 32)
dataset = TensorDataset(x, torch.zeros(len(x)))
loader = DataLoader(dataset, batch_size=4, shuffle=False)
cached_loader = layer.build_dataloader_cached(loader)

for batch_inputs, _ in cached_loader:
    y = layer(batch_inputs)

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

## Scoring Models

Each spec has a scoring model that maps normalized node attributes to one score
per tree node. The default is the layer-owned linear sigmoid gate:

```python
linear_spec = {
    "name": "area_linear",
    "tree_type": "max-tree",
    "attributes": [morphology.AttributeType.AREA],
    "scoring": {"kind": "linear_sigmoid"},
}
```

The linear default keeps trainable parameters under `_weights.<spec_name>` and
`_biases.<spec_name>`, which makes simple linear filters easy to inspect and
export.

Use an MLP scorer when the keep/discard criterion should combine attributes
nonlinearly:

```python
mlp_spec = {
    "name": "shape_mlp",
    "tree_type": "max-tree",
    "attributes": [
        morphology.AttributeType.AREA,
        morphology.AttributeType.COMPACTNESS,
    ],
    "scoring": {
        "kind": "mlp",
        "hidden_units": [8],
        "activation": "tanh",
    },
}
```

MLP parameters are owned by the scorer module and appear in
`get_parameter_contract()["scoring_models"]`.

## Altitude Signal

Scoring decides which nodes contribute to the reconstructed altitude-residue
signal:

```python
altitude_spec = {
    "name": "altitude",
    "tree_type": "max-tree",
    "attributes": [morphology.AttributeType.AREA],
}
```

CFP does not expose alternative signal projections as a Python extension point.
The forward signal is fixed to morphology-tree altitude residues.


## Constraints and Regularizers

Constraints post-process scores before reconstruction. The current built-in
constraint preserves the root score:

```python
constrained_spec = {
    "name": "preserve_root_area",
    "tree_type": "max-tree",
    "attributes": [morphology.AttributeType.AREA],
    "constraints": [{"kind": "preserve_root"}],
}
```

Regularizers add training penalties. They are not included in the inference
contract, so changing a training regularizer does not change forward
semantics.

```python
regularized_spec = {
    "name": "monotone_area",
    "tree_type": "max-tree",
    "attributes": [morphology.AttributeType.AREA],
    "regularizers": [{"kind": "edge_score_monotonicity", "weight": 0.1}],
}

layer = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    filter_specs=[regularized_spec],
)

# Use the same batch_inputs object yielded by build_dataloader_cached(...).
loss = task_loss + layer.regularization_penalty(batch_inputs)
```

Other registered morphological regularizers include
`attribute_order_score_monotonicity`, which penalizes score inversions after sorting
nodes by one normalized attribute, and `path_score_monotonicity`, which penalizes
descendants that score higher than their ancestors.

## Extension Registries

The `mtlearn.layers.cfp` package exposes the default registries used by the
layer:

- `SCORING_MODEL_REGISTRY` for `ScoringModel` factories;
- `SCORE_CONSTRAINT_REGISTRY` for score post-processing constraints;
- `REGULARIZER_REGISTRY` for training penalties.

Register a new component kind by implementing the matching CFP interface and a
factory that accepts serializable config fields. Specs can then reference
registry-backed scoring models, constraints, and regularizers with
`{"kind": "your_kind", ...}`.

## Configs and Contracts

`get_config()` stores the architecture needed by `from_config()`. It includes
tree type, attributes, scoring, constraints, normalization, and training-only
regularizer settings.

```python
config = layer.get_config()
restored = ConnectedFilterPreprocessingLayer.from_config(config)
```

Use named contracts when comparing checkpoints or exported parameters:

```python
contracts = layer.get_contracts()
inference = contracts["inference_contract"]
parameters = contracts["parameter_contract"]
training = contracts["training_contract"]
```

## Normalization and Caching

The default `scale_mode` is `"dataset_clipped_zscore01"`. It uses dataset-level z-score
statistics, clips values to `[-clipped_zscore_radius, clipped_zscore_radius]`, and rescales them into a
positive interval controlled by `clipped_zscore_floor`.

For statistical modes (`"dataset_clipped_zscore01"`, `"dataset_minmax01"`, and `"dataset_zscore"`), fit or
load normalization statistics before normal forward passes. Use
`build_dataloader_cached` on the training split to estimate statistics and
precompute tree payloads.

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=16, shuffle=False)
cached_loader = layer.build_dataloader_cached(loader)

for (x, idx), target in cached_loader:
    y = layer((x, idx))
```

For smoke tests or diagnostics that intentionally avoid a stats prepass, use
`scale_mode="none"` explicitly. In this mode, the scorer receives raw
attributes and the caller is responsible for their scale. Do not use this as
the default training setup for attribute-comparable experiments.

```python
debug_layer = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    filter_specs=filter_specs,
    scale_mode="none",
)
```

## Initialization

Initialize scoring models close to identity when CFP should start by preserving
the input image.

```python
layer.init_identity(p0=0.995)
```

Use an identity-like initialization when CFP is placed before a pretrained or
sensitive downstream network. Use random initialization when the preprocessing
block is meant to discover strong filtering behavior from scratch.

## Inference

`predict` temporarily switches to evaluation mode, runs without gradients, and
uses a caller-provided score sharpness. A large `score_sharpness` makes gates
closer to hard decisions.

```python
with torch.no_grad():
    y_soft = layer(x)
    y_hard = layer.predict(x, score_sharpness=1000.0)
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

layer.export_params("cfp-params.pt")
```

For full model checkpoints, use the helpers documented in
{doc}`pytorch-integration`.
