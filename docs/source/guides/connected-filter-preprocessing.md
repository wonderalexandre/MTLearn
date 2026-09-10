# Connected Filter Preprocessing

`ConnectedFilterPreprocessingLayer` turns morphology-tree attribute filtering
into a trainable PyTorch module. It builds a tree for each sample/channel,
computes node attributes outside autograd, learns node-wise sigmoid scores, and
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
                morphology.AttributeType.GRAY_LEVEL_HEIGHT,
            ],
        },
    ],
    scale_mode="none",
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
| `attributes` | Yes | One scalar attribute, one group, or a list/tuple of scalar attributes and groups. |
| `scoring` | No | Scoring model or configuration; defaults to `linear_sigmoid`. |
| `score_sharpness` | No | Positive sigmoid sharpness for this spec; defaults to the layer setting. |
| `constraints` | No | Score constraints applied before reconstruction. |
| `regularizers` | No | Penalties added explicitly to the training loss. |
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
per tree node. The default is a linear sigmoid model:

```python
linear_spec = {
    "name": "area_linear",
    "tree_type": "max-tree",
    "attributes": [morphology.AttributeType.AREA],
    "scoring": {"kind": "linear_sigmoid"},
}
```

The linear default keeps trainable parameters under the historical
`_weights.<spec_name>` and `_biases.<spec_name>` names for checkpoint
compatibility.

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

## Reconstructed Signal

Scoring determines each node's contribution to the reconstructed image.
CFP reconstructs the filtered altitude signal from node residues and scores:

```python
altitude_spec = {
    "name": "altitude",
    "tree_type": "max-tree",
    "attributes": [morphology.AttributeType.AREA],
}
```

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
contract, so changing a training regularizer does not invalidate checkpoint
weight compatibility.

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
    scale_mode="none",
)

loss = task_loss + layer.regularization_penalty(x)
```

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
tree type, attributes, scoring, constraints, normalization, and
training-only regularizer settings.

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

The default `scale_mode` is `"dataset_clipped_zscore01"`. It uses dataset-level
z-score statistics, clips values to `[-clipped_zscore_radius,
clipped_zscore_radius]`, and rescales them into
`[clipped_zscore_floor, 1]`. The other statistical modes are
`"dataset_minmax01"` and `"dataset_zscore"`.

All three modes require training-set statistics before normal forward passes.
Use `fit_stats` to compute them without retaining per-image morphology, or
`load_stats` to restore a previous fit. Reuse the training statistics for
validation and test data.

```python
from torch.utils.data import DataLoader

layer = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    filter_specs=filter_specs,
    scale_mode="dataset_clipped_zscore01",
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
layer.fit_stats(train_loader)

for x, target in train_loader:
    y = layer(x)
```

If you also want to cache per-image morphology, use `build_dataloader_cached`
in place of `fit_stats`. It computes statistics and returns a loader with
sample indices for reuse:

```python
cached_loader = layer.build_dataloader_cached(train_loader)

for (x, idx), target in cached_loader:
    y = layer((x, idx))
```

For quick experiments without a statistics pass, use `scale_mode="none"`.
This passes raw attribute values to the scoring model.

```python
debug_layer = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    filter_specs=filter_specs,
    scale_mode="none",
)
```

## Initialization

Use `init_identity` to initialize the built-in scoring models with node scores
close to one.

```python
layer.init_identity(p0=0.995)
```

Use an identity-like initialization when CFP is placed before a pretrained or
sensitive downstream network. Use random initialization when the preprocessing
block is meant to discover strong filtering behavior from scratch.

## Inference

`predict` temporarily switches to evaluation mode, runs without gradients, and
uses the requested `score_sharpness`. A large value makes sigmoid scores
closer to binary preservation decisions.

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
    print(spec_report["scoring_model"])
    if "weight" in spec_report:
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
