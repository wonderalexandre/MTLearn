# CFP Regularization Design

This guide describes how CFP training regularizers are designed, configured,
and extended in `mtlearn.layers.cfp`. It focuses on regularization as a
training-only mechanism. For scoring-specific details, see
[CFP Scoring Design](cfp-scoring-design.md). For score post-processing details,
see [CFP Score Constraint Design](cfp-score-constraint-design.md). For the
broader package layout, see [CFP Architecture](cfp-architecture.md).

## What Regularization Does

In CFP, regularization adds a scalar penalty to the training objective. It
observes intermediate CFP quantities, such as node scores and tree metadata,
but it does not change inference by itself.

Typical training loop:

```python
pred = layer(inputs)
task_loss = criterion(pred, targets)
regularization_loss = layer.regularization_penalty(inputs)
loss = task_loss + regularization_loss
```

`regularization_penalty(...)` sums all registered regularizers attached to
active filter specs.

Regularizers are not run automatically by `forward(...)`. The caller must add
the penalty to the training loss explicitly.

## Component Roles

| Component | Input | Output | Learns parameters? | Role |
| --- | --- | --- | --- | --- |
| `ScoringModel` | normalized node attributes | one score per node | yes, when defined by the scorer | defines the soft selection rule |
| `ScoreConstraint` | scores | adjusted scores | no | enforces score-space invariants before reconstruction and regularization |
| `Regularizer` | scores, tree metadata, optional features | scalar penalty | normally no | stabilizes training by penalizing undesired score behavior |
| Task loss | model output and target | scalar loss | no | optimizes the supervised objective |

Regularization should not hide scoring logic. A regularizer should penalize a
property of the learned scores; it should not decide the scores themselves.

## Runtime Contract

The regularization path is:

```text
layer.regularization_penalty(x)
  -> build or reuse tree payloads
  -> compute normalized attributes
  -> ScoringModel(...)
  -> ScoreConstraint(s)
  -> Regularizer(scores, tree_info, features, context)
  -> scalar penalty
```

The base class is:

```python
import torch


class Regularizer(torch.nn.Module):
    def forward(self, scores: torch.Tensor, tree_info, features=None, context=None) -> torch.Tensor:
        raise NotImplementedError
```

Current regularizer requirements:

- accept `scores` with shape `(num_nodes,)`;
- return a scalar tensor;
- keep the result differentiable with respect to scorer parameters;
- keep tensors on compatible devices;
- accept the complete CFP `tree_info` mapping, including `parent`, `tpre`, and
  `tpost`;
- accept `features` and `context` even when unused;
- return zero in a gradient-preserving way when no valid edges or terms exist.

The `scores` passed to a regularizer are already post-constraint scores. For
example, if `preserve_root` is active, the root score seen by the regularizer is
already fixed to `1`.

`context` is a `CFPContext` when the layer calls the regularizer through
`regularization_penalty(...)`. It carries stable metadata for custom
regularizers:

| Field | Meaning |
| --- | --- |
| `sample_key`, `batch_index`, `channel_index` | Location of the channel image inside the current batch or cached loader. |
| `mode` | Current execution mode. For regularizers this is `"regularization_penalty"`. |
| `spec_name`, `spec_index` | Normalized filter-spec identity. |
| `tree_type`, `tree_key` | Tree construction identity used for the payload cache. |
| `attribute_types`, `attribute_names` | Attributes requested by the filter spec, in feature-column order. |
| `image_shape` | Spatial shape `(H, W)` of the channel image. |
| `normalization_mode` | Layer attribute-normalization mode. |
| `score_sharpness` | Effective score sharpness for the current spec. |
| `is_training` | Whether the layer module is in training mode. |
| `raw_attributes`, `normalized_attributes` | Per-attribute node tensors available in the current tree payload. |

Use `features` when a regularizer only needs the normalized feature matrix.
Use `context.raw_attributes` or `context.normalized_attributes` when the
regularizer needs to associate a tensor with a specific attribute type.

The layer averages the accumulated regularization penalty over
`batch_size * channels` and sums the active spec terms. Specs with no effective
regularizers are skipped. If no specs are active, the method returns a zero
penalty without building tree payloads.

## Built-In Regularizers

### Edge Score Monotonicity

`EdgeScoreMonotonicityRegularizer` penalizes child scores that exceed parent scores:

```text
penalty = weight * mean(relu(score(child) - score(parent))^2)
```

Only valid parent-child edges are considered. Inactive nodes are ignored through
the alive-node mask derived from `tpost > tpre`.

Config:

```python
regularized_spec = {
    "name": "monotone_area",
    "tree_type": morphology.TreeType.MAX_TREE,
    "attributes": [morphology.AttributeType.AREA],
    "regularizers": [{"kind": "edge_score_monotonicity", "weight": 0.1}],
}
```

Use `edge_score_monotonicity` when the expected learned filter should become more
selective along tree edges: if a parent component is rejected, descendants
should not become more accepted than the parent.

Do not use `edge_score_monotonicity` as a replacement for task loss, class balancing,
output-image smoothing, or threshold initialization.

### Attribute-Order Monotonicity

`AttributeOrderScoreMonotonicityRegularizer` penalizes score inversions after sorting
nodes by one normalized scoring attribute:

```text
A(i) <= A(j) should imply score(i) <= score(j)
```

For the default increasing direction, the implemented adjacent-order penalty is:

```text
penalty = weight * mean(relu(score_sorted[k] - score_sorted[k + 1])^2)
```

Config:

```python
{
    "regularizers": [
        {
            "kind": "attribute_order_score_monotonicity",
            "weight": 0.05,
            "feature_index": 0,
            "direction": "increasing",
            "min_gap": 0.0,
        }
    ],
}
```

Use it when the scientific hypothesis is an attribute threshold or an
attribute-ordered acceptance rule. The current config uses `feature_index`
because regularizers receive the normalized feature matrix, not the original
attribute names.

### Path Score Monotonicity

`PathScoreMonotonicityRegularizer` penalizes descendants that score higher than
their ancestors:

```text
penalty = weight * mean(relu(score(descendant) - score(ancestor))^2)
```

Config:

```python
{
    "regularizers": [
        {
            "kind": "path_score_monotonicity",
            "weight": 0.05,
            "max_depth": 4,
        }
    ],
}
```

Use it when a whole branch should behave like a coherent pruning. With
`max_depth=1`, this is close to parent-child monotonicity. Larger depths impose
consistency with more distant ancestors. `max_depth=None` checks all ancestors
and is more expensive on deep trees. As with edge monotonicity, inactive nodes
are ignored through the alive-node mask derived from `tpost > tpre`.

## Loss Integration

Regularization is a training objective term:

```python
for batch_inputs, targets in trainloader:
    pred = layer(batch_inputs)
    task_loss = criterion(pred, targets)
    regularization_loss = layer.regularization_penalty(batch_inputs)
    loss = task_loss + regularization_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

When using a cached CFP dataloader, pass the same cached `batch_inputs` object
to both `forward(...)` and `regularization_penalty(...)`. That reuses
precomputed tree payloads and fixed normalization statistics.

Track the terms separately during experiments:

```python
print(
    {
        "task_loss": float(task_loss.detach().cpu()),
        "regularization_loss": float(regularization_loss.detach().cpu()),
    }
)
```

This makes it clear when the regularizer is negligible, dominant, or saturated.

## Notebook Example

One experiment notebook shows regularization as a training-objective term:

- [`CFP_score_regularization_screws.ipynb`](../notebooks/experiments/CFP_score_regularization_screws.ipynb)
  trains two CFP layers on the screws segmentation dataset: one with only
  segmentation loss and one with an `edge_score_monotonicity` regularizer
  added through `regularization_penalty(...)`.

This notebook is a pedagogical regularization example. It is not a benchmark.
It is intended to make the regularizer config, the training-loop addition, and
the score-violation diagnostics easy to inspect.

## Config And Registration

Regularizer specs are config-backed. The current accepted forms are:

```python
# No regularizers.
"regularizers" not in spec

# One regularizer as a mapping.
"regularizers": {"kind": "edge_score_monotonicity", "weight": 0.1}

# One regularizer as a kind string.
"regularizers": "edge_score_monotonicity"

# Multiple regularizers.
"regularizers": [
    {"kind": "edge_score_monotonicity", "weight": 0.1},
    {"kind": "attribute_order_score_monotonicity", "weight": 0.05},
    {"kind": "path_score_monotonicity", "weight": 0.05, "max_depth": 4},
]
```

Unknown kinds raise `ValueError`. Factories should reject unsupported options
rather than silently ignoring them.

To add a registry-backed regularizer, implement a `Regularizer` subclass:

```python
class ScoreEntropyRegularizer(Regularizer):
    def __init__(self, weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.weight = float(weight)
        self.eps = float(eps)

    def forward(self, scores, tree_info, features=None, context=None):
        clipped = torch.clamp(scores, self.eps, 1.0 - self.eps)
        entropy = -(clipped * torch.log(clipped) + (1.0 - clipped) * torch.log(1.0 - clipped))
        return self.weight * entropy.mean()
```

Register a factory:

```python
def create_score_entropy_regularizer(*, weight: float = 1.0, eps: float = 1e-6, **options):
    if options:
        names = ", ".join(sorted(options))
        raise ValueError(f"unsupported score_entropy options: {names}")
    return ScoreEntropyRegularizer(weight=weight, eps=eps)


REGULARIZER_REGISTRY.register(
    "score_entropy",
    create_score_entropy_regularizer,
)
```

Then specs can use:

```python
{
    "name": "area_with_entropy",
    "tree_type": "max-tree",
    "attributes": [morphology.AttributeType.AREA],
    "regularizers": [{"kind": "score_entropy", "weight": 0.01}],
}
```

Direct regularizer objects are not currently accepted through filter-spec
configs. Use the registry path when a regularizer must round-trip through
`get_config()` and `from_config()`.

## Serialization And Contracts

Regularizers are training-only contract entries. `get_config()` includes
registry-backed regularization settings under `regularizers`:

```python
{
    "name": "monotone_area",
    "regularizers": [{"kind": "edge_score_monotonicity", "weight": 0.1}],
}
```

`get_training_contract()` also reports training-only settings. The inference
contract intentionally does not include regularizers:

```text
regularizer change -> training behavior changes
regularizer change -> inference contract unchanged
regularizer change -> checkpoint inference compatibility unchanged
```

This means a model checkpoint can remain inference-compatible even when training
regularization settings change. It does not mean the trained parameters will be
scientifically comparable across experiments; record regularizer configs in
experiment metadata.

## Design Guidelines

Use regularization to encode a training preference, not a hard inference rule.
If a behavior must always hold at inference, prefer a `ScoreConstraint`.

Good regularizer candidates:

- monotonicity preferences over tree edges;
- attribute-order preferences over normalized features;
- ancestor-descendant consistency along branches;
- penalties that keep nonlinear scorers stable on small datasets;
- score-shape penalties such as sparsity or entropy;
- penalties tied to normalized node features.

Poor regularizer candidates:

- losses that need the target image;
- pixel-space losses on reconstructed output;
- behaviors that should modify scores before reconstruction;
- trainable models with parameters that need checkpoint contracts.

Weight-selection guidance:

- start with `0.0` and verify the unregularized baseline;
- add a small weight and compare task loss and regularization loss magnitudes;
- report both terms during training;
- increase weight only when the regularizer changes the undesired score
  behavior without dominating the task loss;
- use fixed weights when comparing scorer families.

## Interaction With Other Components

Scoring controls the values that regularizers observe. Saturated scores can
make a regularizer ineffective because gradients become small.

Constraints run before regularizers. This matters for `preserve_root`: the root
score is fixed to `1` before `edge_score_monotonicity` evaluates parent-child edges.

Regularizers do not receive altitude increments or reconstructed output. If the
penalty must depend on output pixels, keep it as a task-side loss outside CFP
regularizers.

Attribute normalization affects any regularizer that uses `features`. Document
whether the regularizer assumes `dataset_minmax01`, `dataset_clipped_zscore01`,
raw attributes from `scale_mode="none"`, or a normalization-independent
formulation.

## Extension Checklist

When adding a regularizer, verify:

- constructor validation for weights and hyperparameters;
- scalar output shape;
- zero-term behavior on degenerate trees;
- gradient propagation to scorer parameters;
- device compatibility for CPU, CUDA, and MPS tensors when available;
- config validation and registry rejection of unknown options;
- `get_config()` and `from_config()` round trip;
- absence from `get_inference_contract()`;
- interaction with `preserve_root` when tree-edge logic is involved;
- behavior with cached dataloaders and direct image tensors.

Run at least:

```bash
python -m compileall -q mtlearn/python/mtlearn mtlearn/tests/python
PYTHONPATH=mtlearn/python:build/mtlearn/bindings python -m pytest -q mtlearn/tests/python/test_cfp_components.py mtlearn/tests/python/test_cfp_validation.py
git diff --check
```

Use notebook smoke tests when regularization changes experiment workflows or
paper-facing examples.

## Implementation Notes

Regularization code lives under:

```text
mtlearn/python/mtlearn/layers/cfp/regularization/
  path_score_monotonicity.py
  attribute_order_score_monotonicity.py
  base.py
  edge_score_monotonicity.py
```

## Current Boundaries

Current regularizers are local to CFP node scores and tree metadata. This keeps
them cheap and differentiable, but it limits expressiveness:

- no regularizer receives target images;
- no regularizer receives reconstructed output pixels;
- no regularizer receives altitude increments directly;
- regularizers receive tree/spec metadata through `CFPContext`, but should not
  mutate context tensors or use it as hidden global state;
- direct regularizer module objects are not accepted in filter specs;
- trainable regularizer parameters are not represented in the current parameter
  contract;
- the public regularization method is `regularization_penalty(...)`.
