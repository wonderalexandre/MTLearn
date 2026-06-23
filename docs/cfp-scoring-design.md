# CFP Scoring Design

This guide describes how CFP scoring models are designed, configured, trained,
and extended in `mtlearn.layers.cfp`. It focuses on the scoring contract
itself. For the broader package layout, see
[CFP Architecture](cfp-architecture.md). For score post-processing details, see
[CFP Score Constraint Design](cfp-score-constraint-design.md). For how losses
supervise scores indirectly through downstream tasks, see
[CFP Loss Design](cfp-loss-design.md).

## What Scoring Does

In CFP, a scoring model decides how strongly each morphology-tree node
contributes to the altitude-residue signal before reconstruction. For one filter spec
and one input image channel, the layer computes:

```text
features(node) -> ScoringModel -> scores(node)
filtered_signal(node) = altitude_residue(node) * scores(node)
output = reconstruct(filtered_signal)
```

The scorer receives a normalized attribute matrix and returns one
differentiable score per tree node:

```text
features.shape == (num_nodes, num_attributes)
scores.shape == (num_nodes,)
0 <= scores[node] <= 1
```

The usual interpretation is a soft keep probability. Scores near `1` preserve
the node signal; scores near `0` suppress it.

## Component Roles

| Component | Input | Output | Learns parameters? | Role |
| --- | --- | --- | --- | --- |
| `ScoringModel` | normalized node attributes | one score per node | yes, when defined by the scorer | defines the soft rule |
| `ScoreConstraint` | scores | adjusted scores | no | imposes score-space constraints |
| `Regularizer` | scores, features, tree metadata | scalar penalty | no direct parameter ownership | stabilizes training |

The scorer is the only component in this list that maps attributes to scores.
Constraints and regularizers are adjacent mechanisms; they should not be used
to hide scoring logic.

## Runtime Contract

`ConnectedFilterPreprocessingLayer._score_nodes(...)` is the runtime boundary
between the layer and a scoring component:

```text
NormalizedFilterSpec
  -> normalized attribute matrix A_norm
  -> ScoringModel(A_norm, tree_info, context, score_sharpness, clamp)
  -> ScoreConstraint(s)
  -> scores
```

Every scorer subclasses `ScoringModel`, which is a `torch.nn.Module`:

```python
from mtlearn.layers.cfp.scoring import ScoringModel


class ScoringModel(torch.nn.Module):
    def required_features(self) -> tuple[object, ...]:
        return ()

    def init_identity(self, *, score_sharpness: float, p0: float = 0.995, **kwargs):
        raise NotImplementedError

    def forward(self, features, tree_info=None, context=None, **kwargs):
        raise NotImplementedError
```

Current scorer requirements:

- accept `features` with shape `(num_nodes, K)`;
- return one score per node with shape `(num_nodes,)`;
- keep scores differentiable with respect to trainable scorer parameters;
- keep output tensors on the same device as the input features;
- accept optional `tree_info` and `context`;
- accept layer-provided keyword arguments such as `score_sharpness` and
  `clamp`;
- define `init_identity(...)` when the scorer can start from an identity-like
  CFP state.

At the filter-spec and scorer level, the sigmoid gain is named
`score_sharpness`.

The scorer receives normalized attributes only. It does not receive raw pixels,
targets, or reconstructed images directly. Tree metadata is available through
`tree_info`, and execution metadata is available through `CFPContext`.

## Scoring Families

### Linear Sigmoid

`LinearSigmoidScorer` is the default CFP scorer:

```text
score(v) = sigmoid(score_sharpness * (A_norm(v) dot w + b))
```

Default config:

```python
linear_spec = {
    "name": "area_linear",
    "tree_type": morphology.TreeType.MAX_TREE,
    "attributes": [morphology.AttributeType.AREA],
    "score_sharpness": 1.0,
}
```

Equivalent explicit config:

```python
linear_spec = {
    "name": "area_linear",
    "tree_type": morphology.TreeType.MAX_TREE,
    "attributes": [morphology.AttributeType.AREA],
    "scoring": {"kind": "linear_sigmoid"},
}
```

Use it when a weighted linear combination of attributes is enough. With one
attribute, `w` and `b` define a soft threshold in normalized attribute space.
With multiple attributes, the decision boundary is a hyperplane and threshold
interpretation becomes attribute-combination dependent.

### Single-Attribute Soft Threshold

A useful experimental scorer is:

```text
s_A,lambda(v) = sigmoid(score_sharpness * (A_norm(v) - lambda))
```

It learns one scalar `lambda_`. This keeps the model intentionally simple and
makes the learned parameter interpretable as a soft threshold in normalized
attribute space. The notebooks use this scorer to show the same scoring rule in
two problem formulations.

Minimal implementation:

```python
class SingleAttributeSoftThresholdScorer(ScoringModel):
    kind = "single_attribute_soft_threshold"

    def __init__(self, num_features: int, *, initial_lambda: float = 0.0):
        super().__init__()
        if int(num_features) != 1:
            raise ValueError("single_attribute_soft_threshold requires exactly one feature.")
        self.num_features = int(num_features)
        self.initial_lambda = float(initial_lambda)
        self.lambda_ = torch.nn.Parameter(torch.tensor(self.initial_lambda, dtype=torch.float32))

    def forward(self, features, tree_info=None, context=None, *, score_sharpness=None, clamp=None):
        if features.dim() != 2 or features.size(1) != self.num_features:
            raise ValueError(f"expected features with shape (num_nodes, {self.num_features})")

        sharpness = 1.0 if score_sharpness is None else float(score_sharpness)
        lambda_value = self.lambda_.to(dtype=features.dtype, device=features.device)
        logits = sharpness * (features[:, 0] - lambda_value)
        if clamp is not None:
            logits = torch.clamp(logits, clamp[0], clamp[1])
        return torch.sigmoid(logits)

    def to_config(self):
        return {"kind": self.kind, "initial_lambda": self.initial_lambda}
```

### MLP Scoring

`MLPScorer` implements nonlinear attribute scoring:

```text
score(v) = sigmoid(score_sharpness * g_theta(A_norm(v)))
```

Example config:

```python
mlp_spec = {
    "name": "shape_mlp",
    "tree_type": morphology.TreeType.MAX_TREE,
    "attributes": [
        morphology.AttributeType.AREA,
        morphology.AttributeType.COMPACTNESS,
        morphology.AttributeType.CIRCULARITY,
    ],
    "scoring": {
        "kind": "mlp",
        "hidden_units": [8],
        "activation": "tanh",
    },
    "score_sharpness": 1.0,
}
```

Use it when attribute interactions matter, such as "large and compact" or
"small but high contrast". The cost is interpretability: the learned decision
surface is nonlinear, and there is generally no single threshold to project
back to an attribute scale.

## Identity Initialization

The layer exposes:

```python
layer.init_identity(p0=0.995)
```

This asks each scorer to initialize itself so that `score(node) ~= p0` for every
node. Since CFP reconstruction multiplies altitude increments by scores, this
makes the initial preprocessing close to the identity map.

The scorer owns the implementation:

- `LinearSigmoidScorer` sets linear weights to zero and chooses the bias from
  `logit(p0) / score_sharpness`;
- `MLPScorer` sets the final bias from `logit(p0) / score_sharpness` and keeps a small
  output weight scale so the network remains close to constant while gradients
  can still reach hidden layers;
- custom scorers should implement `init_identity` when they have a meaningful
  neutral initialization.

If a custom scorer does not support this contract, `layer.init_identity()` fails
by default. Use `layer.init_identity(strict=False)` only when it is acceptable to
skip unsupported scorers.

## Parameter Interpretation

Parameter interpretation depends on the scorer family:

| Scorer | Learned parameters | Direct threshold interpretation? | Original-scale projection |
| --- | --- | --- | --- |
| `single_attribute_soft_threshold` | `lambda_` | yes, for one normalized attribute | `a_min + lambda_ * (a_max - a_min)` for `dataset_minmax01` |
| `linear_sigmoid` with one attribute | `w`, `b` | yes, if `w != 0` | `a_min + (-b / w) * (a_max - a_min)` for `dataset_minmax01` |
| `linear_sigmoid` with many attributes | vector `w`, scalar `b` | no single attribute threshold | only projections along chosen directions |
| `mlp` | network weights | no | usually not meaningful |

Projection is a reporting tool, not a new model parameter. It depends on the
normalization statistics used by the layer. For `dataset_minmax01`, the projection from
normalized threshold `t` to original attribute scale is:

```text
attribute_original = attribute_min + t * (attribute_max - attribute_min)
```

For other normalization modes, the projection must be derived from that
normalizer's statistics and assumptions.

## Config And Registration

Scoring specs are normalized by `normalize_scoring_model(value, num_features)`.
The accepted forms are:

```python
# Default layer-owned linear sigmoid.
"scoring" not in spec

# Explicit registry-backed scorer.
"scoring": {"kind": "linear_sigmoid"}
"scoring": {"kind": "mlp", "hidden_units": [8], "activation": "tanh"}

# Direct scorer object, useful for experiments.
"scoring": scorer_instance
```

Registry-backed configs are preferred when `get_config()` and `from_config()`
must recreate the layer. Direct scorer objects are useful in notebooks and
small experiments, but they are less portable.

To add a registry-backed scorer, register a factory:

```python
def create_single_attribute_soft_threshold_scorer(
    *,
    num_features: int,
    initial_lambda: float = 0.0,
    **options,
):
    if options:
        names = ", ".join(sorted(options))
        raise ValueError(f"unsupported options: {names}")
    return SingleAttributeSoftThresholdScorer(
        num_features,
        initial_lambda=initial_lambda,
    )


SCORING_MODEL_REGISTRY.register(
    "single_attribute_soft_threshold",
    create_single_attribute_soft_threshold_scorer,
)
```

Then specs can use:

```python
{
    "name": "minor_axis_soft_threshold",
    "tree_type": "max-tree",
    "attributes": [morphology.AttributeType.LENGTH_MINOR_AXIS],
    "scoring": {
        "kind": "single_attribute_soft_threshold",
        "initial_lambda": 0.0,
    },
}
```

Factories should reject unsupported options rather than silently ignoring them.

## Training Stability

Use the smallest scoring model that can express the scientific hypothesis.

For threshold-like scoring:

- prefer the single-attribute soft threshold when the scientific question is
  explicitly "which threshold should this attribute learn?";
- prefer linear sigmoid when attribute weighting matters;
- inspect learned parameters and score histograms, not only task loss;
- set `score_sharpness` high enough to approximate a threshold but low enough to avoid
  immediate sigmoid saturation;
- use `clamp` to bound `score_sharpness * logits` when gradients saturate.

For nonlinear scoring:

- start with one hidden layer and widths such as `[4]` or `[8]`;
- use optimizer weight decay on small datasets;
- consider `edge_score_monotonicity` when the expected filter should be tree-monotone;
- preserve the root when root suppression would distort the image baseline.

Regularization belongs beside the task loss. For the regularizer contract and
extension rules, see [CFP Regularization Design](cfp-regularization-design.md):

```python
loss = task_loss(layer(inputs), targets) + layer.regularization_penalty(inputs)
```

## Notebook Examples

One experiment notebook compares the built-in linear and MLP scorers:

- [`CFP_linear_vs_mlp_scoring_screws_segmentation.ipynb`](../notebooks/experiments/CFP_linear_vs_mlp_scoring_screws_segmentation.ipynb)
  uses the original screws segmentation masks and reports calibrated and
  fixed-threshold segmentation metrics while keeping the CFP layer fixed
  except for the scoring family.

Two additional notebooks show the same simple custom scorer in two tasks:

- [`CFP_soft_threshold_scoring_screws_filtering.ipynb`](../notebooks/experiments/CFP_soft_threshold_scoring_screws_filtering.ipynb)
  uses `SingleAttributeSoftThresholdScorer` for a filtering/regression-style
  target. It learns `lambda_` and reports its projection to the original
  attribute scale.
- [`CFP_soft_threshold_scoring_screws_segmentation.ipynb`](../notebooks/experiments/CFP_soft_threshold_scoring_screws_segmentation.ipynb)
  uses the same scorer for segmentation. It trains with a segmentation loss,
  binarizes the CFP response as a mask, and reports IoU/F1.

These notebooks are pedagogical examples of scoring model design. They are not
full filtering or segmentation benchmarks.

## Extension Checklist

When adding a scorer, verify:

- constructor validation;
- forward output shape `(num_nodes,)`;
- score range if the scorer returns probabilities;
- gradient propagation through scorer parameters;
- config round trip through `get_config()` and `from_config()`, when using the
  registry;
- parameter contract and `state_dict` key shape;
- interaction with `preserve_root` if root behavior matters;
- interaction with `edge_score_monotonicity` if the scorer is nonlinear or tree-aware.

Run at least:

```bash
python -m compileall -q mtlearn/python/mtlearn mtlearn/tests/python
PYTHONPATH=mtlearn/python:build/mtlearn/bindings python -m pytest -q mtlearn/tests/python/test_cfp_components.py mtlearn/tests/python/test_cfp_validation.py
git diff --check
```

Use notebook smoke tests when the scorer changes paper-facing examples or
experiment workflows.

## Adjacent Components

Score constraints post-process scores before reconstruction. The current
built-in constraint is `preserve_root`, which forces alive root nodes to score
`1`. Use a constraint when the behavior changes scores but does not need
trainable parameters. See
[CFP Score Constraint Design](cfp-score-constraint-design.md) for the full
constraint contract.

Score regularizers observe scores and add training penalties. A common built-in
regularizer is `edge_score_monotonicity`, which penalizes child scores that
exceed parent scores:

```text
mean(relu(score(child) - score(parent))^2)
```

The reconstructed node signal is fixed to altitude residues. Changing this
signal is not a current Python extension point.

## Implementation Notes

Scoring code lives under:

```text
mtlearn/python/mtlearn/layers/cfp/scoring/
  base.py
  linear_sigmoid.py
  mlp.py
```

The default `linear_sigmoid` registry factory uses `owns_parameters=False`, so
the layer keeps the trainable linear parameters in `_weights` and `_biases`
dictionaries. This keeps linear scorer parameters easy to inspect and export.

## Current Boundaries

Current scorers are local to attributes of one node. This makes scoring cheap,
differentiable, and cache-friendly, but it limits expressiveness:

- no learned neighborhood aggregation across nodes;
- no direct access to pixels or images inside the scorer;
- no built-in feature-selection mechanism driven by `required_features()`;
- no tree-aware scorer is registered yet.
