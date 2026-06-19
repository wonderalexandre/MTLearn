# CFP Scoring Design

This standalone developer guide documents the scoring design currently
implemented in `mtlearn.layers.cfp`. It is intentionally outside the Sphinx
documentation tree because it describes internal extension contracts and
research design tradeoffs rather than stable user-facing API guarantees.

See also [CFP Architecture](cfp-architecture.md) for the broader package layout.

## Scope

In CFP, scoring decides how strongly each morphology-tree node contributes to a
valuation signal before reconstruction. For one filter spec and one input image
channel, the layer computes:

```text
features(node) -> ScoringModel -> scores(node)
filtered_signal(node) = valuation_signal(node) * scores(node)
output = reconstruct(filtered_signal)
```

The current scoring model returns one differentiable score per tree node:

```text
scores.shape == (num_nodes,)
0 <= scores[node] <= 1
```

The default interpretation is a soft keep probability. Values near `1` preserve
the node signal; values near `0` suppress it.

## Runtime Boundary

`ConnectedFilterPreprocessingLayer._score_nodes(...)` is the runtime boundary
between the layer and scoring components:

```text
NormalizedFilterSpec
  -> normalized attribute matrix A_norm
  -> ScoringModel(A_norm, tree_info, context, beta_f, clamp)
  -> ScoreConstraint(s)
  -> scores
```

The feature matrix has shape `(num_nodes, K)`, where `K` is the number of
attributes declared by the filter spec:

```python
{
    "tree_type": morphology.TreeType.MAX_TREE,
    "attributes": (
        morphology.AttributeType.AREA,
        morphology.AttributeType.COMPACTNESS,
    ),
}
```

The scoring model receives normalized attributes only. It does not receive the
raw image, the target, or pixels directly. Tree metadata is available through
`tree_info` for future tree-aware scorers, and execution metadata is available
through `CFPContext`.

## Package Map

The current implementation lives under:

```text
mtlearn/python/mtlearn/layers/cfp/scoring/
  __init__.py
  base.py
  linear_sigmoid.py
  mlp.py
  legacy_linear_parameter_initializer.py
```

Related scoring-adjacent components live under:

```text
cfp/constraints/
  base.py
  preserve_root.py

cfp/regularization/
  base.py
  monotone_scores.py

cfp/component_registries.py
```

Top-level compatibility shims such as `cfp/mlp_scorer.py` and
`cfp/scoring_model.py` remain importable, but new implementation work should use
the grouped packages.

## ScoringModel Contract

Every scoring model subclasses `ScoringModel`, which is a `torch.nn.Module`:

```python
from mtlearn.layers.cfp.scoring import ScoringModel


class ScoringModel(torch.nn.Module):
    def required_features(self) -> tuple[object, ...]:
        return ()

    def forward(self, features, tree_info=None, context=None, **kwargs):
        raise NotImplementedError
```

Current requirements:

- accept `features` with shape `(num_nodes, K)`;
- return one score per node with shape `(num_nodes,)`;
- keep scores differentiable with respect to trainable scorer parameters;
- keep output tensors on the same device as the input features;
- accept optional `tree_info` and `context` arguments;
- accept extra keyword arguments used by the layer, especially `beta_f` and
  `clamp`.

The `required_features()` hook exists for future component-driven feature
selection. Today the filter spec still owns the scoring attribute list, so a
scorer is expected to match the number of attributes declared by the spec.

## Built-In Scorers

### LinearSigmoidScorer

`LinearSigmoidScorer` implements the original CFP gate:

```text
score(v) = sigmoid(beta_f * (A_norm(v) dot w + b))
```

It is the default scorer when a spec does not define `"scoring"`:

```python
linear_spec = {
    "name": "area_linear",
    "tree_type": morphology.TreeType.MAX_TREE,
    "attributes": [morphology.AttributeType.AREA],
}
```

The equivalent explicit config is:

```python
linear_spec = {
    "name": "area_linear",
    "tree_type": morphology.TreeType.MAX_TREE,
    "attributes": [morphology.AttributeType.AREA],
    "scoring": {"kind": "linear_sigmoid"},
}
```

The implementation is already provided by the API. The class lives in
`mtlearn.layers.cfp.scoring.linear_sigmoid` and is exported as
`mtlearn.layers.cfp.LinearSigmoidScorer`. You do not need to write this class
for normal CFP usage; the simplified code below shows the essential contract:

```python
import torch

from mtlearn.layers.cfp.scoring import ScoringModel


class LinearSigmoidScorer(ScoringModel):
    def __init__(self, num_features: int, *, owns_parameters: bool = True):
        super().__init__()
        self.num_features = int(num_features)
        self.owns_parameters = owns_parameters

        if self.owns_parameters:
            self.weight = torch.nn.Parameter(torch.empty(self.num_features))
            self.bias = torch.nn.Parameter(torch.zeros(1))
        else:
            self.weight = None
            self.bias = None

    def logits(self, features, *, weight=None, bias=None):
        weight = self.weight if weight is None else weight
        bias = self.bias if bias is None else bias
        return features @ weight.view(-1) + bias

    def forward(
        self,
        features: torch.Tensor,
        tree_info=None,
        context=None,
        *,
        weight=None,
        bias=None,
        beta_f=1.0,
        clamp=None,
    ):
        scaled = float(beta_f) * self.logits(features, weight=weight, bias=bias)
        if clamp is not None:
            scaled = torch.clamp(scaled, clamp[0], clamp[1])
        return torch.sigmoid(scaled)

    def to_config(self) -> dict:
        return {"kind": "linear_sigmoid"}
```

The production class adds validation, initialization, dtype/device support, and
stored default `beta_f`/`clamp` values. The conceptual behavior above is the
part extension authors need to understand: compute linear logits, optionally
scale/clamp them, then return sigmoid scores.

The registry path is also already installed by default in
`component_registries.py`. It is shown here so new scorers can follow the same
pattern:

```python
def _create_linear_sigmoid_scorer(*, num_features: int, **options) -> LinearSigmoidScorer:
    unsupported = set(options)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported linear_sigmoid scoring options: {names}")
    return LinearSigmoidScorer(num_features, owns_parameters=False)


SCORING_MODEL_REGISTRY.register(
    "linear_sigmoid",
    _create_linear_sigmoid_scorer,
    aliases=("linear-sigmoid",),
)
```

The important API detail is `owns_parameters=False` in the factory. That tells
the layer to keep the trainable linear scorer parameters in its legacy
`_weights` and `_biases` dictionaries instead of inside the scorer module.

Interpretation:

- with one attribute, `w` and `b` define a soft threshold in normalized
  attribute space;
- `beta_f` controls transition sharpness;
- `clamp` bounds `beta_f * logits` before the sigmoid to avoid saturated
  gradients;
- `preserve_root` can force the root score to `1` after scoring.

Compatibility behavior:

- the registry-created default uses `owns_parameters=False`;
- layer-owned parameters stay under `_weights.<spec_name>` and
  `_biases.<spec_name>`;
- this preserves historical checkpoints, exported params, and notebook access
  to `_weights["spec_000"]` and `_biases["spec_000"]`.

### MLPScorer

`MLPScorer` implements nonlinear attribute scoring:

```text
score(v) = sigmoid(beta_f * g_theta(A_norm(v)))
```

where `g_theta` is a shallow MLP that acts only on the normalized attributes of
the current node.

```python
mlp_spec = {
    "name": "shape_mlp",
    "tree_type": morphology.TreeType.MAX_TREE,
    "attributes": [
        morphology.AttributeType.AREA,
        morphology.AttributeType.COMPACTNESS,
        morphology.AttributeType.CIRCULARITY,
        morphology.AttributeType.GRAY_HEIGHT,
    ],
    "scoring": {
        "kind": "mlp",
        "hidden_channels": [8],
        "activation": "tanh",
    },
}
```

Supported activations are:

```text
relu, tanh, gelu, sigmoid, identity
```

`hidden` is accepted as a config alias for `hidden_channels`, but a config must
not define both names at the same time.

Interpretation:

- the MLP can model interactions such as "large and compact" or "small but high
  contrast";
- the learned decision boundary is a nonlinear surface in normalized attribute
  space;
- direct threshold interpretation is lost because no single weight/bias pair
  maps cleanly to an attribute cutoff.

Parameter behavior:

- MLP parameters are owned by the scorer module;
- they appear under `_scoring_models.<spec_name>.network...` in `state_dict`;
- `get_parameter_contract()["scoring_models"]` lists scorer-owned parameters;
- `_weights` and `_biases` are empty for MLP specs.

## Config Normalization

Scoring specs are normalized by `normalize_scoring_model(value, num_features)`.
The accepted forms are:

```python
# Default legacy linear sigmoid.
"scoring" not in spec

# Explicit legacy linear sigmoid.
"scoring": {"kind": "linear_sigmoid"}

# Registry-backed MLP.
"scoring": {
    "kind": "mlp",
    "hidden_channels": [8],
    "activation": "tanh",
}

# Direct scorer object.
"scoring": scorer_instance
```

Direct scorer objects are useful for experiments, but registry-backed configs
are preferred when `get_config()` and `from_config()` must recreate the layer.

Unknown `kind` values raise `ValueError`. Factories reject unsupported options
rather than silently ignoring them.

## Constraints After Scoring

Score constraints are separate from scoring models. They post-process the score
vector before valuation reconstruction.

Current built-in:

```python
{
    "constraints": [{"kind": "preserve_root"}],
}
```

The legacy shortcut is still accepted:

```python
{"preserve_root": True}
```

`PreserveRootConstraint` forces alive root nodes to score `1`. This keeps the
image baseline stable when suppressing the root would change the whole
reconstruction.

Design rule: if a behavior changes scores but does not require trainable
parameters or feature extraction, implement it as a `ScoreConstraint`, not as a
new scorer.

## Regularization Of Scores

Regularizers observe scores after scoring and constraints. They contribute
training penalties but do not change the inference contract.

Current built-in:

```python
regularized_spec = {
    "name": "monotone_area",
    "tree_type": morphology.TreeType.MAX_TREE,
    "attributes": [morphology.AttributeType.AREA],
    "regularizers": [{"kind": "monotone_scores", "weight": 0.1}],
}

loss = task_loss(layer(inputs), targets) + layer.monotonicity_penalty(inputs)
```

The legacy shortcut is still accepted:

```python
{"monotonicity_weight": 0.1}
```

`MonotoneScoresRegularizer` penalizes child scores that exceed parent scores:

```text
mean(relu(score(child) - score(parent))^2)
```

Use it when the intended filter should become more selective along tree edges.
For MLP scorers, it is especially useful on small datasets because it imposes a
tree-aware shape constraint on a nonlinear decision surface.

## Stability Guidelines

Use the smallest scoring model that can express the scientific hypothesis.

For linear scoring:

- prefer it when threshold interpretability matters;
- inspect learned weights and bias to understand attribute direction and soft
  cutoff;
- use `init_identity_with_bias()` or `init_identity_bias_zero()` when training
  should start close to identity.

For MLP scoring:

- start with one hidden layer and widths such as `[4]` or `[8]`;
- prefer `tanh` when normalized attributes are centered;
- use optimizer weight decay;
- use `clamp` to limit sigmoid saturation;
- add `monotone_scores` if the expected behavior is tree-monotone;
- preserve the root when root suppression would distort the baseline.

Example:

```python
layer = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    filter_specs=[mlp_spec],
    scale_mode="hybrid",
    clamp=8.0,
)

optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3, weight_decay=1e-4)
```

## Adding A New Scorer

Add implementation files under `cfp/scoring/`. Keep one public scorer class per
file.

Minimal pattern for a single-attribute soft threshold scorer:

```python
import torch

from mtlearn.layers.cfp.scoring import ScoringModel


class SingleAttributeSoftThresholdScorer(ScoringModel):
    kind = "single_attribute_soft_threshold"

    def __init__(
        self,
        num_features: int,
        *,
        initial_lambda: float = 0.5,
    ):
        super().__init__()
        if int(num_features) != 1:
            raise ValueError("single_attribute_soft_threshold requires exactly one feature.")
        self.num_features = int(num_features)
        self.initial_lambda = float(initial_lambda)
        self.lambda_ = torch.nn.Parameter(torch.tensor(self.initial_lambda, dtype=torch.float32))

    def forward(self, features, tree_info=None, context=None, *, beta_f=None, clamp=None):
        if features.dim() != 2:
            raise ValueError(f"expected features with shape (num_nodes, K), got {tuple(features.shape)}")
        if features.size(1) != self.num_features:
            raise ValueError(f"expected {self.num_features} features, got {features.size(1)}")

        attribute = features[:, 0]
        lambda_value = self.lambda_.to(dtype=features.dtype, device=features.device)
        logits = attribute - lambda_value
        beta = 1.0 if beta_f is None else float(beta_f)
        scaled = beta * logits
        if clamp is not None:
            scaled = torch.clamp(scaled, clamp[0], clamp[1])
        return torch.sigmoid(scaled)

    def to_config(self):
        return {
            "kind": self.kind,
            "initial_lambda": self.initial_lambda,
        }
```

Register the scorer in `component_registries.py`:

```python
def _create_single_attribute_soft_threshold_scorer(
    *,
    num_features: int,
    initial_lambda: float = 0.5,
    **options,
):
    if options:
        names = ", ".join(sorted(options))
        raise ValueError(f"unsupported single_attribute_soft_threshold scoring options: {names}")
    return SingleAttributeSoftThresholdScorer(
        num_features,
        initial_lambda=initial_lambda,
    )


SCORING_MODEL_REGISTRY.register(
    "single_attribute_soft_threshold",
    _create_single_attribute_soft_threshold_scorer,
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
        "initial_lambda": 0.5,
    },
}
```

The mathematical model is:

```text
s_A,lambda(v) = sigmoid(beta_f * (A_norm(v) - lambda))
```

where `lambda_` is the learned threshold parameter and `beta_f` is the existing
CFP sigmoid gain. This keeps sharpness in the layer-level inference contract
instead of duplicating it inside the scorer config.

Expose it from `cfp/scoring/__init__.py` and optionally from the aggregate
`cfp/__init__.py` if it should be part of the public developer surface.

## Testing Checklist

For any new scorer, add tests for:

- import path and package export;
- constructor validation;
- forward output shape `(num_nodes,)`;
- score range if the scorer is meant to return probabilities;
- gradient propagation through scorer parameters;
- config round trip through `get_config()` and `from_config()`;
- parameter contract and `state_dict` key shape;
- registry rejection of unknown or unsupported options;
- interaction with `preserve_root` if root behavior matters;
- interaction with `monotone_scores` if the scorer is nonlinear or
  tree-aware.

Run at least:

```bash
python -m compileall -q mtlearn/python/mtlearn mtlearn/tests/python
PYTHONPATH=mtlearn/python:build/mtlearn/bindings python -m pytest -q mtlearn/tests/python/test_cfp_components.py mtlearn/tests/python/test_cfp_validation.py
PYTHONPATH=mtlearn/python:build/mtlearn/bindings python -m pytest -q -m gradcheck mtlearn/tests/python
git diff --check
```

Use notebook smoke tests when the scorer changes paper-facing examples or
experiment workflows.

## Current Design Boundaries

The current design deliberately keeps scoring local to attributes of one node.
That makes scoring cheap, differentiable, and easy to cache, but it limits
expressiveness:

- no learned neighborhood aggregation across nodes;
- no direct access to pixels or images inside the scorer;
- no built-in feature-selection mechanism driven by `required_features()`;
- no tree-aware scorer is registered yet.

Future scorer families can relax these constraints, but they should make the
additional dependency explicit in the scorer contract, config, tests, and
architecture guide.
