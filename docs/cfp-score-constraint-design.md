# CFP Score Constraint Design

This guide describes how CFP score constraints are designed, configured, and
extended in `mtlearn.layers.cfp`. It focuses on score post-processing:
transformations applied after a scoring model produces node scores and before
reconstruction or regularization consumes those scores. For scoring-specific
details, see [CFP Scoring Design](cfp-scoring-design.md). For
regularization-specific details, see
[CFP Regularization Design](cfp-regularization-design.md). For the broader
package layout, see [CFP Architecture](cfp-architecture.md).

## What Score Constraints Do

In CFP, a scoring model maps normalized node attributes to one score per tree
node. A score constraint then modifies those scores in score space:

```text
features(node)
  -> ScoringModel
  -> raw score(node)
  -> ScoreConstraint(s)
  -> constrained score(node)
  -> altitude-residue modulation and reconstruction
```

Constraints are part of the forward path. They affect both training and
inference. This is the main difference from regularizers, which only add a
training penalty when the caller explicitly adds `regularization_penalty(...)`
to the loss.

Use a `ScoreConstraint` when a score-space rule should always hold. Examples:

- force root nodes to remain active;
- clamp scores to a fixed interval;
- impose deterministic masks from tree metadata;
- enforce a non-trainable safety bound before reconstruction.

Do not use a `ScoreConstraint` to hide scoring logic. If the behavior maps node
attributes to scores, use a `ScoringModel`. If the behavior is only a training
preference, use a `Regularizer`.

## Component Roles

| Component | Input | Output | Affects inference? | Learns parameters? | Role |
| --- | --- | --- | --- | --- | --- |
| `ScoringModel` | normalized node attributes | raw node scores | yes | yes, when defined by the scorer | learns the soft selection rule |
| `ScoreConstraint` | raw or previously constrained scores | constrained node scores | yes | no in the current contract | enforces deterministic score-space invariants |
| `Regularizer` | constrained scores, tree metadata, optional features | scalar penalty | no | normally no | stabilizes training |

## Runtime Contract

The base class is:

```python
import torch


class ScoreConstraint(torch.nn.Module):
    def forward(self, scores: torch.Tensor, tree_info, context=None) -> torch.Tensor:
        raise NotImplementedError
```

Current constraint requirements:

- accept `scores` with shape `(num_nodes,)`;
- return a tensor with the same shape;
- keep the returned tensor on a compatible device;
- preserve differentiability with respect to upstream scorer parameters when
  possible;
- avoid in-place writes to `scores` or tree tensors used by autograd;
- accept `tree_info` and `context` even when unused;
- avoid rebuilding morphology trees or recomputing attributes;
- keep deterministic behavior from `scores`, `tree_info`, and `context`.

The layer calls constraints inside `_score_nodes(...)`, after the scorer has
returned raw scores:

```text
ScoringModel(...)
  -> ScoreConstraint 0
  -> ScoreConstraint 1
  -> ...
  -> constrained scores
```

Constraints are applied in spec order. Configure them explicitly with the
filter-spec `constraints` list.

## Built-In Constraint

### Preserve Root

`PreserveRootConstraint` forces alive root nodes to score `1`:

```text
score(root) = 1
```

It uses the tree `parent` tensor to find root nodes and, when available,
`tpre/tpost` traversal timestamps to ignore inactive nodes.

Config:

```python
spec = {
    "name": "root_preserved_area",
    "tree_type": morphology.TreeType.MAX_TREE,
    "attributes": [morphology.AttributeType.AREA],
    "constraints": [{"kind": "preserve_root"}],
}
```

Use `preserve_root` when suppressing the root contribution would change the
image baseline in an undesirable way.

## Custom Constraint Example

The example below clamps scores to a minimum value after scoring. This does not
learn a threshold; it only prevents the post-processed score from dropping
below a deterministic floor.

```python
import torch

from mtlearn.layers.cfp import SCORE_CONSTRAINT_REGISTRY
from mtlearn.layers.cfp.constraints import ScoreConstraint


class MinimumScoreFloorConstraint(ScoreConstraint):
    def __init__(self, floor: float = 0.0):
        super().__init__()
        self.floor = float(floor)

    def forward(self, scores, tree_info, context=None):
        return torch.clamp(scores, min=self.floor)


def create_minimum_score_floor(*, floor: float = 0.0, **options):
    if options:
        names = ", ".join(sorted(options))
        raise ValueError(f"unsupported minimum_score_floor options: {names}")
    return MinimumScoreFloorConstraint(floor=floor)


SCORE_CONSTRAINT_REGISTRY.register(
    "minimum_score_floor",
    create_minimum_score_floor,
)
```

Then reference the constraint in a filter spec:

```python
spec = {
    "name": "floored_area",
    "tree_type": morphology.TreeType.MAX_TREE,
    "attributes": [morphology.AttributeType.AREA],
    "constraints": [
        {"kind": "minimum_score_floor", "floor": 0.05},
    ],
}
```

The factory receives only serializable config fields. Keep custom constraints
constructible from plain Python values so `get_config()` and `from_config(...)`
can round-trip the layer.

## Registry And Serialization

Score constraints are registry-backed:

```python
from mtlearn.layers.cfp import SCORE_CONSTRAINT_REGISTRY
```

The registry maps a serializable `kind` string to a factory. During layer
construction:

```text
filter spec constraints
  -> normalize_constraint_configs(...)
  -> SCORE_CONSTRAINT_REGISTRY.resolve(kind)
  -> create_score_constraint(config)
  -> torch.nn.ModuleList(...)
```

Because constraints affect forward semantics, they are part of the architecture
and inference contracts:

```text
constraint change -> get_config() changes
constraint change -> get_inference_contract() changes
constraint change -> checkpoint inference compatibility changes
```

This is different from regularizers, which are training-only settings.

Current boundary: custom constraints should not introduce trainable parameters.
They are `torch.nn.Module` subclasses and would technically be visited by
`layer.parameters()`, but the current parameter contract and `export_params()`
payload do not describe constraint-owned parameters. If a future constraint
needs learned parameters, extend the parameter contract, serialization, tests,
and docs before relying on it.

## Design Guidelines

Good constraint candidates:

- deterministic score clamps;
- root preservation;
- topology-derived fixed masks;
- post-score normalization that does not require attributes;
- safety bounds that must hold at inference.

Poor constraint candidates:

- attribute-to-score models;
- trainable gates with their own learned parameters;
- losses or penalties that should only affect training;
- rules that need target images or reconstructed output pixels;
- changes to the reconstructed morphology signal.

When choosing between extension points:

```text
attribute-dependent score rule -> ScoringModel
always-on score post-processing -> ScoreConstraint
training-only preference -> Regularizer
task/target/output-pixel loss -> external training loss
```

## Testing Checklist

When adding a constraint, verify:

- config creation through `SCORE_CONSTRAINT_REGISTRY`;
- rejection of unknown or unsupported config fields;
- `get_config()` and `from_config()` round trip;
- presence in `get_inference_contract()`;
- output shape, dtype, and device;
- gradient flow through upstream scorer parameters when relevant;
- interaction with `preserve_root` if root behavior matters;
- behavior on small hand-computable trees with inactive nodes when applicable.

Run at least:

```bash
python -m compileall -q mtlearn/python/mtlearn mtlearn/tests/python
PYTHONPATH=mtlearn/python:build/mtlearn/bindings python -m pytest -q mtlearn/tests/python/test_cfp_components.py mtlearn/tests/python/test_cfp_validation.py
git diff --check
```

## Implementation Notes

Constraint code lives under:

```text
mtlearn/python/mtlearn/layers/cfp/constraints/
  base.py
  preserve_root.py
```
