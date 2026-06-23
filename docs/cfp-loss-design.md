# CFP Loss Design Guide

This guide defines training-objective guidelines for
`ConnectedFilterPreprocessingLayer` (CFP). It focuses on loss design. The
implementation contracts for scoring models, score constraints, and
regularizers are documented in
[CFP Scoring Design](cfp-scoring-design.md),
[CFP Score Constraint Design](cfp-score-constraint-design.md), and
[CFP Regularization Design](cfp-regularization-design.md). For the broader
package layout, see [CFP Architecture](cfp-architecture.md).

## Why Loss Design Matters

CFP learns scores on morphology-tree nodes. In the standard use case, those
scores do not receive direct supervision. The training path is:

```text
input image
  -> CFP scores on morphology-tree nodes
  -> filtered image channels
  -> downstream model or task head
  -> task loss
```

The task loss supervises the scorer indirectly through the reconstructed CFP
output and any downstream model. The loss is therefore a critical part of the
scientific model: a low task loss does not prove that CFP learned a meaningful
morphological filter. The downstream model can compensate for weak or harmful
filtering, and the scalar loss becomes dominated by easy background pixels
when the objective does not control that imbalance.

The practical objective is to ensure that gradients reaching node scores
express the intended role of CFP, not only the easiest path to reduce the final
task loss.

## Training Objective

Organize a CFP training objective into three kinds of terms:

```text
loss =
    task_loss(model(CFP(x)), target)
  + auxiliary_cfp_loss(CFP(x), x, target)
  + regularization_penalty(scores, tree_info, features)
```

Only the first term is required. Add auxiliary and regularization terms only
when they encode a clear hypothesis about the desired filtering behavior.

In code, keep the terms explicit:

```python
outputs = model(batch_inputs)
task_loss = criterion(outputs, targets)

regularization_loss = layer.regularization_penalty(batch_inputs)

loss = task_loss + regularization_loss
```

When using a cached CFP dataloader, pass the same cached `batch_inputs` object
to both the forward path and `regularization_penalty(...)`. This reuses the
same tree payloads and normalization statistics.

## Component Roles

| Component | Sees targets? | Affects inference? | Role in the objective |
| --- | --- | --- | --- |
| Task loss | yes | no, except through learned parameters | Optimizes the supervised task. |
| Auxiliary CFP loss | yes, when it supervises the CFP output | no, except through learned parameters | Gives CFP a more direct training signal. |
| `Regularizer` | no in the current contract | no | Penalizes undesired score behavior during training. |
| `ScoreConstraint` | no | yes | Enforces deterministic score-space rules before reconstruction. |
| `ScoringModel` | no | yes | Maps normalized node attributes to scores. |

Do not hide target-dependent losses inside a `ScoringModel`,
`ScoreConstraint`, or current `Regularizer`. If a term needs labels, masks, or
task outputs, keep it in the training loop where the dependency is explicit.

## Common Failure Modes

### Backbone Compensation

When the only objective is the final task loss, a strong downstream model can
learn around CFP. The resulting model obtains acceptable metrics even when CFP
scores remain close to a neutral solution or when CFP learns filters that are
not scientifically interpretable.

Use ablations to detect this:

- downstream model without CFP;
- CFP initialized close to identity and trained;
- CFP frozen at identity;
- CFP output inspected visually or through score diagnostics.

### Background-Dominated Pixel Losses

In segmentation, foreground objects often occupy a small part of the image.
Pixelwise losses tend to decrease mostly by modeling the background. CFP then
receives weak gradients for the structures that drive the filtering decision.

Useful checks:

- report foreground and background losses separately;
- inspect foreground and background contributions to the loss;
- monitor task metrics such as Dice, IoU, F1, or boundary metrics, not only the
  scalar training loss;
- visualize CFP outputs on positive and hard examples.

### Loss Scale And Reduction

`mean` and `sum` reductions change the effective gradient scale. With many
pixels or nodes, a mean reduction makes CFP gradients small; with a sum
reduction, the same learning rate behaves as if it were much larger.

Choose the reduction intentionally. Prefer an explicit normalization rule over
relying on accidental scale differences:

```text
pixel_loss = sum(pixel_terms * weights) / sum(weights)
```

Then tune the CFP learning rate and auxiliary weights against that scale.

### Score Saturation

Sigmoid-based scores saturate near `0` or `1`. Saturation makes gradients
small and makes later recovery difficult. Identity initialization reduces this
risk by starting CFP close to the original image:

```python
layer.init_identity(p0=0.995)
```

Choose `score_sharpness`, clamp bounds, scorer capacity, and learning rates
with this risk in mind. A sharper score function makes thresholds easier to
interpret, but it also increases the risk of early saturation.

## Loss Design Patterns

### Pure Task Loss

Use only the downstream task loss when the goal is to test whether CFP improves
the end task without imposing extra assumptions:

```python
outputs = model(batch_inputs)
loss = criterion(outputs, targets)
```

This is the cleanest baseline, but it is also the least direct supervision for
the scorer. Always compare against a non-CFP baseline and inspect learned
scores.

### Task Loss Plus Regularization

Use regularization when the desired score behavior is described without target
labels:

```python
outputs = model(batch_inputs)
task_loss = criterion(outputs, targets)
reg_loss = cfp_layer.regularization_penalty(batch_inputs)
loss = task_loss + reg_loss
```

Examples include edge score monotonicity, path score monotonicity, and
attribute-order monotonicity. These terms stabilize score behavior, but they
are not substitutes for a task loss.

### Auxiliary CFP Output Loss

Use an auxiliary loss when CFP should produce a directly useful intermediate
image. For example, in segmentation, encourage the CFP output to carry
foreground-relevant contrast:

```python
filtered = cfp_layer(batch_inputs)
logits = head(filtered)

task_loss = segmentation_loss(logits, mask)
aux_loss = cfp_auxiliary_loss(filtered, image, mask)
loss = task_loss + aux_weight * aux_loss
```

Define the auxiliary target from the problem and the morphological hypothesis.
It should describe the intended filtering behavior, not merely make the scalar
loss easier to reduce.

## Diagnostics

Track loss terms separately:

```python
logs = {
    "task_loss": float(task_loss.detach().cpu()),
    "regularization_loss": float(reg_loss.detach().cpu()),
    "aux_loss": float(aux_loss.detach().cpu()),
}
```

Also inspect CFP-specific behavior:

- score histograms per filter spec;
- score mean, min, max, and saturation rate;
- acceptance rate for scores above a chosen threshold;
- filtered image previews at fixed epochs;
- foreground and background metrics separately;
- gradient norms for CFP parameters and downstream parameters;
- learned thresholds or projected parameters when the scorer supports
  interpretation.

The goal is to avoid judging CFP only by the final scalar loss. A model can
reduce the loss while leaving CFP unused, saturated, or semantically
uninterpretable.

## API Boundaries

The current CFP API deliberately keeps target-dependent objective design
outside the layer. The layer exposes:

- differentiable CFP outputs through `forward(...)`;
- training-only score penalties through `regularization_penalty(...)`;
- scoring, constraint, and regularizer plugin points;
- inspection utilities for scores, attributes, and filter parameters.

It does not own the task loss. This keeps CFP reusable across classification,
segmentation, filtering, and analysis experiments. Target-aware CFP extensions
must be explicit training utilities, not behavior hidden inside scoring or
constraint components.

## Notebook Examples

Current experiment notebooks show how scoring and regularization choices are
trained under a task loss:

- [`CFP_soft_threshold_scoring_screws_segmentation.ipynb`](../notebooks/experiments/CFP_soft_threshold_scoring_screws_segmentation.ipynb)
  uses a simple score model in a segmentation setting.
- [`CFP_linear_vs_mlp_scoring_screws_segmentation.ipynb`](../notebooks/experiments/CFP_linear_vs_mlp_scoring_screws_segmentation.ipynb)
  compares linear and nonlinear scoring in segmentation.
- [`CFP_score_regularization_screws.ipynb`](../notebooks/experiments/CFP_score_regularization_screws.ipynb)
  shows how a score regularizer enters the training objective.

Treat these notebooks as modeling examples, not as a universal recommendation
for one loss family. The intended morphological role of CFP in the task defines
the correct objective.
