# CFP Architecture

This page documents the developer-facing architecture of
`ConnectedFilterPreprocessingLayer` and the common definitions shared by the
CFP guides.

Component-specific implementation details live in separate guides:

- [CFP Scoring Design](cfp-scoring-design.md)
- [CFP Score Constraint Design](cfp-score-constraint-design.md)
- [CFP Regularization Design](cfp-regularization-design.md)
- [CFP Loss Design](cfp-loss-design.md)

The public user import remains:

```python
from mtlearn.layers import ConnectedFilterPreprocessingLayer
```

The implementation lives under `mtlearn.layers.cfp`.

## Scope

Use this guide as the architecture map for CFP. It defines the shared runtime
objects, package boundaries, data flow, public utilities, and state contracts.
It intentionally avoids full custom component examples; those belong to the
scoring, score-constraint, and regularization guides.

## Design Goals

The CFP layer is organized so these concerns can evolve independently:

- tree construction and dense tree tensors;
- normalized node features used for scoring;
- node scoring models;
- score post-processing constraints;
- fixed altitude residues, which are the node signal reconstructed by CFP;
- training regularizers;
- serialization contracts for configs, checkpoints, and exported parameters.

New behavior should usually be added as a component under the relevant
`mtlearn/python/mtlearn/layers/cfp/` subpackage, not by growing the layer class
directly. Keep one public component class per file when the component is meant
to be extended or imported by downstream code.

## Common Definitions

| Term | Meaning |
| --- | --- |
| Filter spec | One normalized CFP operator definition: tree type, attributes, scoring model, constraints, regularizers, and related settings. |
| Tree payload | The per-sample, per-channel, per-spec data produced before scoring: tree metadata, raw attributes, and normalized attributes. |
| `tree_info` | Dense tree tensors and metadata used by scoring, constraints, regularization, and reconstruction. |
| Raw node attributes | Scalar morphology attributes computed on tree nodes before dataset normalization. |
| Normalized node features | Tensor with shape `(num_nodes, K)`, where `K` is the number of attributes in the filter spec. |
| Node scores | Differentiable tensor with shape `(num_nodes,)`; one score per tree node. |
| Score constraints | Deterministic post-processing applied to scores before reconstruction. |
| Altitude residues | Fixed backend-provided per-node increments reconstructed by CFP after multiplication by scores. |
| Regularizers | Training-only penalties computed from scores, tree tensors, and optionally normalized features. |
| Loss design | Training-loop objective design that decides how task, auxiliary, and regularization terms supervise CFP. |
| `CFPContext` | Metadata passed to components during normal layer execution. It identifies sample, channel, spec, tree, attributes, image shape, normalization, and execution mode. |
| Inference contract | The part of the layer configuration that changes forward semantics. |
| Training contract | Training-only settings, primarily regularization configuration. |

## Class Relationships

The class architecture has one public layer facade and several focused helper
and extension components. Solid arrows represent layer ownership or delegation,
dashed arrows represent config-backed construction, and green arrows represent
the runtime call path.

![CFP class relationship diagram](assets/cfp-class-architecture.svg)

## Forward Flow

At runtime, `ForwardExecutor` drives the layer. For each batch item, input
channel, and filter spec, the flow is:

```text
input image channel
 -> TreePayloadProvider
 -> AttributeNormalizer
 -> ScoringModel
 -> ScoreConstraint(s)
 -> score * altitude residues
 -> TreeReconstructionFunction
 -> output image channel
```

The output shape is `(B, C * num_specs, H, W)`, where `B` is batch size, `C` is
the number of input channels, `num_specs` is the number of normalized filter
specs, and `H` and `W` are image height and width. Output channels are ordered
by input channel first and filter spec second.

The tree payload contains:

- `info`: dense tree tensors and metadata;
- `base_attrs`: raw scalar attributes by attribute key;
- `norm_attrs`: normalized scalar attributes by attribute key.

The `info` mapping currently contains:

- `residues`: altitude residues from the backend;
- `tpre` and `tpost`: tree traversal entry and exit times;
- `parent`: dense parent ids;
- `node_of_pixel`: proper-part owner node id for each flattened pixel;
- `num_rows` and `num_cols`: image shape;
- `tree_type`: normalized tree type string;
- `order_forward` and `order_backward`: traversal orders for reconstruction.

The differentiable reconstruction boundary is `TreeReconstructionFunction`.
It reconstructs pixels from one scalar per tree node without materializing a
dense region-pixel Jacobian.

## Package Layout

Use this map when deciding where a change belongs:

| Package or module | Responsibility |
| --- | --- |
| `connected_filter_preprocessing_layer.py` | Public layer methods and orchestration hooks. |
| `scoring/` | Scoring base class and built-in scoring models. |
| `constraints/` | Score post-processing constraints. |
| `regularization/` | Training penalties over scores, tree tensors, and normalized features. |
| `normalization/` | Attribute normalization and normalization-stat serialization. |
| `specs/` | Filter-spec dataclasses, validation, normalization, and generic `SpecRegistry`. |
| `runtime/` | Batch input handling, cached dataloaders, forward execution, tree payloads, reconstruction, context, and inspection. |
| `serialization/` | Layer configs, deserialization, checkpoints, saved stats, and parameter exports. |
| `component_registries.py` | Default registries for scoring, constraints, and regularizers. |

New code should import public extension components from the aggregate
`mtlearn.layers.cfp` namespace only when they are exported there: the layer,
specs, scoring models, score constraints, regularizers, and registries. Runtime,
normalization, and serialization infrastructure should be imported from their
grouped packages (`cfp.runtime`, `cfp.normalization`, and `cfp.serialization`)
when advanced or internal work needs them.

The intended package layout is:

```text
cfp/
  connected_filter_preprocessing_layer.py
  component_registries.py
  scoring/
  constraints/
  regularization/
  normalization/
  specs/
  runtime/
  serialization/
```

## Runtime Boundaries

Tree construction, topology, attribute extraction, and altitude-residue
computation are backend-side morphology operations. They are treated as fixed
inputs to the differentiable PyTorch path for a given forward pass.

The differentiable path is:

```text
scoring parameters
 -> scores
 -> constrained scores
 -> residues * constrained scores
 -> reconstruction
 -> loss
```

The following are outside the autograd path:

- tree construction;
- tree topology;
- image quantization used by tree construction;
- raw attribute computation;
- dataset normalization-stat estimation;
- altitude-residue extraction.

The following are inside the PyTorch computation graph when trainable
parameters participate:

- scoring logits and score tensors;
- score constraints that use differentiable tensor operations;
- `residues * scores`;
- differentiable reconstruction;
- regularization penalties.

## Runtime Context

Scoring models, score constraints, and regularizers receive a `CFPContext`
when the layer is running through its normal execution path. The context carries
runtime metadata; it should not be used as hidden mutable state.

The context identifies:

- `sample_key`;
- `batch_index`;
- `channel_index`;
- `mode`;
- `spec_name`;
- `spec_index`;
- `tree_type` and `tree_key`;
- `attribute_types` and `attribute_names`;
- `image_shape`;
- `normalization_mode`;
- `score_sharpness`;
- `is_training`.

It also exposes `raw_attributes` and `normalized_attributes` mappings when the
current tree payload has already computed those tensors. Custom components may
read these mappings, but they should not mutate them.

Current modes are `"forward"` and `"regularization_penalty"`.

## Extension Points

CFP has two extension levels. Python extension points customize the learnable
model around an existing tree payload and reconstruction signal. Full-stack
morphology extension points change what the backend computes or how the
backend signal is exposed to the differentiable layer.

### Python Extension Points

These points are registry-backed components. They are the preferred extension
level when the tree type, node attributes, node payload, normalization semantics,
and reconstructed signal are already sufficient.

| Extension point | Current status | Detailed guide |
| --- | --- | --- |
| `ScoringModel` | Registry-backed and config-roundtrippable. Safe extension point. | [CFP Scoring Design](cfp-scoring-design.md) |
| `ScoreConstraint` | Registry-backed and config-roundtrippable. Safe extension point. | [CFP Score Constraint Design](cfp-score-constraint-design.md) |
| `Regularizer` | Registry-backed and config-roundtrippable. Safe extension point. | [CFP Regularization Design](cfp-regularization-design.md) |

Registry-backed extension points share the same high-level rules:

- preserve tensor shape and device;
- avoid in-place writes to tensors needed by autograd;
- keep construction values serializable through config mappings;
- reject unsupported options instead of silently ignoring them;
- use `CFPContext` as metadata, not as a hidden tensor owner;
- document tree-ordering assumptions when a component depends on them.

`AttributeNormalizer` is shared normalization infrastructure, not a plugin
surface like scoring, constraints, or regularization. Change it only when the
global semantics of CFP attribute scaling should change, and cover the update
with normalization-statistics tests.

### Full-Stack Morphology Extension Points

These points are not simple Python plugin surfaces. They change morphology
semantics or the differentiable boundary and normally require coordinated
changes in backend C++, pybind bindings, the Python facade, CFP runtime code,
tests, and developer documentation.

| Extension point | What changes | Typical affected layers |
| --- | --- | --- |
| New morphology attribute | Add a scalar node attribute such as a shape, contrast, topology, or proper-part attribute. | Backend attribute computer, public `AttributeType`, pybind bindings, `mtlearn.morphology`, filter-spec validation, normalization, CFP tests. |
| New tree type or construction mode | Add a hierarchy or construction variant such as a new self-dual tree, connectivity, boundary rule, or quantization policy. | Backend tree factory, public `TreeType` or construction options, bindings, Python facade, spec normalization, `TreePayloadProvider`, C++ and Python tests. |
| New node payload or topology tensor | Expose additional per-node structure such as depth, ancestors, descendants, contours, ownership, or traversal metadata. | Backend tensor export, `tree_tensors.hpp`, bindings, `tree_info`, consuming scorers/constraints/regularizers, serialization and tests. |
| New reconstructed signal | Replace or complement the current altitude-residue signal with another morphology signal. | Backend signal export, `TreeReconstructionFunction`, backward semantics, inference contract, gradcheck, notebooks that inspect learned parameters. |
| New reconstruction or derivative rule | Change how node scores are mapped back to pixels or how gradients are propagated. | Backend traversal assumptions, `tree_traversal.hpp`, `TreeReconstructionFunction.forward/backward`, C++ tests, Python gradcheck tests, benchmarks. |
| New hard morphology operator | Expose a classical operator that may not be trained through CFP, such as pruning, merge variants, contours, UAO, or another connected filter. | Backend operator, public C++ API, bindings, Python facade, user docs, examples, and regression tests. |

Full-stack morphology extensions should define the mathematical invariant first:
tree type, node-id space, altitude ordering, payload shape, differentiability
assumption, and whether the change affects inference semantics. Do not hide a
new reconstructed signal inside a scorer; if the node signal or derivative
changes, it needs an explicit forward and backward contract.

## Component Boundaries

Scoring decides one score per node from normalized node features. It should not
build trees, recompute morphology attributes, or inspect target labels.

Score constraints post-process scores before reconstruction. They are part of
inference semantics and therefore belong in the inference contract.

Regularizers compute training penalties. They do not change the forward output
unless the training loop explicitly adds `layer.regularization_penalty(x)` to
the task loss.

Loss design belongs to the training loop. Target-dependent terms should remain
explicit in experiment code or dedicated training utilities rather than being
hidden inside scoring, constraints, or current regularizers. See
[CFP Loss Design](cfp-loss-design.md).

Normalization converts raw node attributes into stable feature tensors. It is
shared infrastructure used before scoring and regularization. Statistical
normalization modes (`dataset_clipped_zscore01`, `dataset_minmax01`, and `dataset_zscore`) require
dataset-level statistics fit before training or loaded before inference.
`scale_mode="none"` is the explicit opt-out for tests, diagnostics, or
externally scaled attributes.

Serialization preserves reproducible layer construction, checkpoint parameter
names, normalization statistics, and exported parameter artifacts.

## Altitude Signal

CFP reconstructs a fixed altitude signal. For each tree node, the backend
provides `info["residues"]`; the scorer only learns how strongly each residue
contributes to reconstruction:

```text
filtered_increment(node) = residue(node) * score(node)
output = reconstruct(filtered_increment)
```

This keeps the layer aligned with the classical connected-filter
interpretation and keeps the differentiable boundary clear. The gradient that
reaches a score is scaled by the residue it controls:

```text
dL/dscore(node) = residue(node) * dL/dfiltered_increment(node)
```

Alternative output projections, top-hat outputs, or attribute reconstructions
are not current CFP extension points in Python. Changing the reconstructed
morphology signal requires a separate forward and backward contract; it should
not be hidden inside a filter spec.

## Configs, Contracts, and State

`get_config()` returns the architecture needed by `from_config()`. It includes:

- input channels;
- filter specs;
- scoring config;
- score constraints;
- training regularizers;
- normalization settings;
- tree-of-shapes settings.

`get_contracts()` returns three named views:

- `parameter_contract`: parameter names and shapes;
- `inference_contract`: settings that define forward semantics;
- `training_contract`: training-only penalties.

Do not put caches, dataset statistics, or runtime tensors in `get_config()`.
Use `save_stats()` and `load_stats()` for normalization statistics.

The layer cache is a runtime optimization, not serialized state. Use
`cached_sample_count()` for inspection rather than depending on cache internals.

## Public API and Utilities

Preserve these public entry points unless the project intentionally makes a
breaking API change:

| Entry point | Purpose |
| --- | --- |
| `from mtlearn.layers import ConnectedFilterPreprocessingLayer` | Public layer import. |
| `ConnectedFilterPreprocessingImplicitJacobianFunction` | Public differentiable reconstruction function. |
| `build_dataloader_cached` | Build tree payload cache while estimating normalization statistics. |
| `build_dataloader_cached_fixed_stats` | Build cache while keeping existing normalization statistics fixed. |
| `inspect_training_sample` | Debug tree payloads, attributes, and scores for one sample. |
| `cached_sample_count` | Inspect cache size without accessing cache internals. |
| `get_config` and `from_config` | Round-trip layer architecture. |
| `get_contracts` | Inspect parameter, inference, and training contracts together. |
| `get_parameter_contract` | Inspect trainable CFP parameter names and shapes. |
| `get_inference_contract` | Inspect forward-semantics settings. |
| `get_training_contract` | Inspect training-only settings. |
| `save_stats` and `load_stats` | Persist normalization statistics. |
| `export_params` | Export learned parameter artifacts. |

Default linear scorer parameter names are part of the current parameter
contract: `_weights.<spec_name>` and `_biases.<spec_name>`.

When moving behavior into helpers, keep the layer methods as the public surface
when notebooks or downstream experiments already call them.

## Testing Checklist

Use the smallest test that proves the contract:

- import tests for public paths;
- validation tests for bad config values and unknown kinds;
- forward tests on small hand-computable images;
- shape tests for `(B, C, H, W)` and cached dataloader input;
- gradient tests when a component changes differentiable behavior;
- config round-trip tests for `get_config()` and `from_config()`;
- contract tests for `get_contracts()` and checkpoint parameter names;
- notebook smoke tests for public examples when user-facing workflows change.

Useful commands from the repository root:

```bash
python -m compileall -q mtlearn/python/mtlearn mtlearn/tests/python
PYTHONPATH=mtlearn/python:build/mtlearn/bindings python -m pytest -q -m "not gradcheck" mtlearn/tests/python
PYTHONPATH=mtlearn/python:build/mtlearn/bindings python -m pytest -q -m gradcheck mtlearn/tests/python
python -m sphinx -b html docs/source /tmp/mtlearn-docs-build
git diff --check
```

Use gradcheck when reconstruction or scorer differentiability changes. Use
notebook validation when a change affects published experiments or paper-facing
examples.
