# CFP Morphological Behavior

This note records options in `ConnectedFilterPreprocessingLayer` that make the
learnable connected-filter preprocessing layer closer to classical
morphological behavior.

The CFP layer is still a learnable relaxation: tree construction, attributes,
proper-part ownership, and traversal metadata are computed outside autograd,
while gradients flow through the node scores and the reconstruction map.

## `preserve_root`

```python
from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer

layer = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    filter_specs=[
        {
            "tree_type": morphology.TreeType.MAX_TREE,
            "attributes": (morphology.AttributeType.AREA,),
            "preserve_root": True,
        }
    ],
)
```

`preserve_root=True` in a filter spec forces that spec's root-node gate to one
during the implicit CFP forward pass. This keeps the root altitude increment
unfiltered for that filter only.

The default is `False` per filter spec to preserve historical CFP behavior and
old checkpoints. With the default setting, the root node is scored by the same
sigmoid gate as all other nodes.

### Why It Matters

In component trees and trees of shapes, the root carries the global baseline of
the reconstructed signal. If the root gate is allowed to drop below one, even a
constant image can be attenuated by the learnable score. Preserving the root is
therefore a simple way to keep constant-signal reconstruction aligned with
classical morphological filtering intuition.

### Differentiability

The option remains compatible with backpropagation. Non-root node scores keep
their usual gradient. The root score is fixed to one, so it does not contribute
a score-gradient term for the learnable weight or bias.

### Serialization

`preserve_root` is included inside each serialized filter spec in:

- `get_config()`;
- `get_weight_contract()`;
- `export_params()`;
- `from_config()`.

It is intentionally not a top-level layer option. Specs that do not contain the
field restore it as `False`.

## `monotonicity_weight`

```python
task_loss = criterion(model(inputs), targets)
mono_loss = layer.regularization_penalty(inputs)
loss = task_loss + mono_loss
```

`monotonicity_weight` is an optional non-negative scalar inside each filter
spec. The default is `0.0`, so legacy CFP code is unchanged unless a spec opts
in and the training loop explicitly adds `regularization_penalty(...)` to the
objective.

For every active spec, the regularizer computes node gates from the normalized
attributes and penalizes child gates that are larger than their parent gate:

```text
mean(relu(score_child - score_parent)^2)
```

The result is multiplied by that spec's `monotonicity_weight`. Across an input
batch, the layer averages over samples/channels and sums the active spec terms.
This encourages the learned criterion to behave like a connected attribute
filter: if a parent component is rejected, descendants should not become more
accepted than the parent. The term is differentiable with respect to the spec's
weight vector and bias because it uses the same sigmoid gate scores as the CFP
layer.

When `preserve_root=True`, the root gate is fixed to one for both `forward` and
the monotonicity regularizer. This avoids an artificial root penalty while still
regularizing all non-root parent-child edges.

### Serialization

`monotonicity_weight` is serialized inside each filter spec in `get_config()`
and in the training contract emitted by `export_params()`. It is intentionally
not part of `get_weight_contract()` because regularization changes training
behavior, not inference weight compatibility. It is also intentionally not a
top-level layer option.
