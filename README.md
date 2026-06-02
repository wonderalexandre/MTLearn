# MTLearn

[![CI](https://github.com/wonderalexandre/MTLearn/actions/workflows/ci.yml/badge.svg)](https://github.com/wonderalexandre/MTLearn/actions/workflows/ci.yml)
[![Package](https://github.com/wonderalexandre/MTLearn/actions/workflows/package.yml/badge.svg)](https://github.com/wonderalexandre/MTLearn/actions/workflows/package.yml)

**MTLearn** (*Morphological Tree Learning*) is a C++/Python research library for
learnable connected operators based on morphological trees. The Python package
is published as `mtlearn`.

The library explores a simple idea: connected morphology can become a structural
prior for deep neural networks. Instead of processing images only through local
pixel-wise operations, connected operators reason over components, regions,
shape, contrast, and hierarchy. This makes them naturally interpretable and
well-suited for tasks where structure matters.

Classical connected filters are powerful, but they usually depend on hard
keep/discard decisions and manually selected attribute thresholds. This limits
their integration into end-to-end trainable neural architectures.

**MTLearn** provides a stable implementation platform for this research direction.
It currently includes Connected Filter Preprocessing (CFP), and is intended to
grow toward trainable connected-operator layers, differentiable or learnable
attribute criteria, self-dual tree representations, intermediate network
insertions, and scalable implementations.

## Main Features

- **Connected Filter Preprocessing (CFP):** the current main model, available as
  `mtlearn.layers.ConnectedFilterPreprocessingLayer`. CFP replaces hard
  connected-filter decisions with a differentiable sigmoid gate over normalized
  tree-node attributes.

- **Stable morphology interface:** `mtlearn.morphology` builds max-trees,
  min-trees, and tree-of-shapes through a backend-independent API.

- **Trainable connected morphology:** designed as an implementation platform for
  connected morphology as a learnable structural prior in deep neural networks.

- **Research-ready validation:** includes C++ tests, Python tests, gradient
  checks, reference implementations, notebook validations, and public dataset
  download helpers.

## Install

The Python package is available from PyPI as `mtlearn`:

```bash
pip install mtlearn
```

See [docs/installation.md](docs/installation.md) for installation instructions
and [docs/development.md](docs/development.md) for source builds, validation,
and releases.
For the public scalar attribute and group catalog, see
[docs/source/concepts/attributes.md](docs/source/concepts/attributes.md).

## Quick Start

Build a morphology tree and compute attributes:

```python
import numpy as np
from mtlearn import morphology

image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
tree = morphology.create_max_tree(image)

_, attributes = morphology.compute_attributes(
    tree,
    [morphology.AttributeType.AREA, morphology.AttributeType.COMPACTNESS],
)

print(attributes.shape)
```

Create a CFP layer and run a forward pass:

```python
import torch
from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer

cfp_layer = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    filter_specs=[
        {
            "tree_type": morphology.TreeType.MAX_TREE,
            "attributes": (
                morphology.AttributeType.AREA,
                morphology.AttributeType.CIRCULARITY,
            ),
        }
    ],
    device="cpu",
    scale_mode="none",
)

x = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32)
y = cfp_layer(x)

assert y.shape == x.shape
```

For self-dual preprocessing, use the tree-of-shapes backend explicitly:

```python
cfp_tos = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    filter_specs=[
        {
            "tree_type": morphology.TreeType.TREE_OF_SHAPES,
            "attributes": (morphology.AttributeGroup.SHAPE,),
            "tos_interpolation": "self-dual",
        }
    ],
)
```

## CFP Filter Specs

`ConnectedFilterPreprocessingLayer` is configured with `filter_specs`. Each
specification creates one output channel per input channel and owns the
morphology tree and scoring attributes:

```python
from mtlearn import morphology

filter_specs = [
    {
        "name": "max_area_gray",
        "tree_type": morphology.TreeType.MAX_TREE,
        "attributes": (
            morphology.AttributeType.AREA,
            morphology.AttributeType.GRAY_HEIGHT,
        ),
    },
    {
        "name": "tos_boundary",
        "tree_type": morphology.TreeType.TREE_OF_SHAPES,
        "attributes": (morphology.AttributeGroup.BOUNDARY,),
        "tos_interpolation": "self-dual",
    },
    {
        "name": "min_area",
        "tree_type": morphology.TreeType.MIN_TREE,
        "attributes": (morphology.AttributeType.AREA,),
    },
]
```

`name` is optional. When provided, it becomes the stable parameter key for that
filter spec; when omitted, the layer uses `spec_000`, `spec_001`, and so on.

The sigmoid logits are unclamped by default. Pass `clamp=12` to clamp
`beta_f * logits` to `[-12, 12]`, or pass an explicit pair such as
`clamp=(-8, 10)`.


## Examples and Notebooks

Executable examples are available in `notebooks/`.

Install notebook dependencies with:

```bash
pip install "mtlearn[notebooks]"
```

The main public experiment example is:

```text
notebooks/experiments/Example_screws_filtering.ipynb
```

Representative ICPR 2026 experiment notebooks are available in
[notebooks/ICPR2026](notebooks/ICPR2026/README.md).

## Implementation Notes

`ConnectedFilterPreprocessingLayer` is the recommended implementation for new
CFP experiments.

`ConnectedFilterPreprocessingLayerLegacy` remains available for loading or
reproducing experiments that used the former global tree/output contract.

For PyTorch checkpoints, use the helper functions in `mtlearn.layers`. CFP
trainable weights are regular PyTorch parameters, and the primary layer stores
its serializable config and dataset normalization statistics in PyTorch extra
state:

```python
from mtlearn.layers import (
    ConnectedFilterPreprocessingLayer,
    load_checkpoint,
    save_checkpoint,
)

save_checkpoint("model.pt", model)

def build_model():
    return Model(
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            filter_specs=filter_specs,
        ),
        build_backbone(),
    )

model, checkpoint = load_checkpoint("model.pt", build_model, device=device)
```

The helpers discover CFP layers by module name, save their configs next to the
normal model `state_dict`, and let the CFP extra state validate compatibility
and restore dataset normalization statistics during `load_state_dict`. When the
model constructor cannot hard-code the CFP configuration, the load factory may
instead accept one `cfp_configs` argument and call
`ConnectedFilterPreprocessingLayer.from_config(...)`.

Checkpoints do not persist per-sample tree, attribute, or normalization caches;
those are rebuilt from input data. `export_params()`/`save_params()` are manual
inspection helpers for CFP weights and metadata, not the recommended full-model
checkpoint API.

Tensor operations, trainable parameters, and cached attributes can live on CUDA
when `device="cuda"`. Morphology-tree construction is still performed by the
C++ backend on CPU.

The main implementation uses an implicit Jacobian formulation. The dense
region-pixel matrix is not materialized during normal training; tree-ordering
metadata is used to perform the equivalent reconstruction and backward
accumulation more compactly. This reduces memory pressure compared with
explicit region-pixel Jacobian construction.

Reference implementations based on explicit Jacobians and CPU tree traversals
remain available for gradient checks, comparisons, and debugging.

**MTLearn** uses a C++ morphology backend internally through `mtlearn::morphology`.
User code should interact with morphology through the public Python facade
`mtlearn.morphology`, rather than depending on backend-specific APIs.

The backend is
[`MorphologicalAttributeFilters`](https://github.com/wonderalexandre/MorphologicalAttributeFilters)
/ `mmcfilters`, but the top-level Python package `mmcfilters` is not required
as a runtime dependency of `mtlearn`.

## Current Scope

**MTLearn** is a research-oriented library. CFP is the first validated member of
a broader planned family of trainable connected-operator layers. The current
implementation supports max-tree, min-tree, and tree-of-shapes CFP workflows,
mixed tree types in the same layer,
multi-attribute dataset-level attribute normalization, cached preprocessing,
and PyTorch forward/backward for CFP parameters on CPU or CUDA tensors.

## Citation

If you use the CFP layer in your work, please cite:

> Wonder A. L. Alves, Lucas de P. O. Santos, Ronaldo F. Hashimoto, Nicolas Passat, Anderson H. R. Souza, Dennis J. Silva, Yukiko Kenmochi. **A trainable connected filter preprocessing layer based on component trees.** International Conference on Pattern Recognition (ICPR), 2026, Lyon, France. ⟨[hal-05575141](https://hal.science/hal-05575141/)⟩
