# MTLearn

[![CI](https://github.com/wonderalexandre/MTLearn/actions/workflows/ci.yml/badge.svg)](https://github.com/wonderalexandre/MTLearn/actions/workflows/ci.yml)
[![Package](https://github.com/wonderalexandre/MTLearn/actions/workflows/package.yml/badge.svg)](https://github.com/wonderalexandre/MTLearn/actions/workflows/package.yml)

**MTLearn** (*Morphological Tree Learning*) is a C++/Python research library for learnable connected operators based on morphological trees. The Python package is published as `mtlearn`.

The library explores a simple idea: connected morphology can become a structural prior for deep neural networks. Instead of processing images only through local pixel-wise operations, connected operators reason over components, regions, shape, contrast, and hierarchy. This makes them naturally interpretable and well-suited for tasks where structure matters.

Classical connected filters are powerful, but they usually depend on hard keep/discard decisions and manually selected attribute thresholds. This limits their integration into end-to-end trainable neural architectures.

**MTLearn** provides a stable implementation platform for this research direction. It currently includes Connected Filter Preprocessing (CFP), and is intended to grow toward trainable connected-operator layers, differentiable or learnable attribute criteria, self-dual tree representations, intermediate network insertions, and scalable implementations.

## Main Features

- **Connected Filter Preprocessing (CFP):** the current main model, available as `mtlearn.layers.ConnectedFilterPreprocessingLayer`. CFP replaces hard connected-filter decisions with a differentiable sigmoid gate over normalized tree-node attributes.

- **Stable morphology interface:** `mtlearn.morphology` builds max-trees, min-trees, and tree-of-shapes through a backend-independent API.

- **PyTorch integration:** CFP layers are `torch.nn.Module` objects with trainable parameters, checkpoint helpers, and dataset-level normalization utilities.

- **Validation utilities:** includes C++ tests, Python tests, gradient checks, reference implementations, notebook utilities, and registered dataset download helpers.

## Install

The Python package is available from PyPI as `mtlearn`:

```bash
pip install mtlearn
```

See the guides for installation and source builds:

- [docs/installation.md](docs/installation.md)
- [docs/development.md](docs/development.md)

## Quick Start

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

For tree-of-shapes specs, normalization, caching, checkpoint helpers, and attribute selection, see the guides listed below.

## Documentation

The main documentation is built with Sphinx from `docs/source`.

- [Documentation home](https://wonderalexandre.github.io/MTLearn/)
- [docs/installation.md](docs/installation.md)
- [docs/development.md](docs/development.md)
- [Attribute guide](docs/source/concepts/attributes.md)
- [RRPR guide](notebooks/ICPR2026/README.md)

## Examples and Reproducibility

Executable examples are available in `notebooks/`. Notebook dependency setup is documented in [docs/installation.md](docs/installation.md).

For ICPR 2026 Reproducible Research in Pattern Recognition (RRPR) review, use [RRPR guide](notebooks/ICPR2026/README.md) as the canonical guide for setup, datasets, notebook execution, hardware notes, and current reproducibility scope.

## Scope and API Boundaries

**MTLearn** is a research-oriented library. CFP is the current validated trainable connected-operator layer, with support for max-tree, min-tree, and tree-of-shapes workflows; mixed tree specs; multi-attribute dataset-level normalization; cached preprocessing; and PyTorch forward/backward for CFP parameters on CPU or CUDA tensors.

User code should interact with morphology through the public Python facade `mtlearn.morphology` and the primary CFP layer in `mtlearn.layers`, rather than depending on backend-specific APIs. The backend is [`mmcfilters`](https://github.com/wonderalexandre/mmcfilters), but the top-level Python package `mmcfilters` is not required as a runtime dependency of `mtlearn`.

## Citation

If you use the CFP layer in your work, please cite:

> Wonder A. L. Alves, Lucas de P. O. Santos, Ronaldo F. Hashimoto, Nicolas Passat, Anderson H. R. Souza, Dennis J. Silva, Yukiko Kenmochi. **A trainable connected filter preprocessing layer based on component trees.** International Conference on Pattern Recognition (ICPR), 2026, Lyon, France. ⟨[hal-05575141](https://hal.science/hal-05575141/)⟩
