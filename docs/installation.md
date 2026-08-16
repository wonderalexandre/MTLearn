# MTLearn Installation

This page covers installation paths for **MTLearn**. The published Python
package and import namespace remain `mtlearn`.

For source builds, tests, notebook validation, and releases, see
[development.md](development.md).

## Navigation

- [Development and validation](development.md)

## Install From PyPI

```bash
pip install mtlearn
```

Verify the installation:

```bash
python - <<'PY'
import mtlearn
from mtlearn import morphology

print(mtlearn.__version__)
print(morphology.AttributeType.AREA)
PY
```

## Runtime Dependency Notes

The `mtlearn` package supports NumPy 1.x and 2.x and does not depend on
`scikit-learn` at runtime.

The native `_mtlearn` extension links against LibTorch, so the package declares
tested PyTorch ranges per Python version and platform. PyTorch no longer
publishes recent macOS Intel wheels, so that platform intentionally uses the
newest available 2.2.x line for the supported Python versions.

| Platform | Python | PyTorch requirement |
| --- | --- | --- |
| macOS Intel | 3.9 through 3.12 | `torch>=2.2.2,<2.3` |
| macOS arm64 | 3.9 | `torch>=2.8,<2.9` |
| macOS arm64 | 3.10 through 3.13 | `torch>=2.10,<2.12` |
| macOS arm64 | 3.14 | `torch>=2.11,<2.12` |
| Linux and Windows | 3.9 | `torch>=2.8,<2.9` |
| Linux and Windows | 3.10 through 3.13 | `torch>=2.10,<2.12` |
| Linux and Windows | 3.14 | `torch>=2.11,<2.12` |

Release wheels are built against the lower bound for each row. CI also tests
the generated wheel against supported newer Torch runtimes before upload.

## Notebook Dependencies

Install the notebook extras when you want to run the public examples:

```bash
pip install "mtlearn[notebooks]"
```

The `notebooks` extra installs `ipykernel`, `matplotlib`, `nbformat`,
`papermill`, `pandas`, `scikit-learn`, `scipy`,
`segmentation-models-pytorch`, and `tqdm`. JupyterLab is optional and only
needed for interactive notebook editing:

```bash
pip install jupyterlab
```

Notebook files are not installed with the PyPI package. Clone the repository to
run the public notebooks. The main public example is:

```text
notebooks/experiments/CFP_linear_vs_mlp_scoring_screws_segmentation.ipynb
```

Repository notebooks can download public registered datasets with the dataset
helper:

```bash
python scripts/download_data.py --list
python scripts/download_data.py screws_segmentation
```

For ICPR 2026 RRPR review, dataset access rules, notebook links, and execution
commands, use `notebooks/ICPR2026/README.md` as the canonical guide. To run
those notebooks, clone the repository or use a sparse checkout so the package
sources and helper scripts are available.

## Source Checkout

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/wonderalexandre/MTLearn.git
cd MTLearn
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

The current backend source is expected at:

```text
external/mmcfilters
```

## Editable Install

For local development, either:

- if your environment already has the exact PyTorch stack you want (recommended for
  CUDA/CuDNN-aware setups), use `--torch none`;
- if you created a fresh environment, let the helper install the minimum supported
  Torch build (default) and install the matching CMake/Torch toolchain for you.

Use one of the following flows:

Existing torch stack (keep it):

```bash
python scripts/install_release_dependencies.py --build-tools --torch none
pip install -e . --no-build-isolation --no-deps
```

Fresh environment (installs Torch via helper):

```bash
python scripts/install_release_dependencies.py --build-tools
pip install -e . --no-build-isolation --no-deps
```

For notebooks from a source checkout, use the `notebooks` extra in the editable
install. If your environment already has the desired Torch build, add
`--torch none`:

```bash
python scripts/install_release_dependencies.py --build-tools
pip install -e ".[notebooks]" --no-build-isolation
```

```bash
python scripts/install_release_dependencies.py --build-tools --torch none
pip install -e ".[notebooks]" --no-build-isolation
```

## Next Steps

- See the root [README](../README.md) for the project overview and quick start.
- See [development.md](development.md) for source builds, tests, notebook
  validation, and releases.
