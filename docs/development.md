# MTLearn Development

This page documents source builds, validation, and release checks for
**MTLearn**. The Python package and import namespace remain `mtlearn`.

For basic installation and notebook dependency setup, see
[installation.md](installation.md).

## Navigation

- [Installation](installation.md)

## Build Requirements

Source builds require a working native build environment:

- Python;
- a C++ compiler supported by CMake;
- CMake;
- PyTorch;
- pybind11;
- scikit-build-core.

**MTLearn** uses a native `_mtlearn` extension, so source builds require a C++
build toolchain. Python bindings currently expose Torch tensors, so building the
Python extension also requires Torch support.

Documentation builds use these Python dependencies:

```bash
pip install "myst-parser>=2" "sphinx>=7" "sphinx-autodoc-typehints>=1.25"
```

## Local Dependency Setup

Install the PyTorch build you intend to use before building `mtlearn`. For CUDA
environments, keep `torch`, `torchvision`, and `torchaudio` on matching wheel
builds from the same PyTorch index.

The release dependency helper defaults to the minimum supported Torch version.
On Linux, that default is a CPU wheel and can replace an existing CUDA Torch
installation. For everyday development in an environment that already has the
right Torch build, preserve it with `--torch none`:

```bash
python scripts/install_release_dependencies.py --build-tools --torch none
pip install -e . --no-build-isolation --no-deps
```

Use the helper without `--torch none` only when you explicitly want the
minimum-supported CPU Torch build used by release checks. Use `--no-deps` on
local `mtlearn` installs when preserving an existing Torch stack; otherwise pip
may replace a newer or CUDA-enabled Torch build to satisfy package metadata.

## Wheel Build

```bash
python scripts/install_release_dependencies.py --build-tools --torch none
python -m build --wheel --no-isolation
python -m pip install dist/mtlearn-*.whl --no-deps
```

This wheel build path assumes PyTorch is already installed in the active
environment as described above.

The `mtlearn` Python package uses the native `_mtlearn` extension. The top-level
`mmcfilters` Python package is not a runtime dependency of `mtlearn`.

## CMake Builds

### Minimum C++-Only Build

```bash
cmake -S . -B build-cpp \
      -DMTLEARN_BUILD_PYTHON=OFF \
      -DMTLEARN_WITH_TORCH=OFF
cmake --build build-cpp
```

### C++/Python Test Build

```bash
cmake -S . -B build \
      -DMTLEARN_BUILD_TESTS=ON \
      -DMTLEARN_BUILD_PYTHON=ON \
      -DMTLEARN_WITH_TORCH=ON \
      -DMTLEARN_ENABLE_EMBED=OFF \
      -DPYTHON_EXECUTABLE=$(python -c "import sys; print(sys.executable)") \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch, pybind11; print(torch.utils.cmake_prefix_path + ";" + pybind11.get_cmake_dir())')"
cmake --build build
ctest --test-dir build --output-on-failure
```

The `CMAKE_PREFIX_PATH` expression locates both LibTorch and pybind11 from the
active Python environment.

`MTLEARN_ENABLE_EMBED=ON` activates the embedded-interpreter path in
`mtl_interpreter_test`. Before enabling it, verify that the selected environment
can import PyTorch:

```bash
python -c "import torch"
```

## CMake Options

- `MTLEARN_BUILD_PYTHON`: build the `_mtlearn` pybind11 extension. Default: `ON`.
- `MTLEARN_BUILD_TESTS`: build and register tests. Default: `OFF`.
- `MTLEARN_WITH_TORCH`: enable LibTorch-dependent code. Default: `ON`.
- `MTLEARN_ENABLE_EMBED`: enable embedded Python interpreter test behaviour.
  Default: `OFF`.
- `MTLEARN_ENABLE_ASSERTS`: keep runtime assertions enabled in core C++ code.
  Default: `OFF`.

`MTLEARN_BUILD_PYTHON=ON` currently requires `MTLEARN_WITH_TORCH=ON` because the
bindings expose Torch tensors.

## Documentation Builds

Build the Sphinx documentation directly from the source checkout:

```bash
PYTHONPATH="$PWD/mtlearn/python" \
python -m sphinx -W -b html docs/source build-docs/docs/mtlearn/site
```

The generated site is written to:

```text
build-docs/docs/mtlearn/site
```

The Python API pages are generated from docstrings. The C++ pages document the
installed public facade in `mtlearn/morphology.hpp`; the root CMake project does
not currently define mtlearn-specific Doxygen targets.

The external backend submodule may expose its own documentation targets. Inspect
the configured CMake build if you need backend documentation:

```bash
cmake --build build --target help
```

## Public API Notes

The README shows the main Python entry points. For C++ consumers, the public
weighted-tree type is:

```cpp
mtlearn::morphology::WeightedTree
```

It wraps the current backend so C++ consumers do not depend directly on
`mmcfilters::WeightedMorphologicalTree`. The public header
`mtlearn/morphology.hpp` owns the public morphology enums and does not include
`mmcfilters` headers.

Installed consumers should link only against:

```cmake
find_package(mtlearn CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE mtlearn::core)
```

## Validation

### Direct Python Tests

```bash
pip install "pytest>=8"
pip install -e . --no-build-isolation --no-deps
PYTHONPATH=mtlearn/python:build/mtlearn/bindings python -m pytest -q -m "not gradcheck" mtlearn/tests/python
PYTHONPATH=mtlearn/python:build/mtlearn/bindings python -m pytest -q -m gradcheck mtlearn/tests/python
```

Whitespace and syntax checks:

```bash
python -m compileall -q mtlearn/python/mtlearn mtlearn/tests/python
git diff --check
```

### Notebook Validation

From a source checkout with a local CMake build:

```bash
python scripts/install_release_dependencies.py --build-tools --torch none
pip install -e ".[notebooks]" --no-build-isolation
python scripts/validate_notebooks.py --bindings-dir build/mtlearn/bindings
```

The script executes the full gradcheck notebooks and creates reduced temporary
smoke copies for long experiment notebooks. Source notebooks are not modified.
By default, executed outputs are written to:

```text
/tmp/mtlearn-notebook-validation
```

To validate against an installed wheel instead of the checkout:

```bash
python scripts/validate_notebooks.py --installed-package
```

## Release Process

Releases are built by GitHub Actions, but PyPI publication is manual.

For a production release:

1. Make sure the `CI`, `Package`, and `Notebooks` workflows are green on
   `main`.
2. Choose the release version. Package versions are resolved from Git tags by
   `setuptools_scm`; `pyproject.toml` does not contain a fixed version field.
3. Create and push a semantic version tag matching the resolved package version,
   for example `v1.0.0`.
4. The `Release` workflow builds the source distribution and supported platform
   wheels, checks the package metadata, and attaches the artifacts to a GitHub
   Release.

The workflow rejects a tag when the tag version does not match the package
version resolved by `scripts/resolve_package_version.py`.

The release wheel matrix currently produces 22 wheels and targets Python 3.9
through 3.14 on:

- Linux manylinux x86_64;
- Windows x86_64;
- macOS arm64;
- macOS Intel x86_64 for Python 3.9 through 3.12.

macOS Intel wheels for Python 3.13 and 3.14 are not built because PyTorch does
not publish stable `macosx_x86_64` wheels for those Python versions.

PyPI upload is intentionally manual. Download the release artifacts from the
GitHub Release or from the workflow run, then publish them with:

```bash
python -m pip install --upgrade twine
python -m twine upload dist/*
```

Manual runs of the `Release` workflow build downloadable artifacts without
creating a GitHub Release.

Before publishing from the public repository, validate a clean clone:

```bash
git clone --recurse-submodules https://github.com/wonderalexandre/MTLearn.git
cd MTLearn
cmake -S . -B build-cpp \
      -DMTLEARN_BUILD_PYTHON=OFF \
      -DMTLEARN_WITH_TORCH=OFF \
      -DMTLEARN_BUILD_TESTS=ON
cmake --build build-cpp --parallel
ctest --test-dir build-cpp --output-on-failure
python scripts/install_release_dependencies.py --build-tools
python -m build --wheel --no-isolation
wheel="$(ls dist/mtlearn-*.whl | head -n 1)"
python -m pip install "${wheel}[notebooks]"
python scripts/validate_notebooks.py --installed-package
```
