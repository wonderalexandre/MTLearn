# MTLearn API Documentation

This documentation is organized around the public contracts that mtlearn
intends downstream users to depend on.

## Public Surfaces

- C++ morphology facade: `mtlearn/morphology.hpp`.
- Python morphology facade: `mtlearn.morphology`.
- PyTorch CFP layer: `mtlearn.layers.ConnectedFilterPreprocessingLayer`.
- Dataset and download helpers: `mtlearn.data` and `mtlearn.datasets`.

The current Doxygen target documents the C++ morphology facade first. Python
reference pages are kept here as a source-level map until the Sphinx/autodoc
site is added.

## Stability Rule

Public users should include only `mtlearn/morphology.hpp` on the C++ side and
should import morphology functionality through `mtlearn.morphology` on the
Python side. Backend headers, `_native`, `_backends`, and pybind implementation
details are internal migration points.

## API Sections

- [C++ API](cpp.md)
- [Python API Map](python.md)
