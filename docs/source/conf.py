"""Sphinx configuration for the mtlearn documentation."""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "mtlearn" / "python"))

build_bindings = ROOT / "build" / "mtlearn" / "bindings"
if build_bindings.exists():
    sys.path.insert(0, str(build_bindings))


project = "MTLearn"
author = "Wonder Alexandre Luz Alves"


def _resolve_release() -> str:
    try:
        from setuptools_scm import get_version

        return get_version(
            root=ROOT,
            version_scheme="post-release",
            local_scheme="no-local-version",
            tag_regex=r"^v?(?P<version>\d+\.\d+\.\d+(?:[.-].*)?)$",
        )
    except Exception:
        return os.environ.get("MTLEARN_DOCS_VERSION", "0+unknown")


release = _resolve_release()

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}


def _has_importable_spec(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


# The native extension and heavy runtime dependencies are optional for
# documentation-only builds. Mocking them lets autodoc import the Python facade
# and render docstrings before a full CMake Python-extension build exists.
autodoc_mock_imports = []
if not _has_importable_spec("torch"):
    autodoc_mock_imports.append("torch")
if not _has_importable_spec("numpy"):
    autodoc_mock_imports.append("numpy")
if not _has_importable_spec("cv2"):
    autodoc_mock_imports.append("cv2")
if not _has_importable_spec("_mtlearn"):
    autodoc_mock_imports.extend(["_mtlearn", "mtlearn._mtlearn"])
suppress_warnings = ["autodoc.mocked_object"]

nitpicky = False
