"""Version metadata consistency checks."""

from __future__ import annotations

import ast
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None


REPO_ROOT = Path(__file__).resolve().parents[3]


def _project_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    if tomllib is not None:
        return tomllib.loads(text)["project"]["version"]

    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[project]":
            in_project = True
            continue
        if in_project and stripped.startswith("["):
            break
        if in_project and stripped.startswith("version"):
            _, value = stripped.split("=", 1)
            return ast.literal_eval(value.strip())
    raise AssertionError("project.version not found in pyproject.toml")


def _public_version() -> str:
    module = ast.parse((REPO_ROOT / "mtlearn/python/mtlearn/__init__.py").read_text())
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    raise AssertionError("__version__ not found in mtlearn/__init__.py")


def test_public_version_matches_project_metadata() -> None:
    assert _public_version() == _project_version()
