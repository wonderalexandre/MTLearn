"""Version metadata configuration checks."""

from __future__ import annotations

import ast
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None


REPO_ROOT = Path(__file__).resolve().parents[3]


def _pyproject_text() -> str:
    return (REPO_ROOT / "pyproject.toml").read_text()


def _table_block(text: str, table: str) -> str:
    header = f"[{table}]"
    lines: list[str] = []
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == header:
            in_table = True
            continue
        if in_table and stripped.startswith("["):
            break
        if in_table:
            lines.append(line)
    if not in_table:
        raise AssertionError(f"{header} not found in pyproject.toml")
    return "\n".join(lines)


def test_project_version_is_dynamic() -> None:
    text = _pyproject_text()
    if tomllib is not None:
        metadata = tomllib.loads(text)
        project = metadata["project"]
        assert project["dynamic"] == ["version"]
        assert "version" not in project
        return

    project = _table_block(text, "project")
    assert 'dynamic = ["version"]' in project
    assert not any(line.strip().startswith("version =") for line in project.splitlines())


def test_setuptools_scm_supplies_scikit_build_version() -> None:
    text = _pyproject_text()
    if tomllib is not None:
        metadata = tomllib.loads(text)
        setuptools_scm = metadata["tool"]["setuptools_scm"]
        expected_scm = {
            "write_to": "mtlearn/python/mtlearn/_version.py",
            "version_scheme": "post-release",
            "local_scheme": "no-local-version",
            "tag_regex": r"^v?(?P<version>\d+\.\d+\.\d+(?:[.-].*)?)$",
        }
        for key, value in expected_scm.items():
            assert setuptools_scm[key] == value

        assert metadata["tool"]["scikit-build"]["metadata"]["version"] == {
            "provider": "scikit_build_core.metadata.setuptools_scm",
        }
        return

    setuptools_scm = _table_block(text, "tool.setuptools_scm")
    assert 'write_to = "mtlearn/python/mtlearn/_version.py"' in setuptools_scm
    assert 'version_scheme = "post-release"' in setuptools_scm
    assert 'local_scheme = "no-local-version"' in setuptools_scm
    assert 'tag_regex = "^v?(?P<version>\\\\d+\\\\.\\\\d+\\\\.\\\\d+(?:[.-].*)?)$"' in setuptools_scm

    scikit_metadata = _table_block(text, "tool.scikit-build.metadata")
    assert 'version = { provider = "scikit_build_core.metadata.setuptools_scm" }' in scikit_metadata


def test_public_version_uses_generated_version_module() -> None:
    module = ast.parse((REPO_ROOT / "mtlearn/python/mtlearn/__init__.py").read_text())

    for node in module.body:
        if not isinstance(node, ast.Try):
            continue

        imports_generated_version = any(
            isinstance(statement, ast.ImportFrom)
            and statement.module == "_version"
            and statement.level == 1
            and any(alias.name == "version" and alias.asname == "__version__" for alias in statement.names)
            for statement in node.body
        )
        fallback_unknown = any(
            isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__version__" for target in statement.targets)
            and isinstance(statement.value, ast.Constant)
            and statement.value.value == "0+unknown"
            for handler in node.handlers
            for statement in handler.body
        )
        if imports_generated_version and fallback_unknown:
            return

    raise AssertionError("mtlearn.__version__ must come from generated _version.py with a local fallback")
