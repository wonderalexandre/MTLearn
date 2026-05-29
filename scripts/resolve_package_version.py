#!/usr/bin/env python3
"""Print the mtlearn version resolved from Git tags."""

from __future__ import annotations

from pathlib import Path

from setuptools_scm import get_version


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    version = get_version(
        root=REPO_ROOT,
        version_scheme="post-release",
        local_scheme="no-local-version",
        tag_regex=r"^v?(?P<version>\d+\.\d+\.\d+(?:[.-].*)?)$",
    )
    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
