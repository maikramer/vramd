"""Consistência de versão: o literal __version__ acompanha o pyproject."""

from __future__ import annotations

import re
from pathlib import Path

import vramd


def _pyproject_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, "pyproject.toml sem version"
    return match.group(1)


class TestVersion:
    def test_dunder_version_matches_pyproject(self) -> None:
        # 0.3.2 saiu com __version__="0.3.1": `vramd --version` mentia.
        assert vramd.__version__ == _pyproject_version()
