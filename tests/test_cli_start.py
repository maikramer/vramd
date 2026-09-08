"""Guard de arranque: um registry vazio recusa servir (--allow-empty para tests)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from vramd.cli import cli


class TestStartRefusesEmptyRegistry:
    def _bare_env(self, monkeypatch, tmp_path: Path) -> None:
        """Sem registry: package vazio, sem overlays do utilizador."""
        monkeypatch.delenv("VRAMD_BACKENDS_FILE", raising=False)
        monkeypatch.setenv("VRAMD_BACKENDS_DIR", str(tmp_path / "vazio"))

    def test_empty_registry_exits_nonzero_with_hint(self, tmp_path: Path, monkeypatch) -> None:
        self._bare_env(monkeypatch, tmp_path)
        result = CliRunner().invoke(cli, ["start", "--socket", str(tmp_path / "t.sock")])
        assert result.exit_code == 1
        assert "Nenhum backend configurado" in result.output
        # O diagnóstico diz COMO configurar (senão o operador fica na mesma).
        assert "VRAMD_BACKENDS_FILE" in result.output
        assert "--allow-empty" in result.output

    def test_allow_empty_proceeds_to_server(self, tmp_path: Path, monkeypatch) -> None:
        self._bare_env(monkeypatch, tmp_path)
        server = MagicMock()
        with patch("vramd.server.VramdServer", return_value=server):
            result = CliRunner().invoke(cli, ["start", "--socket", str(tmp_path / "t.sock"), "--allow-empty"])
        assert result.exit_code == 0
        server.serve_forever.assert_called_once()

    def test_registry_with_backends_starts_without_flag(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "base.yaml").write_text(
            "version: 2\nbackends:\n  - name: demo\n    adapter: pkg.demo\n    vram_mib: 1024\n    priority: 10\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("VRAMD_BACKENDS_FILE", str(tmp_path / "base.yaml"))
        monkeypatch.setenv("VRAMD_BACKENDS_DIR", str(tmp_path / "vazio"))
        server = MagicMock()
        with patch("vramd.server.VramdServer", return_value=server):
            result = CliRunner().invoke(cli, ["start", "--socket", str(tmp_path / "t.sock")])
        assert result.exit_code == 0
        assert "demo" in result.output
        server.serve_forever.assert_called_once()
