"""config.yml, token, unit systemd e migração do cache HF."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from vramd.cli import cli
from vramd.config import (
    VramdConfig,
    apply_env,
    ensure_token,
    load_config,
    migrate_models,
    resolve_path_for_user,
    systemd_unit,
    validate_user,
    write_config,
    write_secret,
)


class TestConfigFile:
    def test_roundtrip(self, tmp_path: Path) -> None:
        cfg = VramdConfig.for_current_user()
        cfg.listen = "0.0.0.0:9000"
        cfg.cache_dir = str(tmp_path / "cache")
        cfg.models_home = str(tmp_path / "hf")
        cfg.allow_client_code = False
        path = write_config(cfg, tmp_path / "config.yml")
        loaded = load_config(path, strict=True)
        assert loaded.listen == cfg.listen
        assert loaded.cache_dir == cfg.cache_dir
        assert loaded.models_home == cfg.models_home
        assert loaded.allow_client_code is False

    def test_nested_models(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yml"
        path.write_text(
            "listen: 127.0.0.1:1\nmodels:\n  home: /x/hf\n  policy: off\n  max_gib: 10\nsessions:\n  ttl_hours: 5\n",
            encoding="utf-8",
        )
        cfg = load_config(path, strict=True)
        assert cfg.models_home == "/x/hf"
        assert cfg.models_policy == "off"
        assert cfg.models_max_gib == 10
        assert cfg.session_ttl_hours == 5

    def test_missing_explicit_is_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yml", strict=True)

    def test_missing_default_is_defaults(self) -> None:
        cfg = load_config()
        assert cfg.listen


class TestSecrets:
    def test_token_is_0600_and_stable(self, tmp_path: Path) -> None:
        path = ensure_token(tmp_path / "token")
        assert path.stat().st_mode & 0o777 == 0o600
        first = path.read_text()
        ensure_token(path)
        assert path.read_text() == first

    def test_write_secret(self, tmp_path: Path) -> None:
        path = write_secret(tmp_path / "hf-token", "hf_secret")
        assert path.read_text().strip() == "hf_secret"
        assert path.stat().st_mode & 0o777 == 0o600


class TestEnv:
    def test_apply_env(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("VRAMD_MODELS_POLICY", raising=False)
        cfg = VramdConfig.for_current_user()
        cfg.models_home = str(tmp_path / "hf")
        cfg.models_policy = "off"
        apply_env(cfg)
        assert os.environ["HF_HOME"] == str(tmp_path / "hf")
        assert os.environ["VRAMD_MODELS_POLICY"] == "off"


class TestSystemd:
    def test_unit_fields(self, tmp_path: Path) -> None:
        cfg = VramdConfig.for_current_user()
        cfg.cache_dir = str(tmp_path / "cache")
        cfg.models_home = str(tmp_path / "hf")
        cfg.service_user = "root"
        unit = systemd_unit(cfg, config_file=tmp_path / "config.yml")
        assert "ExecStart=" in unit
        assert "vramd start --config" in unit
        assert "User=root" in unit
        assert "Restart=on-failure" in unit
        assert f"HF_HOME={tmp_path / 'hf'}" in unit


class TestMigration:
    def test_migrate_models(self, tmp_path: Path) -> None:
        old = tmp_path / "old"
        (old / "hub").mkdir(parents=True)
        (old / "hub" / "x").write_text("x")
        cfg = VramdConfig.for_current_user()
        cfg.old_cache = str(old)
        cfg.models_home = str(tmp_path / "hf")
        plan = migrate_models(cfg, dry_run=True)
        assert plan and plan["files"] == 1
        result = migrate_models(cfg)
        assert result and not old.exists()
        assert (tmp_path / "hf" / "hub" / "x").exists()


class TestUsers:
    def test_root_user(self) -> None:
        ids = validate_user("root")
        if ids is None:
            pytest.skip("pwd indisponível")
        assert ids == (0, 0)

    def test_unknown_user(self) -> None:
        with pytest.raises(ValueError):
            validate_user("definitely-not-a-user-xyz")

    def test_resolve_home_for_current_user(self) -> None:
        import getpass

        user = getpass.getuser()
        assert resolve_path_for_user("~/.cache/hf", user) == str(Path.home() / ".cache" / "hf")
        assert resolve_path_for_user("/abs/path", user) == "/abs/path"


class TestConfigCli:
    def test_non_interactive_write(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yml"
        token = tmp_path / "token"
        cache = tmp_path / "cache"
        hf = tmp_path / "hf"
        result = CliRunner().invoke(
            cli,
            [
                "config",
                "--non-interactive",
                "--file",
                str(config),
                "--listen",
                "127.0.0.1:8788",
                "--cache-dir",
                str(cache),
                "--token-file",
                str(token),
                "--hf-home",
                str(hf),
                "--no-install-service",
                "--no-move-models",
                "--no-allow-client-code",
            ],
        )
        assert result.exit_code == 0, result.output
        assert config.is_file()
        assert token.is_file()
        loaded = load_config(config, strict=True)
        assert loaded.listen == "127.0.0.1:8788"
        assert loaded.allow_client_code is False

    def test_show(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yml"
        config.write_text("listen: 127.0.0.1:8789\n", encoding="utf-8")
        result = CliRunner().invoke(cli, ["config", "--show", "--file", str(config)])
        assert result.exit_code == 0
        assert "127.0.0.1:8789" in result.output
