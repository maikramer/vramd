"""Ambientes por lock: describe, system/path e build com uv."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from vramd import protocol as P
from vramd.envs import EnvError, EnvManager, EnvSpec


class TestDescribe:
    def test_system_env(self, tmp_path: Path) -> None:
        manager = EnvManager(tmp_path / "envs")
        info = manager.describe(EnvSpec("main", system=True), tmp_path)
        assert info["kind"] == "system"
        assert info["python_path"] == sys.executable
        assert info["exists"] is True

    def test_path_env(self, tmp_path: Path) -> None:
        manager = EnvManager(tmp_path / "envs")
        info = manager.describe(EnvSpec("main", path=sys.executable), tmp_path)
        assert info["kind"] == "path"
        assert info["python_path"] == str(Path(sys.executable).resolve())
        assert info["exists"] is True

    def test_path_missing(self, tmp_path: Path) -> None:
        manager = EnvManager(tmp_path / "envs")
        with pytest.raises(EnvError):
            manager.describe(EnvSpec("main", path=tmp_path / "nope"), tmp_path)

    def test_lock_hash_and_reuse(self, tmp_path: Path) -> None:
        lock = tmp_path / "main.lock"
        lock.write_text("click==8.1.8\n", encoding="utf-8")
        manager = EnvManager(tmp_path / "envs")
        spec = EnvSpec("main", python="3.12", lock="main.lock")
        first = manager.describe(spec, tmp_path)
        second = manager.describe(spec, tmp_path)
        assert first["hash"] == second["hash"]
        assert first["exists"] is False

    def test_lock_traversal_rejected(self, tmp_path: Path) -> None:
        manager = EnvManager(tmp_path / "envs")
        with pytest.raises(EnvError):
            manager.describe(EnvSpec("main", lock="../x.lock"), tmp_path)


class TestBuild:
    def test_build_lock_with_uv(self, tmp_path: Path) -> None:
        if shutil.which("uv") is None:
            pytest.skip("uv não instalado")
        lock = tmp_path / "tiny.lock"
        lock.write_text("click==8.1.8\n", encoding="utf-8")
        manager = EnvManager(tmp_path / "envs")
        info = manager.build_lock(lock, python="3.12")
        assert Path(info["python_path"]).is_file()
        assert (Path(info["env_dir"]) / ".ready").is_file()
        assert manager.has(info["hash"]) is True
        # Segunda chamada reutiliza (não falha nem duplica).
        again = manager.build_lock(lock, python="3.12")
        assert again["hash"] == info["hash"]

    def test_build_disabled(self, tmp_path: Path) -> None:
        lock = tmp_path / "main.lock"
        lock.write_text("click==8.1.8\n", encoding="utf-8")
        manager = EnvManager(tmp_path / "envs")
        spec = EnvSpec("main", python="3.12", lock="main.lock")
        with pytest.raises(EnvError) as exc:
            manager.ensure(spec, tmp_path, allow_build=False)
        assert exc.value.code == P.ERR_SESSION_ENV_DISABLED


def test_cu13_specs_only(tmp_path: Path) -> None:
    from vramd.envs import _cu13_specs

    lock = tmp_path / "x.lock"
    lock.write_text(
        "--index-url https://pypi.org/simple\n"
        "nvidia-cudnn-cu12==9.10.2.21\n"
        "nvidia-cudnn-cu13==9.20.0.48\n"
        "nvidia-nccl-cu12==2.27.5\n"
        "nvidia-nccl-cu13==2.29.7\n"
        "torch==2.13.0+cu130\n",
        encoding="utf-8",
    )
    assert _cu13_specs(lock) == [
        "nvidia-cudnn-cu13==9.20.0.48",
        "nvidia-nccl-cu13==2.29.7",
    ]
