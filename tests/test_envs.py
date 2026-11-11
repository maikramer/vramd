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

    def test_build_separates_no_build_isolation(self, tmp_path: Path) -> None:
        """nvdiffrast importa torch no setup.py: vai num install sem isolamento.

        O lock é instalado primeiro (com torch/setuptools), e só depois o
        nvdiffrast, senão o build isolado falha com "You have PyTorch
        installed / --no-build-isolation".
        """
        lock = tmp_path / "paint.lock"
        lock.write_text(
            "setuptools==82.0.1\n"
            "wheel==0.47.0\n"
            "torch==2.13.0+cu130\n"
            "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4f\n",
            encoding="utf-8",
        )
        manager = EnvManager(tmp_path / "envs", uv="/usr/bin/uv")
        calls: list[list[str]] = []
        filtered_lock = ""

        def fake_run(cmd: list[str], *, env: dict[str, str], timeout: float, log: list[str]) -> None:
            nonlocal filtered_lock
            calls.append(list(cmd))
            if len(cmd) > 2 and cmd[1] == "venv":
                target = Path(cmd[-1])
                (target / "bin").mkdir(parents=True, exist_ok=True)
                (target / "bin" / "python").write_text("", encoding="utf-8")
            if len(cmd) > 1 and cmd[1] == "pip" and "-r" in cmd:
                # O lock filtrado é temporário: lê-se durante o build.
                filtered_lock = Path(cmd[cmd.index("-r") + 1]).read_text(encoding="utf-8")

        manager._run = fake_run  # type: ignore[method-assign]
        manager._build(
            env_hash="abc",
            env_dir=tmp_path / "envs" / "abc",
            lock_path=lock,
            python="3.13",
            index_url=None,
            extra_index_url=None,
        )
        installs = [cmd for cmd in calls if len(cmd) > 1 and cmd[1] == "pip"]
        assert len(installs) == 2
        main, special = installs
        assert "--no-build-isolation" not in main
        assert "nvdiffrast" not in filtered_lock
        assert "torch==2.13.0+cu130" in filtered_lock
        assert "--no-build-isolation" in special
        assert "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4f" in special
        # setuptools/wheel já vêm no lock: não são repetidos.
        assert "setuptools" not in special and "wheel" not in special
        assert (tmp_path / "envs" / "abc" / ".ready").is_file()
        # O lock filtrado é temporário.
        assert not list((tmp_path / "envs").glob(".*.lock"))


def test_split_no_build_isolation_keeps_continuations() -> None:
    from vramd.envs import _split_no_build_isolation

    content = (
        "torch==2.13.0+cu130\n"
        "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@abc\n"
        "    --hash=sha256:aa\n"
        "click==8.1.8\n"
    )
    kept, special = _split_no_build_isolation(content)
    assert kept == ["torch==2.13.0+cu130", "click==8.1.8"]
    assert special == ["nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@abc\n    --hash=sha256:aa"]


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


def test_lock_with_cu12_rejected(tmp_path: Path) -> None:
    lock = tmp_path / "x.lock"
    lock.write_text("nvidia-nccl-cu12==2.27.5\n", encoding="utf-8")
    manager = EnvManager(tmp_path / "envs")
    with pytest.raises(EnvError, match="CUDA 12"):
        manager.describe(EnvSpec("main", lock="x.lock"), tmp_path)
