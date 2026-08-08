"""Testes do arranque de worker via ``runtime:`` e dos key sets por backend.

Cobre o que o F1 acrescenta ao caminho quente: o comando do worker deixa de ser
derivado só do checkout, e o ambiente/cwd passam a ser declaráveis.
"""

from __future__ import annotations

import pytest

from vramd.backend_manager import _LOAD_KWARG_KEYS, _SHAPE_LOAD_KEYS, BackendManager
from vramd.registry import BackendDescriptor, Registry, RuntimeSpec
from vramd.subprocess_pool import SubprocessWorkerError, SubprocessWorkerPool


class FakeProc:
    def __init__(self) -> None:
        self.pid = 4242
        self.stdin = None
        self.stdout = None

    def poll(self):
        return None


def make_pool(recorder: dict, **kwargs) -> SubprocessWorkerPool:
    def spawn(cmd, stdin=None, stdout=None, stderr=None, **extra):
        recorder["cmd"] = cmd
        recorder["extra"] = extra
        return FakeProc()

    return SubprocessWorkerPool(spawn_fn=spawn, **kwargs)


class TestWorkerCommand:
    def test_falls_back_to_the_monorepo_venv(self, monkeypatch):
        monkeypatch.setattr("vramd.subprocess_pool._resolve_tool_python", lambda tool: f"/venv/{tool}/python")
        pool = make_pool({})
        assert pool._worker_cmd("text3d", "text3d") == [
            "/venv/text3d/python",
            "-m",
            "text3d",
            "serve",
            "--ums-worker",
        ]

    def test_runtime_command_wins(self, monkeypatch):
        monkeypatch.setattr("vramd.subprocess_pool._resolve_tool_python", lambda tool: "/venv/x/python")
        pool = make_pool({})
        pool._runtimes["ext"] = RuntimeSpec(command=("/opt/py", "-m", "ext.worker"))
        assert pool._worker_cmd("ext", "ext") == ["/opt/py", "-m", "ext.worker"]

    def test_python_override_still_wins_over_runtime(self, monkeypatch):
        pool = make_pool({}, python_override={"text3d": "/custom/python"})
        pool._runtimes["text3d"] = RuntimeSpec(command=("/opt/py",))
        assert pool._worker_cmd("text3d", "text3d")[0] == "/custom/python"

    def test_unresolvable_runtime_raises_actionable_error(self, monkeypatch):
        monkeypatch.delenv("NAO_DEFINIDA", raising=False)
        pool = make_pool({})
        pool._runtimes["ext"] = RuntimeSpec(command=("${env:NAO_DEFINIDA}",))
        with pytest.raises(SubprocessWorkerError, match=r"runtime\.command não resolve"):
            pool._worker_cmd("ext", "ext")

    def test_missing_tool_venv_raises_install_hint(self, monkeypatch):
        monkeypatch.setattr("vramd.subprocess_pool._resolve_tool_python", lambda tool: None)
        pool = make_pool({})
        with pytest.raises(SubprocessWorkerError, match=r"install\.sh"):
            pool._worker_cmd("text3d", "text3d")


class TestWorkerEnvironment:
    def test_no_runtime_means_inherit(self):
        assert make_pool({})._worker_env("qualquer") is None

    def test_runtime_env_merges_over_the_inherited(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin")
        pool = make_pool({})
        pool._runtimes["ext"] = RuntimeSpec(env={"HF_HOME": "/data/hf"})
        env = pool._worker_env("ext")
        assert env["HF_HOME"] == "/data/hf"
        assert env["PATH"] == "/usr/bin"  # herdado, não substituído

    def test_empty_env_block_still_inherits(self):
        pool = make_pool({})
        pool._runtimes["ext"] = RuntimeSpec(command=("/x",), env={})
        assert pool._worker_env("ext") is None


class TestSpawnPlumbing:
    def test_spawn_without_runtime_passes_no_extra_kwargs(self, monkeypatch, tmp_path):
        """Duplos antigos de ``spawn_fn`` não podem partir por causa do F1."""
        monkeypatch.setattr("vramd.subprocess_pool._resolve_tool_python", lambda tool: "/venv/python")
        rec: dict = {}
        pool = make_pool(rec, log_path_fn=lambda b: tmp_path / f"{b}.log")
        from vramd.subprocess_pool import _WorkerState

        pool._spawn("text3d", "text3d", _WorkerState(backend="text3d"))
        assert rec["extra"] == {}

    def test_spawn_with_runtime_passes_env_and_cwd(self, monkeypatch, tmp_path):
        rec: dict = {}
        pool = make_pool(rec, log_path_fn=lambda b: tmp_path / f"{b}.log")
        pool._runtimes["ext"] = RuntimeSpec(
            command=("/opt/py", "-m", "ext"), env={"HF_HOME": "/data"}, cwd=str(tmp_path)
        )
        from vramd.subprocess_pool import _WorkerState

        pool._spawn("ext", "ext", _WorkerState(backend="ext"))
        assert rec["cmd"] == ["/opt/py", "-m", "ext"]
        assert rec["extra"]["cwd"] == str(tmp_path)
        assert rec["extra"]["env"]["HF_HOME"] == "/data"

    def test_load_stores_runtime_for_later_respawn(self, monkeypatch, tmp_path):
        rec: dict = {}
        pool = make_pool(rec, log_path_fn=lambda b: tmp_path / f"{b}.log")
        spec = RuntimeSpec(command=("/opt/py",))
        # ``load`` guarda o runtime antes de qualquer spawn; o respawn (que não
        # recebe o descriptor) tem de o encontrar aqui.
        with pytest.raises(Exception):  # noqa: B017 — o worker falso não fala o protocolo
            pool.load("ext", "ext", {}, runtime=spec)
        assert pool._runtimes["ext"] is spec


def registry_with(descs: list[BackendDescriptor]) -> Registry:
    return Registry(descriptors={d.name: d for d in descs})


class TestPerBackendKeySets:
    def test_defaults_to_the_global_allowlist(self):
        manager = BackendManager(registry_with([BackendDescriptor("d", "a", 100, 0)]))
        assert manager.load_keys_for("d") == _LOAD_KWARG_KEYS
        assert manager.shape_keys_for("d") == _SHAPE_LOAD_KEYS

    def test_declared_load_keys_replace_the_global(self):
        desc = BackendDescriptor("d", "a", 100, 0, load_keys=frozenset({"beam_size", "device"}))
        manager = BackendManager(registry_with([desc]))
        assert manager.load_keys_for("d") == frozenset({"beam_size", "device"})

    def test_shape_keys_default_to_the_intersection_with_the_global(self):
        """Declarar load_keys não deve promover chaves que o backend nem usa a 'shape'."""
        desc = BackendDescriptor("d", "a", 100, 0, load_keys=frozenset({"beam_size", "gpu_ids"}))
        manager = BackendManager(registry_with([desc]))
        assert manager.shape_keys_for("d") == frozenset({"gpu_ids"})

    def test_declared_shape_keys_are_used_verbatim(self):
        desc = BackendDescriptor("d", "a", 100, 0, shape_keys=frozenset({"beam_size"}))
        manager = BackendManager(registry_with([desc]))
        assert manager.shape_keys_for("d") == frozenset({"beam_size"})

    def test_unknown_backend_falls_back_to_the_global(self):
        manager = BackendManager(registry_with([]))
        assert manager.load_keys_for("nao-existe") == _LOAD_KWARG_KEYS

    def test_shape_mismatch_honours_the_declared_keys(self):
        manager = BackendManager(registry_with([BackendDescriptor("d", "a", 100, 0)]))
        keys = frozenset({"beam_size"})
        assert manager._shape_mismatch({"beam_size": 5}, {"beam_size": 9}, keys) is True
        assert manager._shape_mismatch({"beam_size": 5}, {"beam_size": 5}, keys) is False
        # Chave fora do conjunto não provoca reload.
        assert manager._shape_mismatch({"beam_size": 5}, {"outra": 1}, keys) is False

    def test_extract_load_shape_uses_the_backend_keys(self):
        desc = BackendDescriptor("d", "a", 100, 0, shape_keys=frozenset({"beam_size"}))
        manager = BackendManager(registry_with([desc]))
        shape = manager._extract_load_shape({"beam_size": 5, "gpu_ids": [0]}, "d")
        assert shape == {"beam_size": 5}

    def test_runtime_for_returns_none_for_unknown_backend(self):
        assert BackendManager(registry_with([]))._runtime_for("x") is None

    def test_runtime_for_returns_the_declared_spec(self):
        spec = RuntimeSpec(command=("/opt/py",))
        manager = BackendManager(registry_with([BackendDescriptor("d", "a", 100, 0, runtime=spec)]))
        assert manager._runtime_for("d") is spec
