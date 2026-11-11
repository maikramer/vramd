"""Backend só com ``runtime.command`` (sem ``tool:``) corre em subprocesso.

Caso real: ``examples/echo-backend`` — o descriptor que o README manda copiar —
não declara ``tool:``. O supervisor só olhava para ``desc.tool`` e despachava-o
in-process; o primeiro ``vramd submit echo`` falhava com
``No module named 'worker'`` (o adapter vive no venv do modelo, não no do
supervisor). A calibração, que fala directamente com o pool, funcionava — por
isso o bug só aparecia em produção.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from vramd.backend_manager import BackendManager
from vramd.registry import BackendDescriptor, Registry, RuntimeSpec
from vramd.subprocess_pool import SubprocessWorkerPool

REPO = Path(__file__).resolve().parent.parent
ECHO_WORKER = REPO / "examples" / "echo-backend" / "worker.py"


def _registry(runtime: RuntimeSpec | None, *, tool: str | None = None) -> Registry:
    desc = BackendDescriptor(name="echo", adapter="worker", vram_mib=512, priority=10, tool=tool, runtime=runtime)
    return Registry(descriptors={"echo": desc})


def _manager(registry: Registry, pool: object) -> BackendManager:
    return BackendManager(registry, query_free_mib=lambda *a, **k: 24000, clear_vram=lambda: None, subprocess_pool=pool)


class TestUseSubprocess:
    def test_runtime_command_without_tool(self) -> None:
        mgr = _manager(_registry(RuntimeSpec(command=("/opt/venv/bin/python", "worker.py"))), object())
        assert mgr._use_subprocess("echo") is True

    def test_runtime_monorepo_tool_without_tool(self) -> None:
        mgr = _manager(_registry(RuntimeSpec(monorepo_tool="text3d")), object())
        assert mgr._use_subprocess("echo") is True

    def test_tool_alone_still_counts(self) -> None:
        assert _manager(_registry(None, tool="text3d"), object())._use_subprocess("echo") is True

    def test_runtime_with_only_env_stays_in_process(self) -> None:
        """Um ``runtime: {env: …}`` sem comando não diz como arrancar um worker."""
        mgr = _manager(_registry(RuntimeSpec(env={"HF_HOME": "/data/hf"})), object())
        assert mgr._use_subprocess("echo") is False

    def test_no_runtime_no_tool_is_in_process(self) -> None:
        assert _manager(_registry(None), object())._use_subprocess("echo") is False

    def test_global_kill_switch(self, monkeypatch) -> None:
        monkeypatch.setenv("VRAMD_SUBPROCESS", "0")
        mgr = _manager(_registry(RuntimeSpec(command=("/x",))), object())
        assert mgr._use_subprocess("echo") is False


class TestEchoExampleEndToEnd:
    def test_readme_example_generates_through_the_supervisor(self, tmp_path: Path) -> None:
        """O exemplo empacotado, tal como documentado, com um worker real."""
        src = str(REPO / "src")
        existing = os.environ.get("PYTHONPATH", "")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": src + (os.pathsep + existing if existing else "")}

        def spawn_fn(cmd, stdin, stdout, stderr, **extra):
            # O comando vem do runtime.command do descriptor — é isso que se testa.
            assert cmd == [sys.executable, str(ECHO_WORKER)]
            return subprocess.Popen(cmd, stdin=stdin, stdout=stdout, stderr=stderr, text=True, bufsize=1, env=env)

        pool = SubprocessWorkerPool(
            spawn_fn=spawn_fn,
            log_path_fn=lambda _backend: tmp_path / "vramd-worker-echo.log",
            load_timeout_sec=20.0,
            event_timeout_sec=20.0,
            abort_timeout_sec=5.0,
            ping_timeout_sec=5.0,
        )
        mgr = _manager(_registry(RuntimeSpec(command=(sys.executable, str(ECHO_WORKER)))), pool)
        progress: list[tuple[float | None, str | None]] = []
        try:
            result = mgr.generate(
                "echo", {"prompt": "olá", "steps": 2, "_progress": lambda pct, msg: progress.append((pct, msg))}
            )
        finally:
            pool.shutdown_all()
        assert result["status"] == "ok", result
        assert result["output"] == "OLÁ"
        assert any(msg == "passo 2/2" for _pct, msg in progress)


@pytest.mark.parametrize("path", [ECHO_WORKER])
def test_example_worker_exists(path: Path) -> None:
    assert path.is_file()
