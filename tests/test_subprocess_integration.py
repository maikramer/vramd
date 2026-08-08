"""Teste de integração: SubprocessWorkerPool ↔ worker real via stdin/stdout JSONL.

Spawna um subprocesso Python real (não mock) que corre
``vramd.worker.serve.run_worker_loop`` com um adapter mock. O
SubprocessWorkerPool fala com ele via stdin/stdout (JSONL), validando o
protocolo end-to-end — cobrindo o caminho real do supervisor para um worker, sem precisar de GPU.

Precisa do ``vramd`` importável no mesmo Python; sem isso é skipado.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

from vramd.subprocess_pool import SubprocessWorkerError, SubprocessWorkerPool

# Script worker: imprime eventos JSONL no stdout, lê cmds do stdin.
WORKER_SCRIPT = textwrap.dedent("""
    import sys
    from vramd.worker.serve import run_worker_loop

    class MockAdapter:
        name = "mock"

        def __init__(self):
            import os

            self._unload_file = os.environ.get("MOCK_UNLOAD_FILE")

        def load(self, **kwargs):
            # VRAM inventada para o UMS ver valor não-nulo.
            return {"loaded": True, **kwargs}

        def generate(self, model, request):
            progress = request.get("_progress")
            if callable(progress):
                progress(0.5, "mid")
            return {"status": "ok", "output": request.get("output", "/tmp/x.glb")}

        def unload(self, model):
            if self._unload_file:
                with open(self._unload_file, "a") as fh:
                    # Escapar o newline: um newline real aqui partia o dedent.
                    fh.write("unloaded\\n")

    run_worker_loop(MockAdapter, backend_name="mock")
""")


def _subprocess_env() -> dict[str, str]:
    """Env do worker com o ``src`` do repo no PYTHONPATH.

    O subprocesso spawnado NÃO herda o ``pythonpath`` do pytest — sem isto os
    7 testes de subprocesso falhavam localmente (``ModuleNotFoundError: vramd``)
    sempre que o package não está instalado no interpretador (no CI passava só
    porque lá corre ``pip install -e .``).
    """
    src = str(Path(__file__).resolve().parent.parent / "src")
    existing = os.environ.get("PYTHONPATH", "")
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": src + (os.pathsep + existing if existing else ""),
    }


def _pool_for_real_subprocess(tmp_path: Path, **overrides) -> tuple[SubprocessWorkerPool, Path]:
    """Cria um pool cujo spawn_fn arranca um worker Python real."""
    script_path = tmp_path / "worker.py"
    script_path.write_text(WORKER_SCRIPT)
    log_path = tmp_path / "worker.log"

    def real_spawn(cmd, stdin, stdout, stderr):
        import subprocess

        return subprocess.Popen(
            [sys.executable, "-u", str(script_path)],
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            text=True,
            bufsize=1,
            env=_subprocess_env(),
        )

    pool = SubprocessWorkerPool(
        spawn_fn=real_spawn,
        log_path_fn=lambda _backend: log_path,
        load_timeout_sec=15.0,
        event_timeout_sec=15.0,
        abort_timeout_sec=5.0,
        ping_timeout_sec=5.0,
        python_override={"mock": "/nonexistent"},  # não usado — spawn_fn custom
        **overrides,
    )
    return pool, log_path


@pytest.fixture(autouse=True)
def _skip_if_no_vramd():
    pytest.importorskip("vramd.worker.serve")


class TestRealSubprocessIntegration:
    """Ciclo completo pool↔worker via JSONL stdin/stdout."""

    def test_load_generate_unload_shutdown(self, tmp_path: Path) -> None:
        pool, _log = _pool_for_real_subprocess(tmp_path)
        try:
            info = pool.load("mock", "mock", {"sdnq_preset": "x"})
            assert info["event"] == "ready"
            assert pool.is_loaded("mock")

            progresses: list[tuple[float | None, str | None]] = []
            result = pool.generate(
                "mock", {"output": "/tmp/out.glb"}, on_progress=lambda p, m: progresses.append((p, m))
            )
            assert result["status"] == "ok"
            assert result["output"] == "/tmp/out.glb"
            # Progress foi emitido pelo worker e capturado pelo pool.
            assert (0.5, "mid") in progresses

            assert pool.unload("mock") is True
            assert not pool.is_loaded("mock")

            assert pool.shutdown("mock") is True
            assert not pool.is_alive("mock")
        finally:
            pool.shutdown_all()

    def test_ping_after_load(self, tmp_path: Path) -> None:
        pool, _log = _pool_for_real_subprocess(tmp_path)
        try:
            pool.load("mock", "mock", {})
            assert pool.ping("mock") is True
            pool.shutdown("mock")
        finally:
            pool.shutdown_all()

    def test_multiple_generates_reuse_worker(self, tmp_path: Path) -> None:
        pool, _log = _pool_for_real_subprocess(tmp_path)
        try:
            pool.load("mock", "mock", {})
            # 3 generates seguidos no mesmo worker.
            for i in range(3):
                r = pool.generate("mock", {"output": f"/tmp/{i}.glb"})
                assert r["status"] == "ok"
                assert r["output"] == f"/tmp/{i}.glb"
            pool.shutdown("mock")
        finally:
            pool.shutdown_all()

    def test_worker_died_raises_on_generate(self, tmp_path: Path) -> None:
        """Worker morre depois do load → generate deve falhar limpo."""
        pool, _log = _pool_for_real_subprocess(tmp_path)
        try:
            pool.load("mock", "mock", {})
            # Matar o subprocesso manualmente.
            with pool._pool_lock:
                state = pool._workers["mock"]
            if state.proc and state.proc.poll() is None:
                state.proc.kill()
                state.proc.wait(timeout=3)
            with pytest.raises(SubprocessWorkerError, match="worker"):
                pool.generate("mock", {})
        finally:
            pool.shutdown_all()

    def test_eof_stdin_unloads_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """EOF no stdin (supervisor morreu sem CMD_SHUTDOWN) descarrega o modelo.

        Regressão: o loop fazia ``break`` no EOF sem ``_safe_unload`` — o
        cleanup do adapter (checkpoints, ficheiros temporários) ficava por fazer.
        """
        unload_file = tmp_path / "unloads.txt"
        monkeypatch.setenv("MOCK_UNLOAD_FILE", str(unload_file))
        pool, _log = _pool_for_real_subprocess(tmp_path)
        try:
            pool.load("mock", "mock", {})
            with pool._pool_lock:
                state = pool._workers["mock"]
            assert state.proc is not None
            state.proc.stdin.close()  # EOF → worker deve fazer unload e sair
            state.proc.wait(timeout=10)
        finally:
            pool.shutdown_all()
        assert unload_file.read_text().strip() == "unloaded"
