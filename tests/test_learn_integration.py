"""Integração learn+hooks no VramdServer real (socket temporário, adapters mock)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

from vramd import protocol as P
from vramd.registry import BackendDescriptor, Registry
from vramd.server import VramdServer

from .conftest_helpers import MockAdapter
from .test_server import _send_request, _start_ums, _stop_ums


def _registry() -> Registry:
    descriptors = {
        n: BackendDescriptor(name=n, adapter=f"_mock_{n}", vram_mib=v, priority=p)
        for n, (v, p) in {"alpha": (1000, 10), "beta": (3000, 30)}.items()
    }
    registry = Registry(descriptors=descriptors)
    for n in descriptors:
        registry._adapter_instances[n] = MockAdapter(name=n)
    return registry


@pytest.fixture
def server(tmp_path: Path):
    srv, sock, thread = _start_ums(tmp_path, registry=_registry())
    yield srv, sock
    _stop_ums(srv, sock, thread)


class TestLearnRpc:
    def test_status_includes_learn_and_hooks_blocks(self, server) -> None:
        _srv, sock = server
        resp = _send_request(sock, {"cmd": P.CMD_STATUS})
        assert "learn" in resp
        assert resp["learn"]["enabled"] is True
        assert "hooks" in resp
        assert resp["hooks"]["configured"] == 0

    def test_learn_reports_backends(self, server) -> None:
        _srv, sock = server
        resp = _send_request(sock, {"cmd": P.CMD_LEARN})
        assert resp["status"] == "ok"
        assert isinstance(resp["backends"], list)
        assert "hint" in resp

    def test_learn_reset(self, server) -> None:
        _srv, sock = server
        resp = _send_request(sock, {"cmd": P.CMD_LEARN, "reset": True})
        assert resp["status"] == "ok"
        assert resp["reset"] is True


class TestHooksWiring:
    def test_job_done_hook_fires_with_payload(self, tmp_path: Path, monkeypatch) -> None:
        out = tmp_path / "hook-payloads.jsonl"
        hooks_file = tmp_path / "hooks.yaml"
        hooks_file.write_text(
            yaml.safe_dump({"hooks": [{"event": "on_job_done", "command": ["tee", "-a", str(out)]}]}),
            encoding="utf-8",
        )
        monkeypatch.setenv("VRAMD_HOOKS_FILE", str(hooks_file))
        # Store de learn isolado por teste (não ~ do runner).
        monkeypatch.setenv("VRAMD_LEARN_INTERVAL_SEC", "0.2")

        srv, sock, thread = _start_ums(tmp_path, registry=_registry())
        try:
            resp = _send_request(sock, {"cmd": P.CMD_GENERATE, "backend": "alpha", "prompt": "x"})
            assert resp["status"] == "ok"
            # O hook corre em thread daemon — esperar pelo ficheiro (≤5s).
            deadline = time.monotonic() + 5.0
            payload = None
            while time.monotonic() < deadline:
                if out.exists():
                    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
                    if lines:
                        payload = json.loads(lines[-1])
                        break
                time.sleep(0.1)
            assert payload is not None, "hook on_job_done não disparou"
            assert payload["event"] == "on_job_done"
            assert payload["backend"] == "alpha"
            assert payload["state"] == "done"
            assert payload["job_id"].startswith(resp["job_id"][:8])
        finally:
            _stop_ums(srv, sock, thread)

        # O status durante a vida reportava o hook configurado.
        assert srv.hooks.status_dict()["configured"] == 1

    def test_hook_config_error_kills_startup(self, tmp_path: Path, monkeypatch) -> None:
        """hooks.yaml malformado falha o arranque — silêncio seria pior."""
        bad = tmp_path / "hooks.yaml"
        bad.write_text(yaml.safe_dump({"hooks": [{"event": "on_nada", "command": ["x"]}]}), encoding="utf-8")
        monkeypatch.setenv("VRAMD_HOOKS_FILE", str(bad))
        with pytest.raises(ValueError):
            VramdServer(registry=_registry(), socket_path=tmp_path / "s.sock")


class TestEvictHookViaManager:
    def test_evict_fires_on_evict_hook(self, tmp_path: Path, monkeypatch) -> None:
        out = tmp_path / "evicts.jsonl"
        hooks_file = tmp_path / "hooks.yaml"
        hooks_file.write_text(
            yaml.safe_dump({"hooks": [{"event": "on_evict", "command": ["tee", "-a", str(out)]}]}),
            encoding="utf-8",
        )
        monkeypatch.setenv("VRAMD_HOOKS_FILE", str(hooks_file))

        srv, sock, thread = _start_ums(tmp_path, registry=_registry())
        try:
            resp = _send_request(sock, {"cmd": P.CMD_GENERATE, "backend": "alpha", "prompt": "x"})
            assert resp["status"] == "ok"
            _send_request(sock, {"cmd": P.CMD_RELEASE, "backend": "alpha"})
            deadline = time.monotonic() + 5.0
            payload = None
            while time.monotonic() < deadline:
                if out.exists():
                    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
                    if lines:
                        payload = json.loads(lines[-1])
                        break
                time.sleep(0.1)
            assert payload is not None, "hook on_evict não disparou"
            assert payload["backend"] == "alpha"
        finally:
            _stop_ums(srv, sock, thread)


def test_queue_on_finish_fires_for_cancelled_queued(tmp_path: Path):
    """Cancelar um job queued também é transição terminal → hook dispara."""
    seen: list[str] = []
    from vramd.job_queue import JobQueue

    queue = JobQueue(max_depth=4, stats=None, wal_path=None, on_finish=lambda job: seen.append(job.state))
    job = queue.enqueue("alpha", {"prompt": "x"})
    resp = queue.cancel(job.job_id)
    assert resp["state"] == P.JOB_CANCELLED
    assert seen == [P.JOB_CANCELLED]
