"""Sessões de código do cliente: bundle, envs e workers efémeros."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import threading
import time
from pathlib import Path

import pytest

from vramd import protocol as P
from vramd import transport
from vramd.registry import Registry
from vramd.server import VramdServer
from vramd.transfer import TransferClient

# Worker mínimo que fala o protocolo JSONL do vramd sem importar o pacote
# (o venv da sessão em teste é o `system`, mas o subprocesso não herda o
# sys.path do pytest).
WORKER = r"""
import json, sys
for line in sys.stdin:
    try:
        req = json.loads(line)
    except Exception:
        continue
    cmd = req.get("cmd")
    if cmd == "load":
        print(json.dumps({"event": "ready", "vram_mib": 0}), flush=True)
    elif cmd == "generate":
        request = req.get("request") or {}
        tag = request.get("tag", "default")
        print(json.dumps({"event": "done", "result": {
            "status": "ok",
            "output": request.get("output"),
            "tag": tag,
            "value": request.get("value"),
        }}), flush=True)
    elif cmd == "unload":
        print(json.dumps({"event": "unloaded"}), flush=True)
    elif cmd == "ping":
        print(json.dumps({"event": "pong"}), flush=True)
    elif cmd == "shutdown":
        break
"""

WORKER_B = WORKER.replace('"tag": tag', '"tag": "worker-b"')


def _start(tmp_path: Path, **kwargs) -> tuple[VramdServer, Path, threading.Thread]:
    socket_path = tmp_path / "vramd.sock"
    kwargs.setdefault("query_free_mib", lambda: 99999)
    kwargs.setdefault("clear_vram", lambda: None)
    kwargs.setdefault("allow_client_code", True)
    srv = VramdServer(registry=Registry(descriptors={}), socket_path=socket_path, **kwargs)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert socket_path.exists(), "vramd não arrancou a tempo"
    return srv, socket_path, thread


def _stop(srv: VramdServer, socket_path: Path, thread: threading.Thread) -> None:
    srv._running = False
    try:
        sock, token = transport.connect(socket_path, timeout=2.0)
        with sock:
            sock.sendall((json.dumps(transport.with_auth({"cmd": P.CMD_SHUTDOWN}, token)) + "\n").encode())
    except OSError:
        pass
    thread.join(timeout=10.0)


def _make_bundle(tmp_path: Path, worker: str, *, mutate_manifest: bool = False) -> tuple[bytes, str, list[dict]]:
    root = tmp_path / "bundle"
    (root / "src" / "mytool").mkdir(parents=True, exist_ok=True)
    (root / "session" / "envs").mkdir(parents=True, exist_ok=True)
    (root / "src" / "mytool" / "__main__.py").write_text(worker, encoding="utf-8")
    (root / "session" / "envs" / "main.lock").write_bytes(b"")
    manifest: list[dict] = []
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        data = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        entry = {"path": rel, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if mutate_manifest and rel.endswith("__main__.py"):
            entry["sha256"] = "0" * 64
        manifest.append(entry)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        for path in files:
            info = tarfile.TarInfo(path.relative_to(root).as_posix())
            info.size = path.stat().st_size
            info.mode = 0o644
            info.mtime = 0
            tf.addfile(info, io.BytesIO(path.read_bytes()))
    data = buffer.getvalue()
    return data, hashlib.sha256(data).hexdigest(), manifest


def _manifest(bundle_sha: str, bundle_size: int, files: list[dict], *, env_system: bool = True) -> dict:
    return {
        "version": 1,
        "bundle": {"sha256": bundle_sha, "size": bundle_size, "format": "tar.gz"},
        "files": files,
        "envs": {"main": {"system": True} if env_system else {"lock": "session/envs/main.lock"}},
        "backends": [
            {
                "name": "echo",
                "tool": "mytool",
                "env": "main",
                "code_roots": ["src"],
                "vram_mib": 512,
                "priority": 0,
                "load_keys": [],
                "shape_keys": [],
            }
        ],
    }


@pytest.fixture
def running(tmp_path: Path):
    srv, socket_path, thread = _start(tmp_path)
    yield srv, socket_path
    _stop(srv, socket_path, thread)


class TestSessionOpen:
    def test_open_run_job_and_close(self, running, tmp_path: Path) -> None:
        srv, sock = running
        bundle, sha, files = _make_bundle(tmp_path, WORKER)
        with TransferClient(sock) as client:
            client.put_blob_bytes(sha, bundle)
            opened = client.request({"cmd": P.CMD_SESSION_OPEN, "manifest": _manifest(sha, len(bundle), files)})
            assert opened["status"] == P.STATUS_OK
            session = opened["session"]
            assert opened["backends"] == ["echo"]
            assert "echo@" + session in srv.registry.names

            out = tmp_path / "out.glb"
            submitted = client.request(
                {
                    "cmd": "submit",
                    "backend": "echo",
                    "session": session,
                    "output": str(out),
                    "value": 42,
                    "priority": "interactive",
                }
            )
            assert submitted["status"] == P.STATUS_OK, submitted
            result = client.request({"cmd": "wait", "job_id": submitted["job_id"], "timeout_sec": 30.0})
            assert result["status"] == P.STATUS_OK, result
            assert result["value"] == 42
            assert result["tag"] == "default"
            assert result["session"] == session
            assert result["bundle_sha"] == sha

            closed = client.request({"cmd": P.CMD_SESSION_CLOSE, "session": session})
            assert closed["closed"] is True
            assert "echo@" + session not in srv.registry.names
            listing = client.request({"cmd": P.CMD_SESSIONS})
            assert listing["count"] == 0

    def test_closed_session_rejects_jobs(self, running, tmp_path: Path) -> None:
        _srv, sock = running
        bundle, sha, files = _make_bundle(tmp_path, WORKER)
        with TransferClient(sock) as client:
            client.put_blob_bytes(sha, bundle)
            opened = client.request({"cmd": P.CMD_SESSION_OPEN, "manifest": _manifest(sha, len(bundle), files)})
            session = opened["session"]
            client.request({"cmd": P.CMD_SESSION_CLOSE, "session": session})
            response = client.request({"cmd": "submit", "backend": "echo", "session": session, "output": "x"})
            assert response["error_code"] == P.ERR_SESSION_UNKNOWN

    def test_unknown_session_is_rejected(self, running) -> None:
        _srv, sock = running
        with TransferClient(sock) as client:
            response = client.request({"cmd": "submit", "backend": "echo", "session": "nope", "output": "x"})
            assert response["error_code"] == P.ERR_SESSION_UNKNOWN


class TestSessionIsolation:
    def test_two_sessions_with_different_code(self, running, tmp_path: Path) -> None:
        _srv, sock = running
        bundle_a, sha_a, files_a = _make_bundle(tmp_path / "a", WORKER)
        bundle_b, sha_b, files_b = _make_bundle(tmp_path / "b", WORKER_B)
        with TransferClient(sock) as client:
            client.put_blob_bytes(sha_a, bundle_a)
            a = client.request({"cmd": P.CMD_SESSION_OPEN, "manifest": _manifest(sha_a, len(bundle_a), files_a)})
            client.put_blob_bytes(sha_b, bundle_b)
            b = client.request({"cmd": P.CMD_SESSION_OPEN, "manifest": _manifest(sha_b, len(bundle_b), files_b)})
            assert a["session"] != b["session"]
            jobs = []
            for opened, tag in ((a, "a"), (b, "b")):
                submitted = client.request(
                    {
                        "cmd": "submit",
                        "backend": "echo",
                        "session": opened["session"],
                        "output": f"/tmp/{tag}.glb",
                        "tag": tag,
                    }
                )
                jobs.append((submitted, tag))
            for submitted, tag in jobs:
                result = client.request({"cmd": "wait", "job_id": submitted["job_id"], "timeout_sec": 30.0})
                assert result["status"] == P.STATUS_OK, result
                if tag == "a":
                    assert result["tag"] == "a"
                else:
                    assert result["tag"] == "worker-b"


class TestBundleRejection:
    def test_manifest_hash_mismatch(self, running, tmp_path: Path) -> None:
        _srv, sock = running
        bundle, sha, files = _make_bundle(tmp_path, WORKER, mutate_manifest=True)
        with TransferClient(sock) as client:
            client.put_blob_bytes(sha, bundle)
            response = client.request({"cmd": P.CMD_SESSION_OPEN, "manifest": _manifest(sha, len(bundle), files)})
            assert response["status"] == P.STATUS_ERROR
            assert response["error_code"] == P.ERR_SESSION_BUNDLE

    def test_bundle_without_blob(self, running, tmp_path: Path) -> None:
        _srv, sock = running
        _bundle, sha, files = _make_bundle(tmp_path, WORKER)
        with TransferClient(sock) as client:
            response = client.request({"cmd": P.CMD_SESSION_OPEN, "manifest": _manifest(sha, 1, files)})
            assert response["status"] == P.STATUS_ERROR
            assert response["error_code"] == P.ERR_SESSION_BUNDLE

    def test_disabled_without_flag(self, tmp_path: Path) -> None:
        srv, socket_path, thread = _start(tmp_path, allow_client_code=False)
        bundle, sha, files = _make_bundle(tmp_path, WORKER)
        try:
            with TransferClient(socket_path) as client:
                client.put_blob_bytes(sha, bundle)
                response = client.request({"cmd": P.CMD_SESSION_OPEN, "manifest": _manifest(sha, len(bundle), files)})
                assert response["error_code"] == P.ERR_SESSION_DISABLED
        finally:
            _stop(srv, socket_path, thread)

    def test_env_build_disabled(self, running, tmp_path: Path) -> None:
        _srv, sock = running
        bundle, sha, files = _make_bundle(tmp_path, WORKER)
        with TransferClient(sock) as client:
            client.put_blob_bytes(sha, bundle)
            response = client.request(
                {
                    "cmd": P.CMD_SESSION_OPEN,
                    "manifest": _manifest(sha, len(bundle), files, env_system=False),
                }
            )
            assert response["status"] == P.STATUS_ERROR
            assert response["error_code"] == P.ERR_SESSION_ENV_DISABLED
