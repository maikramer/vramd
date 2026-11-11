"""Protocolo v2 fim-a-fim: hello, frames persistentes e blobs."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from vramd import protocol as P
from vramd import transport
from vramd.registry import Registry
from vramd.server import VramdServer
from vramd.transfer import TransferClient, TransferError


def _start(tmp_path: Path, **kwargs) -> tuple[VramdServer, Path, threading.Thread]:
    socket_path = tmp_path / "vramd.sock"
    kwargs.setdefault("query_free_mib", lambda: 99999)
    kwargs.setdefault("clear_vram", lambda: None)
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
    thread.join(timeout=5.0)


def _request(socket_path: Path, request: dict, timeout: float = 10.0) -> dict:
    sock, token = transport.connect(socket_path, timeout=timeout)
    with sock:
        sock.sendall((json.dumps(transport.with_auth(request, token)) + "\n").encode())
        data = b""
        while True:
            chunk = sock.recv(8192)
            if not chunk:
                break
            data += chunk
    return json.loads(data.decode().strip().split("\n")[-1])


@pytest.fixture
def running(tmp_path: Path):
    srv, socket_path, thread = _start(tmp_path)
    yield srv, socket_path
    _stop(srv, socket_path, thread)


class TestHello:
    def test_features_and_version(self, running) -> None:
        _srv, sock = running
        with TransferClient(sock) as client:
            hello = client.connect()
            assert hello["protocol"] == P.PROTOCOL_VERSION
            assert hello["vramd_version"]
            assert "binary" in hello["features"]
            assert "blobs" in hello["features"]
            assert "zlib" in hello["compression"]

    def test_multiple_requests_one_connection(self, running) -> None:
        _srv, sock = running
        with TransferClient(sock) as client:
            client.connect()
            first = client.blob_has("a" * 64)
            second = client.blob_has("b" * 64)
            assert first["exists"] is False
            assert second["exists"] is False
            status = client.request({"cmd": P.CMD_STATUS})
            assert status["protocol"] == P.PROTOCOL_VERSION
            assert status["features"]

    def test_legacy_json_still_works(self, running) -> None:
        _srv, sock = running
        resp = _request(sock, {"cmd": P.CMD_STATUS})
        assert resp["status"] == P.STATUS_STATUS
        assert resp["protocol"] == P.PROTOCOL_VERSION


class TestBlobs:
    def test_roundtrip_and_dedupe(self, running, tmp_path: Path) -> None:
        _srv, sock = running
        data = b"glTF" + bytes(range(256)) * 5000
        sha = hashlib.sha256(data).hexdigest()
        with TransferClient(sock) as client:
            assert client.blob_has(sha, len(data))["exists"] is False
            put = client.put_blob_bytes(sha, data)
            assert put["status"] == P.STATUS_OK
            assert put["sha256"] == sha
            assert client.blob_has(sha, len(data))["exists"] is True
            again = client.put_blob_bytes(sha, data)
            assert again.get("skipped") is True
            target = tmp_path / "downloaded.bin"
            size = client.get_blob(sha, target)
            assert size == len(data)
            assert target.read_bytes() == data
            stats = client.blob_stats()
            assert stats["count"] == 1

    def test_missing_blob_get_raises(self, running, tmp_path: Path) -> None:
        _srv, sock = running
        with TransferClient(sock) as client, pytest.raises(TransferError):
            client.get_blob("0" * 64, tmp_path / "x")

    def test_corrupt_upload_is_rejected(self, running) -> None:
        _srv, sock = running
        with TransferClient(sock) as client, pytest.raises(TransferError):
            client.put_blob_bytes("0" * 64, b"conteudo errado")

    def test_stage_materializes_and_clean_unpins(self, running) -> None:
        srv, sock = running
        data = b"mesh" * 10_000
        sha = hashlib.sha256(data).hexdigest()
        with TransferClient(sock) as client:
            client.put_blob_bytes(sha, data)
            staged = client.request({"cmd": P.CMD_STAGE, "blobs": {"in/mesh.bin": sha}})
            assert staged["status"] == P.STATUS_OK
            assert staged["materialized"][0]["sha256"] == sha
            target = Path(staged["root"]) / "in" / "mesh.bin"
            assert target.read_bytes() == data
            assert srv.remote.blobs.pinned(sha) is True
            cleaned = client.request({"cmd": P.CMD_CLEAN, "job": staged["job"]})
            assert cleaned["removed"] is True
            assert srv.remote.blobs.pinned(sha) is False
            assert not target.exists()

    def test_blob_gc_endpoint(self, running) -> None:
        _srv, sock = running
        with TransferClient(sock) as client:
            result = client.blob_gc()
            assert result["status"] == P.STATUS_OK
            assert "removed" in result


class TestJobFiles:
    def test_job_get_file_and_clean(self, running, tmp_path: Path) -> None:
        _srv, sock = running
        data = b"output" + bytes(range(256)) * 4000
        with TransferClient(sock) as client:
            job, root = client.stage()
            target = Path(root) / "out" / "result.bin"
            target.write_bytes(data)
            downloaded = tmp_path / "result.bin"
            size = client.get_file(job, "out/result.bin", downloaded)
            assert size == len(data)
            assert downloaded.read_bytes() == data
            assert client.list(job, "out") == ["result.bin"]
            client.clean(job)
            assert not Path(root).exists()

    def test_job_get_missing_raises(self, running, tmp_path: Path) -> None:
        _srv, sock = running
        with TransferClient(sock) as client:
            job, _root = client.stage()
            with pytest.raises(TransferError):
                client.get_file(job, "out/nope.bin", tmp_path / "nope.bin")


class TestClientResilience:
    def test_timeout_is_not_sticky(self, running) -> None:
        """Um RPC com timeout curto não pode encurtar a ligação persistente."""
        _srv, sock = running
        with TransferClient(sock) as client:
            client.connect()
            assert client._sock is not None
            before = client._sock.gettimeout()
            client.request({"cmd": P.CMD_STATUS}, timeout=1.0)
            assert client._sock.gettimeout() == before

    def test_closed_socket_becomes_transfer_error(self, running) -> None:
        _srv, sock = running
        client = TransferClient(sock)
        try:
            client.connect()
            assert client._sock is not None
            client._sock.close()
            with pytest.raises(TransferError):
                client.request({"cmd": P.CMD_STATUS})
        finally:
            client.close()

    def test_reconnect_same_endpoint(self, running) -> None:
        _srv, sock = running
        with TransferClient(sock) as client:
            client.connect()
            assert client._sock is not None
            client._sock.close()
            client.reconnect()
            assert client.request({"cmd": P.CMD_STATUS})["status"] == P.STATUS_STATUS
