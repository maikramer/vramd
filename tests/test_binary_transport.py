"""Protocolo v2 fim-a-fim: hello, frames persistentes e blobs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from vramd import frames, transport
from vramd import protocol as P
from vramd.registry import Registry
from vramd.server import VramdServer
from vramd.transfer import TransferClient, TransferError, _bounded_chunk_size


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
        with TransferClient(sock) as client:
            with pytest.raises(TransferError) as exc:
                client.put_blob_bytes("0" * 64, b"conteudo errado")
            assert "sha256" in str(exc.value).lower()

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

    def test_put_blob_sends_only_declared_size(self, running, tmp_path: Path, monkeypatch) -> None:
        """Se o ficheiro crescer depois do stat, o cliente não pode enviar o excedente."""
        _srv, sock = running
        path = tmp_path / "cresceu.bin"
        path.write_bytes(b"abcdef")
        declared = b"abc"
        sha = hashlib.sha256(declared).hexdigest()
        real_stat = Path.stat

        class _FrozenStat:
            st_size = len(declared)

        def stat_before_growth(self, *args, **kwargs):
            return _FrozenStat() if self == path else real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", stat_before_growth)
        with TransferClient(sock) as client:
            result = client.put_blob(sha, path, chunk_size=64)
            assert result["size"] == len(declared)
            got = tmp_path / "recuperado.bin"
            client.get_blob(sha, got)
            assert got.read_bytes() == declared

    def test_get_blob_rejects_extra_bytes(self, running, tmp_path: Path, monkeypatch) -> None:
        """Metadados dizem 4 bytes; um servidor buggy não pode encher o disco local."""
        _srv, sock = running
        with TransferClient(sock) as client:
            client.connect()
            monkeypatch.setattr(
                client,
                "blob_has",
                lambda *a, **k: {"status": P.STATUS_OK, "exists": True, "size": 4},
            )
            steps = [
                frames.Frame(
                    frames.TYPE_JSON,
                    frames.FLAG_RAW,
                    client._request_id,
                    frames.encode_json({"status": P.STATUS_OK, "size": 4}),
                ),
                frames.Frame(frames.TYPE_DATA, frames.FLAG_RAW, client._request_id, b"12345678"),
                frames.Frame(frames.TYPE_END, frames.FLAG_RAW, client._request_id, b""),
            ]

            def fake_read():
                frame = steps.pop(0)
                # O request_id real só é conhecido depois do _send_json.
                return frames.Frame(frame.type, frame.flags, client._request_id, frame.payload)

            monkeypatch.setattr(client, "_read_frame", fake_read)
            target = tmp_path / "extra.bin"
            with pytest.raises(TransferError, match="mais de 4 bytes"):
                client.get_blob("a" * 64, target)
            assert not target.exists()
            assert not target.with_name(target.name + ".part").exists()

    def test_get_blob_rejects_size_change(self, running, tmp_path: Path, monkeypatch) -> None:
        """O tamanho nos metadados não pode divergir do que ``blob-has`` disse."""
        _srv, sock = running
        with TransferClient(sock) as client:
            client.connect()
            monkeypatch.setattr(
                client,
                "blob_has",
                lambda *a, **k: {"status": P.STATUS_OK, "exists": True, "size": 4},
            )
            meta = frames.Frame(
                frames.TYPE_JSON,
                frames.FLAG_RAW,
                0,
                frames.encode_json({"status": P.STATUS_OK, "size": 8}),
            )

            def fake_read():
                return frames.Frame(meta.type, meta.flags, client._request_id, meta.payload)

            monkeypatch.setattr(client, "_read_frame", fake_read)
            with pytest.raises(TransferError, match="tamanho do blob mudou"):
                client.get_blob("a" * 64, tmp_path / "mudou.bin")


def test_bounded_chunk_size() -> None:
    """Um chunk acima do frame máximo dessincronizava o upload."""
    assert _bounded_chunk_size(frames.MAX_DATA_BYTES * 2) == frames.MAX_DATA_BYTES
    assert _bounded_chunk_size(0) == 1


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

    def test_rpc_error_closes_connection(self, running, monkeypatch) -> None:
        """Resposta interrompida não pode deixar o socket dessincronizado."""
        _srv, sock = running
        with TransferClient(sock) as client:
            client.connect()

            def boom():
                raise TransferError("falha simulada a meio da resposta")

            monkeypatch.setattr(client, "_read_frame", boom)
            with pytest.raises(TransferError):
                client.request({"cmd": P.CMD_STATUS})
            assert client._sock is None

    def test_reconnect_same_endpoint(self, running) -> None:
        _srv, sock = running
        with TransferClient(sock) as client:
            client.connect()
            assert client._sock is not None
            client._sock.close()
            client.reconnect()
            assert client.request({"cmd": P.CMD_STATUS})["status"] == P.STATUS_STATUS

    def test_failed_upload_does_not_poison_connection(self, running) -> None:
        """Servidor recusa o blob antes de ler os DATA; o cliente tem de fechar."""
        srv, sock = running
        srv.remote.blobs.config = dataclasses.replace(srv.remote.blobs.config, max_blob_bytes=8)
        data = b"x" * 64
        with TransferClient(sock) as client:
            with pytest.raises(TransferError):
                client.put_blob_bytes(hashlib.sha256(data).hexdigest(), data)
            # Sem fechar, os frames DATA/END que o servidor rejeitou ficavam na
            # ligação e o pedido seguinte morria com request_id fora de sequência.
            assert client.request({"cmd": P.CMD_STATUS})["status"] == P.STATUS_STATUS
