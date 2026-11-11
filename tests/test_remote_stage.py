"""Sandbox remoto (stage/put/get/ls/clean) e listener TCP explícito."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pytest

from vramd import protocol as P
from vramd import transport
from vramd.registry import Registry
from vramd.remote import RemoteConfig, RemoteError, RemoteSandbox
from vramd.server import VramdServer


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


@pytest.fixture
def running(tmp_path: Path):
    srv, socket_path, thread = _start(tmp_path)
    yield srv, socket_path
    _stop(srv, socket_path, thread)


# ---------------------------------------------------------------------------
# Integração sobre o protocolo real
# ---------------------------------------------------------------------------


class TestRemoteProtocol:
    def test_stage_put_get_ls_clean(self, running) -> None:
        _, sock = running
        staged = _request(sock, {"cmd": P.CMD_STAGE})
        assert staged["status"] == P.STATUS_OK
        job = staged["job"]

        payload = b"glTF" + bytes(range(256)) * 4000  # ~1 MB, 3 chunks
        chunk = 400_000
        for offset in range(0, len(payload), chunk):
            part = payload[offset : offset + chunk]
            resp = _request(
                sock,
                {
                    "cmd": P.CMD_PUT,
                    "job": job,
                    "path": "in/mesh.bin",
                    "offset": offset,
                    "data": base64.b64encode(part).decode(),
                    "eof": offset + len(part) >= len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
            )
            assert resp["status"] == P.STATUS_OK
        assert resp["size"] == len(payload)
        assert resp["sha256"] == hashlib.sha256(payload).hexdigest()

        listing = _request(sock, {"cmd": P.CMD_LIST, "job": job, "path": "in"})
        assert [e["name"] for e in listing["entries"]] == ["mesh.bin"]

        got = b""
        offset = 0
        while True:
            resp = _request(
                sock, {"cmd": P.CMD_GET, "job": job, "path": "in/mesh.bin", "offset": offset, "length": 500_000}
            )
            assert resp["status"] == P.STATUS_OK
            got += base64.b64decode(resp["data"])
            if resp["eof"]:
                break
            offset = resp["offset"] + len(base64.b64decode(resp["data"]))
        assert got == payload

        cleaned = _request(sock, {"cmd": P.CMD_CLEAN, "job": job})
        assert cleaned["removed"] is True
        assert not (Path(staged["root"])).exists()
        assert _request(sock, {"cmd": P.CMD_LIST, "job": job, "path": "in"})["error_code"] == P.ERR_REMOTE_JOB

    def test_put_traversal_is_rejected(self, running) -> None:
        _, sock = running
        job = _request(sock, {"cmd": P.CMD_STAGE})["job"]
        for bad in ("../x", "/etc/passwd", "in/../../x"):
            resp = _request(
                sock,
                {"cmd": P.CMD_PUT, "job": job, "path": bad, "offset": 0, "data": base64.b64encode(b"x").decode()},
            )
            assert resp["status"] == P.STATUS_ERROR
            assert resp["error_code"] == P.ERR_REMOTE_PATH

    def test_unknown_job_is_rejected(self, running) -> None:
        _, sock = running
        resp = _request(sock, {"cmd": P.CMD_LIST, "job": "a" * 32, "path": "in"})
        assert resp["error_code"] == P.ERR_REMOTE_JOB


# ---------------------------------------------------------------------------
# Unidade: quotas, sha e TTL
# ---------------------------------------------------------------------------


class TestRemoteSandbox:
    def _sandbox(self, tmp_path: Path, **kwargs) -> RemoteSandbox:
        config = RemoteConfig(root=tmp_path / "remote", **kwargs)
        return RemoteSandbox(config)

    def test_offset_mismatch(self, tmp_path: Path) -> None:
        sandbox = self._sandbox(tmp_path)
        job = sandbox.stage()["job"]
        with pytest.raises(RemoteError) as exc:
            sandbox.put(job, "in/x", offset=10, data_b64=base64.b64encode(b"x").decode())
        assert exc.value.code == P.ERR_REMOTE_PATH

    def test_file_quota(self, tmp_path: Path) -> None:
        sandbox = self._sandbox(tmp_path, max_file_bytes=1024)
        job = sandbox.stage()["job"]
        with pytest.raises(RemoteError) as exc:
            sandbox.put(job, "in/big", offset=0, data_b64=base64.b64encode(b"x" * 2048).decode())
        assert exc.value.code == P.ERR_REMOTE_QUOTA

    def test_sha_mismatch(self, tmp_path: Path) -> None:
        sandbox = self._sandbox(tmp_path)
        job = sandbox.stage()["job"]
        with pytest.raises(RemoteError):
            sandbox.put(
                job,
                "in/x",
                offset=0,
                data_b64=base64.b64encode(b"abc").decode(),
                eof=True,
                sha256="0" * 64,
            )
        # O ficheiro rejeitado não fica no sandbox: um retry recomeça em offset 0.
        assert not (sandbox.root / job / "in" / "x").exists()
        sha = hashlib.sha256(b"abc").hexdigest()
        result = sandbox.put(
            job,
            "in/x",
            offset=0,
            data_b64=base64.b64encode(b"abc").decode(),
            eof=True,
            sha256=sha,
        )
        assert result["size"] == 3

    def test_get_offset_beyond_size(self, tmp_path: Path) -> None:
        sandbox = self._sandbox(tmp_path)
        job = sandbox.stage()["job"]
        sandbox.put(job, "in/x", offset=0, data_b64=base64.b64encode(b"abc").decode())
        with pytest.raises(RemoteError) as exc:
            sandbox.get(job, "in/x", offset=4)
        assert exc.value.code == P.ERR_REMOTE_PATH
        # Offset == tamanho continua a ser o fim limpo do download em chunks.
        empty = sandbox.get(job, "in/x", offset=3)
        assert empty["eof"] is True and empty["data"] == ""

    def test_put_into_directory_is_rejected(self, tmp_path: Path) -> None:
        sandbox = self._sandbox(tmp_path)
        job = sandbox.stage()["job"]
        with pytest.raises(RemoteError) as exc:
            sandbox.put(job, "in", offset=0, data_b64=base64.b64encode(b"x").decode())
        assert exc.value.code == P.ERR_REMOTE_PATH

    def test_stage_into_directory_is_rejected(self, tmp_path: Path) -> None:
        sandbox = self._sandbox(tmp_path)
        data = b"materializado"
        sha = hashlib.sha256(data).hexdigest()
        writer = sandbox.blobs.open_writer(sha, len(data))
        writer.write(data)
        writer.finish()
        with pytest.raises(RemoteError) as exc:
            sandbox.stage({"in": sha})
        assert exc.value.code == P.ERR_REMOTE_PATH
        assert sandbox.blobs.pinned(sha) is False

    def test_nul_path_is_rejected(self, tmp_path: Path) -> None:
        sandbox = self._sandbox(tmp_path)
        job = sandbox.stage()["job"]
        with pytest.raises(RemoteError) as exc:
            sandbox.put(job, "in/a\x00b", offset=0, data_b64=base64.b64encode(b"x").decode())
        assert exc.value.code == P.ERR_REMOTE_PATH

    @pytest.mark.skipif(os.name == "nt", reason="symlinks POSIX")
    def test_symlink_loop_is_a_remote_error(self, tmp_path: Path) -> None:
        sandbox = self._sandbox(tmp_path)
        job = sandbox.stage()["job"]
        loop = sandbox.root / job / "in" / "loop"
        loop.symlink_to(loop)
        with pytest.raises(RemoteError) as exc:
            sandbox.get(job, "in/loop")
        assert exc.value.code == P.ERR_REMOTE_PATH

    def test_duplicate_blob_refs_unpin_on_clean(self, tmp_path: Path) -> None:
        sandbox = self._sandbox(tmp_path)
        data = b"repetido"
        sha = hashlib.sha256(data).hexdigest()
        writer = sandbox.blobs.open_writer(sha, len(data))
        writer.write(data)
        writer.finish()
        staged = sandbox.stage({"in/a": sha, "in/b": sha})
        assert sandbox.blobs.pinned(sha) is True
        sandbox.clean(staged["job"])
        # Dois paths = duas referências; o clean tem de libertar as duas.
        assert sandbox.blobs.pinned(sha) is False

    def test_put_over_materialized_blob_does_not_corrupt_cas(self, tmp_path: Path) -> None:
        """Um append no sandbox não pode escrever através do hardlink para o CAS."""
        sandbox = self._sandbox(tmp_path)
        data = b"original"
        sha = hashlib.sha256(data).hexdigest()
        writer = sandbox.blobs.open_writer(sha, len(data))
        writer.write(data)
        writer.finish()
        staged = sandbox.stage({"in/x": sha})
        target = sandbox.root / staged["job"] / "in" / "x"
        blob_path = sandbox.blobs.path(sha)
        assert target.stat().st_ino == blob_path.stat().st_ino

        extra = b"-extra"
        combined = data + extra
        sandbox.put(
            staged["job"],
            "in/x",
            offset=len(data),
            data_b64=base64.b64encode(extra).decode(),
            eof=True,
            sha256=hashlib.sha256(combined).hexdigest(),
        )
        assert blob_path.read_bytes() == data  # CAS intacto
        assert target.read_bytes() == combined
        assert target.stat().st_ino != blob_path.stat().st_ino
        assert sandbox.blobs.pinned(sha) is True

    def test_sweep_ttl(self, tmp_path: Path) -> None:
        sandbox = self._sandbox(tmp_path, ttl_sec=60)
        old = sandbox.stage()["job"]
        fresh = sandbox.stage()["job"]
        old_dir = sandbox.root / old
        past = time.time() - 3600
        for path in [old_dir, *old_dir.rglob("*")]:
            path.touch()
            import os

            os.utime(path, (past, past))
        removed = sandbox.sweep()
        assert removed == [old]
        assert not old_dir.exists()
        assert (sandbox.root / fresh).is_dir()

    def test_sweep_never_deletes_blob_cas(self, tmp_path: Path) -> None:
        """O CAS vive em ``<root>/blobs``; o TTL dos jobs não o pode apagar."""
        sandbox = self._sandbox(tmp_path, ttl_sec=60)
        blob = b"keep-me"
        sha = hashlib.sha256(blob).hexdigest()
        writer = sandbox.blobs.open_writer(sha, len(blob))
        writer.write(blob)
        writer.finish()
        old = time.time() - 3600
        for path in [sandbox.blobs.root, *sandbox.blobs.root.rglob("*")]:
            import os

            os.utime(path, (old, old))
        assert sandbox.sweep() == []
        assert sandbox.blobs.path(sha).read_bytes() == blob

    def test_failed_stage_drops_pins(self, tmp_path: Path) -> None:
        sandbox = self._sandbox(tmp_path)
        good = b"good"
        sha = hashlib.sha256(good).hexdigest()
        writer = sandbox.blobs.open_writer(sha, len(good))
        writer.write(good)
        writer.finish()
        with pytest.raises(RemoteError):
            sandbox.stage({"in/a": sha, "in/missing": "f" * 64})
        assert sandbox.blobs.pinned(sha) is False


# ---------------------------------------------------------------------------
# Listener TCP explícito
# ---------------------------------------------------------------------------


class TestListen:
    def test_listen_descriptor_and_roundtrip(self, tmp_path: Path) -> None:
        srv, socket_path, thread = _start(tmp_path, listen=("127.0.0.1", 0))
        try:
            info = transport.read_endpoint(socket_path)
            assert info is not None
            assert info.host == "127.0.0.1"
            assert info.port and 1 <= info.port <= 65535
            assert info.token
            resp = _request(socket_path, {"cmd": P.CMD_STATUS})
            assert resp["status"] == P.STATUS_STATUS
            assert resp["transport"] == transport.TRANSPORT_TCP
        finally:
            _stop(srv, socket_path, thread)

    def test_stable_token(self, tmp_path: Path) -> None:
        token = "stable-token-for-tests"
        srv, socket_path, thread = _start(tmp_path, listen=("127.0.0.1", 0), token=token)
        try:
            assert transport.read_endpoint(socket_path).token == token
        finally:
            _stop(srv, socket_path, thread)
