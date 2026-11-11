"""Transporte e camada de plataforma — Unix no POSIX, TCP+token no Windows.

O caminho TCP é exercitado em Linux com ``VRAMD_TRANSPORT=tcp``; no Windows é o
default. Assim o CI Linux cobre a mesma lógica que o CI Windows usa, sem
esperar por um runner Windows para descobrir regressões de transporte.
"""

from __future__ import annotations

import io
import json
import os
import socket
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from vramd import host, transport
from vramd import protocol as P
from vramd.subprocess_pool import _ThreadedLineReader

from .test_server import _send_raw, _send_request, _start_ums, _stop_ums


@pytest.fixture
def tcp_mode(monkeypatch):
    """Força o transporte TCP (o default do Windows) em qualquer SO."""
    monkeypatch.setenv(transport.ENV_TRANSPORT, transport.TRANSPORT_TCP)
    return transport.TRANSPORT_TCP


class TestTransportMode:
    def test_env_override_wins(self, monkeypatch) -> None:
        monkeypatch.setenv(transport.ENV_TRANSPORT, transport.TRANSPORT_TCP)
        assert transport.transport_mode() == transport.TRANSPORT_TCP
        monkeypatch.setenv(transport.ENV_TRANSPORT, transport.TRANSPORT_UNIX)
        assert transport.transport_mode() == transport.TRANSPORT_UNIX

    def test_invalid_override_falls_back(self, monkeypatch) -> None:
        monkeypatch.setenv(transport.ENV_TRANSPORT, "rdp")
        assert transport.transport_mode() in (transport.TRANSPORT_UNIX, transport.TRANSPORT_TCP)

    def test_unix_mode_needs_af_unix(self, monkeypatch) -> None:
        monkeypatch.setattr(transport, "has_unix_sockets", lambda: False)
        monkeypatch.delenv(transport.ENV_TRANSPORT, raising=False)
        monkeypatch.setattr(host, "IS_WINDOWS", True)
        assert transport.transport_mode() == transport.TRANSPORT_TCP


class TestDescriptor:
    def test_write_read_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "vramd.sock"
        info = transport.write_endpoint(path, port=43210, token="tok")
        assert info.port == 43210
        read = transport.read_endpoint(path)
        assert read is not None
        assert (read.host, read.port, read.token) == ("127.0.0.1", 43210, "tok")
        assert transport.endpoint_label(path) == "tcp://127.0.0.1:43210"

    def test_bad_descriptor_is_none(self, tmp_path: Path) -> None:
        path = tmp_path / "vramd.sock"
        path.write_text("não é json\n", encoding="utf-8")
        assert transport.read_endpoint(path) is None
        path.write_text(json.dumps({"transport": "tcp", "port": 0}), encoding="utf-8")
        assert transport.read_endpoint(path) is None

    def test_auth_helpers(self) -> None:
        req = {"cmd": "status"}
        assert transport.with_auth(req, None) is req
        signed = transport.with_auth(req, "abc")
        assert signed["auth"] == "abc"
        assert transport.is_authorized(signed, "abc") is True
        assert transport.is_authorized(signed, "outro") is False
        assert transport.is_authorized({"cmd": "status"}, "abc") is False
        assert transport.is_authorized({"cmd": "status"}, None) is True


class TestTcpServer:
    """Servidor real no modo TCP (equivalente ao Windows)."""

    def test_roundtrip_and_unauthorized(self, tmp_path: Path, tcp_mode) -> None:
        srv, sock, thread = _start_ums(tmp_path)
        try:
            info = transport.read_endpoint(sock)
            assert info is not None and info.port and info.token

            status = _send_request(sock, {"cmd": P.CMD_STATUS})
            assert status["status"] == P.STATUS_STATUS
            assert status["transport"] == transport.TRANSPORT_TCP
            assert status["endpoint"] == f"tcp://127.0.0.1:{info.port}"

            # Sem o token do descritor: recusa antes de tocar na fila.
            denied = _send_raw(sock, {"cmd": P.CMD_GENERATE, "backend": "alpha", "prompt": "x"})
            assert denied["status"] == P.STATUS_ERROR
            assert denied["error_code"] == P.ERR_UNAUTHORIZED

            # E o token não vaza para o payload do job.
            ok = _send_request(sock, {"cmd": P.CMD_GENERATE, "backend": "alpha", "prompt": "x"})
            assert ok["status"] == P.STATUS_OK
        finally:
            _stop_ums(srv, sock, thread)

    def test_client_api_uses_token(self, tmp_path: Path, tcp_mode) -> None:
        from vramd.client import send_request

        srv, sock, thread = _start_ums(tmp_path)
        try:
            resp = send_request({"cmd": P.CMD_STATUS}, sock, timeout_sec=5.0)
            assert resp is not None
            assert resp["status"] == P.STATUS_STATUS
        finally:
            _stop_ums(srv, sock, thread)

    def test_endpoint_removed_on_shutdown(self, tmp_path: Path, tcp_mode) -> None:
        srv, sock, thread = _start_ums(tmp_path)
        assert sock.exists()
        _stop_ums(srv, sock, thread)
        assert not sock.exists()


class TestThreadedLineReader:
    def test_reads_lines_then_eof(self) -> None:
        reader = _ThreadedLineReader(io.StringIO('{"event":"ready"}\n{"event":"done"}\n'))
        try:
            assert reader.read(2.0) == '{"event":"ready"}\n'
            assert reader.read(2.0) == '{"event":"done"}\n'
            assert reader.read(2.0) is None
            deadline = time.monotonic() + 2.0
            while not reader.eof and time.monotonic() < deadline:
                time.sleep(0.01)
            assert reader.eof is True
        finally:
            reader.close()

    def test_drains_a_real_subprocess_pipe(self) -> None:
        """Cobre a thread leitora sobre um pipe real (caminho do Windows)."""
        import subprocess

        proc = subprocess.Popen(
            [sys.executable, "-c", "print('um'); print('dois')"],
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        assert proc.stdout is not None
        reader = _ThreadedLineReader(proc.stdout)
        try:
            assert reader.read(5.0).strip() == "um"
            assert reader.read(5.0).strip() == "dois"
            assert reader.read(5.0) is None
        finally:
            reader.close()
            proc.wait(timeout=5.0)


class TestHostLayer:
    @pytest.mark.skipif(not host.IS_WINDOWS, reason="no POSIX o psutil.pid_exists usa os.kill (seguro)")
    def test_pid_exists_never_uses_os_kill(self, monkeypatch) -> None:
        """No Windows ``os.kill(pid, 0)`` mata: o probe não pode passar por lá."""
        monkeypatch.setattr(os, "kill", lambda *a, **k: pytest.fail("os.kill não pode ser usado"))
        assert host.pid_exists(os.getpid()) is True
        assert host.pid_exists(-1) is False

    def test_pid_exists_of_self(self) -> None:
        assert host.pid_exists(os.getpid()) is True
        assert host.pid_exists(-1) is False

    def test_same_user_of_self(self) -> None:
        assert host.same_user(os.getpid()) is True

    def test_cache_dir_env_override(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("VRAMD_CACHE_DIR", str(tmp_path / "cache"))
        assert host.cache_dir() == tmp_path / "cache"
        assert host.default_socket_path() == tmp_path / "cache" / "vramd.sock"

    def test_config_dir_env_override(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("VRAMD_CONFIG_DIR", str(tmp_path / "conf"))
        assert host.config_dir() == tmp_path / "conf"

    def test_signal_handlers_refused_off_main_thread(self) -> None:
        seen: list[int] = []

        def _run() -> None:
            seen.extend(host.install_signal_handlers(lambda *_: None))

        t = threading.Thread(target=_run)
        t.start()
        t.join(timeout=5.0)
        assert seen == []


class TestWindowsLockLayout:
    """O lock do Windows fica fora da região do PID (locks são obrigatórios).

    Um msvcrt falso permite exercitar o caminho ``_acquire_windows`` em Linux;
    no Windows o teste real (:class:`tests.test_process_guard.TestSingletonLock`)
    usa o msvcrt verdadeiro.
    """

    def test_lock_offset_keeps_pid_readable(self, tmp_path: Path, monkeypatch) -> None:
        from vramd import process_guard as pg

        state = {"locked": False}
        fake = types.ModuleType("msvcrt")
        fake.LK_NBLCK = 1  # type: ignore[attr-defined]
        fake.LK_UNLCK = 2  # type: ignore[attr-defined]

        def locking(fd: int, mode: int, nbytes: int) -> None:
            assert os.lseek(fd, 0, os.SEEK_CUR) == pg._WINDOWS_LOCK_OFFSET
            if mode == fake.LK_NBLCK:  # type: ignore[attr-defined]
                if state["locked"]:
                    raise OSError("já bloqueado")
                state["locked"] = True
            else:
                state["locked"] = False

        fake.locking = locking  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "msvcrt", fake)
        monkeypatch.setattr(pg.host, "IS_WINDOWS", True)

        path = tmp_path / "model-server.lock"
        first = pg.SingletonLock(path)
        second = pg.SingletonLock(path)
        assert first.acquire() is True
        assert first.owner_pid() == os.getpid()
        assert path.stat().st_size >= pg._WINDOWS_LOCK_OFFSET + 1
        assert second.acquire() is False
        first.release()
        assert state["locked"] is False
        assert second.acquire() is True
        second.release()


@pytest.mark.skipif(os.name == "nt", reason="SO_REUSEADDR é POSIX")
def test_tcp_listener_sets_reuseaddr(tmp_path: Path) -> None:
    """Restart com porta fixa não pode ficar preso em TIME_WAIT."""
    sock, info = transport.listen(tmp_path / "ep", bind_host="127.0.0.1", port=0)
    try:
        assert info.transport == transport.TRANSPORT_TCP
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) != 0
    finally:
        sock.close()
