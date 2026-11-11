"""Cliente do protocolo v2 (frames binários + blobs).

Uso típico::

    with TransferClient(socket_path) as client:
        client.put_blob(sha, path)
        job = client.request({"cmd": "stage", "blobs": {"in/mesh.glb": sha}})

O ``hello`` vai como linha JSON (compatível com o protocolo antigo); a partir
daí a ligação usa frames. Um servidor 0.6 responde com erro ao ``hello`` e o
cliente levanta ``TransferError`` — quem precisa de compatibilidade usa o
caminho JSON/base64 antigo.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import frames, transport
from . import protocol as P

DEFAULT_TIMEOUT_SEC = 300.0
CHUNK_BYTES = 8 * 1024 * 1024


class TransferError(RuntimeError):
    """Falha de transporte/protocolo no cliente v2."""


class TransferClient:
    """Ligação persistente ao vramd com handshake de capabilities."""

    def __init__(
        self,
        socket_path: Path | str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        endpoints: list[transport.EndpointInfo] | None = None,
    ) -> None:
        self.socket_path = Path(socket_path) if socket_path is not None else P.DEFAULT_SOCKET_PATH
        self.timeout = timeout
        self.endpoints = endpoints if endpoints is not None else transport.remote_endpoints()
        self._sock: socket.socket | None = None
        self._reader: frames.FrameReader | None = None
        self._token: str | None = None
        self._request_id = 0
        self.hello_response: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def __enter__(self) -> TransferClient:
        self.connect()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def connect(self) -> dict[str, Any]:
        if self._sock is not None:
            assert self.hello_response is not None
            return self.hello_response
        last_error: Exception | None = None
        candidates = self.endpoints or [None]
        for info in candidates:
            try:
                if info is None:
                    sock, token = transport.connect(self.socket_path, timeout=self.timeout)
                else:
                    sock, token = transport.connect_remote(info, timeout=self.timeout)
            except OSError as exc:
                last_error = exc
                continue
            reader = frames.FrameReader(sock)
            try:
                request = {"cmd": P.CMD_HELLO, "protocol": P.PROTOCOL_VERSION, "keepalive": True}
                sock.sendall((json.dumps(transport.with_auth(request, token)) + "\n").encode())
                line = reader.read_line(max_bytes=frames.MAX_JSON_BYTES)
                response = json.loads(line) if line else {}
            except (OSError, ValueError, frames.FrameError) as exc:
                sock.close()
                last_error = exc
                continue
            if not isinstance(response, dict) or response.get("status") != P.STATUS_OK:
                sock.close()
                last_error = TransferError(
                    f"hello recusado: {response.get('error') if isinstance(response, dict) else response}"
                )
                continue
            features = set(response.get("features") or ())
            if P.FEATURE_BINARY not in features or P.FEATURE_BLOBS not in features:
                sock.close()
                last_error = TransferError(
                    f"servidor vramd sem protocolo binário/blobs; atualiza o servidor (features={sorted(features)})"
                )
                continue
            self._sock = sock
            self._reader = reader
            self._token = token
            self.hello_response = response
            return response
        raise TransferError(f"vramd remoto inacessível: {last_error}")

    def close(self) -> None:
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._sock = None
            self._reader = None

    # ------------------------------------------------------------------
    # Framing
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._request_id = (self._request_id + 1) & 0xFFFFFFFF
        if self._request_id == 0:
            self._request_id = 1
        return self._request_id

    def _send_json(self, request: dict[str, Any], request_id: int) -> None:
        assert self._sock is not None
        payload = frames.encode_json(transport.with_auth(request, self._token))
        frames.send_frame(self._sock, frames.TYPE_JSON, payload, request_id=request_id)

    def _read_until_end(self, request_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        assert self._reader is not None
        last: dict[str, Any] = {}
        events: list[dict[str, Any]] = []
        while True:
            frame = self._reader.read_frame(max_payload=frames.MAX_DATA_BYTES)
            if frame is None:
                raise TransferError("servidor fechou a ligação antes do END")
            if frame.request_id != request_id:
                raise TransferError(f"request_id fora de sequência: {frame.request_id} != {request_id}")
            if frame.type == frames.TYPE_EVENT:
                events.append(frames.decode_json(frame.payload))
                continue
            if frame.type == frames.TYPE_JSON:
                last = frames.decode_json(frame.payload)
                continue
            if frame.type == frames.TYPE_ERROR:
                raise TransferError(frames.decode_payload(frame.flags, frame.payload).decode("utf-8", "replace"))
            if frame.type == frames.TYPE_END:
                break
            raise TransferError(f"frame inesperado: tipo {frame.type}")
        return last, events

    def rpc(self, request: dict[str, Any], *, timeout: float = DEFAULT_TIMEOUT_SEC) -> dict[str, Any]:
        """Alias compatível com ``RemoteClient.rpc`` (devolve o dict tal como veio)."""
        return self.request(request, timeout=timeout)

    # ------------------------------------------------------------------
    # Pedidos
    # ------------------------------------------------------------------

    def request(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        self.connect()
        if timeout is not None:
            assert self._sock is not None
            self._sock.settimeout(timeout)
        request_id = self._next_id()
        self._send_json(request, request_id)
        response, _events = self._read_until_end(request_id)
        return response

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterator[dict[str, Any]]:
        """Gera eventos e a resposta final (JSON) até ao END."""
        self.connect()
        if timeout is not None:
            assert self._sock is not None
            self._sock.settimeout(timeout)
        request_id = self._next_id()
        self._send_json(request, request_id)
        assert self._reader is not None
        while True:
            frame = self._reader.read_frame(max_payload=frames.MAX_DATA_BYTES)
            if frame is None:
                raise TransferError("servidor fechou a ligação antes do END")
            if frame.request_id != request_id:
                raise TransferError(f"request_id fora de sequência: {frame.request_id} != {request_id}")
            if frame.type == frames.TYPE_EVENT:
                yield frames.decode_json(frame.payload)
                continue
            if frame.type == frames.TYPE_JSON:
                obj = frames.decode_json(frame.payload)
                yield obj
                continue
            if frame.type == frames.TYPE_END:
                return
            if frame.type == frames.TYPE_ERROR:
                raise TransferError(frames.decode_payload(frame.flags, frame.payload).decode("utf-8", "replace"))
            raise TransferError(f"frame inesperado: tipo {frame.type}")

    # ------------------------------------------------------------------
    # Blobs
    # ------------------------------------------------------------------

    def blob_has(self, sha256: str, size: int | None = None) -> dict[str, Any]:
        request: dict[str, Any] = {"cmd": P.CMD_BLOB_HAS, "sha256": sha256}
        if size is not None:
            request["size"] = int(size)
        return self.request(request)

    def blob_stats(self) -> dict[str, Any]:
        return self.request({"cmd": P.CMD_BLOBS})

    def blob_gc(self) -> dict[str, Any]:
        return self.request({"cmd": P.CMD_BLOBS_GC})

    def put_blob(self, sha256: str, path: Path | str, *, chunk_size: int = CHUNK_BYTES) -> dict[str, Any]:
        """Sobe um ficheiro se o servidor ainda não tiver o blob."""
        self.connect()
        source = Path(path)
        size = source.stat().st_size
        existing = self.blob_has(sha256, size)
        if existing.get("exists"):
            return {
                "status": P.STATUS_OK,
                "sha256": existing.get("sha256", sha256),
                "size": size,
                "deduped": True,
                "skipped": True,
            }
        request_id = self._next_id()
        self._send_json({"cmd": P.CMD_BLOB_PUT, "sha256": sha256, "size": size}, request_id)
        assert self._sock is not None
        with source.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                flags, payload = frames.encode_payload(chunk)
                frames.send_frame(
                    self._sock,
                    frames.TYPE_DATA,
                    payload,
                    request_id=request_id,
                    flags=flags,
                )
        frames.send_frame(self._sock, frames.TYPE_END, b"", request_id=request_id)
        response, _events = self._read_until_end(request_id)
        if response.get("status") != P.STATUS_OK:
            raise TransferError(str(response.get("error") or response))
        return response

    def put_blob_bytes(self, sha256: str, data: bytes, *, chunk_size: int = CHUNK_BYTES) -> dict[str, Any]:
        self.connect()
        existing = self.blob_has(sha256, len(data))
        if existing.get("exists"):
            return {
                "status": P.STATUS_OK,
                "sha256": existing.get("sha256", sha256),
                "size": len(data),
                "deduped": True,
                "skipped": True,
            }
        request_id = self._next_id()
        self._send_json({"cmd": P.CMD_BLOB_PUT, "sha256": sha256, "size": len(data)}, request_id)
        assert self._sock is not None
        for start in range(0, len(data), chunk_size):
            chunk = data[start : start + chunk_size]
            flags, payload = frames.encode_payload(chunk)
            frames.send_frame(
                self._sock,
                frames.TYPE_DATA,
                payload,
                request_id=request_id,
                flags=flags,
            )
        frames.send_frame(self._sock, frames.TYPE_END, b"", request_id=request_id)
        response, _events = self._read_until_end(request_id)
        if response.get("status") != P.STATUS_OK:
            raise TransferError(str(response.get("error") or response))
        return response

    def get_blob(self, sha256: str, destination: Path | str, *, chunk_size: int = CHUNK_BYTES) -> int:
        """Descarrega um blob para ``destination`` de forma atómica."""
        self.connect()
        target = Path(destination)
        info = self.blob_has(sha256)
        if not info.get("exists"):
            raise TransferError(f"blob não existe no servidor: {sha256}")
        size = int(info.get("size") or 0)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        request_id = self._next_id()
        self._send_json(
            {"cmd": P.CMD_BLOB_GET, "sha256": sha256, "offset": 0, "length": size or frames.MAX_DATA_BYTES},
            request_id,
        )
        assert self._reader is not None
        digest = hashlib.sha256()
        written = 0
        try:
            with temporary.open("wb") as handle:
                got_meta = False
                while True:
                    frame = self._reader.read_frame(max_payload=frames.MAX_DATA_BYTES)
                    if frame is None:
                        raise TransferError("servidor fechou a ligação antes do END")
                    if frame.request_id != request_id:
                        raise TransferError(f"request_id fora de sequência: {frame.request_id} != {request_id}")
                    if frame.type == frames.TYPE_JSON:
                        meta = frames.decode_json(frame.payload)
                        if meta.get("status") == P.STATUS_ERROR:
                            raise TransferError(str(meta.get("error") or meta))
                        got_meta = True
                        size = int(meta.get("size") or size)
                        continue
                    if frame.type == frames.TYPE_DATA:
                        chunk = frames.decode_payload(frame.flags, frame.payload)
                        handle.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                        continue
                    if frame.type == frames.TYPE_END:
                        break
                    if frame.type == frames.TYPE_ERROR:
                        raise TransferError(
                            frames.decode_payload(frame.flags, frame.payload).decode("utf-8", "replace")
                        )
                    raise TransferError(f"frame inesperado: tipo {frame.type}")
                if not got_meta:
                    raise TransferError("resposta de blob-get sem metadados")
            if written != size:
                raise TransferError(f"download incompleto: {written}/{size} bytes")
            actual = digest.hexdigest()
            if actual != sha256.lower():
                raise TransferError(f"sha256 do download não bate: {actual} != {sha256}")
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return written

    # ------------------------------------------------------------------
    # Compatibilidade com o staging antigo (RemoteClient)
    # ------------------------------------------------------------------

    def stage(self, blobs: dict[str, str] | None = None) -> tuple[str, str]:
        response = self.request({"cmd": P.CMD_STAGE, "blobs": blobs or {}})
        if response.get("status") != P.STATUS_OK or not response.get("job"):
            raise TransferError(f"stage falhou: {response.get('error') or response}")
        return str(response["job"]), str(response.get("root") or "")

    def get_file(self, job: str, rel: str, local: Path | str, *, timeout: float | None = None) -> int:
        """Descarrega ``<job>/<rel>`` para ``local`` via frames (sem base64)."""
        self.connect()
        target = Path(local)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        request_id = self._next_id()
        self._send_json(
            {"cmd": P.CMD_JOB_GET, "job": job, "path": rel, "offset": 0, "length": 0},
            request_id,
        )
        if timeout is not None:
            assert self._sock is not None
            self._sock.settimeout(timeout)
        assert self._reader is not None
        size = 0
        written = 0
        got_meta = False
        try:
            with temporary.open("wb") as handle:
                while True:
                    frame = self._reader.read_frame(max_payload=frames.MAX_DATA_BYTES)
                    if frame is None:
                        raise TransferError("servidor fechou a ligação antes do END")
                    if frame.request_id != request_id:
                        raise TransferError(f"request_id fora de sequência: {frame.request_id} != {request_id}")
                    if frame.type == frames.TYPE_JSON:
                        meta = frames.decode_json(frame.payload)
                        if meta.get("status") == P.STATUS_ERROR:
                            raise TransferError(str(meta.get("error") or meta))
                        size = int(meta.get("size") or 0)
                        got_meta = True
                        continue
                    if frame.type == frames.TYPE_DATA:
                        chunk = frames.decode_payload(frame.flags, frame.payload)
                        handle.write(chunk)
                        written += len(chunk)
                        continue
                    if frame.type == frames.TYPE_END:
                        break
                    if frame.type == frames.TYPE_ERROR:
                        raise TransferError(
                            frames.decode_payload(frame.flags, frame.payload).decode("utf-8", "replace")
                        )
                    raise TransferError(f"frame inesperado: tipo {frame.type}")
                if not got_meta:
                    raise TransferError("resposta de job-get sem metadados")
            if written != size:
                raise TransferError(f"download incompleto: {written}/{size} bytes")
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return written

    def list(self, job: str, rel: str = "out") -> list[str]:
        response = self.request({"cmd": P.CMD_LIST, "job": job, "path": rel})
        if response.get("status") != P.STATUS_OK:
            raise TransferError(f"list {rel} falhou: {response.get('error') or response}")
        return [str(entry["name"]) for entry in response.get("entries") or [] if not entry.get("is_dir")]

    def clean(self, job: str) -> None:
        with contextlib.suppress(TransferError, OSError):
            self.request({"cmd": P.CMD_CLEAN, "job": job})
