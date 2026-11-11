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


def _bounded_chunk_size(value: int) -> int:
    """Chunk nunca maior que um frame DATA (o servidor recusa acima do limite).

    Um ``chunk_size`` maior que ``MAX_DATA_BYTES`` fazia o cliente enviar um
    header que o servidor rejeita e fechar a ligação a meio do upload; o erro
    seguinte era um desalinhamento de request_id, não a causa.
    """
    return max(1, min(int(value), frames.MAX_DATA_BYTES))


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
        # Endpoint que aceitou o hello: uma reconexão tem de voltar ao mesmo
        # servidor (o job/sessão não existe nos peers).
        self.endpoint: transport.EndpointInfo | None = None

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
            self.endpoint = info
            self.hello_response = response
            return response
        raise TransferError(f"vramd remoto inacessível: {last_error}")

    def close(self) -> None:
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._sock = None
            self._reader = None

    def _reset_after_protocol_error(self) -> None:
        """Fecha a ligação depois de um erro a meio de upload/download.

        O servidor pode ter frames por consumir (ex.: recusou o blob antes de
        ler os DATA), pelo que o socket deixa de estar sincronizado; reutilizá-lo
        fazia o pedido seguinte falhar com ``request_id fora de sequência``.
        ``connect()`` reabre na chamada seguinte.
        """
        self.close()

    def reconnect(self) -> dict[str, Any]:
        """Fecha e reabre no **mesmo** endpoint (idle timeout, restart, ...)."""
        endpoint = self.endpoint
        self.close()
        self.hello_response = None
        if endpoint is None:
            return self.connect()
        saved = self.endpoints
        self.endpoints = [endpoint]
        try:
            return self.connect()
        finally:
            self.endpoints = saved

    # ------------------------------------------------------------------
    # Framing
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._request_id = (self._request_id + 1) & 0xFFFFFFFF
        if self._request_id == 0:
            self._request_id = 1
        return self._request_id

    def _send_frame(
        self,
        frame_type: int,
        payload: bytes = b"",
        *,
        request_id: int,
        flags: int = frames.FLAG_RAW,
    ) -> None:
        assert self._sock is not None
        try:
            frames.send_frame(
                self._sock,
                frame_type,
                payload,
                request_id=request_id,
                flags=flags,
            )
        except OSError as exc:
            raise TransferError(f"ligação ao vramd falhou: {exc}") from exc

    def _read_frame(self) -> frames.Frame | None:
        assert self._reader is not None
        try:
            return self._reader.read_frame(max_payload=frames.MAX_DATA_BYTES)
        except OSError as exc:
            raise TransferError(f"ligação ao vramd falhou: {exc}") from exc

    @staticmethod
    def _decode_json(payload: bytes) -> dict[str, Any]:
        try:
            return frames.decode_json(payload)
        except frames.FrameError as exc:
            raise TransferError(f"frame JSON inválido: {exc}") from exc

    @staticmethod
    def _decode_payload(flags: int, payload: bytes) -> bytes:
        try:
            return frames.decode_payload(flags, payload)
        except frames.FrameError as exc:
            raise TransferError(f"payload de frame inválido: {exc}") from exc

    def _send_json(self, request: dict[str, Any], request_id: int) -> None:
        payload = frames.encode_json(transport.with_auth(request, self._token))
        self._send_frame(frames.TYPE_JSON, payload, request_id=request_id)

    def _read_until_end(self, request_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        last: dict[str, Any] = {}
        events: list[dict[str, Any]] = []
        while True:
            frame = self._read_frame()
            if frame is None:
                raise TransferError("servidor fechou a ligação antes do END")
            if frame.request_id != request_id:
                raise TransferError(f"request_id fora de sequência: {frame.request_id} != {request_id}")
            if frame.type == frames.TYPE_EVENT:
                events.append(self._decode_json(frame.payload))
                continue
            if frame.type == frames.TYPE_JSON:
                last = self._decode_json(frame.payload)
                continue
            if frame.type == frames.TYPE_ERROR:
                raise TransferError(self._decode_payload(frame.flags, frame.payload).decode("utf-8", "replace"))
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

    def _set_timeout(self, timeout: float | None) -> float | None:
        """Aplica ``timeout`` e devolve o anterior (para restaurar)."""
        if timeout is None or self._sock is None:
            return None
        previous = self._sock.gettimeout()
        self._sock.settimeout(timeout)
        return previous

    def _restore_timeout(self, previous: float | None) -> None:
        if previous is None or self._sock is None:
            return
        with contextlib.suppress(OSError):
            self._sock.settimeout(previous)

    def request(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        self.connect()
        previous = self._set_timeout(timeout)
        try:
            request_id = self._next_id()
            self._send_json(request, request_id)
            response, _events = self._read_until_end(request_id)
            return response
        except TransferError:
            # Timeout/frame inválido a meio da resposta: o servidor pode ainda
            # estar a enviar frames e o socket deixa de estar sincronizado.
            self._reset_after_protocol_error()
            raise
        finally:
            # Um timeout curto para um RPC não pode ficar colado à ligação
            # persistente e abortar o job seguinte (wait/download longo).
            self._restore_timeout(previous)

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterator[dict[str, Any]]:
        """Gera eventos e a resposta final (JSON) até ao END."""
        self.connect()
        previous = self._set_timeout(timeout)
        try:
            request_id = self._next_id()
            self._send_json(request, request_id)
            while True:
                frame = self._read_frame()
                if frame is None:
                    raise TransferError("servidor fechou a ligação antes do END")
                if frame.request_id != request_id:
                    raise TransferError(f"request_id fora de sequência: {frame.request_id} != {request_id}")
                if frame.type == frames.TYPE_EVENT:
                    yield self._decode_json(frame.payload)
                    continue
                if frame.type == frames.TYPE_JSON:
                    obj = self._decode_json(frame.payload)
                    yield obj
                    continue
                if frame.type == frames.TYPE_END:
                    return
                if frame.type == frames.TYPE_ERROR:
                    raise TransferError(self._decode_payload(frame.flags, frame.payload).decode("utf-8", "replace"))
                raise TransferError(f"frame inesperado: tipo {frame.type}")
        except TransferError:
            # Erro a meio do stream: os eventos/frames restantes ficariam na
            # ligação e o pedido seguinte lia lixo (request_id errado).
            self._reset_after_protocol_error()
            raise
        finally:
            self._restore_timeout(previous)

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
        chunk_size = _bounded_chunk_size(chunk_size)
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
        try:
            self._send_json({"cmd": P.CMD_BLOB_PUT, "sha256": sha256, "size": size}, request_id)
            sent = 0
            with source.open("rb") as handle:
                # Só o tamanho declarado: se o ficheiro crescer durante o upload,
                # o excedente faria o writer do servidor recusar o frame e
                # deixar frames por consumir na ligação.
                while sent < size:
                    chunk = handle.read(min(chunk_size, size - sent))
                    if not chunk:
                        break
                    sent += len(chunk)
                    flags, payload = frames.encode_payload(chunk)
                    self._send_frame(
                        frames.TYPE_DATA,
                        payload,
                        request_id=request_id,
                        flags=flags,
                    )
            self._send_frame(frames.TYPE_END, b"", request_id=request_id)
            response, _events = self._read_until_end(request_id)
        except BaseException:
            self._reset_after_protocol_error()
            raise
        if response.get("status") != P.STATUS_OK:
            # Um erro cedo do servidor (quota/tamanho) chega antes de ele
            # consumir todos os DATA frames; a ligação já não é reutilizável.
            self._reset_after_protocol_error()
            raise TransferError(str(response.get("error") or response))
        return response

    def put_blob_bytes(self, sha256: str, data: bytes, *, chunk_size: int = CHUNK_BYTES) -> dict[str, Any]:
        self.connect()
        chunk_size = _bounded_chunk_size(chunk_size)
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
        try:
            self._send_json({"cmd": P.CMD_BLOB_PUT, "sha256": sha256, "size": len(data)}, request_id)
            for start in range(0, len(data), chunk_size):
                chunk = data[start : start + chunk_size]
                flags, payload = frames.encode_payload(chunk)
                self._send_frame(
                    frames.TYPE_DATA,
                    payload,
                    request_id=request_id,
                    flags=flags,
                )
            self._send_frame(frames.TYPE_END, b"", request_id=request_id)
            response, _events = self._read_until_end(request_id)
        except BaseException:
            self._reset_after_protocol_error()
            raise
        if response.get("status") != P.STATUS_OK:
            self._reset_after_protocol_error()
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
        finished = False
        try:
            with temporary.open("wb") as handle:
                got_meta = False
                while True:
                    frame = self._read_frame()
                    if frame is None:
                        raise TransferError("servidor fechou a ligação antes do END")
                    if frame.request_id != request_id:
                        raise TransferError(f"request_id fora de sequência: {frame.request_id} != {request_id}")
                    if frame.type == frames.TYPE_JSON:
                        meta = self._decode_json(frame.payload)
                        if meta.get("status") == P.STATUS_ERROR:
                            raise TransferError(str(meta.get("error") or meta))
                        reported = meta.get("size")
                        if reported is not None and int(reported) != size:
                            # O sha fixa o conteúdo (e logo o tamanho); um
                            # tamanho diferente é resposta inconsistente e não
                            # pode definir o limite do que se escreve em disco.
                            raise TransferError(
                                f"tamanho do blob mudou entre blob-has e blob-get: {size} -> {reported}"
                            )
                        got_meta = True
                        continue
                    if frame.type == frames.TYPE_DATA:
                        chunk = self._decode_payload(frame.flags, frame.payload)
                        if got_meta and written + len(chunk) > size:
                            # Um servidor buggy/malicioso não pode encher o disco
                            # local: o blob-get declara o tamanho nos metadados.
                            raise TransferError(f"servidor enviou mais de {size} bytes no blob-get")
                        handle.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                        continue
                    if frame.type == frames.TYPE_END:
                        finished = True
                        break
                    if frame.type == frames.TYPE_ERROR:
                        raise TransferError(self._decode_payload(frame.flags, frame.payload).decode("utf-8", "replace"))
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
            if not finished:
                self._reset_after_protocol_error()
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
        previous = self._set_timeout(timeout)
        size = 0
        written = 0
        got_meta = False
        finished = False
        try:
            request_id = self._next_id()
            self._send_json(
                {"cmd": P.CMD_JOB_GET, "job": job, "path": rel, "offset": 0, "length": 0},
                request_id,
            )
            with temporary.open("wb") as handle:
                while True:
                    frame = self._read_frame()
                    if frame is None:
                        raise TransferError("servidor fechou a ligação antes do END")
                    if frame.request_id != request_id:
                        raise TransferError(f"request_id fora de sequência: {frame.request_id} != {request_id}")
                    if frame.type == frames.TYPE_JSON:
                        meta = self._decode_json(frame.payload)
                        if meta.get("status") == P.STATUS_ERROR:
                            raise TransferError(str(meta.get("error") or meta))
                        size = int(meta.get("size") or 0)
                        got_meta = True
                        continue
                    if frame.type == frames.TYPE_DATA:
                        chunk = self._decode_payload(frame.flags, frame.payload)
                        if got_meta and written + len(chunk) > size:
                            raise TransferError(f"servidor enviou mais de {size} bytes no job-get")
                        handle.write(chunk)
                        written += len(chunk)
                        continue
                    if frame.type == frames.TYPE_END:
                        finished = True
                        break
                    if frame.type == frames.TYPE_ERROR:
                        raise TransferError(self._decode_payload(frame.flags, frame.payload).decode("utf-8", "replace"))
                    raise TransferError(f"frame inesperado: tipo {frame.type}")
                if not got_meta:
                    raise TransferError("resposta de job-get sem metadados")
            if written != size:
                raise TransferError(f"download incompleto: {written}/{size} bytes")
            os.replace(temporary, target)
        except BaseException:
            if not finished:
                self._reset_after_protocol_error()
            temporary.unlink(missing_ok=True)
            raise
        finally:
            self._restore_timeout(previous)
        return written

    def list(self, job: str, rel: str = "out") -> list[str]:
        response = self.request({"cmd": P.CMD_LIST, "job": job, "path": rel})
        if response.get("status") != P.STATUS_OK:
            raise TransferError(f"list {rel} falhou: {response.get('error') or response}")
        return [str(entry["name"]) for entry in response.get("entries") or [] if not entry.get("is_dir")]

    def clean(self, job: str) -> None:
        with contextlib.suppress(TransferError, OSError):
            self.request({"cmd": P.CMD_CLEAN, "job": job})
