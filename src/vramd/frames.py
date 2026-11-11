"""Framing binário do protocolo v2.

O protocolo histórico é uma linha JSON (``\\n``) por mensagem. Para blobs
grandes isso obriga a base64 (≈33% de overhead) e a uma ligação nova por
chunk. O protocolo v2 mantém o caminho JSON para compatibilidade e acrescenta
frames binários length-prefixed:

    magic (4s) | versão (B) | tipo (B) | flags (B) | reservado (B)
               | request_id (I) | length (I) | payload

O header tem 16 bytes e usa network byte order. ``length`` conta só o payload.
Um pedido/resposta lógico termina sempre num frame ``END``; eventos de stream
usam ``EVENT``. O payload de um frame ``DATA`` pode ir comprimido (zlib ou
zstd, quando disponível).

O detector de modo é o primeiro byte: ``{`` continua a ser uma linha JSON;
``V`` (de ``VRMD``) entra no modo binário.
"""

from __future__ import annotations

import io
import json
import socket
import struct
import zlib
from dataclasses import dataclass
from typing import Any

MAGIC = b"VRMD"
VERSION = 1

# Tipos de frame.
TYPE_JSON = 1
TYPE_DATA = 2
TYPE_END = 3
TYPE_ERROR = 4
TYPE_EVENT = 5

# Flags de codificação do payload.
FLAG_RAW = 0
FLAG_ZLIB = 1
FLAG_ZSTD = 2

HEADER = struct.Struct("!4sBBBBII")
HEADER_SIZE = HEADER.size  # 16

# Um frame DATA de 8 MiB é o compromisso entre syscalls e latência numa LAN
# gigabit; o cliente pode negociar menor, mas o servidor recusa acima disto
# para um cliente buggy não alocar memória sem limite.
MAX_DATA_BYTES = 16 * 1024 * 1024
# JSON de controlo é pequeno; mantém a folga do protocolo antigo.
MAX_JSON_BYTES = 8 * 1024 * 1024

# zlib é stdlib; zstd é opcional (extra ``vramd[zstd]``). A compressão
# automática só vale a pena acima deste tamanho e se poupar ≥5%.
_COMPRESS_MIN_BYTES = 1 * 1024 * 1024
_COMPRESS_RATIO = 0.95


class FrameError(Exception):
    """Frame inválido, truncado ou acima dos limites do protocolo."""


@dataclass(frozen=True)
class Frame:
    type: int
    flags: int
    request_id: int
    payload: bytes


def zstd_available() -> bool:
    """True se o extra ``zstandard`` estiver importável."""
    try:
        import zstandard  # noqa: F401
    except ImportError:
        return False
    return True


def encode_payload(
    data: bytes,
    *,
    codec: str = "auto",
    min_bytes: int = _COMPRESS_MIN_BYTES,
    ratio: float = _COMPRESS_RATIO,
) -> tuple[int, bytes]:
    """Devolve ``(flags, payload)``; só comprime se compensar.

    ``codec``: ``auto`` (zlib, se poupar), ``zlib``, ``zstd`` ou ``none``.
    Com ``auto`` e dados pequenos, devolve raw sem tocar no compressor.
    """
    if codec in ("none", "raw", None) or len(data) < min_bytes:
        return FLAG_RAW, data
    if codec == "auto":
        # Sonda barata nos primeiros 64 KiB: GLB/texturas já vêm comprimidos e
        # comprimir 8 MiB de ruído só gasta CPU (o benchmark local mostrava
        # 39 MiB/s em vez de ~60 por causa disto).
        probe = data[: 64 * 1024]
        if len(zlib.compress(probe, 1)) >= len(probe) * ratio:
            return FLAG_RAW, data
        codec = "zlib"
    if codec == "zlib":
        compressed = zlib.compress(data, 1)
        if len(compressed) < len(data) * ratio:
            return FLAG_ZLIB, compressed
        return FLAG_RAW, data
    if codec == "zstd":
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise FrameError("zstd pedido mas o extra `zstandard` não está instalado") from exc
        compressed = zstandard.ZstdCompressor(level=1).compress(data)
        if len(compressed) < len(data) * ratio:
            return FLAG_ZSTD, compressed
        return FLAG_RAW, data
    raise FrameError(f"codec desconhecido: {codec!r}")


def decode_payload(flags: int, payload: bytes, *, max_bytes: int = MAX_DATA_BYTES) -> bytes:
    """Inverte :func:`encode_payload`; levanta ``FrameError`` se inválido.

    ``max_bytes`` limita o **resultado** descomprimido. Sem isto, um cliente
    autenticado podia enviar 16 MiB de zlib a expandir para gigabytes e esgotar
    a memória do supervisor (zip bomb). Um payload válido nunca passa de
    ``MAX_DATA_BYTES`` — o transmissor parte em chunks e o header limita o
    tamanho.
    """
    if flags == FLAG_RAW:
        if len(payload) > max_bytes:
            raise FrameError(f"payload {len(payload)} excede o máximo de {max_bytes}")
        return payload
    if flags == FLAG_ZLIB:
        inflater = zlib.decompressobj()
        try:
            out = inflater.decompress(payload, max_bytes + 1)
            if len(out) > max_bytes or inflater.unconsumed_tail:
                raise FrameError(f"payload zlib expande acima de {max_bytes} bytes")
            out += inflater.flush()
        except zlib.error as exc:
            raise FrameError(f"payload zlib inválido: {exc}") from exc
        if len(out) > max_bytes:
            raise FrameError(f"payload zlib expande acima de {max_bytes} bytes")
        return out
    if flags == FLAG_ZSTD:
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise FrameError("frame zstd mas o extra `zstandard` não está instalado") from exc
        try:
            reader = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(payload))
            out = reader.read(max_bytes + 1)
        except Exception as exc:
            raise FrameError(f"payload zstd inválido: {exc}") from exc
        if len(out) > max_bytes:
            raise FrameError(f"payload zstd expande acima de {max_bytes} bytes")
        return out
    raise FrameError(f"flags de codificação desconhecidas: {flags}")


def encode_json(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def decode_json(payload: bytes) -> dict[str, Any]:
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameError(f"payload JSON inválido: {exc}") from exc
    if not isinstance(obj, dict):
        raise FrameError(f"payload JSON deve ser um objecto (recebi {type(obj).__name__})")
    return obj


def send_frame(
    sock: socket.socket,
    frame_type: int,
    payload: bytes = b"",
    *,
    request_id: int = 0,
    flags: int = FLAG_RAW,
) -> None:
    """Escreve um frame completo (header + payload) no socket."""
    if len(payload) > 0xFFFFFFFF:
        raise FrameError("payload não cabe no campo length")
    sock.sendall(HEADER.pack(MAGIC, VERSION, int(frame_type), int(flags), 0, int(request_id), len(payload)) + payload)


class FrameReader:
    """Leitor com buffer interno para sockets em modo binário ou JSON.

    ``peek(1)`` decide o modo sem consumir bytes; ``read_line`` e
    ``read_frame`` partilham o mesmo buffer, pelo que um ``hello`` em JSON
    seguido de frames na mesma ligação funciona.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf = bytearray()

    def _fill(self, need: int) -> bool:
        while len(self._buf) < need:
            chunk = self._sock.recv(65536)
            if not chunk:
                return False
            self._buf.extend(chunk)
        return True

    def peek(self, size: int = 1) -> bytes:
        if not self._fill(size):
            return b""
        return bytes(self._buf[:size])

    def read_exact(self, size: int) -> bytes | None:
        """Lê exactamente ``size`` bytes; ``None`` só em EOF limpo no início."""
        if not self._fill(size):
            if not self._buf:
                return None
            raise FrameError("EOF a meio do frame")
        data = bytes(self._buf[:size])
        del self._buf[:size]
        return data

    def read_line(self, *, max_bytes: int = MAX_JSON_BYTES) -> bytes:
        """Lê até ``\\n`` (sem o newline); ``b""`` em EOF limpo."""
        while True:
            idx = self._buf.find(b"\n")
            if idx >= 0:
                line = bytes(self._buf[:idx])
                del self._buf[: idx + 1]
                return line
            if len(self._buf) > max_bytes:
                raise FrameError(f"linha JSON excede {max_bytes} bytes")
            chunk = self._sock.recv(65536)
            if not chunk:
                if not self._buf:
                    return b""
                # Última linha sem newline: aceita (clientes antigos fecham).
                line = bytes(self._buf)
                self._buf.clear()
                return line
            self._buf.extend(chunk)

    def read_frame(self, *, max_payload: int = MAX_DATA_BYTES) -> Frame | None:
        """Lê um frame; ``None`` em EOF limpo no início."""
        header = self.read_exact(HEADER_SIZE)
        if header is None:
            return None
        magic, version, frame_type, flags, _reserved, request_id, length = HEADER.unpack(header)
        if magic != MAGIC:
            raise FrameError(f"magic inválido: {magic!r}")
        if version != VERSION:
            raise FrameError(f"versão de protocolo não suportada: {version}")
        if frame_type not in (TYPE_JSON, TYPE_DATA, TYPE_END, TYPE_ERROR, TYPE_EVENT):
            raise FrameError(f"tipo de frame desconhecido: {frame_type}")
        if length > max_payload:
            raise FrameError(f"frame de {length} bytes excede o máximo de {max_payload}")
        payload = self.read_exact(length) if length else b""
        if payload is None:
            raise FrameError("EOF a meio do payload")
        return Frame(type=frame_type, flags=flags, request_id=request_id, payload=payload)
