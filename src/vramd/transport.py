"""Transporte do supervisor — socket Unix no POSIX, TCP loopback no Windows.

O protocolo é o mesmo (JSON/NDJSON, uma linha por mensagem). O que muda é como
se chega ao supervisor:

* **POSIX** — o endpoint é o próprio Unix socket (``vramd.sock``), com as
  permissões do filesystem.
* **Windows** — o endpoint é um **descritor JSON** no mesmo path: porta
  efémera em ``127.0.0.1`` + token. O vramd não usa ``AF_UNIX`` no Windows por
  omissão: o caminho tem limite de 108 bytes, sockets UNIX podem não existir
  em versões antigas do SO e o TCP loopback é o que o resto do ecossistema
  Windows espera de um daemon local.

O ficheiro de descritor é escrito **depois** do ``bind`` e apagado no
``cleanup``, pelo que ``Path.exists()`` continua a significar "há endpoint".
O token é exigido pelo servidor em modo TCP (qualquer processo local poderia
ligar-se à porta; no Unix socket o dono do ficheiro já limita o acesso).

``VRAMD_TRANSPORT=unix|tcp`` força o modo — útil para testar o caminho TCP em
CI Linux, que de outra forma só correria no Windows.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import host

TRANSPORT_UNIX = "unix"
TRANSPORT_TCP = "tcp"
ENV_TRANSPORT = "VRAMD_TRANSPORT"

# Descritor TCP (o mesmo nome do socket Unix, para CLI/status/PID file).
DESCRIPTOR_VERSION = 1
DEFAULT_TCP_HOST = "127.0.0.1"


def has_unix_sockets() -> bool:
    """True se a plataforma/Python expõem ``socket.AF_UNIX``."""
    return hasattr(socket, "AF_UNIX")


def transport_mode() -> str:
    """Modo efectivo: ``VRAMD_TRANSPORT`` → TCP no Windows → Unix."""
    raw = os.environ.get(ENV_TRANSPORT, "").strip().lower()
    if raw in (TRANSPORT_UNIX, TRANSPORT_TCP):
        return raw
    if host.IS_WINDOWS:
        return TRANSPORT_TCP
    return TRANSPORT_UNIX if has_unix_sockets() else TRANSPORT_TCP


@dataclass(frozen=True)
class EndpointInfo:
    """Como chegar ao supervisor (e com que credencial)."""

    transport: str
    host: str = ""
    port: int | None = None
    token: str | None = None
    version: int = DESCRIPTOR_VERSION


def _write_atomic(path: Path, text: str) -> None:
    """Escreve o descritor sem deixar um ficheiro a meio a um cliente.

    No Windows um cliente (ou o antivírus) pode estar a ler o descritor antigo
    e impedir o ``os.replace`` durante alguns ms — tenta-se e, em último
    recurso, escreve-se no destino (o leitor valida o JSON e volta a tentar).
    """
    import time

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            break
        except OSError:
            if attempt == 7:
                path.write_text(text, encoding="utf-8")
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
                break
            time.sleep(0.02)


def write_endpoint(path: Path, *, port: int, token: str, pid: int | None = None) -> EndpointInfo:
    """Publica o descritor TCP em ``path`` (chamado após o ``bind``)."""
    info = EndpointInfo(
        transport=TRANSPORT_TCP,
        host=DEFAULT_TCP_HOST,
        port=int(port),
        token=token,
        version=DESCRIPTOR_VERSION,
    )
    payload: dict[str, Any] = {
        "version": info.version,
        "transport": info.transport,
        "host": info.host,
        "port": info.port,
        "token": info.token,
        "pid": os.getpid() if pid is None else int(pid),
    }
    _write_atomic(path, json.dumps(payload, separators=(",", ":")) + "\n")
    if not host.IS_WINDOWS:
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)
    return info


def read_endpoint(path: Path) -> EndpointInfo | None:
    """Lê o descritor TCP; ``None`` se o path não for um descritor válido."""
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("transport") != TRANSPORT_TCP:
        return None
    host_text = str(data.get("host") or DEFAULT_TCP_HOST).strip() or DEFAULT_TCP_HOST
    try:
        port = int(data.get("port"))
    except (TypeError, ValueError):
        return None
    if not (1 <= port <= 65535):
        return None
    token = data.get("token")
    if token is not None and not isinstance(token, str):
        return None
    try:
        version = int(data.get("version", DESCRIPTOR_VERSION))
    except (TypeError, ValueError):
        version = DESCRIPTOR_VERSION
    return EndpointInfo(transport=TRANSPORT_TCP, host=host_text, port=port, token=token, version=version)


def remove_endpoint(path: Path) -> None:
    """Remove o endpoint (socket Unix ou descritor TCP). Idempotente."""
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def set_nodelay(sock: socket.socket) -> None:
    """Desliga Nagle: o protocolo é request/response de linhas curtas."""
    with contextlib.suppress(OSError):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def listen(path: Path) -> tuple[socket.socket, EndpointInfo]:
    """Cria o socket à escuta e devolve ``(socket, info)``.

    Em modo TCP a porta é efémera (``bind`` a ``127.0.0.1:0``) e o descritor
    fica escrito em ``path`` antes do regresso — quando o cliente vê o
    ficheiro, já consegue ligar-se.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = transport_mode()
    if mode == TRANSPORT_UNIX and has_unix_sockets():
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(1.0)
            sock.bind(str(path))
            sock.listen(128)
        except OSError:
            sock.close()
            raise
        return sock, EndpointInfo(transport=TRANSPORT_UNIX)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # SO_EXCLUSIVEADDRUSE (Windows) impede outro processo de raptar a porta
        # enquanto o supervisor corre. Em POSIX não existe; SO_REUSEADDR não é
        # necessário porque a porta é efémera e não a reutilizamos.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind((DEFAULT_TCP_HOST, 0))
        sock.listen(128)
        sock.settimeout(1.0)
        port = int(sock.getsockname()[1])
        token = secrets.token_urlsafe(32)
        info = write_endpoint(path, port=port, token=token)
    except OSError:
        sock.close()
        raise
    return sock, info


def _connect_unix(path: Path, timeout: float | None) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        if timeout is not None:
            sock.settimeout(timeout)
        sock.connect(str(path))
        return sock
    except BaseException:
        sock.close()
        raise


def connect(path: Path, *, timeout: float | None = None) -> tuple[socket.socket, str | None]:
    """Liga ao endpoint; devolve ``(socket, token)``.

    A detecção é pelo **conteúdo**: descritor JSON ⇒ TCP+token; caso contrário
    ⇒ Unix socket quando disponível. Isto permite que um cliente fale com um
    supervisor TCP mesmo sem ``VRAMD_TRANSPORT=tcp`` no seu ambiente (e cobre
    um servidor per-tool legacy que ainda use AF_UNIX).

    Raises:
        OSError, FileNotFoundError: endpoint inexistente/inacessível.
    """
    info = read_endpoint(path)
    if info is not None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if timeout is not None:
                sock.settimeout(timeout)
            sock.connect((info.host, int(info.port or 0)))
            set_nodelay(sock)
            return sock, info.token
        except BaseException:
            sock.close()
            raise

    if has_unix_sockets():
        return _connect_unix(path, timeout), None
    raise FileNotFoundError(f"endpoint inválido: {path}")


def endpoint_label(path: Path) -> str:
    """Representação humana do endpoint para ``status``/logs."""
    info = read_endpoint(path)
    if info is not None:
        return f"tcp://{info.host}:{info.port}"
    return str(path)


def describe_transport(path: Path) -> str:
    """``unix``/``tcp`` para o bloco de status (sem ler o descritor à força)."""
    if transport_mode() == TRANSPORT_UNIX:
        return TRANSPORT_UNIX
    info = read_endpoint(path)
    return info.transport if info is not None else TRANSPORT_TCP


def with_auth(request: dict[str, Any], token: str | None) -> dict[str, Any]:
    """Cópia do request com o token do endpoint (modo TCP)."""
    if not token:
        return request
    return {**request, "auth": token}


def is_authorized(request: dict[str, Any], token: str | None) -> bool:
    """True se o request traz o token correcto (ou não há token — POSIX)."""
    if not token:
        return True
    supplied = request.get("auth")
    if not isinstance(supplied, str):
        return False
    return secrets.compare_digest(supplied, token)
