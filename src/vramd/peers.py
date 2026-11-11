"""Peers — outros supervisores vramd para onde este pode encaminhar jobs.

O modo remoto (``VRAMD_REMOTE``) é uma escolha do *cliente*: fala com outra
máquina em vez da local. Aqui a escolha passa para o supervisor: ele conhece os
seus peers, mede a latência e a largura de banda de cada um, sabe a fila e os
modelos quentes do outro lado, e o router compara "correr cá" com "mandar para
lá" na mesma moeda (segundos até ao resultado).

Três peças:

* :class:`PeerSet` — configuração + monitor (poll de ``status``, RTT por EWMA).
* :class:`RemoteExecutor` — encaminha um job: envia os ficheiros de input para
  o sandbox do peer, corre com stream de progresso, traz os outputs de volta.
* Protecção contra loops — um job encaminhado leva ``_route.hops``; o peer
  corre-o localmente e nunca o reencaminha.

Segurança: o token de um peer é credencial de controlo total sobre ele, e os
jobs (prompts, ficheiros) viajam em claro. Só configurar peers numa rede de
confiança ou sobre WireGuard — ver ``SECURITY.md``.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import socket
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vramd.logging import Logger

from . import host, transport
from . import protocol as P
from .router import DEFAULT_BANDWIDTH_MIB_S, PeerView

_logger = Logger()

ENV_PEERS = "VRAMD_PEERS"
ENV_PEERS_TOKEN = "VRAMD_PEERS_TOKEN"
ENV_PEERS_FILE = "VRAMD_PEERS_FILE"
ERR_PEER_UNREACHABLE = "PEER_UNREACHABLE"

_MIB = 1024 * 1024
_CHUNK_BYTES = _MIB
_EWMA_ALPHA = 0.3
_MAX_LINE_BYTES = 64 * _MIB


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


POLL_INTERVAL_SEC = _env_float("VRAMD_PEER_POLL_SEC", 10.0)
FORWARD_TIMEOUT_SEC = _env_float("VRAMD_FORWARD_TIMEOUT_SEC", 3600.0)
# Um peer que falhou fica fora das rotas este tempo antes de voltar a ser tentado.
PEER_COOLDOWN_SEC = _env_float("VRAMD_PEER_COOLDOWN_SEC", 30.0)


@dataclass(frozen=True)
class PeerSpec:
    """Como chegar a um peer e o que se sabe dele à partida."""

    name: str
    host: str
    port: int
    token: str | None = None
    bandwidth_mib_s: float | None = None
    max_inflight: int = 1

    def endpoint(self) -> transport.EndpointInfo:
        return transport.EndpointInfo(
            transport=transport.TRANSPORT_TCP, host=self.host, port=self.port, token=self.token
        )

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


def _parse_peer_entry(entry: Mapping[str, Any], *, default_token: str | None, index: int) -> PeerSpec:
    address = str(entry.get("address") or "").strip()
    if not address:
        raise ValueError(f"peers[{index}]: falta 'address' (HOST:PORT)")
    try:
        host_text, port = transport.parse_host_port(address)
    except ValueError as exc:
        raise ValueError(f"peers[{index}]: {exc}") from exc
    token = entry.get("token")
    token_file = entry.get("token_file")
    if token is None and token_file:
        path = Path(os.path.expanduser(str(token_file)))
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"peers[{index}]: não consegui ler token_file {path}: {exc}") from exc
    bandwidth = entry.get("bandwidth_mib_s")
    try:
        bandwidth_f = float(bandwidth) if bandwidth is not None else None
        max_inflight = int(entry.get("max_inflight", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"peers[{index}]: bandwidth_mib_s/max_inflight inválido") from exc
    if bandwidth_f is not None and bandwidth_f <= 0:
        raise ValueError(f"peers[{index}]: bandwidth_mib_s deve ser > 0")
    return PeerSpec(
        name=str(entry.get("name") or host_text),
        host=host_text,
        port=port,
        token=str(token) if token else default_token,
        bandwidth_mib_s=bandwidth_f,
        max_inflight=max(1, max_inflight),
    )


def parse_peers_doc(doc: Any, *, default_token: str | None = None) -> list[PeerSpec]:
    """Valida o documento ``peers.yaml``.

    Raises:
        ValueError: estrutura inválida ou nomes repetidos.
    """
    if doc is None:
        return []
    if not isinstance(doc, Mapping):
        raise ValueError("peers.yaml: documento deve ser um mapa")
    entries = doc.get("peers")
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError("peers.yaml: 'peers' deve ser uma lista")
    specs: list[PeerSpec] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"peers[{index}] deve ser um mapa")
        specs.append(_parse_peer_entry(entry, default_token=default_token, index=index))
    names = [s.name for s in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"peers.yaml: nomes repetidos: {', '.join(dupes)}")
    return specs


def default_peers_path() -> Path:
    override = os.environ.get(ENV_PEERS_FILE, "").strip()
    return Path(os.path.expanduser(override)) if override else host.config_dir() / "peers.yaml"


def load_peers(path: Path | str | None = None) -> list[PeerSpec]:
    """Peers de ``VRAMD_PEERS`` (``nome=host:porta,…``) e de ``peers.yaml``.

    O ficheiro manda quando os dois declaram o mesmo nome. Um ``peers.yaml``
    malformado levanta ``ValueError`` — encaminhar jobs para o sítio errado em
    silêncio é pior que não arrancar.
    """
    default_token = (
        os.environ.get(ENV_PEERS_TOKEN, "").strip() or os.environ.get("VRAMD_REMOTE_TOKEN", "").strip() or None
    )
    specs: dict[str, PeerSpec] = {}
    for item in os.environ.get(ENV_PEERS, "").split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, address = item.partition("=")
        if not sep:
            name, address = "", item
        try:
            host_text, port = transport.parse_host_port(address)
        except ValueError as exc:
            _logger.warn(f"[vramd] {ENV_PEERS}: {exc} — a ignorar peer.")
            continue
        spec = PeerSpec(name=name.strip() or host_text, host=host_text, port=port, token=default_token)
        specs[spec.name] = spec

    target = Path(path) if path is not None else default_peers_path()
    if target.is_file():
        import yaml

        try:
            doc = yaml.safe_load(target.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"peers.yaml: YAML inválido: {exc}") from exc
        for spec in parse_peers_doc(doc, default_token=default_token):
            specs[spec.name] = spec
    return list(specs.values())


# ---------------------------------------------------------------------------
# Cliente de peer (uma ligação por pedido, como o protocolo local)
# ---------------------------------------------------------------------------


class PeerUnreachable(Exception):
    """Falha de transporte a falar com um peer (não é erro do job)."""


def _encode(request: Mapping[str, Any], token: str | None) -> bytes:
    return (json.dumps(transport.with_auth(dict(request), token)) + "\n").encode("utf-8")


def peer_request(spec: PeerSpec, request: Mapping[str, Any], *, timeout_sec: float = 30.0) -> dict[str, Any]:
    """Pedido de uma linha a um peer.

    Raises:
        PeerUnreachable: ligação recusada/caída, timeout ou resposta inválida.
    """
    try:
        sock, token = transport.connect_remote(spec.endpoint(), timeout=timeout_sec)
    except OSError as exc:
        raise PeerUnreachable(f"{spec.name} ({spec.address}): {exc}") from exc
    try:
        with sock:
            sock.sendall(_encode(request, token))
            data = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                data += chunk
                if len(data) > _MAX_LINE_BYTES:
                    raise PeerUnreachable(f"{spec.name}: resposta demasiado grande")
    except OSError as exc:
        raise PeerUnreachable(f"{spec.name} ({spec.address}): {exc}") from exc
    lines = [ln for ln in data.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    if not lines:
        raise PeerUnreachable(f"{spec.name}: ligação fechada sem resposta")
    try:
        resp = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise PeerUnreachable(f"{spec.name}: resposta não é JSON") from exc
    if not isinstance(resp, dict):
        raise PeerUnreachable(f"{spec.name}: resposta não é um objecto")
    return resp


def peer_stream(
    spec: PeerSpec,
    request: Mapping[str, Any],
    *,
    timeout_sec: float,
    tick: Callable[[], None] | None = None,
    tick_sec: float = 1.0,
) -> Iterator[dict[str, Any]]:
    """Pedido com stream NDJSON; ``tick`` corre entre leituras (cancelamento).

    Raises:
        PeerUnreachable: falha de transporte ou ``timeout_sec`` sem resultado.
    """
    try:
        sock, token = transport.connect_remote(spec.endpoint(), timeout=30.0)
    except OSError as exc:
        raise PeerUnreachable(f"{spec.name} ({spec.address}): {exc}") from exc
    deadline = time.monotonic() + timeout_sec
    buf = b""
    with sock:
        try:
            sock.sendall(_encode(request, token))
            sock.settimeout(tick_sec)
            while True:
                if time.monotonic() >= deadline:
                    raise PeerUnreachable(f"{spec.name}: timeout de {timeout_sec:.0f}s à espera do resultado")
                try:
                    chunk = sock.recv(65536)
                except TimeoutError:
                    if tick is not None:
                        tick()
                    continue
                if not chunk:
                    return
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text:
                        continue
                    with contextlib.suppress(json.JSONDecodeError):
                        obj = json.loads(text)
                        if isinstance(obj, dict):
                            yield obj
                if tick is not None:
                    tick()
        except OSError as exc:
            raise PeerUnreachable(f"{spec.name} ({spec.address}): {exc}") from exc


# ---------------------------------------------------------------------------
# Estado e monitor
# ---------------------------------------------------------------------------


@dataclass
class PeerState:
    """O que o supervisor sabe de um peer neste momento."""

    spec: PeerSpec
    healthy: bool = False
    last_error: str | None = "ainda sem poll"
    last_ok_at: float | None = None
    cooldown_until: float = 0.0
    rtt_sec: float | None = None
    bandwidth_mib_s: float | None = None
    eta_sec: float = 0.0
    queue_depth: int = 0
    inflight: int = 0
    backends: dict[str, dict[str, Any]] = field(default_factory=dict)
    estimates: dict[str, dict[str, Any]] = field(default_factory=dict)
    gpu_totals_mib: list[int] = field(default_factory=list)
    version: str | None = None
    forwards: int = 0
    forwarded_total: int = 0
    failed_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "address": self.spec.address,
            "healthy": self.healthy,
            "last_error": self.last_error,
            "rtt_ms": round(self.rtt_sec * 1000.0, 1) if self.rtt_sec is not None else None,
            "bandwidth_mib_s": round(self.effective_bandwidth(), 1),
            "bandwidth_measured": self.bandwidth_mib_s is not None,
            "eta_sec": self.eta_sec,
            "queue_depth": self.queue_depth,
            "inflight": self.inflight,
            "backends": sorted(self.backends),
            "loaded": sorted(n for n, row in self.backends.items() if row.get("loaded")),
            "gpu_totals_mib": list(self.gpu_totals_mib),
            "version": self.version,
            "forwards": self.forwards,
            "max_inflight": self.spec.max_inflight,
            "forwarded_total": self.forwarded_total,
            "failed_total": self.failed_total,
        }

    def effective_bandwidth(self) -> float:
        """Medida > declarada no ``peers.yaml`` > default."""
        if self.bandwidth_mib_s is not None:
            return self.bandwidth_mib_s
        if self.spec.bandwidth_mib_s is not None:
            return self.spec.bandwidth_mib_s
        return DEFAULT_BANDWIDTH_MIB_S


def _ewma(previous: float | None, sample: float) -> float:
    return sample if previous is None else (1.0 - _EWMA_ALPHA) * previous + _EWMA_ALPHA * sample


class PeerSet:
    """Peers configurados, o seu estado e a thread que o mantém fresco.

    Args:
        specs: Peers a monitorizar.
        requester: Função de pedido (injectável nos testes).
        clock: Relógio monotónico.
    """

    def __init__(
        self,
        specs: list[PeerSpec] | None = None,
        *,
        requester: Callable[..., dict[str, Any]] = peer_request,
        clock: Callable[[], float] = time.monotonic,
        poll_interval_sec: float = POLL_INTERVAL_SEC,
    ) -> None:
        self._states: dict[str, PeerState] = {s.name: PeerState(spec=s) for s in (specs or [])}
        self._requester = requester
        self._clock = clock
        self._poll_interval = max(1.0, float(poll_interval_sec))
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def __len__(self) -> int:
        return len(self._states)

    def __bool__(self) -> bool:
        return bool(self._states)

    def get(self, name: str) -> PeerState | None:
        return self._states.get(name)

    def states(self) -> list[PeerState]:
        return list(self._states.values())

    @property
    def forward_slots(self) -> int:
        return sum(s.spec.max_inflight for s in self._states.values())

    # -- monitor ------------------------------------------------------------

    def start(self) -> None:
        if not self._states or (self._thread is not None and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="vramd-peers", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self._poll_interval)

    def poll_once(self) -> None:
        """Um ``status`` a cada peer; actualiza saúde, fila, modelos e RTT."""
        for state in self.states():
            started = self._clock()
            try:
                resp = self._requester(state.spec, {"cmd": P.CMD_STATUS}, timeout_sec=5.0)
            except PeerUnreachable as exc:
                self.mark_failed(state.spec.name, str(exc), cooldown=False)
                continue
            except Exception as exc:
                self.mark_failed(state.spec.name, f"{type(exc).__name__}: {exc}", cooldown=False)
                continue
            elapsed = max(0.0, self._clock() - started)
            if resp.get("error_code") == P.ERR_UNAUTHORIZED:
                self.mark_failed(state.spec.name, "token recusado pelo peer", cooldown=False)
                continue
            self._apply_status(state, resp, rtt_sec=elapsed)

    def _apply_status(self, state: PeerState, resp: Mapping[str, Any], *, rtt_sec: float) -> None:
        queue = resp.get("queue") or {}
        backends: dict[str, dict[str, Any]] = {}
        for row in resp.get("backends") or []:
            if isinstance(row, Mapping) and row.get("name"):
                backends[str(row["name"])] = {"loaded": bool(row.get("loaded")), "peak_mib": row.get("peak_mib")}
        routing = resp.get("routing") or {}
        totals: list[int] = []
        for gpu in routing.get("gpus") or []:
            if isinstance(gpu, Mapping) and isinstance(gpu.get("total_mib"), (int, float)):
                totals.append(int(gpu["total_mib"]))
        with self._lock:
            state.healthy = True
            state.last_error = None
            state.last_ok_at = self._clock()
            state.rtt_sec = _ewma(state.rtt_sec, rtt_sec)
            eta = resp.get("eta_sec")
            state.eta_sec = float(eta) if isinstance(eta, (int, float)) else 0.0
            state.queue_depth = int(queue.get("queue_depth") or 0)
            state.inflight = int(queue.get("inflight") or 0)
            state.backends = backends
            estimates = routing.get("estimates")
            state.estimates = dict(estimates) if isinstance(estimates, Mapping) else {}
            state.gpu_totals_mib = totals
            state.version = str(routing["version"]) if routing.get("version") else None

    def mark_failed(self, name: str, error: str, *, cooldown: bool = True) -> None:
        """Peer falhou: sai das rotas (com cooldown se falhou a meio de um job)."""
        state = self._states.get(name)
        if state is None:
            return
        with self._lock:
            if state.healthy:
                _logger.warn(f"[vramd] peer {name}: {error} — fora das rotas.")
            state.healthy = False
            state.last_error = error
            if cooldown:
                state.cooldown_until = self._clock() + PEER_COOLDOWN_SEC
                state.failed_total += 1

    def record_bandwidth(self, name: str, mib: float, seconds: float) -> None:
        """Alimenta a largura de banda medida com uma transferência real."""
        state = self._states.get(name)
        # Transferências minúsculas medem latência, não largura de banda.
        if state is None or mib < 1.0 or seconds <= 0:
            return
        with self._lock:
            state.bandwidth_mib_s = _ewma(state.bandwidth_mib_s, mib / seconds)

    # -- reserva ------------------------------------------------------------

    def reserve(self, name: str) -> bool:
        state = self._states.get(name)
        if state is None:
            return False
        with self._lock:
            if state.forwards >= state.spec.max_inflight:
                return False
            state.forwards += 1
            return True

    def release(self, name: str, *, ok: bool = True) -> None:
        state = self._states.get(name)
        if state is None:
            return
        with self._lock:
            state.forwards = max(0, state.forwards - 1)
            if ok:
                state.forwarded_total += 1

    # -- fotografia para o router --------------------------------------------

    def views(
        self,
        backend: str,
        peak_mib: int,
        *,
        extra_wait: Mapping[str, float] | None = None,
        exclude: frozenset[str] | set[str] = frozenset(),
    ) -> list[PeerView]:
        """Estado dos peers do ponto de vista de um job de ``backend``."""
        now = self._clock()
        out: list[PeerView] = []
        for state in self.states():
            name = state.spec.name
            with self._lock:
                row = state.backends.get(backend)
                est = state.estimates.get(backend) or {}
                totals = list(state.gpu_totals_mib)
                healthy = state.healthy and now >= state.cooldown_until and name not in exclude
                reason = state.last_error or ""
                if name in exclude:
                    reason = "já falhou para este job"
                elif state.healthy and now < state.cooldown_until:
                    reason = f"em cooldown após falha ({state.cooldown_until - now:.0f}s)"
                fits: bool | None = None
                if totals:
                    # Quem admite o job é o peer, com o descriptor DELE (pode estar
                    # calibrado ou quantizado de outra forma): vale o menor dos dois
                    # picos — o nosso para este pedido e o que ele declara.
                    peer_peak = row.get("peak_mib") if row else None
                    need = min(peak_mib, int(peer_peak)) if isinstance(peer_peak, (int, float)) else peak_mib
                    # Mais que uma GPU do lado de lá: ele pode dividir — não afirmar "não cabe".
                    fits = True if need <= max(totals) else (None if len(totals) > 1 else False)
                wait = state.eta_sec + float((extra_wait or {}).get(name, 0.0))
                gen = est.get("gen_sec")
                if state.inflight > 0 and state.eta_sec <= 0:
                    # Peer antigo/sem histórico com um job em curso: não é "livre já".
                    wait += float(gen) if isinstance(gen, (int, float)) else 30.0
                out.append(
                    PeerView(
                        name=name,
                        healthy=healthy,
                        has_backend=row is not None,
                        hot=bool(row and row.get("loaded")),
                        fits=fits,
                        rtt_sec=state.rtt_sec or 0.0,
                        bandwidth_mib_s=state.effective_bandwidth(),
                        wait_sec=wait,
                        slot_free=state.forwards < state.spec.max_inflight,
                        load_sec=float(est["load_sec"]) if isinstance(est.get("load_sec"), (int, float)) else None,
                        gen_sec=float(gen) if isinstance(gen, (int, float)) else None,
                        reason=reason,
                    )
                )
        return out

    def status(self) -> list[dict[str, Any]]:
        with self._lock:
            return [s.to_dict() for s in self._states.values()]


# ---------------------------------------------------------------------------
# Encaminhamento de um job
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_MIB):
            digest.update(block)
    return digest.hexdigest()


class RemoteExecutor:
    """Corre um job num peer e devolve o resultado como se fosse local.

    Args:
        peers: O :class:`PeerSet` (para registar largura de banda medida).
        node_name: Nome deste nó (vai no ``_route.origin`` para diagnóstico).
        requester: Pedido de uma linha (injectável).
        streamer: Pedido com stream (injectável).
    """

    def __init__(
        self,
        peers: PeerSet,
        *,
        node_name: str | None = None,
        requester: Callable[..., dict[str, Any]] = peer_request,
        streamer: Callable[..., Iterator[dict[str, Any]]] = peer_stream,
    ) -> None:
        self._peers = peers
        self._node = node_name or socket.gethostname()
        self._request = requester
        self._stream = streamer

    def _checked(self, spec: PeerSpec, request: Mapping[str, Any], *, timeout_sec: float = 60.0) -> dict[str, Any]:
        resp = self._request(spec, request, timeout_sec=timeout_sec)
        if resp.get("status") != P.STATUS_OK:
            raise PeerUnreachable(f"{spec.name}: {request.get('cmd')} recusado — {resp.get('error', resp)}")
        return resp

    def _upload(self, spec: PeerSpec, sandbox: str, local: Path, remote_rel: str) -> None:
        size = local.stat().st_size
        digest = _sha256_file(local)
        started = time.monotonic()
        offset = 0
        with local.open("rb") as handle:
            while True:
                block = handle.read(_CHUNK_BYTES)
                eof = offset + len(block) >= size
                request: dict[str, Any] = {
                    "cmd": P.CMD_PUT,
                    "job": sandbox,
                    "path": remote_rel,
                    "offset": offset,
                    "data": base64.b64encode(block).decode("ascii"),
                    "eof": eof,
                }
                if eof:
                    request["sha256"] = digest
                self._checked(spec, request)
                offset += len(block)
                if eof:
                    break
        self._peers.record_bandwidth(spec.name, size / _MIB, time.monotonic() - started)

    def _download(self, spec: PeerSpec, sandbox: str, remote_rel: str, local: Path) -> int:
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(f"{local.name}.{os.getpid()}.part")
        started = time.monotonic()
        offset = 0
        try:
            with tmp.open("wb") as handle:
                while True:
                    resp = self._checked(
                        spec,
                        {
                            "cmd": P.CMD_GET,
                            "job": sandbox,
                            "path": remote_rel,
                            "offset": offset,
                            "length": _CHUNK_BYTES,
                        },
                    )
                    block = base64.b64decode(str(resp.get("data") or ""))
                    handle.write(block)
                    offset += len(block)
                    if resp.get("eof") or not block:
                        break
            os.replace(tmp, local)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise
        self._peers.record_bandwidth(spec.name, offset / _MIB, time.monotonic() - started)
        return offset

    def run(
        self,
        peer: PeerState,
        backend: str,
        request: Mapping[str, Any],
        *,
        input_keys: tuple[str, ...] = (),
        output_keys: tuple[str, ...] = (),
        on_progress: Callable[[float | None, str | None], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        timeout_sec: float = FORWARD_TIMEOUT_SEC,
    ) -> dict[str, Any]:
        """Encaminha o job e devolve o resultado (com paths locais repostos).

        Raises:
            PeerUnreachable: o peer falhou **antes** de aceitar o job (seguro
                reencaminhar/correr localmente). Falhas depois de aceite voltam
                como resultado de erro — o job pode ter corrido.
        """
        spec = peer.spec
        payload = {k: v for k, v in request.items() if not str(k).startswith("_") and k not in ("cmd", "stream")}
        sandbox: str | None = None
        outputs: dict[str, tuple[str, Path]] = {}

        def progress(pct: float | None, msg: str | None) -> None:
            if on_progress is not None:
                on_progress(pct, msg)

        uploads = [(k, Path(str(payload[k])).expanduser()) for k in input_keys if isinstance(payload.get(k), str)]
        uploads = [(k, p) for k, p in uploads if p.is_file()]
        wanted_outputs = [k for k in output_keys if isinstance(payload.get(k), str) and payload[k]]
        if uploads or wanted_outputs:
            staged = self._checked(spec, {"cmd": P.CMD_STAGE})
            sandbox = str(staged["job"])
            root = str(staged.get("root") or "").replace("\\", "/")
            for index, (key, local) in enumerate(uploads):
                rel = f"in/{index}_{local.name}"
                progress(None, f"→ {spec.name}: a enviar {local.name}")
                self._upload(spec, sandbox, local, rel)
                payload[key] = f"{root}/{rel}"
            for index, key in enumerate(wanted_outputs):
                local = Path(str(payload[key])).expanduser()
                rel = f"out/{index}_{local.name}"
                outputs[key] = (rel, local)
                payload[key] = f"{root}/{rel}"

        remote_job: dict[str, str] = {}
        abort_sent = False

        def tick() -> None:
            nonlocal abort_sent
            if abort_sent or should_abort is None or not should_abort() or "id" not in remote_job:
                return
            abort_sent = True
            with contextlib.suppress(Exception):
                self._request(spec, {"cmd": P.CMD_CANCEL, "job_id": remote_job["id"]}, timeout_sec=10.0)

        forward = {
            **payload,
            "cmd": P.CMD_GENERATE,
            "backend": backend,
            "stream": True,
            "timeout_sec": timeout_sec,
            "_route": {"hops": 1, "origin": self._node},
        }
        final: dict[str, Any] | None = None
        accepted = False
        try:
            for event in self._stream(spec, forward, timeout_sec=timeout_sec + 30.0, tick=tick):
                kind = event.get("event")
                if kind is None:
                    final = event
                    continue
                accepted = True
                if event.get("job_id"):
                    remote_job["id"] = str(event["job_id"])
                if kind == P.EVENT_QUEUED:
                    progress(None, f"→ {spec.name}: na fila (posição {event.get('queue_position', '?')})")
                elif kind == P.EVENT_PROGRESS:
                    progress(event.get("pct"), f"→ {spec.name}: {event.get('msg') or ''}".rstrip())
                elif kind == P.EVENT_STARTED:
                    progress(None, f"→ {spec.name}: a correr")
        except PeerUnreachable as exc:
            if not accepted:
                self._cleanup(spec, sandbox)
                raise
            final = {
                "status": P.STATUS_ERROR,
                "error": f"ligação ao peer perdida a meio do job: {exc}",
                "error_code": P.ERR_GENERATE_FAILED,
                "hint": f"O job pode ter terminado em {spec.name} — vê `vramd history` nesse nó antes de repetir.",
            }
        if final is None:
            if not accepted:
                self._cleanup(spec, sandbox)
                raise PeerUnreachable(f"{spec.name}: stream terminou sem aceitar o job")
            final = {
                "status": P.STATUS_ERROR,
                "error": "stream do peer terminou sem resultado",
                "error_code": P.ERR_GENERATE_FAILED,
            }

        result = dict(final)
        result.pop("ums_debug", None)
        if result.get("status") == P.STATUS_OK and sandbox is not None:
            try:
                for key, (rel, local) in outputs.items():
                    progress(None, f"← {spec.name}: a receber {local.name}")
                    self._download(spec, sandbox, rel, local)
                    remote_path = str(payload[key])
                    # O adapter costuma devolver o path que lhe demos: repor o local.
                    for field_name, value in list(result.items()):
                        if isinstance(value, str) and value.replace("\\", "/") == remote_path:
                            result[field_name] = str(local)
            except PeerUnreachable as exc:
                result = {
                    "status": P.STATUS_ERROR,
                    "error": f"job concluído em {spec.name} mas o output não chegou: {exc}",
                    "error_code": P.ERR_GENERATE_FAILED,
                    "hint": "O sandbox fica no peer até ao TTL — `vramd` `get` recupera-o.",
                }
                sandbox = None  # não limpar: é a única cópia do output
        self._cleanup(spec, sandbox)
        result["route"] = {"kind": "remote", "target": f"peer:{spec.name}", "remote_job_id": remote_job.get("id")}
        return result

    def _cleanup(self, spec: PeerSpec, sandbox: str | None) -> None:
        if sandbox is None:
            return
        with contextlib.suppress(Exception):
            self._request(spec, {"cmd": P.CMD_CLEAN, "job": sandbox}, timeout_sec=15.0)
