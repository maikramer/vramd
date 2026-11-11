"""RouteEngine — junta o estado real (lanes, peers, estatísticas) e chama o router.

O :mod:`vramd.router` é puro: recebe fotografias e devolve uma decisão. Este
módulo tira as fotografias — pico de admissão do gestor, tempos médios de load
e geração, tamanho dos ficheiros do pedido, estado das GPUs e dos peers — e é
o único sítio que conhece o bloco ``placement:`` dos descritores.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from .lanes import LaneSet
from .router import (
    DEFAULT_GEN_SEC,
    DEFAULT_LOAD_SEC,
    DEFAULT_SPLIT_OVERHEAD,
    DEFAULT_SPLIT_SLOWDOWN,
    Decision,
    JobProfile,
    RoutePolicy,
    plan_routes,
)

_MIB = 1024 * 1024
_HISTORY_TTL_SEC = 60.0

PLACEMENT_KEYS = frozenset(
    {
        "split",
        "split_overhead",
        "split_slowdown",
        "max_replicas",
        "remote",
        "input_keys",
        "output_keys",
        "load_sec",
        "gen_sec",
        "output_mib",
    }
)


def _positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def _str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(v) for v in value)
    return ()


def parse_pinned_devices(request: Mapping[str, Any] | None) -> tuple[int, ...] | None:
    """``gpu_ids`` explícitos do pedido (lista ou CSV) → tuplo ordenado, ou ``None``."""
    raw = (request or {}).get("gpu_ids")
    if raw is None or raw == "" or raw == []:
        return None
    try:
        if isinstance(raw, str):
            ids = [int(x) for x in raw.replace(";", ",").split(",") if x.strip()]
        elif isinstance(raw, (list, tuple)):
            ids = [int(x) for x in raw]
        else:
            ids = [int(raw)]
    except (TypeError, ValueError):
        return None
    return tuple(sorted(set(ids))) or None


class RouteEngine:
    """Decide a rota de um job a partir do estado real do supervisor.

    Args:
        registry: Registry de backends.
        lanes: As GPUs locais.
        peers: Os outros supervisores (``None``/vazio = sem rede).
        policy: Desempates do router (default: env).
        history_reader: ``() -> list[entradas]`` do histórico persistente, para
            estimar a geração depois de um restart (antes de haver médias).
        clock: Relógio monotónico.
    """

    def __init__(
        self,
        registry: Any,
        lanes: LaneSet,
        peers: Any = None,
        *,
        policy: RoutePolicy | None = None,
        history_reader: Callable[[], list[dict[str, Any]]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.registry = registry
        self.lanes = lanes
        self.peers = peers
        self.policy = policy if policy is not None else RoutePolicy.from_env()
        self._history_reader = history_reader
        self._clock = clock
        self._history_cache: tuple[float, dict[str, float]] | None = None
        self._lock = threading.Lock()
        # Contadores para status/métricas: quantas vezes cada tipo de rota ganhou.
        self.counts: dict[str, int] = {"local": 0, "replica": 0, "split": 0, "remote": 0, "wait": 0, "reject": 0}

    @property
    def active(self) -> bool:
        """True se há alguma decisão a tomar (várias GPUs, lane fixada ou peers)."""
        return self.lanes.routed or bool(self.peers)

    # -- estimativas --------------------------------------------------------

    def _placement(self, backend: str) -> Mapping[str, Any]:
        try:
            block = getattr(self.registry.descriptor(backend), "placement", None)
        except Exception:
            return {}
        return block if isinstance(block, Mapping) else {}

    def _history_gen(self) -> dict[str, float]:
        if self._history_reader is None:
            return {}
        now = self._clock()
        with self._lock:
            cached = self._history_cache
            if cached is not None and (now - cached[0]) < _HISTORY_TTL_SEC:
                return cached[1]
        sums: dict[str, list[float]] = {}
        with contextlib.suppress(Exception):
            for entry in self._history_reader():
                gen = entry.get("generate_sec")
                if entry.get("state") == "done" and isinstance(gen, (int, float)) and gen > 0:
                    sums.setdefault(str(entry.get("backend")), []).append(float(gen))
        averages = {name: sum(values) / len(values) for name, values in sums.items()}
        with self._lock:
            self._history_cache = (now, averages)
        return averages

    def _stat(self, backend: str, attr: str, count_attr: str) -> float | None:
        """Média medida (load/generate) na lane que mais vezes a mediu."""
        best: tuple[int, float] | None = None
        for lane in self.lanes.all_lanes():
            stats = getattr(lane.manager, "stats", None)
            row = stats.get(backend) if stats is not None and hasattr(stats, "get") else None
            if row is None:
                continue
            count = int(getattr(row, count_attr, 0) or 0)
            value = float(getattr(row, attr, 0.0) or 0.0)
            if count > 0 and value > 0 and (best is None or count > best[0]):
                best = (count, value)
        return best[1] if best is not None else None

    def load_sec(self, backend: str) -> float:
        measured = self._stat(backend, "avg_load_time_sec", "load_count")
        if measured is not None:
            return measured
        return _positive(self._placement(backend).get("load_sec")) or DEFAULT_LOAD_SEC

    def gen_sec(self, backend: str) -> float:
        measured = self._stat(backend, "avg_generate_time_sec", "generate_count")
        if measured is not None:
            return measured
        historic = self._history_gen().get(backend)
        if historic:
            return historic
        return _positive(self._placement(backend).get("gen_sec")) or DEFAULT_GEN_SEC

    def estimates(self) -> dict[str, dict[str, float]]:
        """Tempos por backend — publicados no ``status`` para os peers usarem."""
        out: dict[str, dict[str, float]] = {}
        for name in getattr(self.registry, "names", []):
            out[name] = {"load_sec": round(self.load_sec(name), 2), "gen_sec": round(self.gen_sec(name), 2)}
        return out

    def _peaks(self, backend: str, request: Mapping[str, Any]) -> tuple[int, int]:
        """``(pico a frio, headroom a quente)`` pelo gestor da lane primária."""
        mgr = self.lanes.primary.manager
        req = dict(request)
        try:
            quant, mem_eff, group_off, streams = mgr.resolve_peak_params(backend, req)
            peak = int(
                mgr.peak_vram_mib(
                    backend,
                    quant_mode=quant,
                    memory_efficient=mem_eff,
                    group_offload=group_off,
                    streams_on_load=streams,
                    footprint_key=req.get("footprint_key"),
                )
            )
            headroom = int(
                mgr.activation_headroom_mib(
                    backend,
                    quant_mode=quant,
                    memory_efficient=mem_eff,
                    group_offload=group_off or streams,
                    footprint_key=req.get("footprint_key"),
                )
            )
            return peak, headroom
        except Exception:
            try:
                vram = int(self.registry.descriptor(backend).vram_mib)
            except Exception:
                vram = 0
            return vram, 0

    def profile(self, backend: str, request: Mapping[str, Any] | None = None, *, forwarded: bool = False) -> JobProfile:
        """Requisitos e estimativas de um job para o router."""
        req = dict(request or {})
        block = self._placement(backend)
        peak, headroom = self._peaks(backend, req)
        input_bytes = 0
        for key in _str_tuple(block.get("input_keys")):
            value = req.get(key)
            if isinstance(value, str) and value:
                with contextlib.suppress(OSError):
                    input_bytes += os.path.getsize(os.path.expanduser(value))
        output_mib = _positive(block.get("output_mib")) or 0.0
        pinned = parse_pinned_devices(req) if self.lanes.routed else None
        try:
            max_replicas = max(0, int(block.get("max_replicas") or 0))
        except (TypeError, ValueError):
            max_replicas = 0
        return JobProfile(
            backend=backend,
            peak_mib=peak,
            headroom_mib=headroom,
            load_sec=self.load_sec(backend),
            gen_sec=self.gen_sec(backend),
            input_bytes=input_bytes,
            output_bytes=int(output_mib * _MIB),
            # Fixar várias GPUs à mão é pedir um split, declare-o o backend ou não.
            allow_split=bool(block.get("split")) or (pinned is not None and len(pinned) > 1),
            split_overhead=_positive(block.get("split_overhead")) or DEFAULT_SPLIT_OVERHEAD,
            split_slowdown=_positive(block.get("split_slowdown")) or DEFAULT_SPLIT_SLOWDOWN,
            max_replicas=max_replicas,
            allow_remote=block.get("remote", True) is not False,
            pinned_devices=pinned,
            local_only=forwarded,
        )

    def io_keys(self, backend: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """``(input_keys, output_keys)`` — campos do request que são ficheiros."""
        block = self._placement(backend)
        return _str_tuple(block.get("input_keys")), _str_tuple(block.get("output_keys"))

    # -- decisão ------------------------------------------------------------

    def decide(
        self,
        backend: str,
        request: Mapping[str, Any] | None = None,
        *,
        forwarded: bool = False,
        lane_extra: Mapping[str, float] | None = None,
        peer_extra: Mapping[str, float] | None = None,
        exclude_peers: frozenset[str] | set[str] = frozenset(),
        record: bool = False,
    ) -> Decision:
        """Rota de um job dado o estado actual.

        Args:
            lane_extra: Segundos de trabalho já prometidos a cada lane por jobs
                à frente na fila (é o que faz o router abrir réplicas num batch).
            peer_extra: O mesmo, por peer.
            exclude_peers: Peers que já falharam para este job.
            record: Contabilizar a decisão nos contadores (só no dispatch real).
        """
        profile = self.profile(backend, request, forwarded=forwarded)
        views = [
            self.lanes.view(lane, backend, request, extra_busy_sec=float((lane_extra or {}).get(lane.lane_id, 0.0)))
            for lane in self.lanes.device_lanes
        ]
        peer_views = []
        if self.peers and not forwarded:
            peer_views = self.peers.views(backend, profile.peak_mib, extra_wait=peer_extra, exclude=exclude_peers)
        decision = plan_routes(profile, views, peer_views, policy=self.policy)
        if record:
            self._count(decision)
        return decision

    def _count(self, decision: Decision) -> None:
        key = decision.action
        if decision.action == "run" and decision.best is not None:
            key = "replica" if decision.best.replica else decision.best.kind
        with self._lock:
            self.counts[key] = self.counts.get(key, 0) + 1

    def record_run(self, decision: Decision) -> None:
        """Conta uma rota efectivamente despachada."""
        self._count(decision)

    def status(self) -> dict[str, Any]:
        with self._lock:
            counts = dict(self.counts)
        return {
            "active": self.active,
            "decisions": counts,
            "policy": {
                "replica_margin_sec": self.policy.replica_margin_sec,
                "remote_penalty_sec": self.policy.remote_penalty_sec,
                "split_penalty_sec": self.policy.split_penalty_sec,
            },
        }
