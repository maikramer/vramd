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
from dataclasses import dataclass, replace
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
    Tier,
    normalize_criticality,
    plan_routes,
    select_tier,
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
        "cost_keys",
    }
)


# Campos do pedido que só o router lê. Nunca chegam ao adapter: um gerador que
# faça ``**request`` rebentaria com uma kwarg desconhecida.
ROUTING_KEYS = ("hw_tiers", "criticality")
_EWMA_ALPHA = 0.3


@dataclass(frozen=True)
class HwTier:
    """Um tier de hardware declarado pelo cliente (``hw_tiers`` do pedido).

    O cliente conhece a sua tool: corre o hw-auto dela para cada tamanho de GPU
    disponível e manda o resultado. O supervisor só escolhe qual se aplica à
    GPU onde o job cai — assim a qualidade escala com o hardware real, não com
    o da máquina que submeteu.
    """

    label: str
    min_vram_mib: int
    rank: int
    set: dict[str, Any]
    gen_factor: float = 1.0


class HwTierError(ValueError):
    """``hw_tiers`` malformado (o pedido é recusado no enqueue)."""


def parse_hw_tiers(raw: Any) -> tuple[HwTier, ...]:
    """Valida ``hw_tiers``: lista de ``{min_vram_mib, rank?, set, label?, gen_factor?}``.

    ``rank`` por omissão é a posição por ``min_vram_mib`` crescente (o tier que
    pede mais VRAM é o melhor).

    Raises:
        HwTierError: estrutura inválida.
    """
    if raw is None or raw == []:
        return ()
    if not isinstance(raw, list):
        raise HwTierError("hw_tiers deve ser uma lista")
    if len(raw) > 16:
        raise HwTierError("hw_tiers: no máximo 16 tiers")
    parsed: list[tuple[int, int | None, str | None, dict[str, Any], float]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise HwTierError(f"hw_tiers[{index}] deve ser um objecto")
        try:
            min_vram = int(entry.get("min_vram_mib", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise HwTierError(f"hw_tiers[{index}].min_vram_mib não é inteiro") from exc
        if min_vram < 0:
            raise HwTierError(f"hw_tiers[{index}].min_vram_mib negativo")
        values = entry.get("set") or {}
        if not isinstance(values, Mapping):
            raise HwTierError(f"hw_tiers[{index}].set deve ser um objecto")
        bad = sorted(k for k in values if str(k).startswith("_") or k in ("cmd", "backend", *ROUTING_KEYS))
        if bad:
            raise HwTierError(f"hw_tiers[{index}].set não pode definir {', '.join(map(str, bad))}")
        rank = entry.get("rank")
        try:
            rank_i = int(rank) if rank is not None else None
            raw_factor = entry.get("gen_factor")
            factor = 1.0 if raw_factor is None else float(raw_factor)
        except (TypeError, ValueError) as exc:
            raise HwTierError(f"hw_tiers[{index}]: rank/gen_factor inválido") from exc
        if not factor > 0:
            raise HwTierError(f"hw_tiers[{index}].gen_factor deve ser > 0")
        label = entry.get("label")
        parsed.append((min_vram, rank_i, str(label) if label else None, dict(values), factor))
    parsed.sort(key=lambda item: item[0])
    return tuple(
        HwTier(
            label=label or f"tier{position}",
            min_vram_mib=min_vram,
            rank=rank if rank is not None else position,
            set=values,
            gen_factor=factor,
        )
        for position, (min_vram, rank, label, values, factor) in enumerate(parsed)
    )


def strip_routing_keys(request: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in request.items() if k not in ROUTING_KEYS}


def allowed_tiers(
    tiers: tuple[HwTier, ...] | list[Any],
    *,
    banned: frozenset[int] | set[int] = frozenset(),
    max_rank: int | None = None,
) -> list[Any]:
    """Tiers utilizáveis: sem os suspensos e até ``max_rank``.

    Nunca vazio quando há tiers: o mais modesto é o chão (é o único que resta
    tentar, e a admissão decide).
    """
    if not tiers:
        return []
    kept = [t for t in tiers if t.rank not in banned and (max_rank is None or t.rank <= max_rank)]
    return kept or [min(tiers, key=lambda t: t.rank)]


def apply_hw_tiers(
    request: Mapping[str, Any],
    total_mib: int | None,
    *,
    banned: frozenset[int] | set[int] = frozenset(),
    max_rank: int | None = None,
) -> tuple[dict[str, Any], HwTier | None]:
    """Pedido pronto para o worker numa GPU com ``total_mib`` de VRAM.

    Escolhe o tier (sem os suspensos por OOM e até ``max_rank``), aplica o
    ``set`` dele e remove os campos de routing. Sem ``hw_tiers`` devolve o
    pedido sem os campos de routing. Tiers inválidos também: o enqueue já os
    recusa, aqui não se rebenta um job a meio.
    """
    try:
        tiers = parse_hw_tiers(request.get("hw_tiers"))
    except HwTierError:
        tiers = ()
    out = strip_routing_keys(request)
    if not tiers:
        return out, None
    tier = select_tier(allowed_tiers(tiers, banned=banned, max_rank=max_rank), total_mib)
    return merge_tier_set(out, tier.set), tier


def merge_tier_set(request: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """Aplica o ``set`` de um tier: valor ``None`` remove a chave.

    O cliente calcula cada tier como a diferença entre o payload desse tamanho
    de GPU e o base; uma chave que só existe no base (ex.: ``sdnq_preset`` num
    tier fp16, onde o builder a omite) tem de poder desaparecer.
    """
    out = dict(request)
    for key, value in values.items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = value
    return out


def parse_cost_keys(raw: Any) -> dict[str, tuple[float, float | None]]:
    """``placement.cost_keys`` → ``{campo: (expoente, valor por omissão)}``.

    Forma curta ``{steps: 1, octree_resolution: 2}`` ou longa
    ``{steps: {exp: 1, default: 50}}``. Entradas inválidas são ignoradas.
    """
    out: dict[str, tuple[float, float | None]] = {}
    if not isinstance(raw, Mapping):
        return out
    for key, spec in raw.items():
        exp_raw: Any = spec
        default: Any = None
        if isinstance(spec, Mapping):
            exp_raw, default = spec.get("exp", 1.0), spec.get("default")
        try:
            exponent = float(exp_raw)
            default_f = float(default) if default is not None else None
        except (TypeError, ValueError):
            continue
        if exponent != 0 and (default_f is None or default_f > 0):
            out[str(key)] = (exponent, default_f)
    return out


def request_cost(cost_keys: Mapping[str, tuple[float, float | None]], request: Mapping[str, Any]) -> float | None:
    """Custo relativo de um pedido: ``Π valor^expoente`` sobre as features declaradas.

    ``None`` quando nenhuma feature tem valor (nada a escalar). Uma feature em
    falta usa o ``default`` declarado — o valor que o worker usaria — ou fica
    de fora.
    """
    cost = 1.0
    used = False
    for key, (exponent, default) in cost_keys.items():
        value = request.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            if default is None:
                continue
            value = default
        cost *= float(value) ** exponent
        used = True
    return cost if used else None


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
        health: Any = None,
    ) -> None:
        self.registry = registry
        self.lanes = lanes
        self.peers = peers
        self.policy = policy if policy is not None else RoutePolicy.from_env()
        # Tiers suspensos depois de OOMs reais (``tier_health.TierHealth``).
        self.health = health
        self._history_reader = history_reader
        self._clock = clock
        self._history_cache: tuple[float, dict[str, float]] | None = None
        self._lock = threading.Lock()
        # Geração medida por (backend, rank de tier): um tier hq de 50 steps não
        # demora o mesmo que o int4 de 30, e a média do backend mistura os dois.
        self._tier_gen: dict[tuple[str, int], float] = {}
        # Segundos por unidade de custo (``placement.cost_keys``) por (backend,
        # rank; -1 sem tiers): um pedido com o dobro dos steps estima o dobro.
        self._cost_ratio: dict[tuple[str, int], float] = {}
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

    def observe(
        self,
        backend: str,
        rank: int | None,
        gen_sec: float,
        request: Mapping[str, Any] | None = None,
    ) -> None:
        """Alimenta as estimativas de geração com um job real (quente).

        Com ``placement.cost_keys`` aprende segundos por unidade de custo do
        pedido; sem elas, a média do tier.
        """
        if not gen_sec or gen_sec <= 0:
            return
        key = (backend, int(rank) if rank is not None else -1)
        cost = request_cost(self._cost_keys(backend), request or {})
        with self._lock:
            if rank is not None:
                previous = self._tier_gen.get(key)
                self._tier_gen[key] = (
                    gen_sec if previous is None else (1 - _EWMA_ALPHA) * previous + _EWMA_ALPHA * gen_sec
                )
            if cost:
                ratio = gen_sec / cost
                previous_ratio = self._cost_ratio.get(key)
                self._cost_ratio[key] = (
                    ratio if previous_ratio is None else (1 - _EWMA_ALPHA) * previous_ratio + _EWMA_ALPHA * ratio
                )

    def _cost_keys(self, backend: str) -> dict[str, tuple[float, float | None]]:
        return parse_cost_keys(self._placement(backend).get("cost_keys"))

    def _scaled_gen(self, backend: str, rank: int, request: Mapping[str, Any]) -> float | None:
        """Geração estimada pelo modelo de custo, se já houver medições."""
        cost = request_cost(self._cost_keys(backend), request)
        if not cost:
            return None
        with self._lock:
            ratio = self._cost_ratio.get((backend, rank))
        return ratio * cost if ratio is not None else None

    def _tier_gen_sec(self, backend: str, tier: HwTier, base: float, request: Mapping[str, Any] | None = None) -> float:
        scaled = self._scaled_gen(backend, tier.rank, request or {})
        if scaled is not None:
            return scaled
        with self._lock:
            measured = self._tier_gen.get((backend, tier.rank))
        return measured if measured is not None else base * tier.gen_factor

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

    def profile(
        self,
        backend: str,
        request: Mapping[str, Any] | None = None,
        *,
        forwarded: bool = False,
        tier_cap: int | None = None,
    ) -> JobProfile:
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
        gen_sec = self.gen_sec(backend)
        scaled = self._scaled_gen(backend, -1, req)
        if scaled is not None:
            gen_sec = scaled
        tiers: list[Tier] = []
        try:
            hw_tiers = parse_hw_tiers(req.get("hw_tiers"))
        except HwTierError:
            hw_tiers = ()
        base_req = strip_routing_keys(req)
        for hw in allowed_tiers(hw_tiers, max_rank=tier_cap):
            tier_req = merge_tier_set(base_req, hw.set)
            tier_peak, tier_headroom = self._peaks(backend, tier_req)
            tiers.append(
                Tier(
                    label=hw.label,
                    min_vram_mib=hw.min_vram_mib,
                    rank=hw.rank,
                    peak_mib=tier_peak,
                    headroom_mib=tier_headroom,
                    gen_sec=self._tier_gen_sec(backend, hw, gen_sec, tier_req),
                )
            )
        if tiers:
            # O perfil "plano" é o do tier mais modesto: é o que qualquer rota
            # sem informação de VRAM vai tentar.
            modest = min(tiers, key=lambda t: (t.min_vram_mib, t.rank))
            peak, headroom, gen_sec = modest.peak_mib, modest.headroom_mib, modest.gen_sec
        return JobProfile(
            backend=backend,
            peak_mib=peak,
            headroom_mib=headroom,
            load_sec=self.load_sec(backend),
            gen_sec=gen_sec,
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
            tiers=tuple(tiers),
            criticality=normalize_criticality(req.get("criticality")),
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
        critical_pending: bool = False,
        tier_cap: int | None = None,
    ) -> Decision:
        """Rota de um job dado o estado actual.

        Args:
            lane_extra: Segundos de trabalho já prometidos a cada lane por jobs
                à frente na fila (é o que faz o router abrir réplicas num batch).
            peer_extra: O mesmo, por peer.
            exclude_peers: Peers que já falharam para este job.
            record: Contabilizar a decisão nos contadores (só no dispatch real).
        """
        profile = self.profile(backend, request, forwarded=forwarded, tier_cap=tier_cap)
        if critical_pending:
            profile = replace(profile, reserve_for_critical=True)
        views = [
            self.lanes.view(lane, backend, request, extra_busy_sec=float((lane_extra or {}).get(lane.lane_id, 0.0)))
            for lane in self.lanes.device_lanes
        ]
        if self.health is not None and profile.tiers:
            views = [replace(v, tier_bans=self.health.banned_ranks(backend, v.total_mib)) for v in views]
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
                "quality_high_sec": self.policy.quality_high_sec,
                "quality_normal_sec": self.policy.quality_normal_sec,
                "reserve_low_sec": self.policy.reserve_low_sec,
                "reserve_low_pending_sec": self.policy.reserve_low_pending_sec,
            },
        }
