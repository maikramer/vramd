"""Router — decide ONDE um job corre: que GPU, várias GPUs, ou outra máquina.

A admissão responde "cabe?". O router responde "onde acaba mais cedo?". Para
cada job enumera as rotas possíveis e estima, em segundos, quando o resultado
estaria pronto em cada uma::

    total = espera + evicção + load + geração + transferência

e escolhe a menor. As três estratégias saem do mesmo modelo, sem regras à parte:

* **Réplica** (o mesmo modelo em várias GPUs, para um batch): carregar o modelo
  numa GPU livre custa ``load``; esperar pela GPU onde ele já está quente custa
  o que falta ao job em curso mais os jobs à frente. Quando a fila acumulada
  ultrapassa o custo do load, a rota fria ganha — e nasce uma réplica. Um job
  isolado nunca paga um load à toa.
* **Split** (o modelo dividido por várias GPUs): só é candidata quando o pico
  não cabe em nenhuma GPU sozinha e o backend declara que sabe dividir-se.
  Paga uma penalização de velocidade, por isso perde sempre para uma GPU onde
  o modelo caiba inteiro.
* **Rede** (outro supervisor vramd): paga latência e a transferência dos
  ficheiros à largura de banda medida, mais a fila do outro lado. Ganha quando
  a máquina local está ocupada ou o modelo não cabe cá.

Este módulo é **puro** (sem GPU, sem sockets, sem relógio) — quem chama
fotografa o estado em :class:`LaneView`/:class:`PeerView` e recebe uma
:class:`Decision` explicável (``vramd route`` mostra a tabela inteira).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

ROUTE_LOCAL = "local"
ROUTE_SPLIT = "split"
ROUTE_REMOTE = "remote"
_KIND_ORDER = {ROUTE_LOCAL: 0, ROUTE_SPLIT: 1, ROUTE_REMOTE: 2}

ACTION_RUN = "run"
ACTION_WAIT = "wait"
ACTION_REJECT = "reject"

# Estimativas quando ainda não há medições (um backend novo, um peer antigo).
DEFAULT_LOAD_SEC = 30.0
DEFAULT_GEN_SEC = 30.0
DEFAULT_BANDWIDTH_MIB_S = 50.0
DEFAULT_SPLIT_OVERHEAD = 0.10
DEFAULT_SPLIT_SLOWDOWN = 1.3
_MIB = 1024 * 1024


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class RoutePolicy:
    """Parâmetros de desempate do router (env overrideáveis).

    Attributes:
        replica_margin_sec: Uma réplica nova só compensa se poupar pelo menos
            isto — evita carregar um modelo de 40 s para ganhar 1 s.
        remote_penalty_sec: Custo fixo de sair da máquina (mais partes móveis:
            rede, outro daemon). Em empate, fica local.
        split_penalty_sec: Custo fixo de dividir um modelo (ocupa várias GPUs
            e mais coisas podem falhar).
    """

    replica_margin_sec: float = 5.0
    remote_penalty_sec: float = 2.0
    split_penalty_sec: float = 5.0

    @classmethod
    def from_env(cls) -> RoutePolicy:
        return cls(
            replica_margin_sec=max(0.0, _env_float("VRAMD_ROUTE_REPLICA_MARGIN_SEC", 5.0)),
            remote_penalty_sec=max(0.0, _env_float("VRAMD_ROUTE_REMOTE_PENALTY_SEC", 2.0)),
            split_penalty_sec=max(0.0, _env_float("VRAMD_ROUTE_SPLIT_PENALTY_SEC", 5.0)),
        )


@dataclass(frozen=True)
class JobProfile:
    """O que o router precisa de saber sobre um job.

    Attributes:
        backend: Nome do backend.
        peak_mib: Pico de admissão a frio (pesos + activação + margem).
        headroom_mib: VRAM livre necessária com os pesos já carregados.
        load_sec: Duração estimada do load a frio.
        gen_sec: Duração estimada da geração.
        input_bytes: Ficheiros de input a transferir numa rota remota.
        output_bytes: Tamanho esperado do output a trazer de volta.
        allow_split: O backend sabe dividir-se por várias GPUs.
        split_overhead: Fracção de VRAM extra quando dividido.
        split_slowdown: Factor de lentidão da geração dividida.
        max_replicas: Teto de réplicas simultâneas (0 = sem teto).
        allow_remote: O job pode sair da máquina.
        pinned_devices: ``gpu_ids`` explícitos do pedido — o router respeita.
        local_only: Job já encaminhado por outro nó (nunca reencaminha: loops).
    """

    backend: str
    peak_mib: int
    headroom_mib: int = 0
    load_sec: float = DEFAULT_LOAD_SEC
    gen_sec: float = DEFAULT_GEN_SEC
    input_bytes: int = 0
    output_bytes: int = 0
    allow_split: bool = False
    split_overhead: float = DEFAULT_SPLIT_OVERHEAD
    split_slowdown: float = DEFAULT_SPLIT_SLOWDOWN
    max_replicas: int = 0
    allow_remote: bool = True
    pinned_devices: tuple[int, ...] | None = None
    local_only: bool = False


@dataclass(frozen=True)
class LaneView:
    """Fotografia de uma GPU local (uma "lane" do supervisor).

    Attributes:
        lane_id: Identificador (``gpu0``).
        devices: Índices físicos da(s) GPU(s) da lane.
        free_mib: VRAM livre agora (``None`` = leitura indisponível).
        total_mib: VRAM total (``None`` = desconhecida).
        evictable_mib: VRAM que a lane liberta evictando os outros backends.
        evict_cost_sec: Custo esperado de recarregar mais tarde o que for evictado.
        hot: O backend está carregado com o mesmo ``load_shape`` do pedido.
        loaded: O backend está carregado (qualquer shape) — conta como réplica.
        busy_sec: ``None`` se livre; senão segundos estimados até ficar livre
            (job em curso + jobs à frente que lhe estão destinados).
    """

    lane_id: str
    devices: tuple[int, ...]
    free_mib: int | None
    total_mib: int | None
    evictable_mib: int = 0
    evict_cost_sec: float = 0.0
    hot: bool = False
    loaded: bool = False
    busy_sec: float | None = None


@dataclass(frozen=True)
class PeerView:
    """Fotografia de outro supervisor vramd na rede.

    Attributes:
        name: Nome do peer.
        healthy: Respondeu ao último poll.
        has_backend: Tem o backend registado.
        hot: Tem-no carregado.
        fits: ``False`` se o pico não cabe em nenhuma GPU dele; ``None`` = não se sabe.
        rtt_sec: Latência de ida e volta medida.
        bandwidth_mib_s: Largura de banda medida/declarada (MiB/s).
        wait_sec: Fila do peer (ETA dele) mais a nossa própria ocupação do slot.
        slot_free: Há capacidade para encaminhar já.
        load_sec: Load estimado no peer (``None`` = usar o nosso).
        gen_sec: Geração estimada no peer (``None`` = usar a nossa).
        reason: Porque está indisponível, quando está.
    """

    name: str
    healthy: bool = True
    has_backend: bool = True
    hot: bool = False
    fits: bool | None = None
    rtt_sec: float = 0.0
    bandwidth_mib_s: float = DEFAULT_BANDWIDTH_MIB_S
    wait_sec: float = 0.0
    slot_free: bool = True
    load_sec: float | None = None
    gen_sec: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class Route:
    """Uma rota candidata, com o custo decomposto (segundos)."""

    kind: str
    target: str
    devices: tuple[int, ...] = ()
    feasible: bool = True
    reason: str = ""
    wait_sec: float = 0.0
    evict_sec: float = 0.0
    load_sec: float = 0.0
    gen_sec: float = 0.0
    transfer_sec: float = 0.0
    penalty_sec: float = 0.0
    hot: bool = False
    replica: bool = False
    shares_mib: dict[int, int] = field(default_factory=dict)

    @property
    def total_sec(self) -> float:
        return self.wait_sec + self.evict_sec + self.load_sec + self.gen_sec + self.transfer_sec + self.penalty_sec

    @property
    def run_now(self) -> bool:
        return self.feasible and self.wait_sec <= 0.0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "target": self.target,
            "devices": list(self.devices),
            "feasible": self.feasible,
            "hot": self.hot,
            "replica": self.replica,
        }
        if self.reason:
            out["reason"] = self.reason
        if self.feasible:
            out.update(
                {
                    "total_sec": round(self.total_sec, 2),
                    "wait_sec": round(self.wait_sec, 2),
                    "evict_sec": round(self.evict_sec, 2),
                    "load_sec": round(self.load_sec, 2),
                    "gen_sec": round(self.gen_sec, 2),
                    "transfer_sec": round(self.transfer_sec, 2),
                    "penalty_sec": round(self.penalty_sec, 2),
                }
            )
        if self.shares_mib:
            out["shares_mib"] = {str(k): v for k, v in self.shares_mib.items()}
        return out


@dataclass(frozen=True)
class Decision:
    """Resultado do router: a rota escolhida e todas as que perdeu.

    Attributes:
        action: ``run`` (despachar já em ``best``), ``wait`` (a melhor rota
            está ocupada — esperar é mais rápido do que qualquer alternativa)
            ou ``reject`` (nenhuma rota é possível).
        best: A rota escolhida (``None`` em ``reject``).
        routes: Todas as candidatas, viáveis primeiro, por custo crescente.
        explanation: Uma frase a justificar a escolha (para logs e ``vramd route``).
    """

    action: str
    best: Route | None
    routes: tuple[Route, ...]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "best": self.best.to_dict() if self.best is not None else None,
            "explanation": self.explanation,
            "routes": [r.to_dict() for r in self.routes],
        }


def _infeasible(kind: str, target: str, devices: tuple[int, ...], reason: str) -> Route:
    return Route(kind=kind, target=target, devices=devices, feasible=False, reason=reason)


def _lane_route(
    profile: JobProfile,
    lane: LaneView,
    *,
    replicas: int,
    policy: RoutePolicy,
) -> Route:
    """Custo de correr o job numa GPU local."""
    pinned = profile.pinned_devices
    if pinned is not None and tuple(pinned) != tuple(lane.devices):
        return _infeasible(ROUTE_LOCAL, lane.lane_id, lane.devices, "pedido fixou outras GPUs (gpu_ids)")

    need = profile.headroom_mib if lane.hot else profile.peak_mib
    if lane.total_mib is not None and profile.peak_mib > lane.total_mib:
        return _infeasible(
            ROUTE_LOCAL,
            lane.lane_id,
            lane.devices,
            f"pico {profile.peak_mib} MiB > VRAM total {lane.total_mib} MiB",
        )
    new_replica = not lane.loaded and replicas > 0
    if new_replica and profile.max_replicas > 0 and replicas >= profile.max_replicas:
        return _infeasible(
            ROUTE_LOCAL, lane.lane_id, lane.devices, f"teto de réplicas atingido ({profile.max_replicas})"
        )

    evict_sec = 0.0
    if lane.free_mib is not None and need > lane.free_mib:
        deficit = need - lane.free_mib
        if deficit > lane.evictable_mib:
            return _infeasible(
                ROUTE_LOCAL,
                lane.lane_id,
                lane.devices,
                f"precisa de {need} MiB; livre {lane.free_mib} + evictável {lane.evictable_mib}",
            )
        # Só se paga a fracção do que realmente tem de sair.
        evict_sec = lane.evict_cost_sec * min(1.0, deficit / max(1, lane.evictable_mib))

    return Route(
        kind=ROUTE_LOCAL,
        target=lane.lane_id,
        devices=lane.devices,
        wait_sec=max(0.0, lane.busy_sec or 0.0),
        evict_sec=evict_sec,
        load_sec=0.0 if lane.hot else profile.load_sec,
        gen_sec=profile.gen_sec,
        penalty_sec=policy.replica_margin_sec if new_replica else 0.0,
        hot=lane.hot,
        replica=new_replica,
    )


def plan_split(profile: JobProfile, lanes: Sequence[LaneView]) -> tuple[tuple[LaneView, ...], dict[int, int]] | None:
    """Menor conjunto de GPUs cuja VRAM somada aguenta o modelo dividido.

    Greedy pelas GPUs com mais VRAM utilizável (livre + evictável): menos GPUs
    envolvidas é menos tráfego entre placas e menos lanes bloqueadas.

    Returns:
        ``(lanes, quota por GPU em MiB)`` ou ``None`` se nem todas juntas chegam.
    """
    need = int(profile.peak_mib * (1.0 + max(0.0, profile.split_overhead)))
    usable: list[tuple[int, LaneView]] = []
    for lane in lanes:
        if len(lane.devices) != 1 or lane.free_mib is None:
            continue
        cap = lane.free_mib + lane.evictable_mib
        if lane.total_mib is not None:
            cap = min(cap, lane.total_mib)
        if cap > 0:
            usable.append((cap, lane))
    usable.sort(key=lambda item: (-item[0], item[1].lane_id))
    chosen: list[tuple[int, LaneView]] = []
    acc = 0
    for cap, lane in usable:
        chosen.append((cap, lane))
        acc += cap
        if acc >= need and len(chosen) >= 2:
            break
    if acc < need or len(chosen) < 2:
        return None
    # Quotas proporcionais à capacidade: a GPU maior leva a fatia maior.
    shares = {lane.devices[0]: max(1, int(need * cap / acc)) for cap, lane in chosen}
    return tuple(lane for _cap, lane in chosen), shares


def _split_route(profile: JobProfile, lanes: Sequence[LaneView], policy: RoutePolicy) -> Route | None:
    if not profile.allow_split:
        return None
    pool = list(lanes)
    if profile.pinned_devices is not None:
        wanted = set(profile.pinned_devices)
        pool = [lane for lane in pool if set(lane.devices) <= wanted]
    plan = plan_split(profile, pool)
    if plan is None:
        devices = tuple(sorted(d for lane in pool for d in lane.devices))
        label = "split:" + "+".join(f"gpu{d}" for d in devices) if devices else "split"
        return _infeasible(ROUTE_SPLIT, label, devices, "VRAM somada das GPUs não chega para o modelo dividido")
    members, shares = plan
    devices = tuple(sorted(shares))
    evict_sec = 0.0
    for lane in members:
        share = shares[lane.devices[0]]
        if lane.free_mib is not None and share > lane.free_mib and lane.evictable_mib > 0:
            evict_sec += lane.evict_cost_sec * min(1.0, (share - lane.free_mib) / lane.evictable_mib)
    return Route(
        kind=ROUTE_SPLIT,
        target="split:" + "+".join(f"gpu{d}" for d in devices),
        devices=devices,
        # O split precisa de TODAS as GPUs do conjunto livres ao mesmo tempo.
        wait_sec=max((lane.busy_sec or 0.0) for lane in members),
        evict_sec=evict_sec,
        load_sec=profile.load_sec,
        gen_sec=profile.gen_sec * max(1.0, profile.split_slowdown),
        penalty_sec=policy.split_penalty_sec,
        shares_mib=shares,
    )


def _peer_route(profile: JobProfile, peer: PeerView, policy: RoutePolicy) -> Route:
    target = f"peer:{peer.name}"
    if profile.local_only:
        return _infeasible(ROUTE_REMOTE, target, (), "job já foi encaminhado por outro nó")
    if not profile.allow_remote:
        return _infeasible(ROUTE_REMOTE, target, (), "backend não permite rota remota")
    if profile.pinned_devices is not None:
        return _infeasible(ROUTE_REMOTE, target, (), "pedido fixou GPUs locais (gpu_ids)")
    if not peer.healthy:
        return _infeasible(ROUTE_REMOTE, target, (), peer.reason or "peer não responde")
    if not peer.has_backend:
        return _infeasible(ROUTE_REMOTE, target, (), "peer não tem este backend")
    if peer.fits is False:
        return _infeasible(ROUTE_REMOTE, target, (), "pico não cabe nas GPUs do peer")
    bandwidth = peer.bandwidth_mib_s if peer.bandwidth_mib_s > 0 else DEFAULT_BANDWIDTH_MIB_S
    payload_mib = (max(0, profile.input_bytes) + max(0, profile.output_bytes)) / _MIB
    wait = max(0.0, peer.wait_sec)
    if not peer.slot_free:
        # Sem slot de encaminhamento livre a rota nunca é "já", por pequena que
        # seja a espera estimada.
        wait = max(wait, 1.0)
    return Route(
        kind=ROUTE_REMOTE,
        target=target,
        wait_sec=wait,
        load_sec=0.0 if peer.hot else (peer.load_sec if peer.load_sec is not None else profile.load_sec),
        gen_sec=peer.gen_sec if peer.gen_sec is not None else profile.gen_sec,
        # Submit + resultado: duas idas e voltas, mais os bytes.
        transfer_sec=2.0 * max(0.0, peer.rtt_sec) + payload_mib / bandwidth,
        penalty_sec=policy.remote_penalty_sec,
        hot=peer.hot,
    )


def _explain(best: Route, routes: Sequence[Route], action: str) -> str:
    where = {
        ROUTE_LOCAL: f"GPU local {best.target}",
        ROUTE_SPLIT: f"modelo dividido por {best.target.removeprefix('split:')}",
        ROUTE_REMOTE: f"rede → {best.target.removeprefix('peer:')}",
    }[best.kind]
    state = "quente" if best.hot else ("réplica nova" if best.replica else "load a frio")
    head = f"{where} ({state}), pronto em ~{best.total_sec:.0f}s"
    if action == ACTION_WAIT:
        head = f"esperar {best.wait_sec:.0f}s por {where} ({state}), pronto em ~{best.total_sec:.0f}s"
    runner_up = next((r for r in routes if r.feasible and r is not best), None)
    if runner_up is not None:
        head += f"; alternativa {runner_up.target} em ~{runner_up.total_sec:.0f}s"
    return head


def plan_routes(
    profile: JobProfile,
    lanes: Sequence[LaneView],
    peers: Sequence[PeerView] = (),
    *,
    policy: RoutePolicy | None = None,
) -> Decision:
    """Enumera as rotas de um job e escolhe a que acaba mais cedo.

    Args:
        profile: Requisitos e estimativas do job.
        lanes: Estado das GPUs locais.
        peers: Estado dos outros supervisores (vazio = sem rede).
        policy: Parâmetros de desempate (default: env).

    Returns:
        A decisão com todas as candidatas — incluindo as inviáveis e porquê.
    """
    policy = policy if policy is not None else RoutePolicy.from_env()
    device_lanes = [lane for lane in lanes if len(lane.devices) == 1]
    replicas = sum(1 for lane in device_lanes if lane.loaded)

    routes: list[Route] = [_lane_route(profile, lane, replicas=replicas, policy=policy) for lane in device_lanes]
    # "Caso precise": dividir só entra em jogo quando o modelo não cabe inteiro
    # em nenhuma GPU local — nunca compete com uma GPU onde caberia.
    if not any(r.feasible for r in routes):
        split = _split_route(profile, device_lanes, policy)
        if split is not None:
            routes.append(split)
    routes.extend(_peer_route(profile, peer, policy) for peer in peers)

    feasible = sorted(
        (r for r in routes if r.feasible),
        key=lambda r: (round(r.total_sec, 3), _KIND_ORDER[r.kind], r.target),
    )
    infeasible = sorted((r for r in routes if not r.feasible), key=lambda r: (_KIND_ORDER[r.kind], r.target))
    ordered = (*feasible, *infeasible)
    if not feasible:
        reasons = "; ".join(f"{r.target}: {r.reason}" for r in infeasible) or "sem GPUs nem peers"
        return Decision(ACTION_REJECT, None, ordered, f"nenhuma rota possível — {reasons}")
    best = feasible[0]
    action = ACTION_RUN if best.run_now else ACTION_WAIT
    return Decision(action, best, ordered, _explain(best, feasible, action))
