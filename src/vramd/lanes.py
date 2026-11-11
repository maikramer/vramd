"""Lanes — uma GPU, um gestor de backends.

O :class:`~vramd.backend_manager.BackendManager` foi escrito para uma placa:
uma leitura de VRAM livre, um plano de evicção. Em vez de o ensinar a
contabilizar N dispositivos, cada GPU ganha a sua **lane** — um gestor próprio,
com o seu pool de workers fixado nessa GPU por ``CUDA_VISIBLE_DEVICES``. Do
ponto de vista de cada gestor (e de cada worker) o mundo continua a ter uma
GPU, e toda a admissão por pico medida e testada vale sem alterações.

Daqui saem as duas formas de multi-GPU:

* **Réplicas** — o mesmo backend carregado em duas lanes são dois workers
  independentes; o router decide quando vale a pena o segundo load.
* **Split** — uma lane especial ("span") cujos workers vêem várias GPUs e cujo
  gestor vê a VRAM somada. Enquanto corre, as lanes das GPUs que ocupa ficam
  reservadas.

Com uma só GPU (e ``VRAMD_GPUS`` por definir) nada disto se activa: há uma lane
não fixada, que é exactamente o supervisor de sempre.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from vramd.logging import Logger

from .router import DEFAULT_LOAD_SEC, LaneView

_logger = Logger()

ENV_GPUS = "VRAMD_GPUS"
KIND_DEVICE = "device"
KIND_SPAN = "span"
_FREE_CACHE_TTL_SEC = 0.5
# Probabilidade de um backend evictado voltar a ser preciso em breve: o custo
# de evicção que o router cobra é o load esperado, não o load certo.
_EVICT_RELOAD_PROBABILITY = 0.5


def parse_gpu_list(raw: str) -> list[int]:
    """``"0,2"`` → ``[0, 2]`` (sem repetidos, ordem preservada).

    Raises:
        ValueError: token não inteiro ou negativo.
    """
    out: list[int] = []
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError(f"índice de GPU negativo: {value}")
        if value not in out:
            out.append(value)
    return out


def resolve_gpu_devices(detect: Callable[[], list[int] | None] | None = None) -> list[int] | None:
    """GPUs que o supervisor gere com lanes fixadas, ou ``None`` (modo de uma lane).

    ``VRAMD_GPUS``:

    * por definir / ``auto`` — todas as GPUs detectadas, **se forem mais que uma**;
      com uma só, ``None`` (comportamento histórico, sem fixar nada);
    * ``legacy`` / ``off`` — ``None`` mesmo com várias GPUs;
    * ``0,2`` — só essas (uma única também vale: fixa o supervisor nessa GPU).
    """
    raw = os.environ.get(ENV_GPUS, "").strip().lower()
    if raw in ("legacy", "off", "none", "false"):
        return None
    if raw and raw != "auto":
        try:
            devices = parse_gpu_list(raw)
        except ValueError as exc:
            _logger.warn(f"[vramd] {ENV_GPUS}={raw!r} inválido ({exc}) — a usar o modo de uma lane.")
            return None
        return devices or None
    if detect is None:
        from .gpu import detect_gpu_ids

        detect = detect_gpu_ids
    try:
        detected = detect()
    except Exception:
        return None
    return list(detected) if detected and len(detected) > 1 else None


def lane_id_for(devices: Iterable[int]) -> str:
    """``(1,)`` → ``gpu1``; ``(0, 1)`` → ``split:gpu0+gpu1``."""
    devs = tuple(devices)
    if len(devs) == 1:
        return f"gpu{devs[0]}"
    return "split:" + "+".join(f"gpu{d}" for d in devs)


def lane_env(devices: Iterable[int]) -> dict[str, str]:
    """Ambiente que fixa os workers de uma lane nas suas GPUs."""
    csv = ",".join(str(d) for d in devices)
    return {
        "CUDA_VISIBLE_DEVICES": csv,
        # PCI_BUS_ID alinha a numeração do CUDA com a do nvidia-smi/NVML — sem
        # isto "GPU 1" pode ser uma placa para o driver e outra para o torch.
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "VRAMD_GPU_IDS": csv,
    }


@dataclass
class Lane:
    """Uma GPU (ou um conjunto, para split) com o seu gestor de backends.

    Attributes:
        lane_id: ``gpu0`` / ``split:gpu0+gpu1``.
        devices: Índices físicos das GPUs.
        manager: O ``BackendManager`` desta lane.
        kind: ``device`` ou ``span``.
        pinned: Os workers estão fixados por ``CUDA_VISIBLE_DEVICES``.
        query_free_mib: Leitura de VRAM livre da lane (somada, num span).
        query_total_mib: VRAM total da lane.
        idle_evictor: Evictor de idle próprio (lanes extra); o da lane
            primária é o do servidor.
    """

    lane_id: str
    devices: tuple[int, ...]
    manager: Any
    kind: str = KIND_DEVICE
    pinned: bool = False
    query_free_mib: Callable[[], int | None] | None = None
    query_total_mib: Callable[[], int | None] | None = None
    idle_evictor: Any = None
    # Estado de execução (guardado pelo lock do LaneSet).
    job: Any = None
    started_at: float | None = None
    est_sec: float = 0.0
    jobs_run: int = 0
    _free_cache: tuple[float, int | None] | None = field(default=None, repr=False)
    _total_cache: int | None = field(default=None, repr=False)

    def overlaps(self, other: Lane) -> bool:
        return bool(set(self.devices) & set(other.devices))


class LaneSet:
    """As lanes do supervisor: reserva, fotografia para o router, evicção cruzada.

    Args:
        lanes: Lanes de dispositivo, a primária primeiro.
        registry: Registry de backends (footprints para o custo de evicção).
        span_factory: Cria a lane de split para um conjunto de GPUs (lazy —
            a maior parte das instalações nunca divide um modelo).
        clock: Relógio monotónico (injectável nos testes).
    """

    def __init__(
        self,
        lanes: list[Lane],
        *,
        registry: Any = None,
        span_factory: Callable[[tuple[int, ...]], Lane] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not lanes:
            raise ValueError("LaneSet precisa de pelo menos uma lane")
        self._lanes = list(lanes)
        self._registry = registry
        self._span_factory = span_factory
        self._spans: dict[tuple[int, ...], Lane] = {}
        self._clock = clock
        self._lock = threading.RLock()

    # -- inventário ---------------------------------------------------------

    @property
    def primary(self) -> Lane:
        return self._lanes[0]

    @property
    def device_lanes(self) -> list[Lane]:
        return list(self._lanes)

    @property
    def span_lanes(self) -> list[Lane]:
        with self._lock:
            return list(self._spans.values())

    def all_lanes(self) -> list[Lane]:
        return [*self.device_lanes, *self.span_lanes]

    @property
    def routed(self) -> bool:
        """True se há decisões de colocação a tomar (várias lanes ou lane fixada)."""
        return len(self._lanes) > 1 or self._lanes[0].pinned

    def lane(self, lane_id: str) -> Lane | None:
        for lane in self.all_lanes():
            if lane.lane_id == lane_id:
                return lane
        return None

    def span_for(self, devices: Iterable[int]) -> Lane | None:
        """Lane de split para ``devices`` (criada na primeira vez)."""
        key = tuple(sorted(devices))
        with self._lock:
            lane = self._spans.get(key)
            if lane is None and self._span_factory is not None:
                lane = self._span_factory(key)
                self._spans[key] = lane
            return lane

    # -- leituras -----------------------------------------------------------

    def free_mib(self, lane: Lane, *, fresh: bool = False) -> int | None:
        """VRAM livre da lane, com cache curto (um passo de claim lê N vezes)."""
        now = self._clock()
        cached = lane._free_cache
        if not fresh and cached is not None and (now - cached[0]) < _FREE_CACHE_TTL_SEC:
            return cached[1]
        value: int | None = None
        if lane.query_free_mib is not None:
            with contextlib.suppress(Exception):
                value = lane.query_free_mib()
        lane._free_cache = (now, value)
        return value

    def total_mib(self, lane: Lane) -> int | None:
        if lane._total_cache is None and lane.query_total_mib is not None:
            with contextlib.suppress(Exception):
                lane._total_cache = lane.query_total_mib()
        return lane._total_cache

    def invalidate(self) -> None:
        """Esquece as leituras de VRAM (depois de um load/evict)."""
        for lane in self.all_lanes():
            lane._free_cache = None

    def _descriptor(self, name: str) -> Any:
        if self._registry is None:
            return None
        try:
            return self._registry.descriptor(name)
        except Exception:
            return None

    def _avg_load_sec(self, manager: Any, name: str) -> float:
        stats = getattr(manager, "stats", None)
        row = stats.get(name) if stats is not None and hasattr(stats, "get") else None
        avg = getattr(row, "avg_load_time_sec", 0.0) if row is not None else 0.0
        return float(avg) if avg and avg > 0 else DEFAULT_LOAD_SEC

    def _evictable(self, lane: Lane, backend: str) -> tuple[int, float]:
        """``(MiB que a lane liberta, custo esperado de recarga)`` excluindo ``backend``."""
        mib = 0
        cost = 0.0
        sources = [(lane, 1)]
        # Um span ocioso com um modelo carregado ocupa VRAM nesta GPU; evictá-lo
        # liberta a fatia que aqui vive (aproximada por divisão igual).
        sources.extend(
            (span, len(span.devices)) for span in self.span_lanes if span.overlaps(lane) and span is not lane
        )
        if lane.kind == KIND_SPAN:
            sources.extend((dev, 1) for dev in self._lanes if dev.overlaps(lane))
        for source, divisor in sources:
            try:
                names = list(source.manager.loaded_names())
            except Exception:
                continue
            for name in names:
                if source is lane and name == backend:
                    continue
                desc = self._descriptor(name)
                if desc is not None and not getattr(desc, "unload_frees_vram", True):
                    continue
                vram = int(getattr(desc, "vram_mib", 0) or 0) if desc is not None else 0
                mib += vram // max(1, divisor)
                cost += self._avg_load_sec(source.manager, name) * _EVICT_RELOAD_PROBABILITY
        return mib, cost

    def remaining_sec(self, lane: Lane) -> float:
        """Segundos estimados até a lane libertar (0 se livre)."""
        with self._lock:
            job, started, est = lane.job, lane.started_at, lane.est_sec
        if job is None or started is None:
            return 0.0
        elapsed = max(0.0, self._clock() - started)
        pct = getattr(job, "progress_pct", None)
        if isinstance(pct, (int, float)) and 0.05 <= float(pct) < 1.0 and elapsed > 0:
            # O progresso reportado pelo worker vale mais que a média histórica.
            return max(1.0, elapsed * (1.0 - float(pct)) / float(pct))
        if est > elapsed:
            return max(1.0, est - elapsed)
        # Passou da estimativa sem dizer quanto falta: não prometer "já".
        return max(1.0, 0.2 * est)

    def busy_sec(self, lane: Lane) -> float:
        """Ocupação da lane contando as que partilham GPUs com ela."""
        worst = self.remaining_sec(lane)
        for other in self.all_lanes():
            if other is not lane and other.overlaps(lane):
                worst = max(worst, self.remaining_sec(other))
        return worst

    def is_busy(self, lane: Lane) -> bool:
        with self._lock:
            if lane.job is not None:
                return True
            return any(o.job is not None for o in self.all_lanes() if o is not lane and o.overlaps(lane))

    def view(
        self,
        lane: Lane,
        backend: str,
        request: Mapping[str, Any] | None = None,
        *,
        extra_busy_sec: float = 0.0,
    ) -> LaneView:
        """Fotografia da lane para o router decidir sobre ``backend``."""
        mgr = lane.manager
        loaded = False
        hot = False
        with contextlib.suppress(Exception):
            loaded = bool(mgr.is_loaded(backend))
        if loaded:
            matcher = getattr(mgr, "shape_matches_loaded", None)
            hot = bool(matcher(backend, dict(request or {}))) if callable(matcher) else True
        with self._lock:
            running = lane.job
        if running is not None and getattr(running, "backend", None) == backend:
            # A lane está a carregar/correr este mesmo backend: quando libertar
            # vai tê-lo quente. Sem isto, durante o primeiro load a frio a lane
            # parecia "sem o modelo" e o job seguinte abria uma réplica sem a
            # margem — dois loads de 30 s em paralelo para um batch de segundos.
            loaded = True
            hot = True
        evictable_mib, evict_cost = self._evictable(lane, backend)
        busy = self.busy_sec(lane) + max(0.0, extra_busy_sec)
        return LaneView(
            lane_id=lane.lane_id,
            devices=lane.devices,
            free_mib=self.free_mib(lane),
            total_mib=self.total_mib(lane),
            evictable_mib=evictable_mib,
            evict_cost_sec=evict_cost,
            hot=hot,
            loaded=loaded,
            busy_sec=busy if busy > 0 else None,
        )

    # -- reserva ------------------------------------------------------------

    def reserve(self, lane: Lane, job: Any, *, est_sec: float) -> bool:
        """Marca a lane como ocupada por ``job``; ``False`` se já não está livre.

        Atómico: duas threads de dispatch que escolham a mesma GPU (ou um
        split e uma das suas GPUs) não a ganham as duas.
        """
        with self._lock:
            if self.is_busy(lane):
                return False
            lane.job = job
            lane.started_at = self._clock()
            lane.est_sec = max(0.0, float(est_sec))
            return True

    def release(self, lane: Lane) -> None:
        with self._lock:
            lane.job = None
            lane.started_at = None
            lane.est_sec = 0.0
            lane.jobs_run += 1
        self.invalidate()

    def make_room(self, lane: Lane) -> list[str]:
        """Evicta o que os OUTROS gestores têm nas GPUs desta lane.

        O gestor de uma lane só evicta os seus backends. Um modelo dividido
        ocioso (noutro gestor) ocupa VRAM que a lane vê como usada mas não
        consegue libertar — e vice-versa quando é o split que precisa das GPUs.
        Só toca em lanes ociosas: quem chama já reservou ``lane``.

        Returns:
            Lanes que foram esvaziadas.
        """
        drained: list[str] = []
        for other in self.all_lanes():
            if other is lane or not other.overlaps(lane):
                continue
            with self._lock:
                if other.job is not None:
                    continue
            # Uma lane de dispositivo só cede a um split; um split cede a qualquer uma.
            if lane.kind == KIND_DEVICE and other.kind == KIND_DEVICE:
                continue
            try:
                if other.manager.loaded_names() and other.manager.evict_all():
                    drained.append(other.lane_id)
            except Exception as exc:
                _logger.warn(f"[vramd] make_room: evict em {other.lane_id} falhou: {exc}")
        if drained:
            self.invalidate()
        return drained

    # -- agregados para os comandos do servidor ------------------------------

    def loaded_names(self) -> list[str]:
        """Backends carregados em qualquer lane (sem repetidos)."""
        seen: list[str] = []
        for lane in self.all_lanes():
            with contextlib.suppress(Exception):
                for name in lane.manager.loaded_names():
                    if name not in seen:
                        seen.append(name)
        return seen

    def lanes_with(self, backend: str) -> list[Lane]:
        out: list[Lane] = []
        for lane in self.all_lanes():
            with contextlib.suppress(Exception):
                if lane.manager.is_loaded(backend):
                    out.append(lane)
        return out

    def status(self) -> list[dict[str, Any]]:
        """Bloco ``lanes:`` do ``status``."""
        rows: list[dict[str, Any]] = []
        for lane in self.all_lanes():
            with self._lock:
                job = lane.job
            try:
                loaded = list(lane.manager.loaded_names())
            except Exception:
                loaded = []
            rows.append(
                {
                    "lane": lane.lane_id,
                    "kind": lane.kind,
                    "devices": list(lane.devices),
                    "pinned": lane.pinned,
                    "free_mib": self.free_mib(lane),
                    "total_mib": self.total_mib(lane),
                    "loaded": loaded,
                    "busy": job is not None,
                    "job_id": getattr(job, "job_id", None),
                    "backend": getattr(job, "backend", None),
                    "remaining_sec": round(self.remaining_sec(lane), 1) if job is not None else None,
                    "jobs_run": lane.jobs_run,
                }
            )
        return rows
