"""WorkerPool — despacha jobs da fila com ``MAX_INFLIGHT`` gerações paralelas.

Default ``MAX_INFLIGHT=1``: uma geração de cada vez na GPU. Com ``MAX_INFLIGHT>1``,
só arranca jobs em paralelo se a VRAM livre couber o footprint do candidato.

``VRAM_INSUFFICIENT`` transitória (livre < pico mas pico ≤ VRAM total): evict +
backoff + requeue até ``MAX_VRAM_RETRIES``. Pico > total GPU → falha imediata.

Com várias GPUs ou peers configurados o despacho passa a ser **roteado**: cada
job pergunta ao :class:`~vramd.placement.RouteEngine` onde acaba mais cedo
(GPU local, réplica noutra GPU, modelo dividido, outro nó) e a pool reserva
esse recurso. Com uma GPU e sem peers o caminho é o de sempre.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vramd.errors import GenerationAborted
from vramd.logging import Logger

from . import protocol as P
from .backend_manager import BackendManager
from .job_queue import Job, JobQueue
from .router import ACTION_REJECT, ACTION_WAIT, ROUTE_LOCAL, ROUTE_REMOTE, ROUTE_SPLIT, Decision
from .scheduler import AffinityScheduler

_logger = Logger()

# Jobs da fila avaliados por passo de claim roteado. Cada avaliação lê o estado
# das lanes; uma fila de centenas não deve custar centenas de fotografias.
_ROUTE_WINDOW = 16
ERR_PEER_UNREACHABLE = "PEER_UNREACHABLE"


@dataclass
class _Claim:
    """Um job despachado pelo router e o recurso que ficou reservado para ele."""

    job: Job
    decision: Decision
    kind: str
    lane: Any = None
    peer: Any = None


def _is_vram_insufficient(result: dict[str, Any]) -> bool:
    if result.get("error_code") == P.ERR_VRAM_INSUFFICIENT:
        return True
    err = str(result.get("error") or "").lower()
    return "vram insuficiente" in err or "out of memory" in err


def _is_worker_dead(result: dict[str, Any]) -> bool:
    """Worker subprocesso morto / load incompleto — requeue transitório."""
    if result.get("error_code") == P.ERR_WORKER_DEAD:
        return True
    err = str(result.get("error") or "").lower()
    return (
        "não está vivo" in err
        or "nao esta vivo" in err
        or "worker fechou stdout" in err
        or "eof no load" in err
        or "worker wedged" in err
    )


def _vram_retry_worthwhile(result: dict[str, Any], total_mib: int | None) -> bool:
    """False se o pico nunca cabe na GPU (sem retry)."""
    peak = result.get("peak_mib")
    if peak is None or total_mib is None:
        return True
    try:
        return int(peak) <= int(total_mib)
    except (TypeError, ValueError):
        return True


def _vram_backoff_sec(retries_done: int) -> float:
    """Backoff exponencial após N retries já consumidos (0-based next attempt)."""
    base = max(0.05, float(P.VRAM_RETRY_BASE_SEC))
    cap = max(base, float(P.VRAM_RETRY_MAX_SEC))
    return min(cap, base * (2 ** max(0, retries_done)))


class WorkerPool:
    """Pool de workers que puxam jobs da ``JobQueue`` via ``AffinityScheduler``."""

    def __init__(
        self,
        queue: JobQueue,
        manager: BackendManager,
        scheduler: AffinityScheduler | None = None,
        *,
        max_inflight: int = P.MAX_INFLIGHT,
        verbose: bool = False,
        query_free_mib: Callable[[], int | None] | None = None,
        engine: Any = None,
        remote_executor: Any = None,
    ) -> None:
        self.queue = queue
        self.manager = manager
        # Router (multi-GPU / peers). ``None`` ou inactivo = despacho clássico.
        self.engine = engine
        self.remote_executor = remote_executor
        self._route_lock = threading.Lock()
        self.scheduler = scheduler if scheduler is not None else AffinityScheduler()
        self.max_inflight = max(1, max_inflight)
        self.verbose = verbose
        self._query_free_mib = query_free_mib
        self._threads: list[threading.Thread] = []
        self._running = False
        self._stop = threading.Event()
        self._affinity_hits = 0
        # ``+=`` em int não é atómico: com MAX_INFLIGHT>1, dois dispatches
        # simultâneos perdiam contagens (métrica mentia para baixo).
        self._metrics_lock = threading.Lock()

    def _log(self, msg: str) -> None:
        if self.verbose:
            _logger.info(f"[vramd-worker] {msg}")

    def _free_mib(self) -> int | None:
        if self._query_free_mib is not None:
            return self._query_free_mib()
        try:
            from vramd.gpu import query_gpu_free_mib

            return query_gpu_free_mib()
        except Exception:
            return None

    def _total_mib(self) -> int | None:
        try:
            from vramd.gpu import query_gpu_snapshot

            snap = query_gpu_snapshot()
            if snap is None:
                return None
            total = getattr(snap, "total_mib", None)
            if total is None:
                total = getattr(snap, "memory_total_mib", None)
            return int(total) if total is not None else None
        except Exception:
            return None

    def _evict_and_clear(self, manager: Any = None) -> None:
        mgr = manager if manager is not None else self.manager
        with contextlib.suppress(Exception):
            mgr.evict_all()
        clear = getattr(mgr, "_clear_cache", None)
        if callable(clear):
            with contextlib.suppress(Exception):
                clear()

    def _sleep_interruptible(self, job: Job, seconds: float) -> bool:
        """Sleep em fatias; ``False`` se cancel pedido."""
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if job.cancel_requested or self._stop.is_set():
                return False
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        return not job.cancel_requested

    @property
    def routed(self) -> bool:
        """True se o despacho passa pelo router (várias GPUs, lane fixada ou peers)."""
        return self.engine is not None and bool(self.engine.active)

    def capacity(self) -> int:
        """Jobs em simultâneo: no modo roteado, um por GPU mais os slots dos peers."""
        if not self.routed:
            return self.max_inflight
        peers = getattr(self.engine, "peers", None)
        slots = int(getattr(peers, "forward_slots", 0) or 0) if peers else 0
        return max(1, len(self.engine.lanes.device_lanes) + slots)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop.clear()
        for i in range(self.capacity()):
            t = threading.Thread(target=self._loop, name=f"vramd-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        self._log(f"Arrancados {self.capacity()} worker(s){' (roteado)' if self.routed else ''}")

    def stop(self) -> None:
        self._running = False
        self._stop.set()
        self.queue.notify()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    def _loop(self) -> None:
        while self._running and not self._stop.is_set():
            if not self.queue.wait_for_work(timeout=0.5):
                continue
            if self.routed:
                claim = self._claim_routed()
                if claim is None:
                    self.queue.wait_for_slot(timeout=0.5)
                    continue
                self._run_claim(claim)
                continue
            job = self._claim_next()
            if job is None:
                # Há trabalho mas nada despachável (slot cheio ou VRAM não
                # chega): esperar por mudança de estado em vez de spin — cada
                # iteração de spin consultava NVML/nvidia-smi e inflacionava
                # affinity_hits/affinity_cuts sem dispatch real.
                self.queue.wait_for_slot(timeout=0.5)
                continue
            self._run_job(job)

    def _backend_peak_mib(self, name: str, request: dict[str, Any] | None = None) -> int:
        """Pico pesos+activação+safety (não o YAML estático)."""
        try:
            quant, mem_eff, group_off, streams = self.manager.resolve_peak_params(name, request or {})
            return int(
                self.manager.peak_vram_mib(
                    name,
                    quant_mode=quant,
                    memory_efficient=mem_eff,
                    group_offload=group_off,
                    streams_on_load=streams,
                    footprint_key=(request or {}).get("footprint_key"),
                )
            )
        except Exception:
            try:
                return int(self.manager._registry.descriptor(name).vram_mib)
            except Exception:
                return 0

    def _fits_parallel(self, job: Job) -> bool:
        """Com already-inflight>0, só permite se VRAM livre couber o pico/headroom."""
        if self.queue.inflight <= 0:
            return True
        if self.max_inflight <= 1:
            return False

        loaded = set(self.manager.loaded_names())
        free = self._free_mib()

        # Hot = backend carregado **e** load_shape compatível (senão = cold reload).
        if job.backend in loaded and self._job_is_hot(job):
            if free is None:
                return True
            try:
                _q, mem_eff, group_off, streams = self.manager.resolve_peak_params(job.backend, job.request)
                needed = self.manager.activation_headroom_mib(
                    job.backend,
                    quant_mode=_q,
                    memory_efficient=mem_eff,
                    group_offload=group_off or streams,
                    footprint_key=job.request.get("footprint_key"),
                )
            except Exception:
                needed = 512
            return free >= needed

        if free is None:
            # Sem leitura VRAM (NVML/smi): não arriscar segundo cold load.
            return False
        return free >= self._backend_peak_mib(job.backend, job.request)

    def _job_is_hot(self, job: Job) -> bool:
        """Backend loaded + load_shape match (duck-typed para mocks de teste)."""
        fn = getattr(self.manager, "shape_matches_loaded", None)
        if callable(fn):
            return bool(fn(job.backend, job.request))
        return job.backend in set(self.manager.loaded_names())

    def _claim_next(self) -> Job | None:
        jobs = self.queue.queued_jobs()
        if not jobs:
            return None
        # Já no máximo efectivo de threads; cada thread só pega se cabe.
        if self.queue.inflight >= self.max_inflight:
            return None
        picked = self.scheduler.pick_next(
            jobs,
            self.manager.loaded_names(),
            is_hot=self._job_is_hot,
        )
        if picked is None:
            return None
        if not self._fits_parallel(picked):
            self._log(
                f"Skip parallel job {picked.job_id[:8]} backend={picked.backend!r} "
                f"(VRAM insuficiente ou free desconhecido; inflight={self.queue.inflight})"
            )
            return None
        # Cap revalidado atomicamente no take(): o check aqui é só fast-path —
        # entre ele e o take() outra thread pode roubar o último slot (TOCTOU).
        job = self.queue.take(picked.job_id, max_inflight=self.max_inflight)
        if job is None:
            # Slot entretanto ocupado — nada para despachar.
            return None
        if self._job_is_hot(job):
            # Métrica: dispatches reais para backend já quente (não avaliações).
            with self._metrics_lock:
                self._affinity_hits += 1
        return job

    # ------------------------------------------------------------------
    # Despacho roteado (multi-GPU / peers)
    # ------------------------------------------------------------------

    def _fallback_lane(self) -> Any:
        """Primeira GPU livre — para onde vai um job que o router rejeitou.

        "Nenhuma rota" não é falha silenciosa: o job corre na lane e é a
        admissão do gestor que produz o erro canónico (``VRAM_INSUFFICIENT``
        com pico, hint e retries), como sempre fez.
        """
        lanes = self.engine.lanes
        for lane in lanes.device_lanes:
            if not lanes.is_busy(lane):
                return lane
        return None

    def _claim_routed(self) -> _Claim | None:
        """Escolhe o próximo job despachável e reserva o recurso da sua rota.

        Percorre a fila por prioridade e ordem de chegada. Um job cuja melhor
        rota está ocupada **espera** — e o trabalho que ele vai trazer fica
        somado a essa GPU/peer para os jobs seguintes. É isso que abre réplicas
        num batch: o terceiro job de um modelo vê a GPU quente com dois à
        frente e, se o load numa GPU livre for mais curto que essa fila, vai
        para lá.
        """
        engine = self.engine
        lanes = engine.lanes
        peers = getattr(engine, "peers", None)
        with self._route_lock:
            jobs = [j for j in self.queue.queued_jobs() if j.state == P.JOB_QUEUED and not j.cancel_requested]
            if not jobs:
                return None
            jobs.sort(key=lambda j: (P.PRIORITY_RANK.get(j.priority, 99), j.seq))
            capacity = self.capacity()
            if self.queue.inflight >= capacity:
                return None
            lane_extra: dict[str, float] = {}
            peer_extra: dict[str, float] = {}
            for job in jobs[:_ROUTE_WINDOW]:
                try:
                    decision = engine.decide(
                        job.backend,
                        job.request,
                        forwarded=job.forwarded,
                        lane_extra=lane_extra,
                        peer_extra=peer_extra,
                        exclude_peers=job.route_excluded,
                    )
                except Exception as exc:
                    _logger.warn(f"[vramd-router] decisão falhou para {job.job_id[:8]}: {exc}")
                    continue
                best = decision.best
                claim: _Claim | None = None
                if decision.action == ACTION_REJECT or best is None:
                    lane = self._fallback_lane()
                    if lane is not None and lanes.reserve(lane, job, est_sec=0.0):
                        claim = _Claim(job, decision, ROUTE_LOCAL, lane=lane)
                elif decision.action == ACTION_WAIT:
                    work = best.load_sec + best.gen_sec
                    if best.kind == ROUTE_REMOTE:
                        name = best.target.removeprefix("peer:")
                        peer_extra[name] = peer_extra.get(name, 0.0) + work
                    else:
                        for device in best.devices:
                            key = f"gpu{device}"
                            lane_extra[key] = lane_extra.get(key, 0.0) + work
                    continue
                elif best.kind == ROUTE_REMOTE:
                    name = best.target.removeprefix("peer:")
                    state = peers.get(name) if peers else None
                    if state is not None and peers.reserve(name):
                        claim = _Claim(job, decision, ROUTE_REMOTE, peer=state)
                else:
                    lane = lanes.span_for(best.devices) if best.kind == ROUTE_SPLIT else lanes.lane(best.target)
                    if lane is not None and lanes.reserve(lane, job, est_sec=best.load_sec + best.gen_sec):
                        claim = _Claim(job, decision, best.kind, lane=lane)
                if claim is None:
                    continue
                taken = self.queue.take(job.job_id, max_inflight=capacity)
                if taken is None:
                    self._release_claim(claim, ok=False)
                    continue
                engine.record_run(decision)
                if best is not None and best.hot:
                    with self._metrics_lock:
                        self._affinity_hits += 1
                return claim
        return None

    def _release_claim(self, claim: _Claim, *, ok: bool = True) -> None:
        if claim.lane is not None:
            self.engine.lanes.release(claim.lane)
        if claim.peer is not None:
            peers = getattr(self.engine, "peers", None)
            if peers:
                peers.release(claim.peer.spec.name, ok=ok)

    def _run_claim(self, claim: _Claim) -> None:
        """Corre um job na rota reservada e liberta-a no fim, aconteça o que acontecer."""
        job = claim.job
        best = claim.decision.best
        route: dict[str, Any] = {
            "kind": claim.kind,
            "target": claim.lane.lane_id if claim.lane is not None else f"peer:{claim.peer.spec.name}",
            "devices": list(claim.lane.devices) if claim.lane is not None else [],
            "explanation": claim.decision.explanation,
        }
        if best is not None:
            route["hot"] = best.hot
            route["replica"] = best.replica
            route["est_sec"] = round(best.total_sec, 1)
        job.route = route
        _logger.info(
            f"[vramd-router] job {job.job_id[:8]} ({job.backend}) → {route['target']}: {claim.decision.explanation}",
            console=self.verbose,
        )
        ok = False
        try:
            if claim.kind == ROUTE_REMOTE:
                self._run_job(job, executor=lambda req: self._execute_remote(job, claim, req), manager=None)
            else:
                lane = claim.lane
                drained = self.engine.lanes.make_room(lane)
                if drained:
                    self._log(f"make_room para {lane.lane_id}: esvaziadas {drained}")
                self._run_job(
                    job,
                    executor=lambda req: lane.manager.generate(job.backend, self._lane_request(req, claim)),
                    manager=lane.manager,
                    free_fn=lambda: self.engine.lanes.free_mib(lane, fresh=True),
                    total_fn=lambda: self.engine.lanes.total_mib(lane),
                )
            ok = True
        finally:
            self._release_claim(claim, ok=ok)
            self.queue.notify()

    @staticmethod
    def _lane_request(req: dict[str, Any], claim: _Claim) -> dict[str, Any]:
        """Ajusta ``gpu_ids`` ao que o worker da lane vê.

        O worker está fixado por ``CUDA_VISIBLE_DEVICES``: para ele as GPUs da
        lane são ``0..k-1``, sejam quais forem os índices físicos. Num split
        recebe também a quota por GPU (``max_memory_mib``) que o router calculou.
        """
        lane = claim.lane
        if lane is None or not getattr(lane, "pinned", False):
            return req
        out = dict(req)
        best = claim.decision.best
        if claim.kind == ROUTE_SPLIT:
            out["gpu_ids"] = list(range(len(lane.devices)))
            shares = best.shares_mib if best is not None else {}
            if shares:
                out["max_memory_mib"] = {
                    str(index): int(shares[device]) for index, device in enumerate(lane.devices) if device in shares
                }
        elif "gpu_ids" in out:
            out["gpu_ids"] = [0]
        return out

    def _execute_remote(self, job: Job, claim: _Claim, req: dict[str, Any]) -> dict[str, Any]:
        """Encaminha o job para o peer reservado; falha de transporte vira resultado."""
        from .peers import PeerUnreachable

        engine = self.engine
        peer = claim.peer
        name = peer.spec.name
        input_keys, output_keys = engine.io_keys(job.backend)
        if self.remote_executor is None:
            return {"status": P.STATUS_ERROR, "error": "sem executor remoto", "error_code": ERR_PEER_UNREACHABLE}
        try:
            return self.remote_executor.run(
                peer,
                job.backend,
                req,
                input_keys=input_keys,
                output_keys=output_keys,
                on_progress=req.get("_progress"),
                should_abort=req.get("_abort"),
            )
        except PeerUnreachable as exc:
            engine.peers.mark_failed(name, str(exc))
            job.route_excluded.add(name)
            return {
                "status": P.STATUS_ERROR,
                "error": f"peer {name} inacessível: {exc}",
                "error_code": ERR_PEER_UNREACHABLE,
            }

    def _run_job(
        self,
        job: Job,
        *,
        executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        manager: Any = ...,
        free_fn: Callable[[], int | None] | None = None,
        total_fn: Callable[[], int | None] | None = None,
    ) -> None:
        """Corre um job e trata os retries transitórios.

        Sem argumentos é o caminho clássico (gestor único). O despacho roteado
        passa o ``executor`` da rota e o ``manager`` da lane (``None`` numa
        rota remota: não há gestor local a evictar).
        """
        mgr = self.manager if manager is ... else manager
        run = executor if executor is not None else (lambda req: self.manager.generate(job.backend, req))
        free_mib = free_fn if free_fn is not None else self._free_mib
        total_mib = total_fn if total_fn is not None else self._total_mib
        if job.cancel_requested:
            self.queue.finish(
                job,
                {
                    "status": P.STATUS_ERROR,
                    "error": "cancelled before start",
                    "error_code": P.ERR_CANCELLED,
                },
            )
            return

        job.mark_started()
        self._log(f"A correr job {job.job_id[:8]} backend={job.backend!r} cuts={job.affinity_cuts} pri={job.priority}")

        def on_progress(pct: float | None = None, msg: str | None = None) -> None:
            if job.cancel_requested:
                return
            job.report_progress(pct, msg)

        def should_abort() -> bool:
            return bool(job.cancel_requested)

        try:
            req = dict(job.request)
            req["_progress"] = on_progress
            req["_abort"] = should_abort
            req["_job_id"] = job.job_id
            if job.cancel_requested:
                result: dict[str, Any] = {
                    "status": P.STATUS_ERROR,
                    "error": "cancelled before generate",
                    "error_code": P.ERR_CANCELLED,
                }
            else:
                on_progress(0.0, "started")
                result = run(req)
                if job.cancel_requested and result.get("status") == P.STATUS_OK:
                    result = {
                        "status": P.STATUS_ERROR,
                        "error": "cancelled during run",
                        "error_code": P.ERR_CANCELLED,
                    }
                elif result.get("status") != P.STATUS_OK:
                    result.setdefault("error_code", P.ERR_GENERATE_FAILED)
        except GenerationAborted:
            result = {
                "status": P.STATUS_ERROR,
                "error": "cancelled during diffusion",
                "error_code": P.ERR_CANCELLED,
            }
        except Exception as e:
            _logger.warn(f"[vramd-worker] job {job.job_id[:8]} falhou: {e}")
            result = {
                "status": P.STATUS_ERROR,
                "error": str(e),
                "error_code": P.ERR_GENERATE_FAILED,
            }

        # Peer inacessível antes de aceitar o job: voltar à fila — o router
        # escolhe outra rota (o peer que falhou fica excluído para este job).
        if (
            result.get("status") != P.STATUS_OK
            and result.get("error_code") == ERR_PEER_UNREACHABLE
            and not job.cancel_requested
            and job.worker_retries < P.MAX_WORKER_DEAD_RETRIES
        ):
            _logger.warn(f"[vramd-router] job {job.job_id[:8]}: {result.get('error')} — a escolher outra rota.")
            job.route = None
            if self.queue.requeue_running(job, reason="peer inacessível", kind="worker"):
                return

        # Worker morto transitório: limpar + requeue (IdleEvictor race / crash).
        if (
            mgr is not None
            and result.get("status") != P.STATUS_OK
            and _is_worker_dead(result)
            and not job.cancel_requested
            and job.worker_retries < P.MAX_WORKER_DEAD_RETRIES
        ):
            msg = f"worker morto — reload+retry ({job.worker_retries + 1}/{P.MAX_WORKER_DEAD_RETRIES})"
            _logger.warn(f"[vramd-worker] job {job.job_id[:8]} {msg}: {result.get('error')}")
            on_progress(None, msg)
            with contextlib.suppress(Exception):
                mgr.evict(job.backend)
            if self.queue.requeue_running(job, reason="worker dead", kind="worker"):
                return
            if job.cancel_requested:
                result = {
                    "status": P.STATUS_ERROR,
                    "error": "cancelled during worker retry",
                    "error_code": P.ERR_CANCELLED,
                }

        # VRAM transitória: evict + backoff + requeue (não matar o batch).
        if (
            mgr is not None
            and result.get("status") != P.STATUS_OK
            and _is_vram_insufficient(result)
            and not job.cancel_requested
            and _vram_retry_worthwhile(result, total_mib())
            and job.vram_retries < P.MAX_VRAM_RETRIES
            and job.vram_flat_retries < P.VRAM_FLAT_RETRY_MAX
        ):
            wait_sec = _vram_backoff_sec(job.vram_retries)
            free = free_mib()
            peak = result.get("peak_mib")
            msg = (
                f"VRAM insuficiente (livre={free} peak={peak}) — "
                f"evict+espera {wait_sec:.1f}s "
                f"(retry {job.vram_retries + 1}/{P.MAX_VRAM_RETRIES})"
            )
            _logger.warn(f"[vramd-worker] job {job.job_id[:8]} {msg}")
            on_progress(None, msg)
            # Progress-guard: retry só compensa se (a) havia backends evictáveis
            # (libertámos pesos agora) ou (b) a VRAM livre se mexeu (processo
            # externo activo — pode libertar). Livre plana + 0 loaded N vezes
            # seguidas = loop improdutivo (ex.: memória presa pelo próprio
            # servidor) → falhar rápido com hint em vez de 8x30s às cegas.
            had_evictable = bool(mgr.loaded_names())
            self._evict_and_clear(mgr)
            free_after = free_mib()
            moved = free is not None and free_after is not None and abs(free_after - free) > P.VRAM_FLAT_SLACK_MIB
            # Leitura de free desconhecida (sem NVML) → conservador: retry normal.
            if had_evictable or moved or free is None or free_after is None:
                job.vram_flat_retries = 0
            else:
                job.vram_flat_retries += 1
                if job.vram_flat_retries >= P.VRAM_FLAT_RETRY_MAX:
                    _logger.warn(
                        f"[vramd-worker] job {job.job_id[:8]} VRAM plana "
                        f"{job.vram_flat_retries}x seguidas sem nada evictável "
                        f"(livre={free_after} peak={peak}) — falha rápida (loop improdutivo)."
                    )
                else:
                    job.last_vram_free_mib = free_after
            if (
                job.vram_flat_retries < P.VRAM_FLAT_RETRY_MAX
                and self._sleep_interruptible(job, wait_sec)
                and self.queue.requeue_running(job, reason=f"livre={free} peak={peak}")
            ):
                return
            if job.cancel_requested:
                result = {
                    "status": P.STATUS_ERROR,
                    "error": "cancelled during vram wait",
                    "error_code": P.ERR_CANCELLED,
                }

        if _is_vram_insufficient(result) and result.get("status") != P.STATUS_OK:
            result.setdefault("error_code", P.ERR_VRAM_INSUFFICIENT)
            if job.vram_flat_retries >= P.VRAM_FLAT_RETRY_MAX:
                result.setdefault(
                    "hint",
                    (
                        "VRAM livre não subiu após retries e não há backends evictáveis "
                        "(memória pode estar presa por processo externo — ver `nvidia-smi` "
                        "e `vramd status`/`debug` para o breakdown do worker). Não mates PIDs GPU."
                    ),
                )
            else:
                result.setdefault(
                    "hint",
                    (
                        "VRAM livre < pico após retries. Espera `vramd queue` libertar, "
                        "fecha processos GPU externos, ou usa sdnq-int4 / quality fast. "
                        "Não mates PIDs — vê Agents/anti-patterns no README UMS."
                    ),
                )
            result["vram_retries"] = job.vram_retries

        if _is_worker_dead(result) and result.get("status") != P.STATUS_OK:
            result.setdefault("error_code", P.ERR_WORKER_DEAD)
            result.setdefault(
                "hint",
                (
                    "Worker subprocesso morreu durante o job. UMS já retentou; "
                    "se persistir: `vramd respawn <backend>` / `vramd doctor`."
                ),
            )
            result["worker_retries"] = job.worker_retries

        self.queue.finish(job, result)


# Tipo auxiliar para testes / DI.
GenerateFn = Callable[[str, dict[str, Any]], dict[str, Any]]
