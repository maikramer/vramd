"""VramdServer — servidor único que roteia pedidos para backends.

Um único processo detém toda a VRAM e escuta num único Unix socket. Pedidos de
geração passam por uma **fila inteligente** (prioridade + afinidade VRAM) e um
worker pool com ``MAX_INFLIGHT`` (default 1).

Protocolo: ver ``protocol.py``. Ciclo de vida: ver ``serve_forever``.

Retrocompatibilidade: o socket único (``vramd.sock``) é descoberto por
``vramd.client.discover_active_sockets`` como qualquer per-tool
legacy server. ``ensure_vram_available`` envia ``ensure-vram`` ao vramd quando
disponível, caindo para o comportamento legacy caso contrário.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

from vramd.client import _ensure_server_dir, _pid_path
from vramd.logging import Logger

from . import __version__, frames, host, transport
from . import protocol as P
from .backend_manager import BackendManager
from .blobs import BlobError
from .dispatcher import WorkerPool
from .history import HISTORY_FILENAME, JobHistory
from .hooks import (
    EVENT_DRIFT,
    EVENT_JOB_CANCELLED,
    EVENT_JOB_DONE,
    EVENT_JOB_FAILED,
    EVENT_SHUTDOWN,
    EVENT_ZERO,
    HookRunner,
    load_hooks,
)
from .job_queue import Job, JobQueue, QueueFullError
from .lanes import KIND_SPAN, Lane, LaneSet, lane_env, lane_id_for
from .learn import PeakLearningStore, PeakTracker
from .models import ModelError, ModelManager, ModelSpec
from .peers import PeerSet, RemoteExecutor
from .placement import RouteEngine
from .process_guard import SingletonLock, lock_path_for
from .registry import Registry
from .remote import RemoteConfig, RemoteError, RemoteSandbox
from .scheduler import AffinityScheduler
from .sessions import SessionError, SessionManager

_logger = Logger()

# Teto de conexões servidas em paralelo (thread-per-connection). Sem isto, um
# cliente que abre centenas de ligações esgota o processo em threads — cada uma
# com a sua stack. Acima do teto: resposta SERVER_BUSY e close imediato.
MAX_CONCURRENT_CLIENTS = P._env_int("VRAMD_MAX_CLIENTS", 64)
# Deadline TOTAL para ler a linha de request (não por recv): um slowloris que
# goteja 1 byte a cada 100s prendia uma thread por tempo indefinido com o
# timeout-per-recv de 600s.
REQUEST_READ_TIMEOUT_SEC = P._env_float("VRAMD_REQUEST_READ_TIMEOUT_SEC", 30.0)
# Ligações persistentes (protocolo v2): tempo de idle entre pedidos. Um job
# GPU pode demorar minutos; o cliente não envia nada durante esse tempo, por
# isso 300s é folgado. 0 desliga o modo persistente (fecha após cada resposta).
CLIENT_IDLE_TIMEOUT_SEC = P._env_float("VRAMD_CLIENT_IDLE_TIMEOUT_SEC", 300.0)


class VramdServer:
    """Servidor único de VRAM — fila inteligente + BackendManager.

    Args:
        registry: Registry de backends. Se ``None``, carrega do ``backends.yaml`` default.
        socket_path: Path do socket. Se ``None``, usa ``DEFAULT_SOCKET_PATH``.
        idle_timeout_min: Minutos de idle (sem pedidos) antes de self-shutdown.
        verbose: Logs detalhados.
    """

    def __init__(
        self,
        registry: Registry | None = None,
        *,
        socket_path: Path | str | None = None,
        idle_timeout_min: int = P.DEFAULT_IDLE_TIMEOUT_MIN,
        idle_evict_sec: float = P.IDLE_EVICT_SEC,
        worker_shutdown_sec: float = P.WORKER_IDLE_SHUTDOWN_SEC,
        max_queue_depth: int = P.MAX_QUEUE_DEPTH,
        max_inflight: int = P.MAX_INFLIGHT,
        max_affinity_cuts: int = P.MAX_AFFINITY_CUTS,
        verbose: bool = False,
        query_free_mib: Any = None,
        clear_vram: Any = None,
        subprocess_pool: Any = None,
        listen: tuple[str, int] | None = None,
        token: str | None = None,
        gpu_devices: list[int] | None = None,
        peers: Any = None,
        lane_factory: Any = None,
        node_name: str | None = None,
        allow_client_code: bool = False,
        allow_client_env: bool = False,
    ) -> None:
        self.registry = registry if registry is not None else Registry()
        # Hooks de eventos (``~/.config/vramd/hooks.yaml``): criados ANTES do
        # manager para que este possa disparar on_evict desde o primeiro evict.
        # Config malformada falha no arranque — um hook mal escrito que
        # corresse em silêncio seria pior que não arrancar.
        self.hooks = HookRunner(load_hooks())
        # Pool de subprocessos para backends com 'tool:' definido (Fase 3-4).
        # Injectável para testes; por defeito cria um SubprocessWorkerPool real
        # (fica inerte se nenhum backend usar subprocesso).
        # Multi-GPU: com ``gpu_devices`` cada GPU ganha uma lane com o seu
        # gestor e os workers fixados nela. Sem ele (default) há uma lane não
        # fixada — o supervisor de uma GPU de sempre. Quem detecta as GPUs é o
        # CLI (``vramd start``), nunca o construtor: um teste não pode mudar de
        # comportamento por correr numa máquina com duas placas.
        devices = tuple(dict.fromkeys(int(d) for d in (gpu_devices or ())))
        self._gpu_devices = devices
        self._clear_vram_arg = clear_vram
        self._idle_cfg = (float(idle_evict_sec), float(worker_shutdown_sec))
        if devices and query_free_mib is None:
            query_free_mib = self._device_free_query((devices[0],))
        if subprocess_pool is None:
            from .subprocess_pool import SubprocessWorkerPool

            subprocess_pool = SubprocessWorkerPool(env_overrides=lane_env((devices[0],)) if devices else None)
        self.manager = BackendManager(
            self.registry,
            query_free_mib=query_free_mib,
            clear_vram=clear_vram,
            subprocess_pool=subprocess_pool,
            reap_strays=self.reap_strays,
            on_evict=lambda name, payload: self.hooks.dispatch("on_evict", payload),
        )
        self.socket_path = Path(socket_path) if socket_path else P.DEFAULT_SOCKET_PATH
        self.ppid_path = _pid_path(self.socket_path)
        self.idle_timeout_sec = idle_timeout_min * 60
        self.verbose = verbose

        wal_path = self.socket_path.parent / P.WAL_FILENAME
        self.queue = JobQueue(
            max_depth=max_queue_depth,
            stats=self.manager.stats,
            wal_path=wal_path,
            on_finish=self._on_job_finished,
        )
        self.scheduler = AffinityScheduler(
            max_cuts=max_affinity_cuts,
            starvation_timeout_sec=P.STARVATION_TIMEOUT_SEC,
        )
        # Lanes (GPUs), peers (outros nós) e o router que escolhe entre eles.
        self._lane_factory = lane_factory
        primary_devices = (devices[0],) if devices else (0,)
        primary = Lane(
            lane_id=lane_id_for(primary_devices),
            devices=primary_devices,
            manager=self.manager,
            pinned=bool(devices),
            query_free_mib=self.manager._free_mib,
            query_total_mib=self._device_total_query(primary_devices),
        )
        extra = [self._make_lane((d,)) for d in devices[1:]]
        self.lanes = LaneSet(
            [primary, *extra],
            registry=self.registry,
            span_factory=self._make_lane if devices else None,
        )
        self.peers = peers if isinstance(peers, PeerSet) else PeerSet(list(peers or []))
        # OOMs reais por tier/tamanho de GPU — ao lado do socket, como o
        # histórico: um supervisor de teste nunca suspende tiers do real.
        from .tier_health import HEALTH_FILENAME, TierHealth

        self.tier_health = TierHealth(self.socket_path.parent / HEALTH_FILENAME)
        self.router = RouteEngine(
            self.registry,
            self.lanes,
            self.peers,
            history_reader=lambda: self.history.read(limit=500),
            health=self.tier_health,
        )
        self.remote_executor = RemoteExecutor(self.peers, node_name=node_name)
        # Prewarm preditivo: o cliente avisa o que vem aí (``forecast``) e o
        # modelo carrega na GPU que o router escolheria, antes do job chegar.
        from .forecast import Prewarmer

        self.prewarmer = Prewarmer(self.router, self._prewarm_load)
        self.workers = WorkerPool(
            self.queue,
            self.manager,
            self.scheduler,
            max_inflight=max_inflight,
            verbose=verbose,
            engine=self.router,
            remote_executor=self.remote_executor,
        )
        # Histórico persistente de jobs terminados (``vramd history``): ao lado
        # do socket, como o WAL — um supervisor de teste nunca escreve no do real.
        self.history = JobHistory(self.socket_path.parent / HISTORY_FILENAME)

        from .idle_evictor import IdleEvictor

        self.idle_evictor = IdleEvictor(
            self.manager,
            idle_timeout_sec=idle_evict_sec,
            check_interval_sec=P.IDLE_EVICT_CHECK_SEC,
            worker_shutdown_sec=worker_shutdown_sec,
            health_check_sec=P.WORKER_HEALTH_CHECK_SEC,
            queue=self.queue,
        )

        # Aprendizagem contínua de picos: observa a VRAM real dos jobs em
        # curso, mede drift vs o declarado e alimenta ``vramd learn``. Thread
        # observadora — nunca interfere com a fila (intervalo 0 desliga).
        self.learner = PeakTracker(
            self.queue,
            self.manager,
            store=PeakLearningStore(),
            on_drift=lambda report: self.hooks.dispatch(
                EVENT_DRIFT,
                {
                    "backend": report.backend,
                    "verdict": report.verdict,
                    "declared_peak_mib": report.declared_peak_mib,
                    "observed_p95_mib": report.observed_p95_mib,
                    "suggested_mib": report.suggested_mib,
                    "hint": "vramd learn para ver; vramd learn --apply para corrigir",
                },
            ),
        )

        self._server_sock: socket.socket | None = None
        self._bound = False  # True só após bind+listen OK (protege cleanup em double-start)
        # Modo remoto (``--listen``): TCP numa interface explícita. O token é
        # estável (ficheiro/env) para os clientes da rede não perderem acesso
        # quando o supervisor reinicia; sem token gera-se um por arranque.
        self._listen = listen
        self._token = token
        # Sandbox do modo remoto: um diretório por job para staged inputs e
        # outputs. Fica inerte (e vazio) quando ninguém usa os comandos remotos.
        self.remote = RemoteSandbox(RemoteConfig.from_env(self.socket_path.parent))
        # Cache HF normal (gerido aqui): `vramd models` e os `models:` das
        # sessões usam a mesma instância para pins/GC coerentes.
        self.models = ModelManager()
        # Sessões de código do cliente: bundles content-addressed + envs + um
        # worker por (sessão, backend). Inerte sem `--allow-client-code`.
        self.sessions = SessionManager(
            cache_dir=self.socket_path.parent,
            registry=self.registry,
            manager=self.manager,
            job_queue=self.queue,
            blobs=self.remote.blobs,
            allow_client_code=bool(allow_client_code),
            allow_client_env=bool(allow_client_env),
            models=self.models,
        )
        self._endpoint_info: transport.EndpointInfo = transport.EndpointInfo(transport=transport.transport_mode())
        self._auth_token: str | None = None
        self._running = False
        self._last_activity = time.monotonic()
        self._requests_served = 0
        self._served_lock = threading.Lock()  # `_requests_served += 1` de N threads perde contagens
        self._pid = os.getpid()
        # Singleton: só um supervisor por socket. flock é libertado pelo kernel
        # na morte do processo — não há pid-file stale a limpar.
        self._singleton = SingletonLock(lock_path_for(self.socket_path))

    # ------------------------------------------------------------------
    # Lanes (multi-GPU)
    # ------------------------------------------------------------------

    @staticmethod
    def _device_free_query(devices: tuple[int, ...]) -> Any:
        """Leitura de VRAM livre somada das GPUs dadas (``None`` se alguma falhar)."""

        def query() -> int | None:
            from .gpu import query_gpu_free_mib

            values = [query_gpu_free_mib(d) for d in devices]
            return None if any(v is None for v in values) else int(sum(v for v in values if v is not None))

        return query

    @staticmethod
    def _device_total_query(devices: tuple[int, ...]) -> Any:
        def query() -> int | None:
            from .gpu import query_gpu_snapshot

            total = 0
            for device in devices:
                snap = query_gpu_snapshot(device)
                if snap is None:
                    return None
                total += int(snap.total_mib)
            return total

        return query

    def _make_lane(self, devices: tuple[int, ...]) -> Lane:
        """Lane extra (outra GPU) ou de split (várias GPUs), com gestor próprio."""
        devices = tuple(devices)
        if self._lane_factory is not None:
            lane: Lane = self._lane_factory(devices)
            return lane
        from .idle_evictor import IdleEvictor
        from .subprocess_pool import SubprocessWorkerPool

        lane_id = lane_id_for(devices)
        suffix = lane_id.replace(":", "-")
        pool = SubprocessWorkerPool(
            env_overrides=lane_env(devices),
            # Um log por (backend, lane): duas réplicas não escrevem no mesmo ficheiro.
            log_path_fn=lambda backend, _suffix=suffix: host.worker_log_path(f"{backend}@{_suffix}"),
        )
        free_query = self._device_free_query(devices)
        manager = BackendManager(
            self.registry,
            query_free_mib=free_query,
            clear_vram=self._clear_vram_arg,
            subprocess_pool=pool,
            reap_strays=self.reap_strays,
            on_evict=lambda name, payload, _lane=lane_id: self.hooks.dispatch("on_evict", {**payload, "lane": _lane}),
        )
        # Estatísticas partilhadas: os tempos médios de um backend valem para
        # todas as réplicas, e ``vramd stats`` mostra um total, não N parciais.
        manager.stats = self.manager.stats
        idle_evict_sec, worker_shutdown_sec = self._idle_cfg
        evictor = IdleEvictor(
            manager,
            idle_timeout_sec=idle_evict_sec,
            check_interval_sec=P.IDLE_EVICT_CHECK_SEC,
            worker_shutdown_sec=worker_shutdown_sec,
            health_check_sec=P.WORKER_HEALTH_CHECK_SEC,
            queue=getattr(self, "queue", None),
        )
        if getattr(self, "_running", False):
            evictor.start()
        return Lane(
            lane_id=lane_id,
            devices=devices,
            manager=manager,
            kind=KIND_SPAN if len(devices) > 1 else "device",
            pinned=True,
            query_free_mib=free_query,
            query_total_mib=self._device_total_query(devices),
            idle_evictor=evictor,
        )

    def _prewarm_load(self, lane: Lane, backend: str, request: Any) -> None:
        """Carrega ``backend`` na lane com o tier e as GPUs que o job real teria."""
        from .placement import apply_hw_tiers

        total = self.lanes.total_mib(lane)
        banned = self.tier_health.banned_ranks(backend, total)
        req, _tier = apply_hw_tiers(dict(request or {}), total, banned=banned)
        keys = lane.manager.load_keys_for(backend)
        load_kwargs = {k: v for k, v in req.items() if k in keys}
        if lane.pinned and "gpu_ids" in load_kwargs:
            load_kwargs["gpu_ids"] = list(range(len(lane.devices)))
        lane.manager.ensure_loaded(backend, **load_kwargs)

    def _extra_lanes(self) -> list[Lane]:
        return [lane for lane in self.lanes.all_lanes() if lane.manager is not self.manager]

    def _loaded_names(self) -> list[str]:
        """Backends carregados em qualquer lane (a primária primeiro)."""
        return self.lanes.loaded_names()

    def _evict_everywhere(self, backend: str, *, hard: bool = False) -> bool:
        evicted = False
        for lane in self.lanes.all_lanes():
            with contextlib.suppress(Exception):
                evicted = bool(lane.manager.evict(backend)) or evicted
                if hard:
                    # ``evict`` só descarrega os pesos: o worker fica com o
                    # contexto CUDA/allocator e a VRAM não sobe o suficiente
                    # para a fase seguinte (text2d→text3d era recusado com
                    # ~5 GiB presos). O respawn lazy mata o processo; o próximo
                    # generate arranca um worker fresco.
                    lane.manager.respawn(backend, lazy=True)
        self.lanes.invalidate()
        return evicted

    def _evict_all_lanes(self) -> int:
        count = 0
        for lane in self.lanes.all_lanes():
            with contextlib.suppress(Exception):
                count += int(lane.manager.evict_all() or 0)
        self.lanes.invalidate()
        return count

    def _topology(self) -> dict[str, Any]:
        """O que um cliente precisa para alimentar o router (resposta de ``topology``).

        ``capacity`` é quantos jobs correm ao mesmo tempo (uma geração por GPU
        mais os slots de encaminhamento para peers saudáveis): um batch que
        mantém pelo menos isto em voo nunca deixa uma GPU parada.
        ``gpu_sizes_mib`` são os tamanhos distintos de GPU onde um job pode
        cair — locais e dos peers — e são esses que o cliente usa para calcular
        os ``hw_tiers``.
        """
        lanes = [
            {
                "lane": lane.lane_id,
                "devices": list(lane.devices),
                "total_mib": self.lanes.total_mib(lane),
                "busy": self.lanes.is_busy(lane),
            }
            for lane in self.lanes.device_lanes
        ]
        peers = []
        peer_slots = 0
        sizes: set[int] = {int(row["total_mib"]) for row in lanes if isinstance(row["total_mib"], int)}
        for state in self.peers.states():
            totals = list(state.gpu_totals_mib)
            peers.append(
                {
                    "name": state.spec.name,
                    "healthy": state.healthy,
                    "gpu_totals_mib": totals,
                    "max_inflight": state.spec.max_inflight,
                }
            )
            if state.healthy:
                peer_slots += state.spec.max_inflight
                sizes.update(int(t) for t in totals)
        capacity = self.workers.capacity()
        return {
            "routed": self.router.active,
            "capacity": capacity,
            "local_capacity": len(self.lanes.device_lanes) if self.workers.routed else self.workers.max_inflight,
            "peer_capacity": peer_slots,
            "lanes": lanes,
            "peers": peers,
            "gpu_sizes_mib": sorted(sizes),
            "features": sorted(P.FEATURES),
        }

    def _routing_status(self) -> dict[str, Any]:
        """Bloco ``routing:`` do status — é o que um peer lê para decidir rotas.

        Cache de 2 s: dashboards e peers fazem poll do status, e sem NVML cada
        leitura de VRAM é um ``nvidia-smi`` (o mesmo cuidado do bug M1).
        """
        now = time.monotonic()
        cached = getattr(self, "_routing_cache", None)
        if cached is not None and (now - cached[0]) < 2.0:
            return {**cached[1], **self.router.status()}
        gpus = [
            {
                "index": lane.devices[0],
                "lane": lane.lane_id,
                "total_mib": self.lanes.total_mib(lane),
                "free_mib": self.lanes.free_mib(lane),
            }
            for lane in self.lanes.device_lanes
        ]
        from . import __version__

        block = {"version": __version__, "gpus": gpus, "estimates": self.router.estimates()}
        self._routing_cache = (now, block)
        return {**block, **self.router.status()}

    def _log(self, msg: str) -> None:
        # Sempre ficheiro; consola só com --verbose.
        _logger.info(f"[vramd] {msg}", console=self.verbose)

    def _on_job_finished(self, job: Any) -> None:
        """Transição terminal de job (via JobQueue) → evento de hook.

        Corre na thread que finalizou o job; o ``HookRunner.dispatch`` é
        non-blocking (threads daemon), pelo que hooks nunca atrasam a fila.
        """
        result = getattr(job, "result", None) or {}
        history = getattr(self, "history", None)
        if history is not None:
            history.record_job(job)
        event = {
            P.JOB_DONE: EVENT_JOB_DONE,
            P.JOB_FAILED: EVENT_JOB_FAILED,
            P.JOB_CANCELLED: EVENT_JOB_CANCELLED,
        }.get(getattr(job, "state", ""), EVENT_JOB_DONE)
        timing = job.timing_dict() if hasattr(job, "timing_dict") else {}
        self.hooks.dispatch(
            event,
            {
                "job_id": job.job_id,
                "backend": job.backend,
                "state": job.state,
                "priority": job.priority,
                "duration_sec": timing.get("total_sec"),
                "queue_wait_sec": timing.get("queue_wait_sec"),
                "generate_sec": timing.get("generate_sec"),
                "error_code": result.get("error_code"),
                "error": result.get("error"),
                "output": result.get("output"),
            },
        )

    # ------------------------------------------------------------------
    # Órfãos (supervisores/workers de runs anteriores)
    # ------------------------------------------------------------------

    def reap_strays(self, *, dry_run: bool = False) -> dict[str, Any]:
        """Mata processos UMS que não são nossos (só seguro com o lock detido).

        Sem o :class:`SingletonLock` não há garantia de que os outros processos
        sejam lixo — pode ser um supervisor legítimo, e matá-lo racaria a fila.
        """
        if not self._singleton.held:
            return {
                "count": 0,
                "reaped": [],
                "vram_mib_freed": 0,
                "skipped": "sem singleton lock — reap inseguro",
            }
        # Import local: os nomes do módulo colidem com estes métodos.
        from .process_guard import reap_strays as _reap_strays

        return _reap_strays(self_pid=self._pid, dry_run=dry_run)

    def stray_report(self) -> dict[str, Any]:
        """Órfãos + VRAM que seguram (informativo; não mata)."""
        from .process_guard import stray_report as _stray_report

        return _stray_report(self_pid=self._pid)

    # ------------------------------------------------------------------
    # Despacho de comandos
    # ------------------------------------------------------------------

    def _resolve_backend(self, request: dict[str, Any]) -> str | None:
        """Resolve o nome do backend a partir do request.

        Ordem: ``backend`` explícito → ``tool`` field → único backend carregado.
        Retorna ``None`` se não for possível determinar.
        """
        backend = request.get("backend")
        if backend:
            return str(backend)
        tool = request.get("tool")
        if tool:
            return str(tool)
        loaded = self._loaded_names()
        if len(loaded) == 1:
            return loaded[0]
        return None

    def _error(
        self,
        error: str,
        *,
        error_code: str,
        hint: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Resposta de erro estruturada para debug."""
        out: dict[str, Any] = {
            "status": P.STATUS_ERROR,
            "error": error,
            "error_code": error_code,
        }
        if hint:
            out["hint"] = hint
        out.update(extra)
        return out

    def _backend_error(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Valida backend; devolve dict de erro ou None se OK. Também define backend resolvido."""
        session_id = request.get("session")
        if session_id:
            session_id = str(session_id)
            session = self.sessions.get(session_id)
            if session is None:
                return self._error(
                    f"sessão desconhecida: {session_id}",
                    error_code=P.ERR_SESSION_UNKNOWN,
                    hint="Abre a sessão com `session-open` antes de submeter jobs.",
                )
            base = str(request.get("backend") or request.get("tool") or "").strip()
            if not base:
                return self._error(
                    "pedido com `session` requer `backend` explícito",
                    error_code=P.ERR_INVALID_REQUEST,
                    hint=f"Backends da sessão: {[spec.name for spec in session.backend_specs]}",
                )
            synthetic = f"{base}@{session_id}"
            if not self.registry.has(synthetic):
                return self._error(
                    f"backend {base!r} não existe na sessão {session_id}",
                    error_code=P.ERR_BACKEND_UNKNOWN,
                    hint=f"Backends da sessão: {[spec.name for spec in session.backend_specs]}",
                )
            self.sessions.touch(session_id)
            request["_resolved_backend"] = synthetic
            return None
        backend = self._resolve_backend(request)
        if backend is None:
            loaded = self._loaded_names()
            known = list(self.registry.names)
            hint = (
                f'Backends carregados: {loaded or "nenhum"}. Registados: {known}. Passa "backend": "<nome>" no request.'
            )
            return self._error(
                "backend ambíguo — especifica 'backend' no request",
                error_code=P.ERR_BACKEND_AMBIGUOUS,
                hint=hint,
                loaded_backends=loaded,
                known_backends=known,
            )
        if not self.registry.has(backend):
            known = list(self.registry.names)
            return self._error(
                f"backend desconhecido: {backend}",
                error_code=P.ERR_BACKEND_UNKNOWN,
                hint=f"Backends válidos: {known}",
                known_backends=known,
            )
        request["_resolved_backend"] = backend
        return None

    def _enqueue_from_request(self, request: dict[str, Any]) -> Job | dict[str, Any]:
        """Enfileira a partir de generate/submit. Retorna Job ou dict de erro."""
        err = self._backend_error(request)
        if err is not None:
            return err
        backend = str(request["_resolved_backend"])
        # ``hw_tiers`` malformado falha já, não a meio do batch na GPU errada.
        from .placement import HwTierError, parse_hw_tiers, parse_pinned_devices

        try:
            parse_hw_tiers(request.get("hw_tiers"))
        except HwTierError as exc:
            return self._error(
                str(exc),
                error_code=P.ERR_INVALID_REQUEST,
                hint='Formato: [{"min_vram_mib": 10240, "rank": 2, "label": "hq", "set": {...}}, …]',
            )
        # Não persistir campos internos no job request.
        job_req = {k: v for k, v in request.items() if not k.startswith("_") and k not in ("cmd", "stream")}
        # Multi-GPU: ``gpu_ids`` físicos fora das lanes geridas ficariam presos
        # no fallback (ou correriam na GPU errada). Falha no enqueue.
        if len(self.lanes.device_lanes) > 1:
            pinned = parse_pinned_devices(job_req)
            if pinned is not None:
                known = {device for lane in self.lanes.device_lanes for device in lane.devices}
                unknown = sorted(set(pinned) - known)
                if unknown:
                    return self._error(
                        f"gpu_ids inexistente(s): {unknown}",
                        error_code=P.ERR_INVALID_REQUEST,
                        hint=f"GPUs geridas: {sorted(known)}",
                    )
        # Job encaminhado por outro supervisor (``_route.hops``): corre cá.
        route_meta = request.get("_route")
        forwarded = False
        if isinstance(route_meta, dict):
            try:
                forwarded = int(route_meta.get("hops") or 0) > 0
            except (TypeError, ValueError):
                return self._error(
                    "_route.hops inválido (inteiro esperado)",
                    error_code=P.ERR_INVALID_REQUEST,
                )
        try:
            job = self.queue.enqueue(backend, job_req, priority=request.get("priority"), forwarded=forwarded)
        except QueueFullError as e:
            snap = self.queue.snapshot()
            loaded = self._loaded_names()
            return {
                "status": P.STATUS_QUEUE_FULL,
                "error": str(e),
                "error_code": P.ERR_QUEUE_FULL,
                "hint": (
                    "Espera jobs terminarem, cancela com vramd cancel <job_id>, ou aumenta VRAMD_MAX_QUEUE_DEPTH."
                ),
                "queue_depth": self.queue.depth,
                "max_depth": self.queue.max_depth,
                "inflight": self.queue.inflight,
                "queued": snap.get("queued", []),
                "running": snap.get("running", []),
                "ums_debug": {
                    "backend": backend,
                    "priority": P.normalize_priority(request.get("priority")),
                    "queue_depth": self.queue.depth,
                    "max_depth": self.queue.max_depth,
                    "inflight": self.queue.inflight,
                    "loaded_backends": loaded,
                    "queued_backends": [j.get("backend") for j in snap.get("queued", [])],
                    "running_backends": [j.get("backend") for j in snap.get("running", [])],
                },
            }
        # Close concorrente: a sessão pode ter começado a fechar entre o
        # ``_backend_error`` (que a viu aberta) e o enqueue. Sem este recheck o
        # job corria contra um backend efémero já desregistado.
        session_id = job_req.get("session")
        if session_id and self.sessions.get(str(session_id)) is None:
            self.queue.cancel(job.job_id)
            return self._error(
                f"sessão {session_id} fechada durante o pedido",
                error_code=P.ERR_SESSION_UNKNOWN,
                hint="Abre a sessão novamente com `session-open` e re-submete.",
            )
        self._log(f"Enfileirado job {job.job_id[:8]} backend={backend!r} pri={job.priority}")
        return job

    def _ums_debug_for_job(self, job: Job) -> dict[str, Any]:
        """Bloco de debug anexado a generate/wait/poll."""
        timing = job.timing_dict()
        return {
            "job_id": job.job_id,
            "backend": job.backend,
            "priority": job.priority,
            "state": job.state,
            "affinity_cuts": job.affinity_cuts,
            "seq": job.seq,
            "queue_wait_sec": timing["queue_wait_sec"],
            "generate_sec": timing["generate_sec"],
            "total_sec": timing["total_sec"],
            "queue_depth": self.queue.depth,
            "inflight": self.queue.inflight,
            "loaded_backends": self._loaded_names(),
            "cancel_requested": job.cancel_requested,
            "cancel_reason": job.cancel_reason,
        }

    def _estimate_eta_sec(self) -> float | None:
        """ETA aproximado: soma avg_generate dos jobs queued + remaining do running.

        Itera os Jobs directamente (sem ``snapshot()`` — serializar cada job com
        ``to_public_dict`` a cada status/queue/stats é desperdício).
        """
        total = 0.0
        any_est = False
        for j in self.queue.running_jobs():
            avg = self.manager.stats.avg_generate_sec(j.backend)
            if avg is None:
                continue
            # Heurística: metade do avg ainda a correr.
            total += avg * 0.5
            any_est = True
        for j in self.queue.queued_jobs():
            avg = self.manager.stats.avg_generate_sec(j.backend)
            if avg is None:
                avg = 30.0  # fallback conservador sem histórico
            total += avg
            any_est = True
        if not any_est:
            return None
        return round(total, 1)

    @staticmethod
    def _coerce_mib(value: Any) -> int | None:
        """``needed_mib`` do request → int. ``"4096"``/``4096.0`` OK; lixo → None.

        Sem isto, ``int(needed)`` com um float-string (``"4096.5"``) ou um
        dict/list rebentava com ValueError/TypeError críptico no handler
        genérico em vez de INVALID_REQUEST acionável.
        """
        if isinstance(value, bool):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_int(value: Any, *, default: int | None = None) -> int | None:
        """Inteiro de um campo do protocolo; ``None`` se inválido (não rebenta)."""
        if value is None:
            return default
        if isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _request_timeout_sec(self, request: dict[str, Any]) -> float:
        raw = request.get("timeout_sec")
        if raw is None:
            return P.DEFAULT_GENERATE_TIMEOUT_SEC
        try:
            return max(1.0, float(raw))
        except (TypeError, ValueError):
            return P.DEFAULT_GENERATE_TIMEOUT_SEC

    def _result_from_job(self, job: Job) -> dict[str, Any]:
        result = job.result or {"status": P.STATUS_ERROR, "error": "sem resultado"}
        out = dict(result)
        out.setdefault("job_id", job.job_id)
        out.setdefault("backend", job.backend)
        request = job.request if isinstance(job.request, dict) else {}
        session_id = request.get("session")
        if session_id:
            out.setdefault("session", session_id)
            bundle_sha = self.sessions.bundle_sha(str(session_id))
            if bundle_sha:
                out.setdefault("bundle_sha", bundle_sha)
        out["priority"] = job.priority
        out["ums_debug"] = self._ums_debug_for_job(job)

        # Normalizar error_code em falhas.
        if out.get("status") != P.STATUS_OK:
            err_txt = str(out.get("error", "")).lower()
            if "cancel" in err_txt:
                out.setdefault("error_code", P.ERR_CANCELLED)
                out.setdefault("hint", "Job cancelado (queued ou durante generate).")
            elif "oom" in err_txt:
                out.setdefault("error_code", P.ERR_GENERATE_FAILED)
                out.setdefault(
                    "hint",
                    "OOM/spin no worker — vê o log vramd-worker-<backend>.log; "
                    "considera reduzir tamanho/qualidade do pedido.",
                )
            elif "timeout" in err_txt:
                out.setdefault("error_code", P.ERR_TIMEOUT)
                out.setdefault("hint", "Aumenta timeout do cliente ou inspecciona vramd queue.")
            else:
                out.setdefault("error_code", P.ERR_GENERATE_FAILED)
                out.setdefault(
                    "hint",
                    "Vê ums_debug + vramd stats (last_error do backend). "
                    "Se OOM, evict outros backends ou usa --no-ums.",
                )

        if out.get("status") == P.STATUS_OK:
            # Check-and-set sob o lock do job: um generate bloqueante + wait(s)
            # concorrentes para o mesmo job não podem contar 2x.
            with job._lock:
                if not job.counted_served:
                    job.counted_served = True
                    with self._served_lock:
                        self._requests_served += 1
        return out

    def _hello_response(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handshake de versão/capabilities do protocolo v2.

        Um cliente antigo nunca envia ``hello``; um cliente novo usa a resposta
        para decidir entre frames binários/blobs e o caminho JSON/base64. Se o
        cliente anunciar uma versão superior à do servidor, é melhor falhar
        aqui do que deixar o job correr com um contrato diferente.
        """
        raw = request.get("protocol")
        try:
            client_protocol = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            client_protocol = None
        if client_protocol is not None and client_protocol > P.PROTOCOL_VERSION:
            return self._error(
                f"cliente fala protocolo {client_protocol}; este vramd suporta até {P.PROTOCOL_VERSION}",
                error_code=P.ERR_PROTOCOL,
                hint="Atualiza o vramd do servidor (`pip install -U vramd`).",
                protocol=P.PROTOCOL_VERSION,
                features=sorted(P.FEATURES),
                vramd_version=__version__,
            )
        features = set(P.FEATURES)
        compression = ["zlib"]
        if frames.zstd_available():
            features.add("zstd")
            compression.append("zstd")
        return {
            "status": P.STATUS_OK,
            "protocol": P.PROTOCOL_VERSION,
            "client_protocol": client_protocol,
            "features": sorted(features),
            "compression": compression,
            "max_payload": frames.MAX_DATA_BYTES,
            "vramd_version": __version__,
            "transport": self._endpoint_info.transport if self._endpoint_info is not None else None,
        }

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Despacha um request já parseado. Retorna a resposta (dict).

        Nota: ``generate`` / ``wait`` com stream são tratados em ``_handle_client``
        (precisam do socket para NDJSON). Aqui ``generate`` sem stream bloqueia.
        """
        cmd = request.get("cmd", P.DEFAULT_CMD)
        self._log(f"Comando: {cmd}")

        if cmd == P.CMD_HELLO:
            return self._hello_response(request)

        if cmd == P.CMD_SESSION_OPEN:
            try:
                return self.sessions.open(request)
            except SessionError as exc:
                return self._error(str(exc), error_code=exc.code, hint=exc.hint)

        if cmd == P.CMD_SESSION_CLOSE:
            try:
                return self.sessions.close(
                    str(request.get("session", "")),
                    force=bool(request.get("force")),
                )
            except SessionError as exc:
                return self._error(str(exc), error_code=exc.code, hint=exc.hint)

        if cmd == P.CMD_SESSIONS:
            return self.sessions.list()

        if cmd == P.CMD_MODELS:
            try:
                return self.models.list()
            except ModelError as exc:
                return self._error(str(exc), error_code=exc.code, hint=exc.hint)

        if cmd == P.CMD_MODELS_ENSURE:
            raw = request.get("models")
            if not isinstance(raw, list) or not raw:
                return self._error(
                    "models-ensure requer 'models': [{repo, revision?}, …]",
                    error_code=P.ERR_INVALID_REQUEST,
                )
            try:
                specs = [ModelSpec.from_dict(item) for item in raw]
                return {"status": P.STATUS_OK, "models": self.models.ensure(specs)}
            except (ModelError, ValueError) as exc:
                code = exc.code if isinstance(exc, ModelError) else P.ERR_MODEL
                return self._error(str(exc), error_code=code)

        if cmd == P.CMD_MODELS_GC:
            try:
                return self.models.gc()
            except ModelError as exc:
                return self._error(str(exc), error_code=exc.code, hint=exc.hint)

        if cmd == P.CMD_SHUTDOWN:
            self._running = False
            return {"status": P.STATUS_OK, "message": "shutting down"}

        if cmd == P.CMD_STATUS:
            mgr_status = self.manager.status()
            strays = self.stray_report()
            qsnap = self.queue.snapshot()
            backend_stats = self.manager.stats.get_all()
            last_errors = {name: s["last_error"] for name, s in backend_stats.items() if s.get("last_error")}
            eta = self._estimate_eta_sec()
            qsnap["eta_sec"] = eta
            return {
                "status": P.STATUS_STATUS,
                "pid": self._pid,
                "socket": str(self.socket_path),
                "transport": self._endpoint_info.transport,
                "endpoint": transport.endpoint_label(self.socket_path),
                "platform": host.platform_label(),
                "tool": "vramd",
                "vramd_version": __version__,
                "protocol": P.PROTOCOL_VERSION,
                "features": sorted(P.FEATURES),
                "blobs": self.remote.blobs.stats(),
                "sessions": self.sessions.summary(),
                "models": {
                    "home": str(self.models.home),
                    "policy": self.models.policy,
                    "pinned": sorted(self.models.pinned()),
                },
                "requests_served": self._requests_served,
                "max_affinity_cuts": self.scheduler.max_cuts,
                "max_inflight": self.workers.capacity(),
                "starvation_timeout_sec": self.scheduler.starvation_timeout_sec,
                "queue": qsnap,
                "queue_metrics": self.manager.stats.queue_dict(),
                "eta_sec": eta,
                "idle_evict_timeout_sec": self.idle_evictor.idle_timeout_sec,
                "worker_shutdown_sec": self.idle_evictor.worker_shutdown_sec,
                "strays": strays,
                "learn": self.learner.status_dict(),
                "hooks": self.hooks.status_dict(),
                "lanes": self.lanes.status(),
                "peers": self.peers.status(),
                "routing": self._routing_status(),
                "tier_health": self.tier_health.status(),
                "forecasts": self.prewarmer.status(),
                "debug": {
                    "loaded_backends": self._loaded_names(),
                    "last_errors": last_errors,
                    "affinity_cuts_max": self.scheduler.max_cuts,
                    "queue_depth": qsnap.get("queue_depth", 0),
                    "inflight": qsnap.get("inflight", 0),
                    "affinity_hits": getattr(self.workers, "_affinity_hits", 0),
                    "process_vram_mib": mgr_status.get("process_vram_mib"),
                    "worker_vram_mib": mgr_status.get("worker_vram_mib"),
                    "stray_vram_mib": strays.get("vram_mib"),
                    "workers": self.manager.worker_diagnostics_all(),
                },
                **mgr_status,
            }

        if cmd == P.CMD_REAP:
            if request.get("report_only"):
                return {"status": P.STATUS_OK, **self.stray_report()}
            result = self.reap_strays(dry_run=bool(request.get("dry_run")))
            return {"status": P.STATUS_OK, **result}

        if cmd == P.CMD_QUEUE:
            snap = self.queue.snapshot()
            eta = self._estimate_eta_sec()
            snap["eta_sec"] = eta
            return {
                "status": P.STATUS_OK,
                "eta_sec": eta,
                "debug": {
                    "loaded_backends": self._loaded_names(),
                    "max_affinity_cuts": self.scheduler.max_cuts,
                    "max_inflight": self.workers.capacity(),
                    "starvation_timeout_sec": self.scheduler.starvation_timeout_sec,
                },
                **snap,
            }

        if cmd == P.CMD_LIST_BACKENDS:
            return {
                "status": P.STATUS_OK,
                "backends": [
                    {
                        "name": desc.name,
                        "vram_mib": desc.vram_mib,
                        "priority": desc.priority,
                        "loaded": bool(self.lanes.lanes_with(desc.name)),
                    }
                    for desc in self.registry
                ],
            }

        if cmd == P.CMD_STATS:
            # Reset in-process — NÃO para o vramd nem cancela jobs.
            if request.get("reset"):
                self.manager.stats.reset()
                return {
                    "status": P.STATUS_OK,
                    "reset": True,
                    "message": "stats reset (jobs/backends intactos)",
                    "pid": self._pid,
                    "queue_metrics": self.manager.stats.queue_dict(),
                    "backends": {},
                }
            stats = self.manager.stats.get_all()
            qsnap = self.queue.snapshot()
            qsnap["eta_sec"] = self._estimate_eta_sec()
            return {
                "status": P.STATUS_OK,
                "pid": self._pid,
                "requests_served": self._requests_served,
                "idle_evict_timeout_sec": self.idle_evictor.idle_timeout_sec,
                "max_affinity_cuts": self.scheduler.max_cuts,
                "max_inflight": self.workers.capacity(),
                "starvation_timeout_sec": self.scheduler.starvation_timeout_sec,
                "queue": qsnap,
                "queue_metrics": self.manager.stats.queue_dict(),
                "affinity_hits": getattr(self.workers, "_affinity_hits", 0),
                "backends": stats,
                "debug": {
                    "loaded_backends": self._loaded_names(),
                    "last_errors": {name: s["last_error"] for name, s in stats.items() if s.get("last_error")},
                    "last_runtime_budgets": {
                        name: s["last_runtime_budget"] for name, s in stats.items() if s.get("last_runtime_budget")
                    },
                    "workers": self.manager.worker_diagnostics_all(),
                },
            }

        if cmd == P.CMD_RELEASE:
            backend = request.get("backend")
            if backend:
                hard = bool(request.get("hard"))
                evicted = self._evict_everywhere(str(backend), hard=hard)
                scrub = self.manager.scrub_dead_vram() if not self._loaded_names() else None
                msg = f"backend {backend} {'evicted' if evicted else 'não estava carregado'}"
                if hard:
                    msg += " (worker parado)"
                if scrub and scrub.get("dead_vram"):
                    msg += f" (residual process={scrub.get('process_vram_mib_after')} MiB — contexto/cache)"
                return {
                    "status": P.STATUS_OK if evicted else P.STATUS_ERROR,
                    "message": msg,
                    "error": None if evicted else msg,
                    "scrub": scrub,
                }
            count = self._evict_all_lanes()
            proc = self.manager._process_vram_mib()
            return {
                "status": P.STATUS_OK,
                "message": f"{count} backend(s) evicted (cache scrubbed)",
                "scrub": {"process_vram_mib": proc},
            }

        if cmd == P.CMD_PRELOAD:
            backend = request.get("backend")
            if not backend:
                return self._error("preload requer 'backend'", error_code=P.ERR_INVALID_REQUEST)
            name = str(backend)
            if not self.registry.has(name):
                return self._error(
                    f"backend desconhecido: {name}",
                    error_code=P.ERR_BACKEND_UNKNOWN,
                    hint=f"Backends válidos: {list(self.registry.names)}",
                )
            try:
                from .backend_manager import _LOAD_KWARG_KEYS, InsufficientVramError

                load_kwargs = {k: v for k, v in request.items() if k in _LOAD_KWARG_KEYS}
                self.manager.ensure_loaded(name, **load_kwargs)
                quant, mem_eff, group_off, streams = self.manager.resolve_peak_params(name, load_kwargs)
                return {
                    "status": P.STATUS_OK,
                    "message": f"backend {name} pré-carregado",
                    "ums_debug": {
                        "backend": name,
                        "loaded_backends": self._loaded_names(),
                        "peak_mib": self.manager.peak_vram_mib(
                            name,
                            quant_mode=quant,
                            memory_efficient=mem_eff,
                            group_offload=group_off,
                            streams_on_load=streams,
                            footprint_key=load_kwargs.get("footprint_key"),
                        ),
                        "quant_mode": quant,
                        "memory_efficient": mem_eff,
                        "group_offload": group_off,
                    },
                }
            except InsufficientVramError as e:
                return self._error(
                    str(e),
                    error_code=P.ERR_VRAM_INSUFFICIENT,
                    hint=("Pico = pesos + activação + safety. Em ~6 GB usa sdnq-int4; não mates GPU — `vramd queue`."),
                    backend=name,
                    peak_mib=e.peak_mib,
                    free_mib=e.free_mib,
                )
            except Exception as e:
                return self._error(
                    f"falha ao pré-carregar {name}: {e}",
                    error_code=P.ERR_PRELOAD_FAILED,
                    hint="Verifica deps do backend (vramd doctor) e VRAM livre.",
                    backend=name,
                )

        if cmd == P.CMD_RESPAWN:
            # Respawn só faz sentido com workers subprocesso activos. Recusar se
            # há jobs na fila/inflight — matar um worker mid-generate rouba o job.
            # ``is_busy()`` lê inflight+depth sob o mesmo lock (TOCTOU-safe).
            if self.queue.is_busy():
                return self._error(
                    "fila ocupada — espera os jobs terminarem antes de respawn",
                    error_code=P.ERR_RESPAWN_BUSY,
                    hint="vramd queue / vramd wait <job_id> — não mates o worker a meio de um job.",
                    queue=self.queue.snapshot(),
                )
            from .backend_manager import ShapeBusyError

            backend = request.get("backend")
            lazy_req = request.get("lazy", True)
            lazy = lazy_req is None or self.manager._as_bool(lazy_req) is not False
            try:
                if backend:
                    name = str(backend)
                    if not self.registry.has(name):
                        return self._error(
                            f"backend desconhecido: {name}",
                            error_code=P.ERR_BACKEND_UNKNOWN,
                            hint=f"Backends válidos: {list(self.registry.names)}",
                        )
                    results = [self.manager.respawn(name, lazy=lazy)]
                    for lane in self._extra_lanes():
                        results.append({**lane.manager.respawn(name, lazy=lazy), "lane": lane.lane_id})
                else:
                    results = self.manager.respawn_all(lazy=lazy)
                    for lane in self._extra_lanes():
                        results.extend({**r, "lane": lane.lane_id} for r in lane.manager.respawn_all(lazy=lazy))
                self.lanes.invalidate()
            except ShapeBusyError as e:
                return self._error(
                    str(e),
                    error_code=P.ERR_RESPAWN_BUSY,
                    hint="Backend com job a correr (ref_count>0). Espera o job terminar.",
                )
            except KeyError as e:
                return self._error(f"backend desconhecido: {e}", error_code=P.ERR_BACKEND_UNKNOWN)
            except Exception as e:
                return self._error(
                    f"falha no respawn: {e}",
                    error_code=P.ERR_RESPAWN_FAILED,
                    hint="Worker provavelmente já morto — o próximo generate arranca um novo.",
                )
            # Scrub residual CUDA depois de matar workers (pode libertar contexto).
            scrub = self.manager.scrub_dead_vram() if not self._loaded_names() else None
            return {
                "status": P.STATUS_OK,
                "message": f"{sum(1 for r in results if r.get('respawned'))}/{len(results)} worker(s) reiniciado(s)",
                "results": results,
                "lazy": lazy,
                "scrub": scrub,
                "loaded_backends": self._loaded_names(),
            }

        if cmd == P.CMD_ZERO:
            # Zero liberta TODA a VRAM do vramd sem parar o supervisor: mata os
            # workers vivos (só a morte do processo devolve o contexto CUDA) e
            # scrubba caches. Mesmo busy-guard do respawn — nunca matar um
            # worker mid-generate. ``is_busy()`` lê inflight+depth sob o mesmo
            # lock (TOCTOU-safe).
            if self.queue.is_busy():
                return self._error(
                    "fila ocupada — espera os jobs terminarem antes de zerar a VRAM",
                    error_code=P.ERR_ZERO_BUSY,
                    hint="vramd queue / vramd wait <job_id> — não mates o worker a meio de um job.",
                    queue=self.queue.snapshot(),
                )
            from .backend_manager import ShapeBusyError

            try:
                summary = self.manager.zero_vram()
                for lane in self._extra_lanes():
                    extra = lane.manager.zero_vram()
                    summary["workers_killed"] = int(summary.get("workers_killed") or 0) + int(
                        extra.get("workers_killed") or 0
                    )
                    summary.setdefault("lanes", {})[lane.lane_id] = extra
                self.lanes.invalidate()
            except ShapeBusyError as e:
                return self._error(
                    str(e),
                    error_code=P.ERR_ZERO_BUSY,
                    hint="Backend com job a correr (ref_count>0). Espera o job terminar.",
                )
            except Exception as e:
                return self._error(
                    f"falha ao zerar VRAM: {e}",
                    error_code=P.ERR_ZERO_BUSY,
                    hint="Worker provavelmente já morto — o próximo generate arranca um novo.",
                )
            killed = int(summary.get("workers_killed") or 0)
            fb, fa = summary.get("free_mib_before"), summary.get("free_mib_after")
            freed = (fa - fb) if (isinstance(fa, int) and isinstance(fb, int)) else None
            freed_str = f"; ~{freed} MiB recuperados" if freed else ""
            self.hooks.dispatch(
                EVENT_ZERO,
                {
                    "workers_killed": killed,
                    "free_mib_before": fb,
                    "free_mib_after": fa,
                    "freed_mib": freed,
                },
            )
            return {
                "status": P.STATUS_OK,
                "message": f"{killed} worker(s) terminado(s) — VRAM zerada, supervisor intacto{freed_str}",
                **summary,
                "loaded_backends": self._loaded_names(),
            }

        if cmd == P.CMD_LEARN:
            # Relatório de drift (pico observado na produção vs declarado).
            # ``reset`` limpa as observações — útil depois de mudar de GPU ou
            # de reinstalar um backend com outro footprint.
            if request.get("reset"):
                backend = request.get("backend")
                count = self.learner.reset(str(backend) if backend else None)
                return {
                    "status": P.STATUS_OK,
                    "message": f"{count} ficheiro(s) de observações removido(s)",
                    "reset": True,
                }
            reports = self.learner.report_all()
            return {
                "status": P.STATUS_OK,
                "enabled": self.learner.enabled,
                "interval_sec": round(self.learner.interval_sec, 3),
                "probe": self.learner.probe_status(),
                "backends": [r.to_dict() for r in reports],
                "hint": "vramd learn --apply escreve as correcções como overlay YAML",
            }

        if cmd == P.CMD_ENSURE_VRAM:
            needed = self._coerce_mib(request.get("needed_mib"))
            if needed is None or needed <= 0:
                return self._error(
                    "ensure-vram requer 'needed_mib' (número > 0 de MiB)",
                    error_code=P.ERR_INVALID_REQUEST,
                )
            backend = request.get("backend")
            bname = str(backend) if backend else ""
            group_off = False
            allow_go = self.manager._as_bool(request.get("allow_group_offload"))
            if backend and self.registry.has(bname):
                quant, mem_eff, group_off, streams = self.manager.resolve_peak_params(bname, request)
            else:
                quant = self.manager.resolve_quant_mode(request)
                mem_eff = self.manager._as_bool(request.get("memory_efficient")) is True
            before = self._loaded_names()
            # Admit/evict room: pesos completos no load frio — EXCEPTO backends
            # com load já streaming (diffusers offload → streams_on_load).
            target = needed
            if backend and self.registry.has(bname):
                target = max(
                    target,
                    self.manager.peak_vram_mib(
                        bname,
                        quant_mode=quant,
                        memory_efficient=mem_eff,
                        group_offload=group_off,
                        streams_on_load=streams,
                        footprint_key=request.get("footprint_key"),
                    ),
                )
            ok = self.manager.ensure_vram(
                needed,
                backend=bname if backend else None,
                quant_mode=quant,
                memory_efficient=mem_eff,
                allow_group_offload=allow_go,
            )
            after = self._loaded_names()
            return {
                "status": P.STATUS_OK if ok else P.STATUS_ERROR,
                "needed_mib": needed,
                "target_mib": target,
                "error_code": None if ok else P.ERR_VRAM_INSUFFICIENT,
                "ums_debug": {
                    "loaded_before": before,
                    "loaded_after": after,
                    "evicted": sorted(set(before) - set(after)),
                    "backend": backend,
                    "quant_mode": quant,
                    "memory_efficient": mem_eff,
                    "group_offload": group_off,
                    "peak_mib": (
                        self.manager.peak_vram_mib(
                            str(backend),
                            quant_mode=quant,
                            memory_efficient=mem_eff,
                            group_offload=group_off,
                            streams_on_load=streams,
                            footprint_key=request.get("footprint_key"),
                        )
                        if backend and self.registry.has(str(backend))
                        else None
                    ),
                },
            }

        if cmd == P.CMD_SUBMIT:
            outcome = self._enqueue_from_request(request)
            if isinstance(outcome, dict):
                return outcome
            return {
                "status": P.STATUS_OK,
                "job_id": outcome.job_id,
                "backend": outcome.backend,
                "priority": outcome.priority,
                "state": outcome.state,
                "queue_position": self.queue.depth,
            }

        if cmd == P.CMD_POLL:
            job_id = request.get("job_id")
            if not job_id:
                return self._error("poll requer 'job_id'", error_code=P.ERR_INVALID_REQUEST)
            job = self.queue.get(str(job_id))
            if job is None:
                return self._error(
                    f"job desconhecido: {job_id}",
                    error_code=P.ERR_JOB_UNKNOWN,
                    hint="Lista jobs com vramd queue",
                )
            payload = job.to_public_dict()
            payload["ums_debug"] = self._ums_debug_for_job(job)
            if job.result is not None:
                payload["result"] = job.result
            return {"status": P.STATUS_OK, **payload}

        if cmd == P.CMD_CANCEL:
            if request.get("all") or str(request.get("job_id") or "").lower() in ("all", "*"):
                return self.queue.cancel_all(include_running=not bool(request.get("queued_only")))
            job_id = request.get("job_id")
            if not job_id:
                return self._error(
                    "cancel requer 'job_id' ou all=true",
                    error_code=P.ERR_INVALID_REQUEST,
                    hint="ums cancel <job_id|prefixo> | ums cancel --all | ums flush",
                )
            return self.queue.cancel(str(job_id))

        if cmd == P.CMD_FLUSH:
            return self.queue.cancel_all(include_running=not bool(request.get("queued_only")))

        if cmd == P.CMD_WAIT:
            job_id = request.get("job_id")
            if not job_id:
                return self._error("wait requer 'job_id'", error_code=P.ERR_INVALID_REQUEST)
            resolved, err = self.queue.resolve_job_id(str(job_id))
            if resolved is None:
                return self._error(
                    err or f"job desconhecido: {job_id}",
                    error_code=P.ERR_JOB_UNKNOWN,
                    hint="Lista jobs com vramd queue",
                )
            job_id = resolved
            timeout = self._request_timeout_sec(request)
            job = self.queue.wait(str(job_id), timeout_sec=timeout)
            if job is None:
                return self._error(
                    f"job desconhecido: {job_id}",
                    error_code=P.ERR_JOB_UNKNOWN,
                    hint="Lista jobs com vramd queue",
                )
            if not job.done_event.is_set():
                return self._error(
                    "timeout à espera do job",
                    error_code=P.ERR_TIMEOUT,
                    hint="Inspecciona vramd queue / status",
                    job_id=job_id,
                    ums_debug=self._ums_debug_for_job(job),
                )
            return self._result_from_job(job)

        if cmd == P.CMD_GENERATE:
            outcome = self._enqueue_from_request(request)
            if isinstance(outcome, dict):
                return outcome
            job = outcome
            timeout = self._request_timeout_sec(request)
            job.done_event.wait(timeout=timeout)
            if not job.done_event.is_set():
                return self._error(
                    "timeout à espera do job na fila",
                    error_code=P.ERR_TIMEOUT,
                    hint="Inspeciona vramd queue / status",
                    job_id=job.job_id,
                    ums_debug=self._ums_debug_for_job(job),
                )
            return self._result_from_job(job)

        if cmd == P.CMD_FORECAST:
            err = self._backend_error(request)
            if err is not None:
                return err
            from .placement import HwTierError, parse_hw_tiers

            sample = request.get("request") or {}
            if not isinstance(sample, dict):
                return self._error("forecast: 'request' deve ser um objecto", error_code=P.ERR_INVALID_REQUEST)
            try:
                parse_hw_tiers(sample.get("hw_tiers"))
                count = int(request.get("count") or 1)
                eta = float(request.get("eta_sec") or 0.0)
                ttl = float(request.get("ttl_sec") or 900.0)
            except (HwTierError, TypeError, ValueError) as exc:
                return self._error(f"forecast inválido: {exc}", error_code=P.ERR_INVALID_REQUEST)
            forecast = self.prewarmer.add(
                str(request["_resolved_backend"]), count=count, eta_sec=eta, request=sample, ttl_sec=ttl
            )
            # Sem esperar pelo loop: se já está na janela, aquece agora (em background).
            threading.Thread(target=self.prewarmer.tick, name="vramd-prewarm-now", daemon=True).start()
            return {"status": P.STATUS_OK, "forecast": forecast.forecast_id, "backend": forecast.backend}

        if cmd == P.CMD_TIERS:
            # Tiers suspensos depois de OOMs (``reset`` limpa — ex. depois de
            # trocar de driver ou de reduzir o tamanho das malhas).
            if request.get("reset"):
                backend = request.get("backend")
                count = self.tier_health.reset(str(backend) if backend else None)
                return {"status": P.STATUS_OK, "reset": True, "cleared": count}
            return {"status": P.STATUS_OK, "tiers": self.tier_health.status()}

        if cmd == P.CMD_TOPOLOGY:
            return {"status": P.STATUS_OK, **self._topology()}

        if cmd == P.CMD_ROUTE:
            # Dry-run do router: onde correria este job agora, e porquê. Não
            # enfileira nem reserva nada.
            err = self._backend_error(request)
            if err is not None:
                return err
            backend = str(request["_resolved_backend"])
            job_req = {k: v for k, v in request.items() if not k.startswith("_") and k not in ("cmd", "stream")}
            decision = self.router.decide(backend, job_req)
            profile = self.router.profile(backend, job_req)
            return {
                "status": P.STATUS_OK,
                "backend": backend,
                "routed": self.router.active,
                "profile": {
                    "peak_mib": profile.peak_mib,
                    "headroom_mib": profile.headroom_mib,
                    "load_sec": round(profile.load_sec, 2),
                    "gen_sec": round(profile.gen_sec, 2),
                    "input_mib": round(profile.input_bytes / (1024 * 1024), 2),
                    "allow_split": profile.allow_split,
                    "allow_remote": profile.allow_remote,
                    "max_replicas": profile.max_replicas,
                    "pinned_devices": list(profile.pinned_devices) if profile.pinned_devices else None,
                    "criticality": profile.criticality,
                    "tiers": [
                        {
                            "label": t.label,
                            "min_vram_mib": t.min_vram_mib,
                            "rank": t.rank,
                            "peak_mib": t.peak_mib,
                            "gen_sec": round(t.gen_sec, 2),
                        }
                        for t in profile.tiers
                    ],
                },
                **decision.to_dict(),
                "lanes": self.lanes.status(),
                "peers": self.peers.status(),
            }

        if cmd in (
            P.CMD_STAGE,
            P.CMD_PUT,
            P.CMD_GET,
            P.CMD_LIST,
            P.CMD_CLEAN,
            P.CMD_BLOB_HAS,
            P.CMD_BLOB_PUT,
            P.CMD_BLOB_GET,
            P.CMD_BLOB_DELETE,
            P.CMD_BLOBS,
            P.CMD_BLOBS_GC,
            P.CMD_JOB_GET,
        ):
            return self._dispatch_remote(cmd, request)

        return self._error(
            f"comando desconhecido: {cmd}",
            error_code=P.ERR_INVALID_REQUEST,
            hint=f"Comandos: {sorted(P.KNOWN_COMMANDS)}",
        )

    def _dispatch_remote(self, cmd: str, request: dict[str, Any]) -> dict[str, Any]:
        """Transferência de ficheiros do modo remoto (sandbox por job).

        Nota de segurança: ``stage`` materializa blobs do CAS por hardlink.
        ``put`` quebra o link antes de escrever, mas um worker (ou código do
        cliente) que escreva directamente num input ``in/*`` pode alterar o
        inode partilhado e corromper o CAS. O contrato é que inputs são
        read-only; não há confinamento do worker ao sandbox. Copiar sempre no
        ``stage`` protegeria o CAS mas duplicaria cada asset por job — fica
        documentado como limitação conhecida.
        """
        try:
            if cmd == P.CMD_STAGE:
                raw = request.get("blobs") or {}
                if not isinstance(raw, dict):
                    return self._error("'blobs' deve ser um mapa path→sha256", error_code=P.ERR_INVALID_REQUEST)
                blobs = {str(path): str(sha) for path, sha in raw.items()}
                return self.remote.stage(blobs)
            if cmd == P.CMD_PUT:
                offset = self._coerce_int(request.get("offset"), default=0)
                if offset is None:
                    return self._error("offset inválido (inteiro esperado)", error_code=P.ERR_INVALID_REQUEST)
                return self.remote.put(
                    str(request.get("job", "")),
                    str(request.get("path", "")),
                    offset=offset,
                    data_b64=str(request.get("data", "")),
                    eof=bool(request.get("eof")),
                    sha256=request.get("sha256"),
                )
            if cmd == P.CMD_GET:
                offset = self._coerce_int(request.get("offset"), default=0)
                length = self._coerce_int(request.get("length"), default=2 * 1024 * 1024)
                if offset is None or length is None:
                    return self._error("offset/length inválidos (inteiros esperados)", error_code=P.ERR_INVALID_REQUEST)
                if length == 0:
                    # Compat: ``length`` ausente/0 sempre significou o chunk default.
                    length = 2 * 1024 * 1024
                return self.remote.get(
                    str(request.get("job", "")),
                    str(request.get("path", "")),
                    offset=offset,
                    length=length,
                )
            if cmd == P.CMD_LIST:
                return self.remote.ls(str(request.get("job", "")), str(request.get("path", ".") or "."))
            if cmd == P.CMD_BLOB_HAS:
                raw_size = request.get("size")
                size = self._coerce_int(raw_size)
                if raw_size is not None and (size is None or size < 0):
                    return self._error("size inválido (inteiro >= 0)", error_code=P.ERR_INVALID_REQUEST)
                return self.remote.blob_has(str(request.get("sha256", "")), size)
            if cmd == P.CMD_BLOB_DELETE:
                return self.remote.blob_delete(str(request.get("sha256", "")), force=bool(request.get("force")))
            if cmd == P.CMD_BLOBS:
                return self.remote.blob_stats()
            if cmd == P.CMD_BLOBS_GC:
                return self.remote.blob_gc()
            if cmd in (P.CMD_BLOB_PUT, P.CMD_BLOB_GET, P.CMD_JOB_GET):
                return self._error(
                    f"{cmd} requer o transporte binário (frames)",
                    error_code=P.ERR_PROTOCOL,
                    hint="Liga com `hello` e envia frames `DATA`/`END`.",
                )
            return self.remote.clean(str(request.get("job", "")))
        except RemoteError as exc:
            return self._error(str(exc), error_code=exc.code, hint=exc.hint)
        except BlobError as exc:
            return self._error(str(exc), error_code=exc.code, hint=exc.hint)

    # ------------------------------------------------------------------
    # Handle de 1 ligação (com suporte a NDJSON stream)
    # ------------------------------------------------------------------

    def _send_json(self, conn: socket.socket, obj: dict[str, Any]) -> None:
        conn.sendall((json.dumps(obj) + "\n").encode())

    def _handle_streaming_job(
        self,
        conn: socket.socket,
        job: Job,
        *,
        timeout_sec: float | None = None,
    ) -> None:
        """Envia eventos NDJSON até o job terminar; última linha = resultado."""
        events: list[dict[str, Any]] = []
        ready = threading.Event()
        timeout = timeout_sec if timeout_sec is not None else P.DEFAULT_GENERATE_TIMEOUT_SEC

        def on_event(event: dict[str, Any]) -> None:
            events.append(event)
            ready.set()

        job.add_listener(on_event)
        try:
            pos0 = self.queue.queue_position(job.job_id) or max(1, self.queue.depth)
            self._send_json(
                conn,
                {
                    "event": P.EVENT_QUEUED,
                    "job_id": job.job_id,
                    "backend": job.backend,
                    "priority": job.priority,
                    "state": job.state,
                    "queue_position": pos0,
                },
            )

            deadline = time.monotonic() + timeout
            sent = 0
            last_pos: int | None = pos0
            last_pos_emit = time.monotonic()
            while time.monotonic() < deadline:
                if job.done_event.is_set() and sent >= len(events):
                    break
                ready.wait(timeout=0.25)
                ready.clear()
                while sent < len(events):
                    ev = events[sent]
                    sent += 1
                    if ev.get("event") in (P.EVENT_DONE, P.EVENT_ERROR, P.EVENT_CANCELLED):
                        continue
                    with contextlib.suppress(OSError):
                        self._send_json(conn, ev)

                # Actualizar queue_position enquanto ainda queued.
                if job.state == P.JOB_QUEUED and (time.monotonic() - last_pos_emit) >= 1.0:
                    pos = self.queue.queue_position(job.job_id)
                    if pos is not None and pos != last_pos:
                        last_pos = pos
                        with contextlib.suppress(OSError):
                            self._send_json(
                                conn,
                                {
                                    "event": P.EVENT_QUEUED,
                                    "job_id": job.job_id,
                                    "backend": job.backend,
                                    "priority": job.priority,
                                    "state": job.state,
                                    "queue_position": pos,
                                },
                            )
                    last_pos_emit = time.monotonic()

            if not job.done_event.is_set():
                self._send_json(
                    conn,
                    self._error(
                        "timeout à espera do job",
                        error_code=P.ERR_TIMEOUT,
                        hint="Inspecciona vramd queue / status",
                        job_id=job.job_id,
                        ums_debug=self._ums_debug_for_job(job),
                    ),
                )
                return
            self._send_json(conn, self._result_from_job(job))
        finally:
            # Sem isto, o listener (e a lista ``events`` que retém) ficava preso
            # ao job para sempre — incluindo em timeout / disconnect do cliente.
            job.remove_listener(on_event)

    def _serve_json_request(self, conn: socket.socket, request: dict[str, Any]) -> None:
        """Processa um pedido que chegou como linha JSON (legacy ou hello)."""
        try:
            cmd = request.get("cmd", P.DEFAULT_CMD)
            want_stream = bool(request.get("stream"))

            # generate/wait com stream: caminho NDJSON.
            if cmd == P.CMD_GENERATE and want_stream:
                outcome = self._enqueue_from_request(request)
                if isinstance(outcome, dict):
                    self._send_json(conn, outcome)
                    return
                self._handle_streaming_job(conn, outcome, timeout_sec=self._request_timeout_sec(request))
                return

            if cmd == P.CMD_WAIT and want_stream:
                job_id = request.get("job_id")
                if not job_id:
                    self._send_json(
                        conn,
                        self._error("wait requer 'job_id'", error_code=P.ERR_INVALID_REQUEST),
                    )
                    return
                resolved, err = self.queue.resolve_job_id(str(job_id))
                job = self.queue.get(resolved) if resolved is not None else None
                if job is None:
                    self._send_json(
                        conn,
                        self._error(
                            err or f"job desconhecido: {job_id}",
                            error_code=P.ERR_JOB_UNKNOWN,
                            hint="Lista jobs com vramd queue",
                        ),
                    )
                    return
                self._handle_streaming_job(conn, job, timeout_sec=self._request_timeout_sec(request))
                return

            response = self._dispatch(request)
            self._send_json(conn, response)

        except json.JSONDecodeError as e:
            with contextlib.suppress(OSError):
                self._send_json(
                    conn,
                    self._error(f"JSON inválido: {e}", error_code=P.ERR_INVALID_REQUEST),
                )
        except Exception as e:
            self._log(f"Erro ao processar cliente: {e}")
            with contextlib.suppress(OSError):
                self._send_json(
                    conn,
                    self._error(str(e), error_code=P.ERR_GENERATE_FAILED),
                )

    def _send_frame_json(self, conn: socket.socket, obj: dict[str, Any], request_id: int) -> None:
        """Resposta lógica completa: JSON + END (o cliente lê até END)."""
        frames.send_frame(
            conn,
            frames.TYPE_JSON,
            frames.encode_json(obj),
            request_id=request_id,
        )
        frames.send_frame(conn, frames.TYPE_END, b"", request_id=request_id)

    def _send_frame_error(
        self,
        conn: socket.socket,
        message: str,
        request_id: int,
        code: str = P.ERR_INVALID_REQUEST,
    ) -> None:
        self._send_frame_json(conn, self._error(message, error_code=code), request_id)

    def _serve_frame_request(
        self,
        reader: frames.FrameReader,
        conn: socket.socket,
        request: dict[str, Any],
        request_id: int,
    ) -> bool | None:
        """Processa um pedido que chegou num frame JSON.

        Devolve ``False`` quando a ligação ficou dessincronizada e tem de ser
        fechada; ``True``/``None`` mantêm o modo persistente.
        """
        cmd = request.get("cmd", P.DEFAULT_CMD)
        try:
            if cmd == P.CMD_GENERATE and request.get("stream"):
                outcome = self._enqueue_from_request(request)
                if isinstance(outcome, dict):
                    self._send_frame_json(conn, outcome, request_id)
                    return
                self._handle_streaming_job_frames(
                    conn, outcome, request_id, timeout_sec=self._request_timeout_sec(request)
                )
                return

            if cmd == P.CMD_WAIT and request.get("stream"):
                job_id = request.get("job_id")
                if not job_id:
                    self._send_frame_error(conn, "wait requer 'job_id'", request_id, P.ERR_INVALID_REQUEST)
                    return
                resolved, err = self.queue.resolve_job_id(str(job_id))
                job = self.queue.get(resolved) if resolved is not None else None
                if job is None:
                    self._send_frame_error(
                        conn,
                        err or f"job desconhecido: {job_id}",
                        request_id,
                        P.ERR_JOB_UNKNOWN,
                    )
                    return
                self._handle_streaming_job_frames(conn, job, request_id, timeout_sec=self._request_timeout_sec(request))
                return

            if cmd == P.CMD_BLOB_PUT:
                return self._handle_blob_put_stream(reader, conn, request, request_id)

            if cmd == P.CMD_BLOB_GET:
                self._handle_blob_get_stream(conn, request, request_id)
                return

            if cmd == P.CMD_JOB_GET:
                self._handle_job_get_stream(conn, request, request_id)
                return

            response = self._dispatch(request)
            self._send_frame_json(conn, response, request_id)
            return True

        except frames.FrameError as e:
            with contextlib.suppress(OSError):
                self._send_frame_error(conn, str(e), request_id, P.ERR_PROTOCOL)
            return False if cmd == P.CMD_BLOB_PUT else None
        except Exception as e:
            self._log(f"Erro ao processar frame: {e}")
            with contextlib.suppress(OSError):
                self._send_frame_error(conn, str(e), request_id, P.ERR_GENERATE_FAILED)
            return False if cmd == P.CMD_BLOB_PUT else None

    def _handle_streaming_job_frames(
        self,
        conn: socket.socket,
        job: Job,
        request_id: int,
        *,
        timeout_sec: float | None = None,
    ) -> None:
        """Variante de ``_handle_streaming_job`` para o protocolo binário."""
        events: list[dict[str, Any]] = []
        ready = threading.Event()
        timeout = timeout_sec if timeout_sec is not None else P.DEFAULT_GENERATE_TIMEOUT_SEC

        def on_event(event: dict[str, Any]) -> None:
            events.append(event)
            ready.set()

        job.add_listener(on_event)
        try:
            pos0 = self.queue.queue_position(job.job_id) or max(1, self.queue.depth)
            frames.send_frame(
                conn,
                frames.TYPE_EVENT,
                frames.encode_json(
                    {
                        "event": P.EVENT_QUEUED,
                        "job_id": job.job_id,
                        "backend": job.backend,
                        "priority": job.priority,
                        "state": job.state,
                        "queue_position": pos0,
                    }
                ),
                request_id=request_id,
            )

            deadline = time.monotonic() + timeout
            sent = 0
            last_pos: int | None = pos0
            last_pos_emit = time.monotonic()
            while time.monotonic() < deadline:
                if job.done_event.is_set() and sent >= len(events):
                    break
                ready.wait(timeout=0.25)
                ready.clear()
                while sent < len(events):
                    ev = events[sent]
                    sent += 1
                    if ev.get("event") in (P.EVENT_DONE, P.EVENT_ERROR, P.EVENT_CANCELLED):
                        continue
                    with contextlib.suppress(OSError):
                        frames.send_frame(
                            conn,
                            frames.TYPE_EVENT,
                            frames.encode_json(ev),
                            request_id=request_id,
                        )

                if job.state == P.JOB_QUEUED and (time.monotonic() - last_pos_emit) >= 1.0:
                    pos = self.queue.queue_position(job.job_id)
                    if pos is not None and pos != last_pos:
                        last_pos = pos
                        with contextlib.suppress(OSError):
                            frames.send_frame(
                                conn,
                                frames.TYPE_EVENT,
                                frames.encode_json(
                                    {
                                        "event": P.EVENT_QUEUED,
                                        "job_id": job.job_id,
                                        "backend": job.backend,
                                        "priority": job.priority,
                                        "state": job.state,
                                        "queue_position": pos,
                                    }
                                ),
                                request_id=request_id,
                            )
                    last_pos_emit = time.monotonic()

            if not job.done_event.is_set():
                self._send_frame_json(
                    conn,
                    self._error(
                        "timeout à espera do job",
                        error_code=P.ERR_TIMEOUT,
                        hint="Inspecciona vramd queue / status",
                        job_id=job.job_id,
                        ums_debug=self._ums_debug_for_job(job),
                    ),
                    request_id,
                )
                return
            self._send_frame_json(conn, self._result_from_job(job), request_id)
        finally:
            job.remove_listener(on_event)

    def _handle_blob_put_stream(
        self,
        reader: frames.FrameReader,
        conn: socket.socket,
        request: dict[str, Any],
        request_id: int,
    ) -> bool:
        """Recebe frames DATA até END e publica o blob no CAS.

        Returns:
            ``False`` quando a ligação tem de ser fechada: um erro antes do
            ``END`` deixa frames DATA/END por consumir, e reutilizar a ligação
            faria o servidor interpretá-los como pedidos novos (dessincronização).
            Um erro depois do ``END`` (sha/tamanho) mantém a ligação utilizável.
        """
        writer = None
        stream_ended = False
        try:
            sha = str(request.get("sha256", ""))
            size = int(request.get("size", -1))
            writer = self.remote.blobs.open_writer(sha, size)
            while True:
                frame = reader.read_frame(max_payload=frames.MAX_DATA_BYTES)
                if frame is None:
                    writer.abort()
                    return False
                if frame.request_id != request_id:
                    raise BlobError(
                        f"request_id do frame ({frame.request_id}) != pedido ({request_id})",
                        code=P.ERR_PROTOCOL,
                    )
                if frame.type == frames.TYPE_DATA:
                    writer.write(frames.decode_payload(frame.flags, frame.payload))
                elif frame.type == frames.TYPE_END:
                    stream_ended = True
                    break
                else:
                    raise BlobError(
                        f"frame inesperado no upload: tipo {frame.type}",
                        code=P.ERR_PROTOCOL,
                    )
            result = writer.finish()
        except (BlobError, frames.FrameError, ValueError, OSError) as exc:
            if writer is not None:
                writer.abort()
            code = exc.code if isinstance(exc, BlobError) else P.ERR_BLOB_INVALID
            with contextlib.suppress(OSError):
                self._send_frame_error(conn, str(exc), request_id, code)
            if not stream_ended:
                # Não há forma barata de saber onde acaba o upload: o cliente
                # pode ainda estar a enviar muitos DATA frames. Fechar é melhor
                # que deixar a ligação dessincronizada (o cliente reconecta).
                self._log("Upload de blob abortado antes do END — a fechar a ligação")
                return False
        self._send_frame_json(conn, result, request_id)
        return True

    def _handle_blob_get_stream(
        self,
        conn: socket.socket,
        request: dict[str, Any],
        request_id: int,
    ) -> None:
        """Envia metadados JSON, frames DATA e END a partir do CAS."""
        try:
            sha = str(request.get("sha256", ""))
            info = self.remote.blobs.stat(sha)
            offset = max(0, int(request.get("offset", 0)))
            length = max(1, int(request.get("length", frames.MAX_DATA_BYTES)))
            size = int(info["size"])
            if offset > size:
                raise BlobError(
                    f"offset {offset} além do tamanho {size}",
                    code=P.ERR_BLOB_SIZE,
                )
            meta = {
                "status": P.STATUS_OK,
                "sha256": info["sha256"],
                "size": size,
                "offset": offset,
            }
            frames.send_frame(conn, frames.TYPE_JSON, frames.encode_json(meta), request_id=request_id)
            remaining = min(length, size - offset)
            with self.remote.blobs.open_reader(sha) as handle:
                handle.seek(offset)
                while remaining > 0:
                    chunk = handle.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        break
                    flags, payload = frames.encode_payload(chunk)
                    frames.send_frame(
                        conn,
                        frames.TYPE_DATA,
                        payload,
                        request_id=request_id,
                        flags=flags,
                    )
                    remaining -= len(chunk)
            frames.send_frame(conn, frames.TYPE_END, b"", request_id=request_id)
        except (BlobError, frames.FrameError, ValueError, OSError) as exc:
            code = exc.code if isinstance(exc, BlobError) else P.ERR_BLOB_INVALID
            with contextlib.suppress(OSError):
                self._send_frame_error(conn, str(exc), request_id, code)

    def _handle_job_get_stream(
        self,
        conn: socket.socket,
        request: dict[str, Any],
        request_id: int,
    ) -> None:
        """Download binário de um ficheiro do sandbox (outputs de jobs)."""
        try:
            job = str(request.get("job", ""))
            path = str(request.get("path", ""))
            offset = max(0, int(request.get("offset", 0)))
            length = int(request.get("length", 0) or 0)
            handle, size, offset, remaining = self.remote.open_get(job, path, offset=offset, length=length)
            with handle:
                meta = {
                    "status": P.STATUS_OK,
                    "job": job,
                    "path": path,
                    "size": size,
                    "offset": offset,
                }
                frames.send_frame(conn, frames.TYPE_JSON, frames.encode_json(meta), request_id=request_id)
                while remaining > 0:
                    chunk = handle.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        break
                    flags, payload = frames.encode_payload(chunk)
                    frames.send_frame(
                        conn,
                        frames.TYPE_DATA,
                        payload,
                        request_id=request_id,
                        flags=flags,
                    )
                    remaining -= len(chunk)
            frames.send_frame(conn, frames.TYPE_END, b"", request_id=request_id)
        except (RemoteError, frames.FrameError, ValueError, OSError) as exc:
            code = exc.code if isinstance(exc, RemoteError) else P.ERR_REMOTE_PATH
            with contextlib.suppress(OSError):
                self._send_frame_error(conn, str(exc), request_id, code)

    @staticmethod
    def _abort_pending_read(conn: socket.socket) -> None:
        """Acorda um ``recv`` bloqueado sem fechar o lado de escrita."""
        with contextlib.suppress(OSError):
            conn.shutdown(socket.SHUT_RD)

    def _read_with_deadline(self, conn: socket.socket, read: Any) -> Any:
        """Executa ``read`` com um deadline TOTAL (não por-``recv``).

        ``conn.settimeout`` limita cada ``recv``: um cliente que goteja um byte
        a cada ``REQUEST_READ_TIMEOUT_SEC`` mantinha a thread presa para sempre
        (slowloris). O timer corta a leitura no total; ``read_line``/``read_frame``
        vêem EOF e devolvem o que têm, e o handler fecha a ligação.
        """
        timeout = float(REQUEST_READ_TIMEOUT_SEC)
        if timeout <= 0:
            return read()
        timer = threading.Timer(timeout, self._abort_pending_read, args=(conn,))
        timer.daemon = True
        timer.start()
        try:
            return read()
        finally:
            timer.cancel()

    def _handle_client(self, conn: socket.socket) -> None:
        self._last_activity = time.monotonic()
        transport.set_nodelay(conn)
        reader = frames.FrameReader(conn)
        authorized = False
        persistent = False
        try:
            while self._running:
                # A primeira leitura tem o deadline curto contra slowloris; as
                # seguintes usam o idle timeout da ligação persistente.
                conn.settimeout(CLIENT_IDLE_TIMEOUT_SEC if persistent else REQUEST_READ_TIMEOUT_SEC)
                try:
                    head = reader.peek(1)
                except TimeoutError:
                    return
                if not head:
                    return

                if head != b"V":
                    # Qualquer outro byte é tratado como linha JSON (o caminho
                    # antigo). Isto preserva o erro "JSON inválido" para lixo
                    # textual em vez de um "protocolo desconhecido" opaco.
                    try:
                        line = self._read_with_deadline(conn, lambda: reader.read_line(max_bytes=P.MAX_REQUEST_BYTES))
                    except frames.FrameError as exc:
                        with contextlib.suppress(OSError):
                            self._send_json(conn, self._error(str(exc), error_code=P.ERR_INVALID_REQUEST))
                        return
                    if not line:
                        return
                    try:
                        request = json.loads(line)
                    except json.JSONDecodeError as e:
                        with contextlib.suppress(OSError):
                            self._send_json(
                                conn,
                                self._error(f"JSON inválido: {e}", error_code=P.ERR_INVALID_REQUEST),
                            )
                        return
                    if not isinstance(request, dict):
                        with contextlib.suppress(OSError):
                            self._send_json(
                                conn,
                                self._error(
                                    f"request deve ser um objecto JSON (recebi {type(request).__name__})",
                                    error_code=P.ERR_INVALID_REQUEST,
                                ),
                            )
                        return
                    if not authorized:
                        if not transport.is_authorized(request, self._auth_token):
                            with contextlib.suppress(OSError):
                                self._send_json(
                                    conn,
                                    self._error(
                                        "token do endpoint ausente ou inválido",
                                        error_code=P.ERR_UNAUTHORIZED,
                                        hint=(
                                            "Usa o cliente vramd (lê o token do descritor) — não ligues à porta à mão."
                                        ),
                                    ),
                                )
                            return
                        authorized = True
                    request.pop("auth", None)
                    keep = bool(request.pop("keepalive", False))
                    # Resposta pode ser longa (stream NDJSON): restaura o
                    # timeout generoso; o loop seguinte volta ao idle timeout.
                    conn.settimeout(P.DEFAULT_GENERATE_TIMEOUT_SEC)
                    self._serve_json_request(conn, request)
                    # Uma ligação persistente activa renova o idle: sem isto o
                    # supervisor fazia auto-shutdown a meio de um cliente que
                    # continua a mandar pedidos (só contava o accept inicial).
                    self._last_activity = time.monotonic()
                    if not keep:
                        return
                    persistent = True
                    continue

                if head == b"V":
                    try:
                        frame = self._read_with_deadline(
                            conn, lambda: reader.read_frame(max_payload=frames.MAX_DATA_BYTES)
                        )
                    except frames.FrameError as exc:
                        with contextlib.suppress(OSError):
                            self._send_frame_error(conn, str(exc), 0, P.ERR_PROTOCOL)
                        return
                    if frame is None:
                        return
                    if frame.type != frames.TYPE_JSON:
                        with contextlib.suppress(OSError):
                            self._send_frame_error(
                                conn,
                                f"esperava frame JSON, recebi {frame.type}",
                                frame.request_id,
                                P.ERR_PROTOCOL,
                            )
                        continue
                    try:
                        request = frames.decode_json(frame.payload)
                    except frames.FrameError as exc:
                        with contextlib.suppress(OSError):
                            self._send_frame_error(conn, str(exc), frame.request_id, P.ERR_PROTOCOL)
                        continue
                    if not authorized:
                        if not transport.is_authorized(request, self._auth_token):
                            with contextlib.suppress(OSError):
                                self._send_frame_error(
                                    conn,
                                    "token do endpoint ausente ou inválido",
                                    frame.request_id,
                                    P.ERR_UNAUTHORIZED,
                                )
                            return
                        authorized = True
                    request.pop("auth", None)
                    request.pop("keepalive", None)
                    conn.settimeout(P.DEFAULT_GENERATE_TIMEOUT_SEC)
                    keep_connection = self._serve_frame_request(reader, conn, request, frame.request_id)
                    self._last_activity = time.monotonic()
                    if keep_connection is False:
                        # Upload abortado antes do END: há frames pendentes.
                        return
                    persistent = True
                    continue

        except (ConnectionError, OSError) as e:
            self._log(f"Cliente desligou: {e}")
        except Exception as e:
            self._log(f"Erro ao processar cliente: {e}")
            with contextlib.suppress(OSError):
                self._send_json(
                    conn,
                    self._error(str(e), error_code=P.ERR_GENERATE_FAILED),
                )
        finally:
            with contextlib.suppress(OSError):
                conn.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        self._log("Cleanup...")
        # Hook de shutdown primeiro: ainda há estado para o payload reportar.
        # drain(): sem isto o interpretador matava a thread do hook ao meio —
        # o evento que existe para integrações de encerramento raramente corria.
        with contextlib.suppress(Exception):
            self.hooks.dispatch(EVENT_SHUTDOWN, {"pid": self._pid})
            self.hooks.drain(timeout_sec=3.0)
        with contextlib.suppress(Exception):
            self.learner.stop()
        with contextlib.suppress(Exception):
            self.remote.stop_janitor()
            self.sessions.stop_janitor()
        with contextlib.suppress(Exception):
            self.workers.stop()
        with contextlib.suppress(Exception):
            self.idle_evictor.stop()
        with contextlib.suppress(Exception):
            self.peers.stop()
        for lane in self._extra_lanes():
            if lane.idle_evictor is not None:
                with contextlib.suppress(Exception):
                    lane.idle_evictor.stop()
        with contextlib.suppress(Exception):
            self._evict_all_lanes()
        # Terminar workers subprocesso (subprocess-per-backend, Fase 3-4) — de
        # todas as lanes: um worker de GPU 1 esquecido fica a segurar a placa.
        for lane in self.lanes.all_lanes():
            pool = getattr(lane.manager, "_subprocess_pool", None)
            if pool is not None:
                with contextlib.suppress(Exception):
                    pool.shutdown_all()
        if self._bound:
            # Só remover socket/pid se fomos nós a criá-los (bind OK) — um
            # double-start que falhou o bind não pode apagar os do vramd vivo.
            transport.remove_endpoint(self.socket_path)
            with contextlib.suppress(OSError):
                self.ppid_path.unlink(missing_ok=True)
            self._bound = False
        self._auth_token = None
        if self._server_sock is not None:
            with contextlib.suppress(OSError):
                self._server_sock.close()
            self._server_sock = None
        self._singleton.release()

    def _signal_handler(self, signum: int, frame: Any) -> None:
        self._log(f"Sinal {signum} recebido — a encerrar.")
        self._running = False

    def serve_forever(self) -> None:
        """Arranca o vramd (bloqueante). Graceful shutdown via SIGTERM/SIGINT."""
        _ensure_server_dir()

        # Singleton ANTES de mexer no socket: o probe ``is_server_running``
        # falha num supervisor vivo mas ocupado, e o unlink+bind a seguir criava
        # supervisores paralelos invisíveis (cada um com os seus workers e VRAM).
        if not self._singleton.acquire():
            owner = self._singleton.owner_pid()
            raise RuntimeError(
                f"[{P.ERR_ALREADY_RUNNING}] UMS já ativo (PID {owner or '?'}) — "
                f"lock {self._singleton.path}. "
                "Usa `vramd status` / `vramd stop`; não arranques um segundo supervisor."
            )

        # Com o lock nas mãos, qualquer outro processo da família UMS é lixo de
        # uma run anterior (supervisor zombie ou worker órfão a segurar VRAM).
        if P.REAP_ON_START:
            with contextlib.suppress(Exception):
                report = self.reap_strays()
                if report.get("count"):
                    _logger.warn(
                        f"[vramd] arranque: {report['count']} processo(s) órfão(s) terminado(s) "
                        f"(~{report.get('vram_mib_freed')} MiB de VRAM recuperados)."
                    )

        if self.socket_path.exists():
            # Detemos o lock ⇒ não existe supervisor legítimo; socket é stale.
            with contextlib.suppress(OSError):
                self.socket_path.unlink(missing_ok=True)
                self.ppid_path.unlink(missing_ok=True)

        self._server_sock, self._endpoint_info = transport.listen(
            self.socket_path,
            bind_host=self._listen[0] if self._listen else None,
            port=self._listen[1] if self._listen else 0,
            token=self._token,
        )
        self._auth_token = self._endpoint_info.token
        self._bound = True
        # PID file só depois do bind: um double-start não pode clobber o pid do vramd vivo.
        self.ppid_path.write_text(str(self._pid))

        host.install_signal_handlers(self._signal_handler)

        self._running = True
        replayed = self.queue.replay_from_wal()
        if replayed:
            _logger.info(f"WAL replay: {replayed} job(s) re-enfileirados")
        self.workers.start()
        self.remote.start_janitor()
        self.sessions.start_janitor()
        self.peers.start()
        for lane in self._extra_lanes():
            if lane.idle_evictor is not None:
                lane.idle_evictor.start()
        if self.router.active:
            lanes_txt = ", ".join(lane.lane_id for lane in self.lanes.device_lanes)
            peers_txt = ", ".join(s.spec.name for s in self.peers.states()) or "nenhum"
            # Parênteses, não [..]: o rich leria os nomes como markup e apagava-os.
            _logger.info(f"Router activo: GPUs ({lanes_txt}); peers ({peers_txt})")
        label = transport.endpoint_label(self.socket_path)
        _logger.info(f"vramd ativo em {label} ({self._endpoint_info.transport}, {self.socket_path}) (PID {self._pid})")
        _logger.info(f"Backends registados: {', '.join(self.registry.names)}")
        _logger.info(
            f"Fila: max_depth={self.queue.max_depth}, max_inflight={self.workers.capacity()}, "
            f"affinity_cuts={self.scheduler.max_cuts}"
        )
        _logger.info(f"Idle timeout: {self.idle_timeout_sec / 60:.0f} min")
        _logger.info(
            f"Idle evictor: unload após {self.idle_evictor.idle_timeout_sec:.0f}s sem uso, "
            f"worker terminado após {self.idle_evictor.worker_shutdown_sec:.0f}s"
        )

        self.idle_evictor.start()
        self.learner.start()
        if self.learner.enabled:
            _logger.info(
                f"Learn: amostragem de picos a cada {self.learner.interval_sec:.2f}s "
                "(vramd learn para o drift; VRAMD_LEARN_INTERVAL_SEC=0 desliga)"
            )
        if self.hooks.specs:
            names = sorted({e for spec in self.hooks.specs for e in spec.events})
            _logger.info(f"Hooks ativos: {', '.join(names)}")

        # Teto de handlers concorrentes: thread-per-connection sem limite é
        # exaurível por um cliente que abre ligações em rajada (DoS de threads).
        client_slots = threading.BoundedSemaphore(MAX_CONCURRENT_CLIENTS)
        try:
            while self._running:
                idle = time.monotonic() - self._last_activity
                # NUNCA auto-shutdown com trabalho em curso: `_last_activity` só
                # é renovado no início de cada conexão, e um generate/wait longo
                # (> idle_timeout) sem outro tráfego mataria o worker a meio.
                if (
                    self.idle_timeout_sec > 0
                    and self._requests_served > 0
                    and idle > self.idle_timeout_sec
                    and not self.queue.is_busy()
                ):
                    _logger.info(f"Idle {idle / 60:.0f}min > timeout — a encerrar.")
                    break
                try:
                    conn, _ = self._server_sock.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                if not client_slots.acquire(blocking=False):
                    # Saturado: recusar já com resposta estruturada em vez de
                    # deixar o cliente à espera de um timeout silencioso.
                    with contextlib.suppress(OSError):
                        self._send_json(
                            conn,
                            self._error(
                                f"servidor saturado ({MAX_CONCURRENT_CLIENTS} conexões) — tenta já",
                                error_code=P.ERR_SERVER_BUSY,
                            ),
                        )
                    with contextlib.suppress(OSError):
                        conn.close()
                    continue

                def _serve(c: socket.socket, slots: threading.BoundedSemaphore) -> None:
                    try:
                        self._handle_client(c)
                    finally:
                        slots.release()

                try:
                    t = threading.Thread(target=_serve, args=(conn, client_slots), daemon=True)
                    t.start()
                except RuntimeError:
                    # Não há recursos para mais threads (RLIMIT/OOM a caminho):
                    # libertar o slot e a conexão em vez de matar o supervisor.
                    client_slots.release()
                    with contextlib.suppress(OSError):
                        conn.close()
        finally:
            self._cleanup()
            _logger.info("vramd encerrado.")
