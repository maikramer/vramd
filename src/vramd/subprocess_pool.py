"""SubprocessWorkerPool — gestão de workers subprocesso persistentes por backend.

Cada backend com ``tool:`` definido no ``backends.yaml`` corre num subprocesso
próprio, no venv da tool (ex.: ``Text3D/.venv/bin/python -m text3d serve --ums-worker``).
O worker é **persistente**: carrega o modelo no arranque (``load``) e mantém-se
vivo entre jobs (``generate``); evict = ``unload`` (descarrega pesos); idle
timeout = ``shutdown``.

Comunicação via stdin/stdout JSONL (ver :mod:`vramd.worker.protocol`):

- UMS → Worker (stdin): ``{"cmd":"load","kwargs":{...}}``, ``{"cmd":"generate",...}``,
  ``{"cmd":"unload"}``, ``{"cmd":"abort"}``, ``{"cmd":"shutdown"}``.
- Worker → UMS (stdout): ``{"event":"ready","vram_mib":...}``, ``{"event":"progress",...}``,
  ``{"event":"done","result":{...}}``, etc.
- stderr → ficheiro de log por backend (``~/.cache/vramd/vramd-worker-<backend>.log``).

Abort cooperativo (``{"cmd":"abort"}`` no stdin); SIGTERM como fallback após
``abort_timeout_sec`` sem ``done``. Worker morto inesperadamente → re-spawn +
requeue do job (ver :meth:`SubprocessWorkerPool.send_generate`).
"""

from __future__ import annotations

import contextlib
import io
import os
import queue
import select
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vramd import host
from vramd.logging import Logger
from vramd.worker.protocol import (
    CMD_ABORT,
    CMD_GENERATE,
    CMD_LOAD,
    CMD_PING,
    CMD_SHUTDOWN,
    CMD_UNLOAD,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_PONG,
    EVENT_READY,
    EVENT_UNLOADED,
    decode,
)

_logger = Logger()

# Idle entre eventos de progresso (reset a cada progress). Hunyuan pode
# demorar muitos minutos no total — NÃO é wall-clock do generate inteiro.
DEFAULT_EVENT_TIMEOUT_SEC = 600.0

# Watcher de OOM-spin: o allocator CUDA em retry infinito não emite eventos —
# cresce no stderr do worker (~0.7 linhas/s de "expandable_segments: memory
# mapping failed"). 8 linhas + 30s sem progress denunciam o wedge em ~40s
# em vez dos 600s do idle-timeout (que depois virava "timeout do cliente").
_OOM_SPIN_PATTERNS = ("expandable_segments: memory mapping failed", "CUDA out of memory")
_OOM_SPIN_MIN_LINES = 8
_OOM_SPIN_QUIET_SEC = 30.0
# Janela rolante: só contam OOMs dos últimos N segundos. OOMs transitórios
# espaçados (ex. 8 em 266s durante o xatlas — o pipeline é mudo por dentro)
# não podem matar um job saudável; o spin real produz ~0.7 linhas/s.
_OOM_SPIN_WINDOW_SEC = 120.0
# Reciclagem do worker persistente: fragmentação/estado CUDA acumulam entre
# jobs — após N jobs o worker morre e o próximo load re-spawna fresco.
DEFAULT_WORKER_MAX_JOBS = 12  # 10 min sem progress → force abort
# Após progress pct≥1.0 / msg "done": o adapter já acabou o GPU work — só
# falta ``EVENT_DONE``. NÃO renovar o idle de 600s (era o hang de ~7-10 min
# com UI a 100% done e ``completed=0``). Grace curto para scrub+emit.
DEFAULT_POST_DONE_TIMEOUT_SEC = 90.0
# Após CMD_ABORT: se worker não emitir done/error, SIGTERM (antes: ficava
# preso até EVENT_TIMEOUT — text3d a 10% engolia cancel --all / text2d).
DEFAULT_ABORT_TIMEOUT_SEC = 15.0
DEFAULT_LOAD_TIMEOUT_SEC = 300.0  # 5 min para carregar modelo
DEFAULT_PING_TIMEOUT_SEC = 5.0


# Linha máxima aceitável do stdout do worker. Uma linha maior (worker runaway
# a escrever sem \n) é lixo de protocolo: saltar em vez de alocar sem limite.
_MAX_LINE_BYTES = 1024 * 1024


def _readline_nonblocking(stdout: Any, fd: int | None) -> str:
    """Lê 1 linha se já estiver em buffer user-space; senão ``\"\"`` sem bloquear.

    ``select`` no fileno **não** vê linhas que o ``TextIOWrapper`` já puxou
    do kernel no ``readline`` anterior (progress+done no mesmo ``read``). Com
    o write-end aberto (worker vivo), ``select`` nunca acorda e o job trava a
    100% ``done``. ``O_NONBLOCK`` + ``readline`` drena esse buffer; vazio → ``\"\"``.
    """
    if fd is None:
        # Fakes de teste (sem fileno): readline já é não-bloqueante.
        line = stdout.readline()
        if isinstance(line, bytes):
            return line.decode("utf-8", errors="replace")
        return line or ""
    if host.IS_WINDOWS:
        # No Windows o caminho é a thread leitora (``state.reader``); se um
        # stream real chegasse aqui, um readline bloquearia o worker pool.
        return ""
    import fcntl

    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    try:
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            line = stdout.readline()
        except BlockingIOError:
            return ""
    finally:
        with contextlib.suppress(Exception):
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)
    if isinstance(line, bytes):
        return line.decode("utf-8", errors="replace")
    return line or ""


def _progress_is_terminal(pct: Any, msg: Any) -> bool:
    """Progress que significa "GPU work acabou; espera-se EVENT_DONE já"."""
    try:
        if pct is not None and float(pct) >= 1.0:
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(msg, str) and msg.strip().lower() == "done"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


SpawnFn = Callable[[list[str], Any, Any, Any], subprocess.Popen]
LogPathFn = Callable[[str], Path]


def _default_log_path(backend: str) -> Path:
    return host.worker_log_path(backend)


class _ThreadedLineReader:
    """Leitor de stdout por thread — o ``select`` do Windows não vê pipes.

    O POSIX usa ``select`` + ``O_NONBLOCK`` (sem thread por worker). No Windows
    ``select`` só aceita sockets, pelo que a leitura bloqueante vive numa thread
    daemon e o supervisor consome linhas com timeout de uma ``queue.Queue``.
    """

    _EOF = object()

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=4096)
        self._closed = False
        self._eof_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="vramd-worker-stdout", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                line = self._stream.readline()
                if not line:
                    break
                try:
                    self._queue.put(line, timeout=1.0)
                except queue.Full:
                    # Worker a inundar o protocolo: bloquear para trás é melhor
                    # que crescer sem limite. Só desiste se o leitor fechou.
                    while not self._closed:
                        with contextlib.suppress(queue.Full):
                            self._queue.put(line, timeout=0.5)
                            break
        except (OSError, ValueError):
            pass
        finally:
            self._eof_event.set()
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(self._EOF)

    def read(self, timeout: float) -> str | None:
        """Próxima linha, ``None`` se o timeout expirou ou o stream fechou."""
        try:
            item = self._queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None
        if item is self._EOF:
            # Voltar a pôr o EOF: leitores seguintes também o vêem.
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(self._EOF)
            return None
        return item

    @property
    def eof(self) -> bool:
        """True se o stream terminou (o worker fechou o stdout)."""
        return self._closed or self._eof_event.is_set()

    def close(self) -> None:
        self._closed = True


def _default_spawn(
    cmd: list[str],
    stdin: Any,
    stdout: Any,
    stderr: Any,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> subprocess.Popen:
    """Spawn por omissão: sessão própria, stdout=PIPE, stderr=ficheiro, stdin=PIPE.

    A sessão própria isola o worker do Ctrl+C do terminal do vramd (o abort é
    cooperativo, via ``{"cmd":"abort"}``). Para o worker não sobreviver à morte
    do supervisor há duas redes: EOF no stdin e o watchdog de PPID em
    :func:`vramd.worker.serve.start_parent_watchdog`. Não se usa
    ``PR_SET_PDEATHSIG`` porque no Linux dispara com a morte da *thread* que
    criou o processo — e o spawn acontece nas threads do ``WorkerPool``.
    """
    env_out = env if env is not None else os.environ.copy()
    # O protocolo é JSON com ``ensure_ascii=False``: um prompt/mensagem com
    # caracteres fora do cp1252 rebentava o worker no Windows se os pipes
    # ficassem na encoding do locale. UTF-8 explícito nos dois lados.
    env_out.setdefault("PYTHONUTF8", "1")
    env_out.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.Popen(
        cmd,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,  # line-buffered
        **host.worker_popen_kwargs(),
        env=env_out,
        cwd=cwd,
    )


@dataclass
class _WorkerState:
    """Estado de um worker persistente (um por backend)."""

    backend: str
    proc: subprocess.Popen | None = None
    load_shape: dict[str, Any] = field(default_factory=dict)
    loaded: bool = False  # modelo carregado (após ``ready``)
    vram_mib: int | None = None  # último reportado pelo worker
    lock: threading.RLock = field(default_factory=threading.RLock)
    log_path: Path | None = None
    # File handle do stderr do worker (aberto em _spawn, fechado em
    # shutdown/_force_abort — antes ficava órfão e só era recuperado por GC).
    log_fh: Any = None
    jobs_served: int = 0
    # Diagnóstico de ciclo de vida (status/debug e mensagens de erro).
    spawn_count: int = 0
    recycle_count: int = 0
    spawned_at: float | None = None
    last_exit_code: int | None = None
    last_abort_reason: str | None = None
    last_job_id: str | None = None
    last_progress_pct: float | None = None
    last_progress_msg: str | None = None
    last_event_at: float | None = None
    # PGID capturado no spawn: continua válido depois de o líder morrer e
    # permite matar filhos/netos órfãos (crash/OOM do worker).
    pgid: int | None = None
    # Leitor por thread (Windows): ``select`` não funciona em pipes.
    reader: _ThreadedLineReader | None = None


class SubprocessWorkerError(Exception):
    """Erro de comunicação com worker subprocesso."""


class SubprocessWorkerPool:
    """Gere workers subprocesso persistentes para backends com ``tool:`` definido.

    Threading: cada chamada bloqueante (``load``/``generate``/``unload``/
    ``shutdown``) obtém o lock do backend; o vramd chama-as a partir da
    ``WorkerPool`` que já tem ``MAX_INFLIGHT`` (default 1).
    """

    def __init__(
        self,
        *,
        spawn_fn: SpawnFn = _default_spawn,
        log_path_fn: LogPathFn = _default_log_path,
        load_timeout_sec: float = DEFAULT_LOAD_TIMEOUT_SEC,
        event_timeout_sec: float | None = None,
        abort_timeout_sec: float | None = None,
        post_done_timeout_sec: float | None = None,
        ping_timeout_sec: float = DEFAULT_PING_TIMEOUT_SEC,
        python_override: dict[str, str] | None = None,
        worker_max_jobs: int | None = None,
    ) -> None:
        self._spawn_fn = spawn_fn
        self._log_path_fn = log_path_fn
        self._load_timeout = float(load_timeout_sec)
        self._event_timeout = float(
            event_timeout_sec
            if event_timeout_sec is not None
            else _env_float("VRAMD_EVENT_TIMEOUT_SEC", DEFAULT_EVENT_TIMEOUT_SEC)
        )
        self._abort_timeout = float(
            abort_timeout_sec
            if abort_timeout_sec is not None
            else _env_float("VRAMD_ABORT_TIMEOUT_SEC", DEFAULT_ABORT_TIMEOUT_SEC)
        )
        self._post_done_timeout = float(
            post_done_timeout_sec
            if post_done_timeout_sec is not None
            else _env_float("VRAMD_POST_DONE_TIMEOUT_SEC", DEFAULT_POST_DONE_TIMEOUT_SEC)
        )
        self._worker_max_jobs = int(
            worker_max_jobs
            if worker_max_jobs is not None
            else _env_float("VRAMD_WORKER_MAX_JOBS", DEFAULT_WORKER_MAX_JOBS)
        )
        self._ping_timeout = float(ping_timeout_sec)
        # Override do interpretador python por backend (testes / ambientes exóticos).
        self._python_override: dict[str, str] = dict(python_override or {})
        # Estado por backend (criado on-demand).
        self._workers: dict[str, _WorkerState] = {}
        # RuntimeSpec por backend (bloco ``runtime:`` do YAML v2). Guardado no
        # ``load`` e reutilizado no re-spawn, que não recebe o descriptor.
        self._runtimes: dict[str, Any] = {}
        self._pool_lock = threading.RLock()

    # ------------------------------------------------------------------
    # API pública — chamada pelo BackendManager
    # ------------------------------------------------------------------

    def _safe_send(self, backend: str, state: _WorkerState, cmd: str, **payload: Any) -> None:
        """Envia um comando ao worker; pipe morto → ``SubprocessWorkerError``.

        A mensagem contém «não está vivo» de propósito: o dispatcher classifica
        por string e faz requeue transitório — um ``BrokenPipeError`` cru lá
        chegava como GENERATE_FAILED e o job falhava sem o retry de respawn.
        """
        from vramd.worker.protocol import send_cmd

        stdin = getattr(state.proc, "stdin", None) if state.proc is not None else None
        if stdin is None:
            raise SubprocessWorkerError(f"{backend}: worker não está vivo (pipe fechado)")
        try:
            send_cmd(stdin, cmd, **payload)
        except (OSError, ValueError) as e:
            raise SubprocessWorkerError(f"{backend}: worker não está vivo (pipe fechado: {e})") from e

    @staticmethod
    def _close_proc_pipes(state: _WorkerState) -> None:
        """Fecha os pipes stdin/stdout do Popen (o GC sozinho deixava fds órfãos).

        Também faz ``wait()`` final para recolher o zombie — sem isto cada
        abort/shutdown deixava um <defunct> até ao GC do Popen (que ainda
        rebentava com BrokenPipeError no flush do stdin — warning no pytest).
        """
        proc = state.proc
        if proc is None:
            return
        if state.reader is not None:
            with contextlib.suppress(Exception):
                state.reader.close()
            state.reader = None
        for pipe in (proc.stdin, proc.stdout):
            with contextlib.suppress(OSError, ValueError, AttributeError):
                pipe.close()
        with contextlib.suppress(Exception):
            if proc.poll() is not None:
                proc.wait(timeout=1.0)

    def load(
        self,
        backend: str,
        tool: str,
        kwargs: dict[str, Any],
        *,
        on_progress: Callable[[float | None, str | None], None] | None = None,
        runtime: Any = None,
    ) -> dict[str, Any]:
        """Arranca o worker e carrega o modelo; retorna o evento ``ready``.

        Reutiliza worker já vivo com a mesma ``load_shape``; recarrega (sem re-spawn)
        se o worker está vivo mas descarregado; re-spawn se o worker morreu.

        Args:
            runtime: :class:`~vramd.registry.RuntimeSpec` do backend. Define
                comando/cwd/env do worker; ``None`` = comando derivado do
                checkout (comportamento legado, backends do monorepo).
        """
        if runtime is not None:
            self._runtimes[backend] = runtime
        with self._pool_lock:
            state = self._workers.get(backend)
            if state is None:
                state = _WorkerState(backend=backend, log_path=self._log_path_fn(backend))
                self._workers[backend] = state

        with state.lock:
            # Worker vivo e carregado com mesma shape → noop.
            if (
                state.proc
                and state.proc.poll() is None
                and state.loaded
                and not _shape_mismatch(state.load_shape, kwargs)
            ):
                return {"ready": True, "vram_mib": state.vram_mib, "reused": True}

            # Spawn se necessário.
            if state.proc is None or state.proc.poll() is not None:
                try:
                    self._spawn(backend, tool, state)
                except OSError as e:
                    raise SubprocessWorkerError(f"{backend}: spawn do worker falhou ({e})") from e

            # Enviar load.
            self._safe_send(backend, state, CMD_LOAD, kwargs=kwargs)
            event = self._wait_event(
                state,
                expected={EVENT_READY, EVENT_ERROR},
                timeout=self._load_timeout,
                on_progress=on_progress,
            )
            if event is None:
                if state.proc is not None and state.proc.poll() is None:
                    # Timeout com processo VIVO: o worker está wedged a meio do
                    # load. Deixá-lo vivo era a desincronização de protocolo —
                    # o próximo load mandava um 2.º CMD_LOAD para um worker
                    # ainda a processar o 1.º. Matar agora; o próximo load
                    # faz spawn fresco.
                    _logger.warn(
                        f"[vramd] worker {backend}: load timeout ({self._load_timeout:.0f}s) — a forçar respawn."
                    )
                    self._force_abort(state, backend, reason="load_timeout")
                    raise SubprocessWorkerError(f"{backend}: load timeout ({self._load_timeout:.0f}s) — worker morto")
                raise SubprocessWorkerError(f"{backend}: EOF no load (worker morreu)")
            if event["event"] == EVENT_ERROR:
                raise SubprocessWorkerError(f"{backend}: load falhou — {event.get('error')}")
            state.loaded = True
            state.load_shape = dict(kwargs)
            state.vram_mib = event.get("vram_mib")
            return event

    def generate(
        self,
        backend: str,
        request: dict[str, Any],
        *,
        on_progress: Callable[[float | None, str | None], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Envia ``generate`` e bloqueia até ``done``.

        ``on_progress``/``should_abort`` são chamados a partir dos eventos do
        worker (``progress``) e do poll a ``should_abort`` (que envia
        ``{"cmd":"abort"}`` quando True). ``job_id`` (ou ``request["_job_id"]``)
        é só diagnóstico: entra nas mensagens de timeout/abort/OOM para
        correlacionar com a fila.
        """
        job_id = job_id or request.get("_job_id")
        if job_id is not None:
            job_id = str(job_id)
        with self._pool_lock:
            state = self._workers.get(backend)
        if state is None or state.proc is None or state.proc.poll() is not None:
            raise SubprocessWorkerError(f"{backend}: worker não está vivo — faz load primeiro")

        with state.lock:
            # Re-verificar sob o lock: entre o check exterior e aqui um shutdown
            # concorrente pode ter posto state.proc a None (TOCTOU → AttributeError).
            if state.proc is None or state.proc.poll() is not None:
                raise SubprocessWorkerError(f"{backend}: worker não está vivo — faz load primeiro")

            state.last_job_id = job_id
            state.last_progress_pct = None
            state.last_progress_msg = None
            state.last_event_at = None

            # Strip hooks in-process antes de serializar para JSONL — o worker
            # recebe-os via request e reconstrói os seus próprios a partir do
            # estado interno (state["abort"] + emissor de progress).
            serializable_request = {k: v for k, v in request.items() if not k.startswith("_") and not callable(v)}
            self._safe_send(backend, state, CMD_GENERATE, request=serializable_request)
            # Idle timeout por backend (RuntimeSpec.event_timeout_sec) — senão default.
            rt = self._runtimes.get(backend)
            ev_timeout = float(getattr(rt, "event_timeout_sec", None) or 0.0) or self._event_timeout
            # Idle timeout: renovado a cada progress (generate longo OK).
            idle_deadline = time.monotonic() + ev_timeout
            abort_sent = False
            abort_deadline: float | None = None
            oom_ts: deque[float] = deque()
            oom_offset = self._worker_log_size(state)
            last_progress = time.monotonic()
            try:
                while True:
                    now = time.monotonic()
                    if now >= idle_deadline:
                        self._force_abort(state, backend, reason="idle")
                        raise SubprocessWorkerError(
                            f"{backend}: worker wedged — idle {ev_timeout:.0f}s sem progress{self._ctx(state)}"
                        )
                    # Poll cooperativo: se o caller pediu abort, enviar ao worker.
                    if not abort_sent and should_abort and should_abort():
                        with contextlib.suppress(OSError, ValueError):
                            # Worker pode ter morrido entretanto — o idle/poll
                            # check a seguir apanha-o; não deixar BrokenPipe
                            # saltar daqui.
                            from vramd.worker.protocol import send_cmd

                            send_cmd(state.proc.stdin, CMD_ABORT)  # type: ignore[union-attr]
                        abort_sent = True
                        abort_deadline = now + self._abort_timeout
                        _logger.info(
                            f"[vramd] worker {backend}: abort pedido — SIGTERM se sem done"
                            f" em {self._abort_timeout:.0f}s{self._ctx(state)}"
                        )
                    # Escalação: abort cooperativo ignorado (text3d mid image_to_3d).
                    if abort_sent and abort_deadline is not None and now >= abort_deadline:
                        self._force_abort(state, backend, reason="abort_timeout")
                        raise SubprocessWorkerError(
                            f"{backend}: abort timeout ({self._abort_timeout:.0f}s) — "
                            f"worker forçado (SIGTERM){self._ctx(state)}"
                        )
                    wait = min(idle_deadline - now, 1.0)
                    if abort_deadline is not None:
                        wait = min(wait, max(0.05, abort_deadline - now))
                    event = self._read_event_with_timeout(state, timeout=wait)
                    if event is None:
                        # Silêncio ≠ morte: só falha se o processo já saiu.
                        if state.proc is None or state.proc.poll() is not None:
                            state.loaded = False
                            state.last_exit_code = state.proc.poll() if state.proc is not None else None
                            raise SubprocessWorkerError(
                                f"{backend}: worker fechou stdout mid-generate{self._ctx(state)}"
                            )
                        # Watcher de OOM-spin: o allocator em retry não emite
                        # eventos — as linhas de OOM crescem no stderr.
                        n, oom_offset = self._scan_worker_oom(state, oom_offset)
                        if n:
                            oom_ts.extend([now] * n)
                        # Janela rolante: OOMs antigos expiram — densidade
                        # recente é o sinal do spin, não o total acumulado.
                        while oom_ts and now - oom_ts[0] > _OOM_SPIN_WINDOW_SEC:
                            oom_ts.popleft()
                        if len(oom_ts) >= _OOM_SPIN_MIN_LINES and (now - last_progress) >= _OOM_SPIN_QUIET_SEC:
                            self._force_abort(state, backend, reason="oom_spin")
                            raise SubprocessWorkerError(
                                f"{backend}: worker wedged — OOM-spin ({len(oom_ts)} allocs, "
                                f"{now - last_progress:.0f}s sem progress){self._ctx(state)}"
                            )
                        continue
                    ev = event["event"]
                    state.last_event_at = time.monotonic()
                    if ev == "progress":
                        last_progress = time.monotonic()
                        pct, msg = event.get("pct"), event.get("msg")
                        if pct is not None:
                            with contextlib.suppress(TypeError, ValueError):
                                state.last_progress_pct = float(pct)
                        if msg is not None:
                            state.last_progress_msg = str(msg)
                        # Progress terminal NÃO renova os 600s — senão um DONE
                        # perdido/atrasado segura inflight ~10 min com UI a 100%.
                        if _progress_is_terminal(pct, msg):
                            idle_deadline = time.monotonic() + self._post_done_timeout
                        else:
                            idle_deadline = time.monotonic() + ev_timeout
                        if on_progress:
                            with contextlib.suppress(Exception):
                                on_progress(pct, msg)
                        continue
                    if ev == "vram_budget":
                        state.vram_mib = event.get("vram_mib", state.vram_mib)
                        continue
                    if ev == EVENT_DONE:
                        result = event.get("result", {})
                        return result
                    if ev == EVENT_ERROR:
                        raise SubprocessWorkerError(
                            f"{backend}: generate erro — {event.get('error')} ({event.get('error_code')})"
                        )
                    # Evento inesperado (pong, ready, unloaded): ignorar.
            finally:
                # Reciclagem: worker persistente acumula fragmentação/estado
                # CUDA entre jobs (OOM residual, alocador partido) — após N
                # jobs morre e o próximo load re-spawna fresco.
                state.jobs_served += 1
                max_jobs = self._max_jobs_for(backend)
                alive = state.proc is not None and state.proc.poll() is None
                if max_jobs > 0 and state.jobs_served >= max_jobs and alive:
                    state.recycle_count += 1
                    self._terminate_worker(state, backend, reason="recycle")

    def unload(self, backend: str) -> bool:
        """Manda o worker descarregar o modelo (worker persiste vivo)."""
        with self._pool_lock:
            state = self._workers.get(backend)
        if state is None or state.proc is None or state.proc.poll() is not None:
            return False
        with state.lock:
            try:
                self._safe_send(backend, state, CMD_UNLOAD)
                event = self._wait_event(state, expected={EVENT_UNLOADED, EVENT_ERROR}, timeout=60.0)
            except SubprocessWorkerError:
                # Pipe morto a meio do unload: pesos ficam no limbo até o
                # worker morrer/idle-shutdown — marcar não-loaded para o
                # planner não contar com VRAM que não está cá.
                state.loaded = False
                return False
            state.loaded = False
            return event is not None and event.get("event") == EVENT_UNLOADED

    def shutdown(self, backend: str) -> bool:
        """Manda o worker terminar (gracioso)."""
        with self._pool_lock:
            state = self._workers.pop(backend, None)
        if state is None or state.proc is None:
            return False
        with state.lock:
            proc = state.proc
            proc_pid = getattr(proc, "pid", None) if proc is not None else None
            children = host.process_tree_pids(proc_pid) if proc_pid is not None else []
            pgid = host.process_group_id(proc_pid) if proc_pid is not None else None
            try:
                from vramd.worker.protocol import send_cmd

                send_cmd(state.proc.stdin, CMD_SHUTDOWN)
            except Exception:
                pass
            # Espera graciosa 5s; SIGTERM depois.
            try:
                state.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                # O worker pode ter filhos (ex.: _xatlas_worker) que não morrem
                # com o pai; POSIX usa o grupo, Windows a árvore do psutil.
                if proc_pid is not None:
                    with contextlib.suppress(Exception):
                        host.terminate_process_tree(proc_pid, grace_sec=3.0)
                with contextlib.suppress(Exception):
                    state.proc.terminate()
                try:
                    state.proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    if pgid is not None:
                        host.signal_process_group(pgid, signal.SIGKILL)
                    else:
                        with contextlib.suppress(Exception):
                            state.proc.kill()
                    # Recolher o zombie do SIGKILL (wait sem timeout: já é
                    # inevitavelmente mortal a esta altura).
                    with contextlib.suppress(Exception):
                        state.proc.wait(timeout=2.0)
            finally:
                # Mesmo num shutdown gracioso, um filho pode ter ficado vivo
                # (o pai saiu sem o esperar). Matar os descendentes capturados
                # antes de largar o estado; logar se algum resistiu.
                survivors = [
                    p for p in children if p != (proc_pid if proc_pid is not None else -1) and host.pid_exists(p)
                ]
                for pid in survivors:
                    with contextlib.suppress(Exception):
                        host.kill_pid(pid)
                # Qualquer membro do grupo que tenha escapado à captura (um
                # filho criado durante o shutdown) também não pode sobreviver.
                if pgid is not None:
                    host.signal_process_group(pgid, signal.SIGKILL)
                if survivors:
                    _logger.warn(
                        f"[vramd] worker {backend}: {len(survivors)} descendente(s) órfão(s) "
                        f"no shutdown: {survivors} — terminados à força"
                    )
                self._close_proc_pipes(state)
                state.proc = None
                state.loaded = False
                # Fechar o handle do stderr do worker (antes ficava órfão).
                if state.log_fh is not None:
                    with contextlib.suppress(Exception):
                        state.log_fh.close()
                    state.log_fh = None
                state.pgid = None
            return True

    def shutdown_all(self) -> None:
        """Termina todos os workers (shutdown do vramd)."""
        with self._pool_lock:
            backends = list(self._workers)
        for b in backends:
            with contextlib.suppress(Exception):
                self.shutdown(b)

    def is_loaded(self, backend: str) -> bool:
        """True se o worker existe, está vivo E tem modelo carregado."""
        with self._pool_lock:
            state = self._workers.get(backend)
        if state is None or state.proc is None:
            return False
        return state.loaded and state.proc.poll() is None

    def is_alive(self, backend: str) -> bool:
        """True se o subprocesso está vivo (mesmo sem modelo carregado)."""
        with self._pool_lock:
            state = self._workers.get(backend)
        return state is not None and state.proc is not None and state.proc.poll() is None

    def worker_pid(self, backend: str) -> int | None:
        """PID do worker vivo, ou ``None``.

        Usado pela calibração para atribuir VRAM por processo (NVML só reporta
        por PID; sem isto a medição incluiria compositor e vizinhos).
        """
        with self._pool_lock:
            state = self._workers.get(backend)
        if state is None or state.proc is None or state.proc.poll() is not None:
            return None
        return int(state.proc.pid)

    def vram_mib(self, backend: str) -> int | None:
        """Última VRAM reportada pelo worker (None se desconhecida)."""
        with self._pool_lock:
            state = self._workers.get(backend)
        return state.vram_mib if state else None

    def loaded_backends(self) -> set[str]:
        """Backends com modelo carregado (para o planner do BackendManager)."""
        with self._pool_lock:
            return {b for b, s in self._workers.items() if s.loaded and s.proc and s.proc.poll() is None}

    def ping(self, backend: str) -> bool:
        """Health check: envia ``ping`` e espera ``pong``."""
        with self._pool_lock:
            state = self._workers.get(backend)
        if state is None or state.proc is None or state.proc.poll() is not None:
            return False
        with state.lock:
            try:
                self._safe_send(backend, state, CMD_PING)
                event = self._wait_event(state, expected={EVENT_PONG, EVENT_ERROR}, timeout=self._ping_timeout)
            except SubprocessWorkerError:
                return False
            return event is not None and event.get("event") == EVENT_PONG

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _worker_cmd(self, backend: str, tool: str) -> list[str]:
        """Argv do worker: ``runtime:`` do YAML, senão o venv da tool no checkout.

        Raises:
            SubprocessWorkerError: Nenhuma das vias resolve — sem comando não há
                worker, e falhar aqui dá uma mensagem acionável em vez de um
                ``FileNotFoundError`` do ``Popen``.
        """
        override = self._python_override.get(tool)
        if override:
            return [override, "-m", tool, "serve", "--ums-worker"]

        runtime = self._runtimes.get(backend)
        if runtime is not None:
            cmd = runtime.resolve_command(tool=tool)
            if cmd:
                return cmd
            raise SubprocessWorkerError(
                f"{backend}: runtime.command não resolve "
                f"(referência a env var ou venv inexistente): {list(runtime.command or [])}"
            )

        python = _resolve_tool_python(tool)
        if python is None:
            raise SubprocessWorkerError(f"{backend}: venv da tool {tool!r} não encontrado — corre ./install.sh {tool}")
        return [python, "-m", tool, "serve", "--ums-worker"]

    def _worker_env(self, backend: str) -> dict[str, str] | None:
        """Ambiente do worker (``None`` = herdar o do supervisor sem alterações)."""
        runtime = self._runtimes.get(backend)
        extra = runtime.resolve_env() if runtime is not None else {}
        if not extra:
            return None
        return {**os.environ, **extra}

    def _spawn(self, backend: str, tool: str, state: _WorkerState) -> None:
        """Arranca o subprocesso worker. Fecha o anterior se morto."""
        if state.proc is not None and state.proc.poll() is None:
            return
        cmd = self._worker_cmd(backend, tool)
        # Log stderr para ficheiro (captura imports / warnings torch).
        state.log_path = state.log_path or self._log_path_fn(backend)
        state.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Fechar handle anterior se existir (respawn sem shutdown limpo).
        if state.log_fh is not None:
            with contextlib.suppress(Exception):
                state.log_fh.close()
        log_fh = open(state.log_path, "ab")  # noqa: SIM115 — fechado no shutdown/_force_abort
        state.log_fh = log_fh
        _logger.info(f"[vramd] spawn worker {backend}: {' '.join(cmd)} (log: {state.log_path})")
        # ``spawn_fn`` recebe kwargs stdin/stdout/stderr explícitos — o fake
        # em testes ignora-os (já tem StringIO próprios); o default passa-os
        # ao subprocess.Popen como PIPE/log_fh.
        # env/cwd só entram quando o ``runtime:`` os define — spawn_fn falsos em
        # testes não precisam de aceitar kwargs que o caminho comum não usa.
        extra: dict[str, Any] = {}
        env = self._worker_env(backend)
        if env is not None:
            extra["env"] = env
        runtime = self._runtimes.get(backend)
        cwd = runtime.resolve_cwd() if runtime is not None else None
        if cwd:
            extra["cwd"] = cwd

        state.proc = self._spawn_fn(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_fh,
            **extra,
        )
        state.loaded = False
        # O contador é POR worker: sem reset, um worker que já serviu N jobs
        # reciclava após o primeiro job de cada spawn novo (12→13→14…) e pagava
        # um load inteiro por job para sempre.
        state.jobs_served = 0
        state.spawn_count += 1
        state.spawned_at = time.monotonic()
        state.pgid = host.process_group_id(getattr(state.proc, "pid", 0)) if state.proc is not None else None
        state.last_exit_code = None
        state.last_abort_reason = None
        state.last_job_id = None
        state.last_progress_pct = None
        state.last_progress_msg = None
        state.last_event_at = None
        _logger.info(
            f"[vramd] worker {backend}: spawn #{state.spawn_count} (recicla a cada {self._max_jobs_for(backend)} jobs)"
        )
        # Windows: select() só aceita sockets — o stdout do worker (pipe) é
        # drenado por uma thread. Fakes de teste sem fileno ficam no caminho
        # antigo (readline não-bloqueante + sleep).
        if host.IS_WINDOWS and state.proc.stdout is not None:
            try:
                state.proc.stdout.fileno()
            except (AttributeError, io.UnsupportedOperation, ValueError, OSError):
                state.reader = None
            else:
                state.reader = _ThreadedLineReader(state.proc.stdout)

    def _read_event_with_timeout(self, state: _WorkerState, *, timeout: float) -> dict[str, Any] | None:
        """Lê 1 linha do stdout do worker com timeout real.

        ``subprocess.PIPE.readline()`` bloqueia para sempre sem dados — o poll
        antigo com ``sleep`` nunca corria (cancel/abort/SIGTERM mortos). Usa
        ``select`` no fd real; fakes de teste (sem fileno) fazem readline não
        bloqueante + sleep.

        **Crítico:** drenar linhas já no buffer do ``TextIOWrapper`` *antes*
        de ``select`` (via ``O_NONBLOCK``) — senão ``EVENT_DONE`` atrás de
        ``progress`` no mesmo ``read`` do kernel fica preso para sempre.
        """
        if state.proc is None or state.proc.stdout is None:
            return None
        deadline = time.monotonic() + max(0.0, timeout)
        stdout = state.proc.stdout

        # Windows: consome a thread leitora (select não vê pipes). A thread
        # bloqueia no readline e o timeout fica do lado do supervisor.
        reader = state.reader
        if reader is not None:
            while time.monotonic() < deadline:
                remaining = max(0.0, deadline - time.monotonic())
                line = reader.read(min(0.5, remaining))
                if line:
                    if len(line) > _MAX_LINE_BYTES:
                        _logger.warn(
                            f"[vramd] worker {state.backend}: linha >{_MAX_LINE_BYTES // 1024} KiB "
                            "sem newline — saltada."
                        )
                        continue
                    try:
                        return decode(line)
                    except ValueError as e:
                        _logger.warn(f"[vramd] worker {state.backend}: linha inválida: {e}")
                        continue
                # EOF só depois de drenar a queue: um ``done`` escrito antes de
                # o worker morrer ainda está no buffer.
                if reader.eof or state.proc.poll() is not None:
                    return None
            return None

        fd: int | None
        try:
            fd = stdout.fileno()
        except (AttributeError, io.UnsupportedOperation, ValueError, OSError):
            fd = None

        while time.monotonic() < deadline:
            # 1) Drenar buffer user-space PRIMEIRO (select não o vê) — e ANTES
            # do poll(): um worker que escreve done e morre no instante seguinte
            # deixava o evento buffered no pipe; verificar poll() primeiro
            # descartava-o e o job falhava apesar de ter completado.
            line = _readline_nonblocking(stdout, fd)
            if line:
                if len(line) > _MAX_LINE_BYTES:
                    _logger.warn(
                        f"[vramd] worker {state.backend}: linha >{_MAX_LINE_BYTES // 1024} KiB sem newline — saltada."
                    )
                    continue
                try:
                    return decode(line)
                except ValueError as e:
                    _logger.warn(f"[vramd] worker {state.backend}: linha inválida: {e}")
                    continue
            if state.proc.poll() is not None:
                # Morto e buffer drenado: EOF real.
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # 2) Nada buffered — esperar dados novos no fd.
            if fd is not None:
                ready, _, _ = select.select([fd], [], [], min(remaining, 0.5))
                if not ready:
                    continue
                line = stdout.readline()
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if line:
                    if len(line) > _MAX_LINE_BYTES:
                        _logger.warn(
                            f"[vramd] worker {state.backend}: linha >{_MAX_LINE_BYTES // 1024} KiB "
                            "sem newline — saltada."
                        )
                        continue
                    try:
                        return decode(line)
                    except ValueError as e:
                        _logger.warn(f"[vramd] worker {state.backend}: linha inválida: {e}")
                        continue
            else:
                # Fake stdout: readline vazio = sem dados ainda.
                time.sleep(min(0.05, remaining))
        return None

    def _wait_event(
        self,
        state: _WorkerState,
        *,
        expected: set[str],
        timeout: float,
        on_progress: Callable[[float | None, str | None], None] | None = None,
    ) -> dict[str, Any] | None:
        """Lê eventos até um dos ``expected`` (ignorando progress/vram_budget)."""
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            event = self._read_event_with_timeout(state, timeout=min(remaining, 1.0))
            if event is None:
                # Silêncio com proc vivo ≠ EOF — continua até deadline.
                if state.proc is None or state.proc.poll() is not None:
                    return None
                continue
            ev = event["event"]
            if ev == "progress" and on_progress:
                with contextlib.suppress(Exception):
                    on_progress(event.get("pct"), event.get("msg"))
                continue
            if ev == "vram_budget":
                state.vram_mib = event.get("vram_mib", state.vram_mib)
                continue
            if ev in expected:
                return event
            # Evento inesperado: logar e continuar.
            _logger.info(f"[vramd] worker {state.backend}: evento inesperado {ev} (à espera de {expected})")
        return None

    def _worker_log_size(self, state: _WorkerState) -> int:
        """Tamanho actual do stderr do worker (0 se ainda não existe)."""
        if state.log_path is None:
            return 0
        try:
            return state.log_path.stat().st_size
        except OSError:
            return 0

    def _scan_worker_oom(self, state: _WorkerState, offset: int) -> tuple[int, int]:
        """Linhas OOM novas no stderr do worker desde ``offset``.

        Returns:
            ``(contagem de padrões OOM no trecho novo, novo offset)``.
        """
        path = state.log_path
        if path is None:
            return 0, offset
        try:
            size = path.stat().st_size
            if size <= offset:
                return 0, offset
            with open(path, "rb") as fh:
                fh.seek(offset)
                chunk = fh.read(size - offset)
        except OSError:
            return 0, offset
        text = chunk.decode("utf-8", errors="ignore")
        return sum(text.count(pat) for pat in _OOM_SPIN_PATTERNS), size

    def _max_jobs_for(self, backend: str) -> int:
        """Limite de jobs antes de reciclar (``runtime.max_jobs`` ou global)."""
        runtime = self._runtimes.get(backend)
        override = getattr(runtime, "max_jobs", None) if runtime is not None else None
        return int(override) if override is not None else self._worker_max_jobs

    def _ctx(self, state: _WorkerState) -> str:
        """Sufixo compacto de diagnóstico para mensagens de fila/erro.

        Inclui job/pid/último progresso/idade do último evento — sem isto um
        ``abort timeout`` ou ``OOM-spin`` no log não diz o que estava a correr.
        """
        parts: list[str] = []
        proc = state.proc
        if proc is not None:
            with contextlib.suppress(Exception):
                parts.append(f"pid={proc.pid}")
        if state.last_job_id:
            parts.append(f"job={state.last_job_id[:8]}")
        if state.last_progress_msg:
            parts.append(f"stage={state.last_progress_msg!r}")
        if state.last_progress_pct is not None:
            parts.append(f"pct={state.last_progress_pct:.0%}")
        if state.last_event_at is not None:
            parts.append(f"event_age={time.monotonic() - state.last_event_at:.0f}s")
        if state.jobs_served:
            parts.append(f"jobs={state.jobs_served}")
        return (" [" + " ".join(parts) + "]") if parts else ""

    def worker_diagnostics(self, backend: str) -> dict[str, Any] | None:
        """Snapshot do worker para ``status``/``debug``/resultado de erro."""
        with self._pool_lock:
            state = self._workers.get(backend)
        if state is None:
            return None
        proc = state.proc
        alive = proc is not None and proc.poll() is None
        exit_code = state.last_exit_code
        if exit_code is None and proc is not None:
            exit_code = proc.poll()
        return {
            "backend": backend,
            "pid": getattr(proc, "pid", None) if proc is not None else None,
            "alive": alive,
            "loaded": bool(state.loaded and alive),
            "jobs_served": state.jobs_served,
            "max_jobs": self._max_jobs_for(backend),
            "spawn_count": state.spawn_count,
            "recycle_count": state.recycle_count,
            "uptime_sec": (round(time.monotonic() - state.spawned_at, 1) if state.spawned_at else None),
            "last_exit_code": exit_code,
            "last_abort_reason": state.last_abort_reason,
            "last_job_id": state.last_job_id,
            "last_progress_pct": state.last_progress_pct,
            "last_progress_msg": state.last_progress_msg,
            "last_event_age_sec": (
                round(time.monotonic() - state.last_event_at, 1) if state.last_event_at is not None else None
            ),
            "log_path": str(state.log_path) if state.log_path is not None else None,
            "process_group": state.pgid,
        }

    def kill_orphan_group(self, backend: str) -> int | None:
        """SIGKILL ao grupo de um worker morto — filhos/netos não ficam órfãos.

        Chamado quando o supervisor detecta um worker morto (crash/OOM) fora
        do caminho normal de ``_terminate_worker``. O PGID foi capturado no
        spawn e continua válido depois de o líder morrer, desde que o grupo
        tenha membros.
        """
        with self._pool_lock:
            state = self._workers.get(backend)
        if state is None or state.pgid is None:
            return None
        pgid = state.pgid
        host.signal_process_group(pgid, signal.SIGKILL)
        state.pgid = None
        return pgid

    def worker_log_tail(self, backend: str, *, lines: int = 20) -> list[str]:
        """Últimas linhas do stderr do worker (diagnóstico; nunca lança)."""
        with self._pool_lock:
            state = self._workers.get(backend)
        path = state.log_path if state is not None else None
        if path is None:
            return []
        try:
            size = path.stat().st_size
            with open(path, "rb") as fh:
                fh.seek(max(0, size - 8192))
                chunk = fh.read()
        except OSError:
            return []
        text = chunk.decode("utf-8", errors="replace")
        return text.splitlines()[-max(1, lines) :]

    def _force_abort(self, state: _WorkerState, backend: str, *, reason: str = "abort") -> None:
        """Compat: abort forçado com uma razão explícita."""
        self._terminate_worker(state, backend, reason=reason)

    def _terminate_worker(self, state: _WorkerState, backend: str, *, reason: str) -> None:
        """Termina o worker e limpa o estado, registando a razão.

        ``reason="recycle"`` é operação normal (INFO, sem WARN e sem deixar
        ``loaded=True`` para o BackendManager acusar um worker morto).
        """
        context = self._ctx(state)
        proc = state.proc
        proc_pid = getattr(proc, "pid", None) if proc is not None else None
        children = host.process_tree_pids(proc_pid) if proc_pid is not None else []
        pgid = state.pgid or (host.process_group_id(proc_pid) if proc_pid is not None else None)
        tree = f" tree={pgid}" if pgid is not None else ""
        kids = f" children={len(children) - 1}" if len(children) > 1 else ""
        if reason == "recycle":
            _logger.info(
                f"[vramd] worker {backend}: reciclar após {state.jobs_served} jobs "
                f"(spawn #{state.spawn_count}){context}{tree}{kids}"
            )
        else:
            _logger.warn(f"[vramd] worker {backend}: abort forçado ({reason}){context}{tree}{kids}")
        state.loaded = False
        state.last_abort_reason = reason
        if proc is not None and proc.poll() is None:
            # POSIX: SIGTERM ao grupo inteiro (PID==PGID nos workers com
            # start_new_session) para apanhar o _xatlas_worker e afins; no
            # Windows/fallback o psutil percorre a árvore do worker.
            grace = min(2.0, self._abort_timeout) if reason == "recycle" else self._abort_timeout
            if proc_pid is not None:
                with contextlib.suppress(Exception):
                    host.terminate_process_tree(proc_pid, grace_sec=grace)
            with contextlib.suppress(Exception):
                proc.terminate()
            try:
                proc.wait(timeout=self._abort_timeout)
            except subprocess.TimeoutExpired:
                if pgid is not None:
                    host.signal_process_group(pgid, signal.SIGKILL)
                else:
                    with contextlib.suppress(Exception):
                        proc.kill()
                # Recolher o zombie pós-SIGKILL — sem wait(), o Popen ficava ao
                # des-cuidado do GC e rebentava com BrokenPipe no flush (fd leak
                # + <defunct> por cada abort).
                with contextlib.suppress(Exception):
                    proc.wait(timeout=2.0)
            state.last_exit_code = proc.poll()
            # O bug que motivou isto: o filho xatlas continuava a 299% de CPU
            # depois de o worker morrer. Se algum descendente sobreviveu,
            # gritar no log (e deixar o `vramd reap`/doctor visível).
            survivors = [p for p in children if p != proc_pid and host.pid_exists(p)]
            if survivors:
                _logger.warn(
                    f"[vramd] worker {backend}: {len(survivors)} descendente(s) sobreviveram "
                    f"à terminação: {survivors} — verifica com `ps`/`vramd doctor`"
                )
        elif proc is not None:
            state.last_exit_code = proc.poll()
        # Fechar pipes stdin/stdout ANTES de largar a referência (fd leak).
        self._close_proc_pipes(state)
        state.proc = None
        # Fechar o handle do stderr do worker (antes ficava órfão).
        if state.log_fh is not None:
            with contextlib.suppress(Exception):
                state.log_fh.close()
            state.log_fh = None
        state.pgid = None


# ---------------------------------------------------------------------------
# Helpers de resolução
# ---------------------------------------------------------------------------


def _resolve_tool_python(tool: str) -> str | None:
    """Descobre o ``python`` do venv da tool ."""
    from .toolchain import resolve_tool_python

    return resolve_tool_python(tool)


def _shape_mismatch(stored: dict[str, Any], new: dict[str, Any]) -> bool:
    """True se kwargs relevantes mudaram (ex.: max_num_view, sdnq_preset).

    Reutiliza o ``_SHAPE_LOAD_KEYS`` do BackendManager (source of truth) para
    garantir que o reuse-fast-path do pool decide "mesma shape" com os mesmos
    critérios do manager — antes o pool tinha um subset divergente, o que era
    um latent footgun (bug M4): podia reusar um worker com shape errada se um
    caller futuro dependesse só da lógica do pool.
    """
    from .backend_manager import _SHAPE_LOAD_KEYS

    return any(stored.get(k) != new.get(k) for k in _SHAPE_LOAD_KEYS)
