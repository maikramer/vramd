"""Guardas de processo do vramd — singleton, reap de órfãos e VRAM alheia.

Três problemas históricos que este módulo resolve:

1. **Supervisores duplicados.** ``serve_forever`` apagava o socket quando o
   probe ``is_server_running`` falhava (supervisor vivo mas ocupado) e fazia
   bind por cima. Resultado observado: 3 supervisores vivos, um deles com um
   worker ``text3d`` a segurar 3.5 GiB — invisível no ``vramd status`` (que só
   fala com o supervisor dono do socket). :class:`SingletonLock` usa ``flock``
   exclusivo: o kernel liberta o lock na morte do processo, logo nunca fica
   stale e nunca há dois supervisores.

2. **Órfãos.** Workers descendem por stdin (EOF ⇒ saem), mas um supervisor
   zombie mantém-nos vivos, e um ``kill -9`` no supervisor deixa o worker sem
   quem lhe feche o pipe. :func:`find_strays` + :func:`reap` limpam de forma
   determinística: quem detém o lock é, por definição, o único legítimo.

3. **VRAM alheia invisível.** O admit contabilizava só o PID do próprio
   supervisor. :func:`gpu_vram_by_pid` / :func:`stray_report` expõem a VRAM
   presa por supervisores/workers estranhos, para o ``status`` mostrar e o
   ``ensure_vram`` poder recuperá-la em vez de recusar o job.

Linux-first (``/proc``); no Windows o varrimento usa ``psutil`` e as funções
de ``/proc`` só correm quando o ``proc_root`` existe (testes). O resto do vramd
continua a funcionar.
"""

from __future__ import annotations

import contextlib
import errno
import os
import signal
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from vramd import host
from vramd.logging import Logger

_logger = Logger()

LOCK_FILENAME = "model-server.lock"

# Assinaturas de cmdline da família UMS.
_WORKER_FLAG = "--ums-worker"

KIND_SUPERVISOR = "supervisor"
KIND_WORKER = "worker"


@dataclass(frozen=True)
class UmsProcess:
    """Processo da família UMS encontrado no sistema."""

    pid: int
    kind: str
    backend: str | None
    cmdline: str
    vram_mib: int | None = None

    def describe(self) -> str:
        who = f"{self.kind}"
        if self.backend:
            who += f":{self.backend}"
        vram = f", {self.vram_mib} MiB" if self.vram_mib else ""
        return f"PID {self.pid} ({who}{vram})"


# ---------------------------------------------------------------------------
# Singleton por flock
# ---------------------------------------------------------------------------


def lock_path_for(socket_path: Path | str) -> Path:
    """Path do lockfile ao lado do socket."""
    return Path(socket_path).parent / LOCK_FILENAME


# Lock process-local: no Windows duas handles do mesmo processo podem não
# colidir no ``LockFile``; no POSIX o ``flock`` de dois fds separados colide,
# mas o guard torna o contrato ("um supervisor por path") explícito nos dois.
_held_locks: set[Path] = set()
_held_lock_mutex = threading.Lock()

# No Windows os locks de bytes são **obrigatórios**: ler a região bloqueada por
# outro handle falha com ``ERROR_LOCK_VIOLATION``. O PID é escrito no início do
# ficheiro e o lock fica num byte muito à frente, para ``owner_pid()`` (outro
# handle, possivelmente outro processo) conseguir ler o dono.
_WINDOWS_LOCK_OFFSET = 4096


class SingletonLock:
    """Lock exclusivo de supervisor (``flock`` POSIX, ``msvcrt`` Windows).

    O fd fica aberto enquanto o processo vive; o SO liberta o lock quando o
    processo morre (inclusive com ``SIGKILL``/``TerminateProcess``), pelo que
    não existe estado stale a limpar — ao contrário de pid-files.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> bool:
        """Tenta obter o lock. ``False`` se outro supervisor o detém."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = self.path.resolve()
        with _held_lock_mutex:
            if key in _held_locks:
                return False

        if host.IS_WINDOWS:
            return self._acquire_windows(key)

        try:
            import fcntl
        except ImportError:  # pragma: no cover — plataforma exótica
            with _held_lock_mutex:
                _held_locks.add(key)
            return True
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        os.truncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.fsync(fd)
        self._fd = fd
        with _held_lock_mutex:
            _held_locks.add(key)
        return True

    def _acquire_windows(self, key: Path) -> bool:
        """Lock de 1 byte via ``msvcrt.locking`` (libertado pelo SO na morte).

        O lock fica no offset :data:`_WINDOWS_LOCK_OFFSET`; o PID no início.
        Assim ``owner_pid()`` não lê a região bloqueada (no Windows a violação
        de lock é um erro de leitura, não uma espera).
        """
        import msvcrt

        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0))
        try:
            os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            os.close(fd)
            return False
        try:
            data = f"{os.getpid()}\n".encode()
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, data)
            # Padding até ao byte do lock: manter a região do PID legível mesmo
            # que o ficheiro fosse truncado noutra run.
            size = os.lseek(fd, 0, os.SEEK_END)
            if size < _WINDOWS_LOCK_OFFSET + 1:
                os.write(fd, b" " * (_WINDOWS_LOCK_OFFSET + 1 - size))
            os.fsync(fd)
        except OSError:
            # Lock adquirido mas o pid não ficou legível: libertar é melhor que
            # deixar um supervisor sem diagnóstico do dono.
            with contextlib.suppress(OSError):
                os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            os.close(fd)
            return False
        self._fd = fd
        with _held_lock_mutex:
            _held_locks.add(key)
        return True

    def owner_pid(self) -> int | None:
        """PID escrito no lockfile pelo detentor actual (best-effort)."""
        # ``os.read`` sem buffering: um ``open(..., "rb").read(64)`` lê um
        # bloco de 8 KiB e toca no byte bloqueado (offset 4096), que no
        # Windows devolve ERROR_LOCK_VIOLATION em vez de esperar.
        fd: int | None = None
        try:
            fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 64).decode("ascii", "ignore")
        except OSError:
            return None
        finally:
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
        text = raw.splitlines()[0].strip() if raw.strip() else ""
        try:
            return int(text) if text else None
        except ValueError:
            return None

    def release(self) -> None:
        """Liberta o lock (idempotente)."""
        if self._fd is None:
            return
        if host.IS_WINDOWS:
            with contextlib.suppress(Exception):
                import msvcrt

                os.lseek(self._fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
        else:
            with contextlib.suppress(ImportError, OSError):
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(self._fd)
        self._fd = None
        with _held_lock_mutex:
            _held_locks.discard(self.path.resolve())


# ---------------------------------------------------------------------------
# Varrimento de /proc
# ---------------------------------------------------------------------------


def _read_cmdline(pid: int, proc_root: Path) -> list[str]:
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except OSError:
        return []
    return [part for part in raw.split(b"\x00") if part]  # type: ignore[misc]


def _decode_cmdline(pid: int, proc_root: Path) -> list[str]:
    return [p.decode("utf-8", "replace") if isinstance(p, bytes) else str(p) for p in _read_cmdline(pid, proc_root)]


def _ppid(pid: int, proc_root: Path) -> int | None:
    try:
        for line in (proc_root / str(pid) / "status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def classify_cmdline(parts: Sequence[str]) -> tuple[str, str | None] | None:
    """Classifica um cmdline como ``(kind, backend)`` da família UMS, ou ``None``.

    Worker: ``<python> -m <tool> serve --ums-worker`` → ``("worker", tool)``.
    Supervisor: ``-m vramd start`` / ``vramd start`` /
    ``python /path/vramd start``.
    """
    if not parts:
        return None
    if _WORKER_FLAG in parts:
        backend = None
        if "-m" in parts:
            idx = parts.index("-m")
            if idx + 1 < len(parts):
                backend = parts[idx + 1]
        return KIND_WORKER, backend
    if "start" not in parts:
        return None
    if "-m" in parts:
        idx = parts.index("-m")
        if idx + 1 < len(parts) and parts[idx + 1] in ("vramd", "vramd.cli"):
            return KIND_SUPERVISOR, None
    exe = Path(parts[0]).name
    exe_stem = Path(exe).stem.lower()
    if exe_stem in ("vramd", "ums"):
        return KIND_SUPERVISOR, None
    if exe_stem.startswith("python"):
        # ``python /path/vramd start`` / ``python /repo/vramd/cli.py start``.
        # NÃO basta a substring "vramd" na cmdline: o path do checkout do
        # próprio vramd (D:\a\vramd\...) faria qualquer pytest/IDE passar por
        # supervisor e o ``reap`` matava-o.
        for part in parts[1:]:
            name = Path(part).name.lower()
            if name in ("vramd", "vramd.exe", "vramd.py") or part.replace("\\", "/").endswith("vramd/cli.py"):
                return KIND_SUPERVISOR, None
    return None


def _resolve_proc(proc_root: Path | str | None) -> tuple[Path | None, bool]:
    """Resolve ``/proc`` vs ``psutil`` preservando o contrato dos testes.

    Returns:
        ``(root, use_psutil)``. Um ``proc_root`` **explícito** que não existe
        (fixtures de teste) devolve ``(None, False)`` — vazio, nunca uma
        varredura do sistema real. Só o default ``None`` cai para ``psutil``
        quando ``/proc`` não existe (Windows/macOS).
    """
    if proc_root is None:
        root = Path("/proc")
        return (root, False) if root.is_dir() else (None, True)
    root = Path(proc_root)
    return (root, False) if root.is_dir() else (None, False)


def _list_processes_psutil() -> list[UmsProcess]:
    """Varrimento portátil (Windows/macOS/sem ``/proc``) via ``psutil``."""
    try:
        import psutil
    except ImportError:  # pragma: no cover — psutil é dependência
        return []
    out: list[UmsProcess] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            parts = proc.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        kind_backend = classify_cmdline([str(p) for p in parts])
        if kind_backend is None:
            continue
        kind, backend = kind_backend
        out.append(UmsProcess(pid=int(proc.pid), kind=kind, backend=backend, cmdline=" ".join(str(p) for p in parts)))
    return sorted(out, key=lambda p: p.pid)


def list_processes(*, proc_root: Path | str | None = None) -> list[UmsProcess]:
    """Lista todos os processos da família UMS (supervisores + workers)."""
    root, use_psutil = _resolve_proc(proc_root)
    if use_psutil:
        return _list_processes_psutil()
    if root is None:
        return []
    out: list[UmsProcess] = []
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        parts = _decode_cmdline(pid, root)
        kind_backend = classify_cmdline(parts)
        if kind_backend is None:
            continue
        kind, backend = kind_backend
        out.append(UmsProcess(pid=pid, kind=kind, backend=backend, cmdline=" ".join(parts)))
    return sorted(out, key=lambda p: p.pid)


def descendants(pid: int, *, proc_root: Path | str | None = None) -> set[int]:
    """PIDs descendentes de ``pid`` (transitivo)."""
    root, use_psutil = _resolve_proc(proc_root)
    if use_psutil:
        try:
            import psutil

            return {int(p) for p in psutil.Process(int(pid)).children(recursive=True)}
        except Exception:
            return set()
    if root is None:
        return set()
    parent_of: dict[int, int] = {}
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        child = int(entry.name)
        parent = _ppid(child, root)
        if parent is not None:
            parent_of[child] = parent
    out: set[int] = set()
    for child in parent_of:
        cursor: int | None = child
        seen: set[int] = set()
        while cursor is not None and cursor > 1 and cursor not in seen:
            seen.add(cursor)
            parent = parent_of.get(cursor)
            if parent == pid:
                out.add(child)
                break
            cursor = parent
    return out


def ancestors(pid: int, *, proc_root: Path | str | None = None) -> set[int]:
    """PIDs ascendentes de ``pid`` (até ``init``).

    Lançadores como ``timeout 900 python -m vramd start``, ``nohup`` ou um
    ``bash -c`` carregam a nossa cmdline no argv deles e seriam classificados
    como supervisores — matá-los mataria quem nos lançou.
    """
    root, use_psutil = _resolve_proc(proc_root)
    if use_psutil:
        out: set[int] = set()
        try:
            import psutil

            cursor = psutil.Process(int(pid)).parent()
            while cursor is not None and cursor.pid > 1 and cursor.pid not in out:
                out.add(int(cursor.pid))
                cursor = cursor.parent()
        except Exception:
            return out
        return out
    if root is None:
        return set()
    out: set[int] = set()
    cursor = _ppid(int(pid), root)
    while cursor is not None and cursor > 1 and cursor not in out:
        out.add(cursor)
        cursor = _ppid(cursor, root)
    return out


def find_strays(
    *,
    self_pid: int | None = None,
    keep: Iterable[int] = (),
    proc_root: Path | str | None = None,
    with_vram: bool = True,
) -> list[UmsProcess]:
    """Processos UMS que não somos nós, nem descendentes, nem ascendentes.

    Chamado pelo supervisor **depois** de ganhar o :class:`SingletonLock`: nesse
    ponto qualquer outro supervisor/worker é lixo de uma run anterior.
    """
    me = os.getpid() if self_pid is None else int(self_pid)
    protected = (
        {me, *(int(p) for p in keep)} | descendants(me, proc_root=proc_root) | ancestors(me, proc_root=proc_root)
    )
    strays = [p for p in list_processes(proc_root=proc_root) if p.pid not in protected]
    if with_vram and strays:
        strays = annotate_vram(strays)
    return strays


# ---------------------------------------------------------------------------
# VRAM por PID
# ---------------------------------------------------------------------------


def gpu_vram_by_pid() -> dict[int, int]:
    """``{pid: MiB}`` dos processos compute na GPU (NVML → nvidia-smi)."""
    try:
        from vramd.gpu import list_nvidia_compute_apps

        apps = list_nvidia_compute_apps()
    except Exception:
        return {}
    out: dict[int, int] = {}
    for pid, _name, used in apps or []:
        if used is None:
            continue
        out[int(pid)] = out.get(int(pid), 0) + int(used)
    return out


def annotate_vram(procs: Sequence[UmsProcess]) -> list[UmsProcess]:
    """Preenche ``vram_mib`` de cada processo com a leitura da GPU."""
    by_pid = gpu_vram_by_pid()
    return [replace(p, vram_mib=by_pid.get(p.pid)) for p in procs]


def stray_report(
    *,
    self_pid: int | None = None,
    keep: Iterable[int] = (),
    proc_root: Path | str | None = None,
) -> dict[str, Any]:
    """Sumário de órfãos + VRAM que eles seguram (para ``status`` / ``doctor``)."""
    strays = find_strays(self_pid=self_pid, keep=keep, proc_root=proc_root)
    total = sum(p.vram_mib or 0 for p in strays)
    return {
        "count": len(strays),
        "vram_mib": total,
        "processes": [
            {
                "pid": p.pid,
                "kind": p.kind,
                "backend": p.backend,
                "vram_mib": p.vram_mib,
                "cmdline": p.cmdline,
            }
            for p in strays
        ],
    }


# ---------------------------------------------------------------------------
# Reap
# ---------------------------------------------------------------------------


def pid_alive(pid: int, *, kill_fn: Callable[[int, int], None] | None = None) -> bool:
    """True se o PID existe.

    Sem ``kill_fn`` usa :func:`vramd.host.pid_exists` (seguro no Windows, onde
    ``os.kill(pid, 0)`` mataria o processo). ``kill_fn`` mantém-se injectável
    para os testes do contrato POSIX.
    """
    if kill_fn is None:
        return host.pid_exists(int(pid))
    try:
        kill_fn(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def _default_kill(pid: int, sig: int) -> None:
    """Mata com a semântica de ``os.kill``, mas portátil.

    ``SIGTERM`` → pedido de terminação; qualquer outro sinal (``SIGKILL`` ou
    ``signal 0`` usado como probe) → força/existência, respectivamente.
    Quando o processo é líder do próprio grupo, o sinal vai para o grupo —
    senão um neto (``_xatlas_worker``) ficava órfão ao reapar o worker.
    Um worker antigo (ou lançado sem ``start_new_session``) não é líder:
    nesse caso a árvore do psutil apanha os descendentes na mesma.
    """
    if sig == 0:
        if not host.pid_exists(pid):
            raise ProcessLookupError(errno.ESRCH, f"processo {pid} não existe")
        return
    pgid = host.process_group_id(pid)
    if pgid is not None and host.signal_process_group(pgid, sig):
        return
    sigterm = int(getattr(signal, "SIGTERM", 15))
    tree = host.process_tree_pids(pid)
    if sig == sigterm:
        # Sem grupo próprio: TERM à árvore do psutil garante que os
        # descendentes (dataloaders/exporters) não ficam vivos depois do pai.
        # Sem escalar para KILL — a janela do ``reap`` continua a mandar.
        for target in tree or [pid]:
            with contextlib.suppress(OSError):
                host.terminate_pid(target)
        return
    # KILL: descendentes primeiro, o próprio PID no fim (ESRCH propaga, como
    # no contrato do os.kill).
    for target in tree or [pid]:
        if target != pid:
            with contextlib.suppress(OSError):
                host.kill_pid(target)
    host.kill_pid(pid)


def reap(
    procs: Sequence[UmsProcess],
    *,
    term_wait_sec: float = 3.0,
    kill_fn: Callable[[int, int], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    poll_sec: float = 0.2,
) -> list[dict[str, Any]]:
    """Termina os processos (SIGTERM → SIGKILL). Devolve o que foi feito por PID."""
    kill = kill_fn or _default_kill
    results: list[dict[str, Any]] = []
    pending: list[UmsProcess] = []
    # PGID por líder de grupo: mesmo que o líder morra com o TERM, os filhos
    # que ignoraram o sinal têm de levar KILL no fim da janela.
    group_ids: dict[int, int] = {}
    # Worker antigo sem ``start_new_session`` não tem grupo próprio: a árvore
    # do psutil é a única forma de o apanhar depois de o líder morrer (os
    # filhos são reparentados e o PID do pai já não os liga).
    trees: dict[int, list[int]] = {}
    for proc in procs:
        if kill_fn is None:
            pgid = host.process_group_id(proc.pid)
            if pgid is not None:
                group_ids[proc.pid] = pgid
            else:
                kids = [p for p in host.process_tree_pids(proc.pid) if p != proc.pid]
                if kids:
                    trees[proc.pid] = kids
        try:
            kill(proc.pid, signal.SIGTERM)
            pending.append(proc)
        except OSError as exc:
            results.append({"pid": proc.pid, "kind": proc.kind, "signal": "none", "error": str(exc)})
    deadline = term_wait_sec
    while pending and deadline > 0:
        sleep_fn(poll_sec)
        deadline -= poll_sec
        still = [p for p in pending if pid_alive(p.pid, kill_fn=None if kill_fn is None else kill)]
        for gone in [p for p in pending if p not in still]:
            results.append({"pid": gone.pid, "kind": gone.kind, "signal": "SIGTERM", "vram_mib": gone.vram_mib})
        pending = still
    # Limpar descendentes que sobreviveram ao TERM (o líder pode já ter saído).
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    for pgid in group_ids.values():
        host.signal_process_group(pgid, sigkill)
    for kids in trees.values():
        for child in kids:
            if host.pid_exists(child):
                with contextlib.suppress(OSError):
                    kill(child, sigkill)
    for proc in pending:
        with contextlib.suppress(OSError):
            kill(proc.pid, sigkill)
        results.append({"pid": proc.pid, "kind": proc.kind, "signal": "SIGKILL", "vram_mib": proc.vram_mib})
    return results


def reap_strays(
    *,
    self_pid: int | None = None,
    keep: Iterable[int] = (),
    proc_root: Path | str | None = None,
    dry_run: bool = False,
    term_wait_sec: float = 3.0,
) -> dict[str, Any]:
    """Encontra e mata órfãos UMS; devolve sumário para logs/CLI/protocolo."""
    strays = find_strays(self_pid=self_pid, keep=keep, proc_root=proc_root)
    freed = sum(p.vram_mib or 0 for p in strays)
    if not strays:
        return {"reaped": [], "count": 0, "vram_mib_freed": 0, "dry_run": dry_run}
    if dry_run:
        return {
            "reaped": [],
            "count": len(strays),
            "vram_mib_freed": freed,
            "dry_run": True,
            "would_reap": [p.describe() for p in strays],
        }
    for proc in strays:
        _logger.warn(f"[vramd] reap órfão: {proc.describe()} — {proc.cmdline[:120]}")
    results = reap(strays, term_wait_sec=term_wait_sec)
    return {
        "reaped": results,
        "count": len(results),
        "vram_mib_freed": freed,
        "dry_run": False,
    }
