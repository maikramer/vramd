"""Camada de plataforma do vramd — o único sítio que sabe em que SO corre.

O supervisor foi escrito Linux-first. Este módulo concentra as diferenças que
importam para correr também no Windows:

* **Paths** — ``~/.cache/vramd`` no POSIX, ``%LOCALAPPDATA%\\vramd`` no Windows
  (e ``%APPDATA%\\vramd`` para configuração). ``VRAMD_CACHE_DIR`` e
  ``VRAMD_CONFIG_DIR`` continuam a mandar em qualquer SO.
* **Processos** — ``psutil`` é a via comum; no POSIX mantém-se ``os.kill`` onde
  faz falta. ``os.kill(pid, 0)`` **não** é um probe seguro no Windows: a
  implementação da stdlib chama ``TerminateProcess`` e mataria o supervisor.
* **Sinais** — ``SIGTERM``/``SIGKILL`` não são sinais reais no Windows. O
  encerramento gracioso continua a ser por RPC (``shutdown``); o caminho de
  força usa ``psutil.terminate()``/``kill()``.
* **Pipes** — o Windows não faz ``select`` sobre pipes; quem lê stdout de um
  worker usa :class:`vramd.subprocess_pool` com uma thread leitora.
* **Detach** — ``start_new_session`` é POSIX; no Windows usa-se
  ``CREATE_NEW_PROCESS_GROUP``/``CREATE_NO_WINDOW``.

Nada aqui importa ``torch`` nem toca na GPU.
"""

from __future__ import annotations

import contextlib
import errno
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

IS_WINDOWS = os.name == "nt"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def cache_dir() -> Path:
    """Directório de estado/cache do vramd.

    Precedência: ``VRAMD_CACHE_DIR`` → ``XDG_CACHE_HOME`` (POSIX) →
    ``%LOCALAPPDATA%`` (Windows) → ``~/.cache``.
    """
    env = os.environ.get("VRAMD_CACHE_DIR", "").strip()
    if env:
        return Path(os.path.expanduser(env))
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA", "").strip() or os.environ.get("APPDATA", "").strip()
        if base:
            return Path(base) / "vramd"
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        return Path(xdg) / "vramd"
    return Path.home() / ".cache" / "vramd"


def config_dir() -> Path:
    """Directório de configuração (``backends.d``, ``hooks.yaml``).

    ``VRAMD_CONFIG_DIR`` manda; senão ``%APPDATA%\\vramd`` no Windows e
    ``~/.config/vramd`` no POSIX.
    """
    env = os.environ.get("VRAMD_CONFIG_DIR", "").strip()
    if env:
        return Path(os.path.expanduser(env))
    if IS_WINDOWS:
        base = os.environ.get("APPDATA", "").strip()
        if base:
            return Path(base) / "vramd"
    return Path.home() / ".config" / "vramd"


def default_socket_path() -> Path:
    """Endpoint canónico do supervisor (socket Unix ou descritor TCP)."""
    return cache_dir() / "vramd.sock"


def autostart_log_path() -> Path:
    """Log do auto-start (o supervisor destacado não tem consola)."""
    override = os.environ.get("VRAMD_AUTO_START_LOG", "").strip()
    return Path(os.path.expanduser(override)) if override else cache_dir() / "ums-autostart.log"


def worker_log_path(backend: str) -> Path:
    """stderr do worker de um backend."""
    return cache_dir() / f"vramd-worker-{backend}.log"


def learn_dir() -> Path:
    """Observações do ``vramd learn``."""
    return cache_dir() / "learn"


# ---------------------------------------------------------------------------
# Processos
# ---------------------------------------------------------------------------


def pid_exists(pid: int) -> bool:
    """True se ``pid`` está vivo.

    No Windows **nunca** se usa ``os.kill(pid, 0)``: a stdlib trata qualquer
    sinal diferente de ``CTRL_C_EVENT``/``CTRL_BREAK_EVENT`` como
    ``TerminateProcess``, incluindo o 0. ``psutil.pid_exists`` é seguro e já é
    dependência do pacote.
    """
    if pid <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        if IS_WINDOWS:
            return False
        try:
            os.kill(int(pid), 0)
        except OSError as exc:
            return exc.errno == errno.EPERM
        return True


def process_user(pid: int) -> str | None:
    """Username dono do processo (``None`` se não determinável)."""
    if pid <= 0:
        return None
    try:
        import psutil

        return psutil.Process(int(pid)).username()
    except Exception:
        return None


def current_user() -> str | None:
    """Username do processo actual."""
    return process_user(os.getpid())


def same_user(pid: int) -> bool:
    """True se o processo corre com o mesmo utilizador que o supervisor.

    No POSIX isto é equivalente ao ``UID`` de ``/proc``; no Windows compara o
    ``DOMAIN\\user`` devolvido pelo psutil.
    """
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if IS_WINDOWS:
        me = current_user()
        them = process_user(pid)
        return bool(me and them and me.casefold() == them.casefold())
    if hasattr(os, "getuid"):
        try:
            import psutil

            return psutil.Process(int(pid)).uids().real == os.getuid()
        except Exception:
            pass
    return True


def _translate_psutil_error(exc: Exception) -> OSError:
    """Converte as excepções do psutil no contrato ``OSError`` dos callers."""
    import psutil

    if isinstance(exc, psutil.NoSuchProcess):
        return ProcessLookupError(errno.ESRCH, f"processo {getattr(exc, 'pid', '?')} não existe")
    if isinstance(exc, psutil.AccessDenied):
        return PermissionError(errno.EPERM, f"sem permissão para o processo {getattr(exc, 'pid', '?')}")
    if isinstance(exc, OSError):
        return exc
    return OSError(str(exc))


def terminate_pid(pid: int) -> None:
    """Pedido de terminação (``SIGTERM`` no POSIX; ``TerminateProcess`` no Windows).

    Raises:
        ProcessLookupError, PermissionError, OSError: como ``os.kill``.
    """
    if pid <= 0:
        raise ProcessLookupError(errno.ESRCH, "PID inválido")
    if IS_WINDOWS:
        try:
            import psutil

            psutil.Process(int(pid)).terminate()
        except Exception as exc:
            raise _translate_psutil_error(exc) from exc
        return
    os.kill(int(pid), signal.SIGTERM)


def kill_pid(pid: int) -> None:
    """Terminação forçada (``SIGKILL`` no POSIX; ``kill()`` no Windows)."""
    if pid <= 0:
        raise ProcessLookupError(errno.ESRCH, "PID inválido")
    if IS_WINDOWS:
        try:
            import psutil

            psutil.Process(int(pid)).kill()
        except Exception as exc:
            raise _translate_psutil_error(exc) from exc
        return
    sigkill = getattr(signal, "SIGKILL", None)
    if sigkill is None:  # pragma: no cover — POSIX sem SIGKILL é teórico
        os.kill(int(pid), signal.SIGTERM)
    else:
        os.kill(int(pid), sigkill)


def process_create_time(pid: int) -> float | None:
    """Instante de criação do processo (protecção contra reutilização de PID)."""
    try:
        import psutil

        return float(psutil.Process(int(pid)).create_time())
    except Exception:
        return None


def parent_alive(pid: int, create_time: float | None = None) -> bool:
    """True se o processo pai ainda é *o mesmo* (PID + instante de criação).

    No POSIX o reparenting muda o PPID; no Windows o PPID fica gravado, pelo
    que só a criação distingue um PID reutilizado. O watchdog do worker usa
    esta função.

    Falha fechada? Não: se o psutil não consegue inspeccionar o pai (ACLs,
    processo protegido), devolve ``True`` — matar o worker por falta de
    visibilidade seria pior que deixá-lo vivo até ao EOF do stdin.
    """
    if pid <= 0:
        return False
    try:
        import psutil
    except ImportError:  # pragma: no cover — psutil é dependência
        return True
    try:
        proc = psutil.Process(int(pid))
    except psutil.NoSuchProcess:
        return False
    except Exception:
        return True
    try:
        if not proc.is_running():
            return False
        if create_time is None:
            return True
        return abs(float(proc.create_time()) - float(create_time)) < 0.001
    except Exception:
        return True


def process_tree_pids(pid: int) -> list[int]:
    """PIDs do processo e dos seus descendentes (filhos primeiro)."""
    if pid <= 0:
        return []
    try:
        import psutil

        root = psutil.Process(int(pid))
        children = root.children(recursive=True)
        # Filhos primeiro: matar o pai não deixa órfãos sem stdio.
        return [p.pid for p in children] + [root.pid]
    except Exception:
        return []


def process_group_id(pid: int) -> int | None:
    """PGID do processo se ele for líder do próprio grupo; senão ``None``.

    Os workers do vramd são spawnados com ``start_new_session=True`` (POSIX),
    logo ``PID == PGID == SID`` e ``killpg`` alcança filhos/netos como o
    ``_xatlas_worker`` que sobreviviam ao ``terminate()`` do pai. Só devolve o
    PGID quando é o próprio PID, para nunca matar o grupo do supervisor ou de
    um processo alheio.
    """
    if pid <= 0 or IS_WINDOWS:
        return None
    try:
        pgid = os.getpgid(int(pid))
    except OSError:
        return None
    return pgid if pgid == int(pid) else None


def signal_process_group(pgid: int, sig: int) -> bool:
    """Envia um sinal ao grupo ``pgid``; ``False`` se já não existe/sem permissão."""
    if pgid <= 0 or IS_WINDOWS:
        return False
    try:
        os.killpg(int(pgid), int(sig))
        return True
    except OSError:
        return False


def process_group_members(pgid: int) -> set[int]:
    """PIDs **vivos** (não-zombie) do grupo ``pgid`` — leitura de ``/proc``.

    ``os.killpg(pgid, 0)`` também acerta em zombies (o líder pode ficar
    ``<defunct>`` à espera do ``wait`` do ``Popen``); usá-lo para decidir um
    SIGKILL ao grupo mataria o líder cedo demais e queimava a janela graciosa.
    Aqui os zombies são ignorados e o líder é excluído pelos callers.
    """
    if pgid <= 0 or IS_WINDOWS:
        return set()
    root = Path("/proc")
    if not root.is_dir():
        return set()
    out: set[int] = set()
    try:
        entries = list(root.iterdir())
    except OSError:  # pragma: no cover — /proc inacessível
        return set()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # ``comm`` pode conter espaços/parênteses: tudo depois do ÚLTIMO ')'.
        try:
            fields = stat[stat.rfind(")") + 2 :].split()
            if len(fields) < 3:
                continue
            state, pgrp = fields[0], int(fields[2])
        except (ValueError, IndexError):
            continue
        if pgrp == int(pgid) and state != "Z":
            out.add(int(entry.name))
    return out


def process_group_exists(pgid: int) -> bool:
    """True se o grupo ainda tem membros (inclui zombies) — ``killpg(pgid, 0)``.

    Fallback para plataformas sem ``/proc`` (macOS): sem lista de membros este
    é o único probe disponível para decidir o SIGKILL final do grupo. No Linux
    prefere-se :func:`process_group_members` (ignora zombies).
    """
    if pgid <= 0 or IS_WINDOWS:
        return False
    try:
        os.killpg(int(pgid), 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def terminate_process_tree(
    pid: int,
    *,
    grace_sec: float = 3.0,
    include_root: bool = True,
    pgid: int | None = None,
) -> list[int]:
    """Termina o processo e os descendentes; devolve os PIDs atingidos.

    No Windows não há grupos de processos/sessões como no POSIX: um worker que
    lança dataloaders deixa-os vivos se só o pai for terminado. Esta função
    fecha esse buraco nos dois SOs: no POSIX usa ``killpg`` quando o worker é
    líder de sessão (achilles: ``_xatlas_worker`` a 299% CPU depois de o paint
    worker morrer); no Windows e como fallback usa ``psutil``.

    Args:
        pgid: PGID capturado no spawn. Obrigatório para apanhar descendentes
            quando o líder **já morreu** (crash/OOM): nesse estado ``getpgid``
            falha e o ``/proc`` já não liga os filhos ao PID do pai.
    """
    pids = process_tree_pids(pid)
    if not include_root:
        pids = [p for p in pids if p != pid]
    pids = [p for p in pids if p != os.getpid()]
    group = process_group_id(pid) if include_root else None
    if group is None and include_root and pgid is not None and int(pgid) > 0:
        group = int(pgid)
    if not pids and group is None:
        return []
    try:
        import psutil
    except ImportError:  # pragma: no cover — psutil é dependência
        return []

    if group is not None:
        # Grupo próprio: um único SIGTERM alcança netos que já não são filhos
        # directos do worker (dataloaders, _xatlas_worker).
        signal_process_group(group, signal.SIGTERM)
        children = [p for p in pids if p != pid]
        deadline = time.monotonic() + max(0.0, grace_sec)
        while children and time.monotonic() < deadline:
            if not any(pid_exists(p) for p in children):
                break
            time.sleep(0.1)
        # Há membro vivo além do líder? Então o TERM foi ignorado (ou nasceu
        # um filho depois do snapshot) — KILL ao grupo. Membros vivos são
        # lidos de /proc (zombies não contam), por isso um líder `defunct` à
        # espera do wait não encurta a janela graciosa.
        stubborn = any(pid_exists(p) for p in children) or bool(process_group_members(group) - {pid})
        if not stubborn and not Path("/proc").is_dir():
            # macOS: sem /proc não há lista de membros; killpg(0) é o probe.
            stubborn = process_group_exists(group)
        if stubborn:
            signal_process_group(group, getattr(signal, "SIGKILL", signal.SIGTERM))
        return pids

    procs: list[Any] = []
    for p in pids:
        try:
            procs.append(psutil.Process(p))
        except Exception:
            continue
    for proc in procs:
        with contextlib.suppress(Exception):
            proc.terminate()
    if procs:
        _, alive = psutil.wait_procs(procs, timeout=max(0.0, grace_sec))
        for proc in alive:
            with contextlib.suppress(Exception):
                proc.kill()
    return pids


# ---------------------------------------------------------------------------
# GPU / ferramentas externas
# ---------------------------------------------------------------------------


def nvidia_smi_path() -> str | None:
    """``nvidia-smi`` no PATH ou nos caminhos canónicos do Windows.

    O instalador da NVIDIA não põe sempre ``NVSMI`` no PATH (drivers novos
    trazem o binário em ``System32``; antigos em ``Program Files``).
    """
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if IS_WINDOWS:
        program_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
        candidates = (
            Path(program_files) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
            Path(system_root) / "System32" / "nvidia-smi.exe",
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    return None


# ---------------------------------------------------------------------------
# Criação de processos
# ---------------------------------------------------------------------------


def detached_popen_kwargs() -> dict[str, Any]:
    """Kwargs para lançar o supervisor destacado do processo chamador."""
    if IS_WINDOWS:
        flags = 0
        flags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        flags |= int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {"creationflags": flags} if flags else {}
    return {"start_new_session": True}


def worker_popen_kwargs() -> dict[str, Any]:
    """Kwargs para o worker subprocesso (stdio já vai por PIPE/ficheiro).

    ``CREATE_NO_WINDOW`` evita uma janela de consola por worker; o novo grupo
    de processos impede que o Ctrl+C da consola do supervisor chegue ao worker
    (o abort é cooperativo, pelo protocolo).
    """
    if IS_WINDOWS:
        flags = 0
        flags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        flags |= int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {"creationflags": flags} if flags else {}
    return {"start_new_session": True}


# ---------------------------------------------------------------------------
# Sinais
# ---------------------------------------------------------------------------


def install_signal_handlers(
    handler: Callable[[int, Any], None],
    *,
    signals: tuple[int, ...] | None = None,
) -> list[int]:
    """Instala handlers de paragem na main thread; devolve os que pegaram.

    No Windows não existem ``SIGTERM``/``SIGKILL`` entregues pela consola; o
    ``SIGBREAK`` é o equivalente ao Ctrl+Break. O encerramento normal continua
    a ser o RPC ``shutdown``, por isso a falha de um handler nunca é fatal.
    """
    if threading.current_thread() is not threading.main_thread():
        return []
    if signals is None:
        signals = (signal.SIGINT, signal.SIGTERM)
        if IS_WINDOWS and hasattr(signal, "SIGBREAK"):
            signals = (*signals, signal.SIGBREAK)  # type: ignore[attr-defined]
    installed: list[int] = []
    for sig in signals:
        with contextlib.suppress(OSError, ValueError, RuntimeError):
            signal.signal(sig, handler)
            installed.append(sig)
    return installed


def platform_label() -> str:
    """Etiqueta curta para logs/status (``linux``, ``win32``, ``darwin``)."""
    return sys.platform
