"""Hooks de eventos — o vramd reage e integra-se sem editar código.

Um hook é um comando shell disparado quando algo acontece no supervisor:
job termina, backend é evicted, VRAM é zerada, o learn detecta drift, o
supervisor encerra. O payload viaja como JSON no stdin e como variáveis de
ambiente (``VRAMD_EVENT``, ``VRAMD_HOOK``), e pode ser interpolado no argv
com ``${campo}``::

    # ~/.config/vramd/hooks.yaml
    hooks:
      - event: on_job_failed
        command: ["notify-send", "-u", "critical", "vramd", "${backend}: ${error_code}"]
      - event: on_drift
        command: ["/usr/local/bin/recalibrate-and-notify.sh"]   # JSON no stdin
      - event: on_job_done
        events: [on_job_done, on_job_cancelled]
        command: ["curl", "-sS", "-XPOST", "https://hooks.exemplo/vramd", "-d@-"]
        timeout_sec: 5

Contrato: os hooks nunca podem derrubar nem atrasar o supervisor. Correm em
threads daemon, com timeout, throttling por hook e falhas apenas logadas.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .logging import Logger

_logger = Logger()

# Eventos disparados pelo supervisor (ver VramdServer).
EVENT_JOB_DONE = "on_job_done"
EVENT_JOB_FAILED = "on_job_failed"
EVENT_JOB_CANCELLED = "on_job_cancelled"
EVENT_EVICT = "on_evict"
EVENT_ZERO = "on_zero"
EVENT_DRIFT = "on_drift"
EVENT_SHUTDOWN = "on_shutdown"

KNOWN_EVENTS = (
    EVENT_JOB_DONE,
    EVENT_JOB_FAILED,
    EVENT_JOB_CANCELLED,
    EVENT_EVICT,
    EVENT_ZERO,
    EVENT_DRIFT,
    EVENT_SHUTDOWN,
)

ENV_HOOKS_FILE = "VRAMD_HOOKS_FILE"
DEFAULT_HOOKS_PATH = Path.home() / ".config" / "vramd" / "hooks.yaml"
DEFAULT_HOOK_TIMEOUT_SEC = 10.0


@dataclass(frozen=True)
class HookSpec:
    """Um hook declarado: eventos + comando argv."""

    events: frozenset[str]
    command: tuple[str, ...]
    timeout_sec: float = DEFAULT_HOOK_TIMEOUT_SEC
    name: str = ""

    def matches(self, event: str) -> bool:
        return event in self.events


@dataclass
class HookStats:
    """Contadores para o ``status`` (observabilidade dos próprios hooks)."""

    fired: int = 0
    succeeded: int = 0
    failed: int = 0
    throttled: int = 0
    last_error: str | None = None
    last_fired_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "fired": self.fired,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "throttled": self.throttled,
                "last_error": self.last_error,
            }


def parse_hooks(raw: Mapping[str, Any]) -> list[HookSpec]:
    """Valida o bloco ``hooks:`` de um documento YAML.

    Raises:
        ValueError: estrutura inválida (hook sem comando/evento, timeout não
            numérico). Falhar no arranque é correcto: um hook mal escrito que
            corresse em silêncio seria pior.
    """
    entries = raw.get("hooks")
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError("hooks.yaml: 'hooks' deve ser uma lista")
    specs: list[HookSpec] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"hooks.yaml: hooks[{i}] deve ser um mapa")
        events: set[str] = set()
        single = entry.get("event")
        if single:
            events.add(str(single))
        multiple = entry.get("events")
        if multiple:
            if not isinstance(multiple, list):
                raise ValueError(f"hooks.yaml: hooks[{i}].events deve ser uma lista")
            events.update(str(e) for e in multiple)
        unknown = events.difference(KNOWN_EVENTS)
        if unknown:
            raise ValueError(f"hooks.yaml: hooks[{i}] evento(s) desconhecido(s): {sorted(unknown)}")
        if not events:
            raise ValueError(f"hooks.yaml: hooks[{i}] precisa de 'event' ou 'events'")
        command = entry.get("command")
        if isinstance(command, str):
            argv = tuple(shlex.split(command))
        elif isinstance(command, (list, tuple)):
            argv = tuple(str(c) for c in command)
        else:
            raise ValueError(f"hooks.yaml: hooks[{i}].command em falta ou não é str/lista")
        if not argv:
            raise ValueError(f"hooks.yaml: hooks[{i}].command vazio")
        timeout_raw = entry.get("timeout_sec", DEFAULT_HOOK_TIMEOUT_SEC)
        try:
            timeout_sec = float(timeout_raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"hooks.yaml: hooks[{i}].timeout_sec não numérico: {timeout_raw!r}") from e
        import math

        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            # NaN passava as comparações de timeout todas (hook eterno, um por
            # janela de throttle); <=0 matava o hook no arranque (silêncio).
            raise ValueError(f"hooks.yaml: hooks[{i}].timeout_sec deve ser > 0 e finito: {timeout_raw!r}")
        timeout_sec = min(timeout_sec, 300.0)  # teto: um hook não pendura o boot de integrations
        specs.append(
            HookSpec(
                events=frozenset(events),
                command=argv,
                timeout_sec=timeout_sec,
                name=str(entry.get("name") or argv[0]),
            )
        )
    return specs


def load_hooks(path: Path | None = None) -> list[HookSpec]:
    """Lê ``~/.config/vramd/hooks.yaml`` (ou ``$VRAMD_HOOKS_FILE``).

    Ficheiro inexistente → ``[]`` (hooks são opt-in). Ficheiro malformado →
    :class:`ValueError` no arranque do supervisor.
    """
    raw_path = path or Path(os.environ.get(ENV_HOOKS_FILE, "") or DEFAULT_HOOKS_PATH)
    raw_path = Path(os.path.expanduser(str(raw_path)))
    if not raw_path.is_file():
        return []
    try:
        doc = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        # Normalizar para o ValueError do contrato: o doctor (e o caller do
        # arranque) apanham (ValueError, OSError) — um yaml.YAMLError cru
        # passava ao lado e o traceback comia o modo de falha desenhado.
        raise ValueError(f"hooks.yaml: YAML inválido: {e}") from e
    if doc is None:
        return []
    if not isinstance(doc, Mapping):
        raise ValueError(f"hooks.yaml: documento deve ser um mapa ({raw_path})")
    return parse_hooks(doc)


def _substitute(token: str, payload: Mapping[str, Any]) -> str:
    """``"--title=${backend}"`` com ``{"backend": "text3d"}`` → preenchido.

    Campos desconhecidos viram string vazia — um hook não deve rebentar porque
    um evento não tem o campo que ele menciona.
    """

    def _repl(match: re.Match[str]) -> str:
        key = match.group(1)
        value = payload.get(key)
        return "" if value is None else str(value)

    return re.sub(r"\$\{([A-Za-z0-9_.]+)\}", _repl, token)


class HookRunner:
    """Despacha eventos aos hooks declarados — async, com timeout e throttle."""

    def __init__(
        self,
        specs: list[HookSpec] | None = None,
        *,
        min_interval_sec: float = 1.0,
        runner: Any = None,
    ) -> None:
        self.specs = list(specs or [])
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        # ``runner(cmd, timeout_sec, env, input_text)`` injectável para testes;
        # default: subprocess.run real.
        self._runner = runner or self._run_subprocess
        self.stats = HookStats()
        self._last_fired: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()
        # Threads de hooks vivas — para o drain() do shutdown (o on_shutdown
        # despachado em fire-and-forget morria ao meio quando o interpretador saía).
        self._live: set[threading.Thread] = set()

    # ------------------------------------------------------------------

    @staticmethod
    def _run_subprocess(argv: list[str], timeout_sec: float, env: dict[str, str], input_text: str) -> int:
        """Corre o hook; retorna o exit code (excepções viram code≠0).

        Output → DEVNULL: o output do hook não é usado para nada e o
        ``capture_output=True`` anterior buferizava TUDO em RAM do supervisor —
        um hook tagarela/spinner (`yes`, `dmesg -w`) a correr até ao timeout de
        10s enchia GBs no processo que gere a VRAM (OOM kill do supervisor).
        """
        try:
            proc = subprocess.run(
                argv,
                input=input_text,
                text=True,  # sem isto, input=str lança TypeError (silenciado)
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_sec,
                env=env,
            )
            return int(proc.returncode)
        except subprocess.TimeoutExpired:
            return -9
        except Exception:
            return -1

    def _hook_name(self, spec: HookSpec) -> str:
        """Nome estável do hook (env/throttle/stats) — specs construídos à mão
        podem não ter ``name``; o argv[0] serve."""
        return spec.name or (spec.command[0] if spec.command else "?")

    def dispatch(self, event: str, payload: Mapping[str, Any] | None = None) -> int:
        """Dispara os hooks que matcham o evento. Non-blocking; retorna nº lançados.

        Cada match corre numa thread daemon: o supervisor nunca espera pelo
        ``curl`` de alguém. Throttle por (evento, hook): rajadas de jobs não
        geram rajadas de notificações.
        """
        payload = dict(payload or {})
        launched = 0
        now = time.monotonic()
        for spec in self.specs:
            if not spec.matches(event):
                continue
            name = self._hook_name(spec)
            key = (event, name)
            with self._lock:
                if now - self._last_fired.get(key, 0.0) < self.min_interval_sec:
                    self.stats.throttled += 1
                    continue
                self._last_fired[key] = now
                self.stats.fired += 1
                self.stats.last_fired_at = now
            try:
                t = threading.Thread(
                    target=self._run_hook_tracked,
                    args=(spec, event, payload),
                    name=f"vramd-hook-{name}",
                    daemon=True,
                )
                t.start()
            except RuntimeError:
                # Sem recursos para threads (RLIMIT a caminho): contar como
                # falha em vez de propagar — um dispatch no caminho de sucesso
                # do zero-vram convertia-o em erro para o cliente.
                with self.stats._lock:
                    self.stats.failed += 1
                    self.stats.last_error = f"{name}({event}) thread-start falhou"
                continue
            launched += 1
        return launched

    def _run_hook_tracked(self, spec: HookSpec, event: str, payload: Mapping[str, Any]) -> None:
        t = threading.current_thread()
        with self._lock:
            self._live.add(t)
        try:
            self._run_hook(spec, event, payload)
        finally:
            with self._lock:
                self._live.discard(t)

    def drain(self, timeout_sec: float = 3.0) -> None:
        """Espera (bounded) pelos hooks vivos — shutdown gracioso.

        Sem isto, o hook ``on_shutdown`` despachado pelo ``_cleanup`` era morto
        ao meio pelo exit do interpretador: o evento que existe precisamente
        para integrações de encerramento raramente chegava a correr.
        """
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while True:
            with self._lock:
                threads = list(self._live)
            if not threads:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            for t in threads:
                t.join(timeout=max(0.0, min(0.5, remaining)))

    def _run_hook(self, spec: HookSpec, event: str, payload: Mapping[str, Any]) -> None:
        name = self._hook_name(spec)
        argv = [_substitute(tok, payload) for tok in spec.command]
        env = dict(os.environ)
        env["VRAMD_EVENT"] = event
        env["VRAMD_HOOK"] = name
        input_text = json.dumps({"event": event, "hook": name, **payload}, ensure_ascii=False, default=str)
        try:
            code = self._runner(argv, spec.timeout_sec, env, input_text)
        except Exception as e:  # runner injectado pode levantar
            code = -1
            _logger.warn(f"[vramd-hooks] {name}({event}) exc: {e}")
        with self.stats._lock:
            if code == 0:
                self.stats.succeeded += 1
            else:
                self.stats.failed += 1
                self.stats.last_error = f"{name}({event}) exit={code}"
        if code != 0:
            _logger.warn(f"[vramd-hooks] {name}({event}) falhou com exit={code}: {' '.join(argv)}")

    # ------------------------------------------------------------------

    def dispatch_sync(self, event: str, payload: Mapping[str, Any] | None = None) -> int:
        """Versão síncrona para testes / ferramentas CLI (espera os hooks)."""
        payload = dict(payload or {})
        launched = 0
        for spec in self.specs:
            if not spec.matches(event):
                continue
            with self._lock:
                self._last_fired[(event, self._hook_name(spec))] = time.monotonic()
                self.stats.fired += 1
            self._run_hook(spec, event, payload)
            launched += 1
        return launched

    def status_dict(self) -> dict[str, Any]:
        return {
            "configured": len(self.specs),
            "events": sorted({e for spec in self.specs for e in spec.events}),
            **self.stats.to_dict(),
        }
