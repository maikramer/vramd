"""Histórico persistente de jobs terminados.

A fila só guarda jobs terminados enquanto alguém pode estar à espera deles; o
WAL só serve para recuperar pendentes. Depois de um batch nocturno a pergunta
"o que correu, quanto demorou e o que falhou?" não tinha resposta sem ler logs.

Este módulo mantém um JSONL append-only, limitado em tamanho, ao lado do socket
(``<cache>/vramd-history.jsonl``). O supervisor só escreve; a leitura é feita
directamente do ficheiro, por isso ``vramd history`` funciona com o supervisor
parado — que é exactamente quando se quer saber porque parou.

O request do job **não** é guardado: prompts podem ser privados e grandes. Fica
só o que descreve a execução (backend, estado, tempos, erro, output).
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from . import protocol as P

HISTORY_FILENAME = "vramd-history.jsonl"
ENV_HISTORY_MAX = "VRAMD_HISTORY_MAX"
DEFAULT_HISTORY_MAX = 2000
_ERROR_MAX_CHARS = 400


def default_history_path() -> Path:
    """Ficheiro de histórico do supervisor canónico."""
    return P.DEFAULT_SOCKET_PATH.parent / HISTORY_FILENAME


def history_max_entries() -> int:
    """Teto de entradas (``VRAMD_HISTORY_MAX``; 0 desliga o histórico)."""
    raw = os.environ.get(ENV_HISTORY_MAX, "").strip()
    if not raw:
        return DEFAULT_HISTORY_MAX
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_HISTORY_MAX


def entry_from_job(job: Any, *, now: float | None = None) -> dict[str, Any]:
    """Entrada de histórico a partir de um job em estado terminal."""
    result = getattr(job, "result", None) or {}
    timing = job.timing_dict() if hasattr(job, "timing_dict") else {}
    error = result.get("error")
    entry: dict[str, Any] = {
        "ts": round(time.time() if now is None else now, 3),
        "job_id": str(getattr(job, "job_id", "")),
        "backend": str(getattr(job, "backend", "")),
        "state": str(getattr(job, "state", "")),
        "priority": getattr(job, "priority", None),
        "queue_wait_sec": timing.get("queue_wait_sec"),
        "generate_sec": timing.get("generate_sec"),
        "total_sec": timing.get("total_sec"),
        "vram_retries": int(getattr(job, "vram_retries", 0) or 0),
        "affinity_cuts": int(getattr(job, "affinity_cuts", 0) or 0),
    }
    if error:
        entry["error"] = str(error)[:_ERROR_MAX_CHARS]
    if result.get("error_code"):
        entry["error_code"] = str(result["error_code"])
    output = result.get("output")
    if isinstance(output, (str, int, float)):
        entry["output"] = output if not isinstance(output, str) else output[:_ERROR_MAX_CHARS]
    return entry


class JobHistory:
    """JSONL append-only com teto de entradas.

    Args:
        path: Ficheiro de histórico.
        max_entries: Entradas mantidas; ``0`` desliga (``record`` é no-op).

    A compactação reescreve o ficheiro (tmp + replace) quando este excede o
    teto por uma folga — nunca a cada append, para o custo ser amortizado.
    """

    def __init__(self, path: Path | str | None = None, *, max_entries: int | None = None) -> None:
        self.path = Path(path) if path is not None else default_history_path()
        self.max_entries = history_max_entries() if max_entries is None else max(0, int(max_entries))
        self._lock = threading.Lock()
        self._count: int | None = None  # linhas no ficheiro (lazy)

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0

    def _slack(self) -> int:
        return max(16, self.max_entries // 4)

    def _count_lines(self) -> int:
        try:
            with self.path.open("rb") as fh:
                return sum(1 for _ in fh)
        except OSError:
            return 0

    def record(self, entry: dict[str, Any]) -> bool:
        """Acrescenta uma entrada. Nunca lança (o histórico não pode partir a fila)."""
        if not self.enabled:
            return False
        try:
            line = json.dumps(entry, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return False
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self._count is None:
                    self._count = self._count_lines()
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                self._count += 1
                if self._count > self.max_entries + self._slack():
                    self._compact_locked()
        except OSError:
            return False
        return True

    def record_job(self, job: Any) -> bool:
        """Atalho: constrói a entrada a partir do job e grava-a."""
        if not self.enabled:
            return False
        try:
            return self.record(entry_from_job(job))
        except Exception:
            return False

    def _compact_locked(self) -> None:
        keep: deque[str] = deque(maxlen=self.max_entries)
        with self.path.open(encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                if raw.strip():
                    keep.append(raw if raw.endswith("\n") else raw + "\n")
        tmp = self.path.with_suffix(f".jsonl.{os.getpid()}.tmp")
        try:
            tmp.write_text("".join(keep), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise
        self._count = len(keep)

    def clear(self) -> int:
        """Apaga o histórico; devolve quantas entradas existiam."""
        with self._lock:
            count = self._count_lines()
            with contextlib.suppress(OSError):
                self.path.unlink(missing_ok=True)
            self._count = 0
        return count

    def read(
        self,
        *,
        limit: int | None = 50,
        backend: str | None = None,
        state: str | None = None,
        since_sec: float | None = None,
    ) -> list[dict[str, Any]]:
        """Entradas mais recentes primeiro, com filtros opcionais."""
        return read_history(self.path, limit=limit, backend=backend, state=state, since_sec=since_sec)


def read_history(
    path: Path | str | None = None,
    *,
    limit: int | None = 50,
    backend: str | None = None,
    state: str | None = None,
    since_sec: float | None = None,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Lê o histórico (mais recentes primeiro). Linhas corrompidas são saltadas.

    Args:
        path: Ficheiro (default: o do supervisor canónico).
        limit: Máximo de entradas devolvidas (``None`` = todas).
        backend: Só este backend.
        state: Só este estado (``done``/``failed``/``cancelled``).
        since_sec: Só entradas terminadas nos últimos N segundos.
        now: Relógio injectável (testes).
    """
    target = Path(path) if path is not None else default_history_path()
    cutoff = None if since_sec is None else (time.time() if now is None else now) - float(since_sec)
    keep: deque[dict[str, Any]] = deque(maxlen=limit if limit and limit > 0 else None)
    try:
        with target.open(encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                if backend and entry.get("backend") != backend:
                    continue
                if state and entry.get("state") != state:
                    continue
                if cutoff is not None and float(entry.get("ts") or 0.0) < cutoff:
                    continue
                keep.append(entry)
    except OSError:
        return []
    return list(reversed(keep))


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((p / 100.0) * (len(ordered) - 1))))
    return round(ordered[idx], 3)


def summarize(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Agregado por backend: contagens por estado e tempos de geração.

    Só jobs ``done`` contam para os tempos — um job falhado aos 2 s puxaria a
    média para baixo e faria o backend parecer mais rápido do que é.
    """
    out: dict[str, dict[str, Any]] = {}
    durations: dict[str, list[float]] = {}
    waits: dict[str, list[float]] = {}
    for entry in entries:
        name = str(entry.get("backend") or "?")
        row = out.setdefault(name, {"jobs": 0, "done": 0, "failed": 0, "cancelled": 0})
        row["jobs"] += 1
        state = str(entry.get("state") or "")
        if state in ("done", "failed", "cancelled"):
            row[state] += 1
        gen = entry.get("generate_sec")
        if state == "done" and isinstance(gen, (int, float)):
            durations.setdefault(name, []).append(float(gen))
        wait = entry.get("queue_wait_sec")
        if isinstance(wait, (int, float)):
            waits.setdefault(name, []).append(float(wait))
    for name, row in out.items():
        gens = durations.get(name, [])
        row["generate_avg_sec"] = round(sum(gens) / len(gens), 3) if gens else None
        row["generate_p95_sec"] = _percentile(gens, 95)
        row["generate_total_sec"] = round(sum(gens), 3) if gens else 0.0
        row["queue_wait_p95_sec"] = _percentile(waits.get(name, []), 95)
        finished = row["done"] + row["failed"]
        row["success_rate"] = round(row["done"] / finished, 4) if finished else None
    return out
