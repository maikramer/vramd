"""Saúde dos tiers de hardware: qualidade que se auto-repara depois de um OOM.

O hw-auto escolhe o tier pela VRAM total da GPU — uma regra estática. Na
prática há assets que estouram um tier que "cabia" (malhas enormes, refs a
mais, fragmentação). Sem isto, um OOM a fp16 numa GPU de 12 GB falhava o job
e o seguinte, igual, voltava a estourar.

Regra (por backend, rank do tier e tamanho de GPU):

* **OOM real** durante a geração → o tier fica *suspenso* nesse tamanho de
  GPU por um cooldown que dobra a cada reincidência (30 min, 1 h, 2 h… até
  24 h). O job é re-tentado no tier abaixo, automaticamente.
* **Fim do cooldown** → o tier volta a ser tentado (uma sonda). Se passar, o
  histórico zera; se voltar a estourar, o cooldown seguinte é o dobro.
* Recusas de **admissão** (livre < pico porque há outros modelos em VRAM) não
  contam: são transitórias e o retry de VRAM do dispatcher já as trata.

O estado persiste em ``<cache>/learn/tier_health.json`` — sobrevive a restarts,
que é quando um operador mais tenta "outra vez igual".
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import protocol as P

HEALTH_FILENAME = "tier_health.json"
DEFAULT_COOLDOWN_SEC = 1800.0
MAX_COOLDOWN_SEC = 24 * 3600.0

# Texto que denuncia um OOM *durante* a geração (não uma recusa de admissão).
_OOM_MARKERS = ("out of memory", "outofmemoryerror", "oom-spin", "cudaerrormemoryallocation", "cublas_status_alloc")


def is_runtime_oom(result: Mapping[str, Any]) -> bool:
    """O worker ficou sem VRAM a meio do job (e não o supervisor a recusá-lo)."""
    if result.get("status") == P.STATUS_OK:
        return False
    text = str(result.get("error") or "").lower()
    if "vram insuficiente" in text:
        # Recusa de admissão do BackendManager: pico > livre agora.
        return False
    return any(marker in text for marker in _OOM_MARKERS)


def size_bucket(total_mib: int | None) -> int | None:
    """GPUs do mesmo modelo reportam totais ligeiramente diferentes: agrupa por GiB."""
    if total_mib is None or total_mib <= 0:
        return None
    return max(1, round(int(total_mib) / 1024))


@dataclass
class TierRecord:
    backend: str
    rank: int
    gib: int
    strikes: int = 0
    banned_until: float = 0.0
    last_error: str | None = None
    last_oom_at: float = 0.0
    successes: int = 0

    def key(self) -> str:
        return f"{self.backend}|{self.rank}|{self.gib}"


class TierHealth:
    """Registo de OOMs por tier; responde "que ranks evitar nesta GPU agora".

    Args:
        path: Ficheiro JSON (``None`` = só em memória — testes).
        clock: Relógio de parede (os cooldowns sobrevivem a restarts).
        base_cooldown_sec: Primeiro cooldown; dobra a cada reincidência.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        clock: Callable[[], float] = time.time,
        base_cooldown_sec: float | None = None,
        max_cooldown_sec: float = MAX_COOLDOWN_SEC,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._clock = clock
        raw = os.environ.get("VRAMD_TIER_COOLDOWN_SEC", "").strip()
        default = DEFAULT_COOLDOWN_SEC
        with contextlib.suppress(ValueError):
            default = float(raw) if raw else default
        # Cooldown negativo/NaN violava o contrato "suspenso por N segundos":
        # o tier ficava imediatamente elegível e as strikes cresciam sem fim.
        base = float(base_cooldown_sec if base_cooldown_sec is not None else default)
        limit = float(max_cooldown_sec)
        self.base_cooldown_sec = base if math.isfinite(base) else 0.0
        self.base_cooldown_sec = max(0.0, self.base_cooldown_sec)
        self.max_cooldown_sec = max(0.0, limit if math.isfinite(limit) else 0.0)
        self._lock = threading.Lock()
        self._records: dict[str, TierRecord] = {}
        self._load()

    # -- persistência --------------------------------------------------------

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for entry in raw.get("tiers", []) if isinstance(raw, Mapping) else []:
            with contextlib.suppress(TypeError, ValueError, KeyError):
                record = TierRecord(
                    backend=str(entry["backend"]),
                    rank=int(entry["rank"]),
                    gib=int(entry["gib"]),
                    strikes=int(entry.get("strikes", 0)),
                    banned_until=float(entry.get("banned_until", 0.0)),
                    last_error=entry.get("last_error"),
                    last_oom_at=float(entry.get("last_oom_at", 0.0)),
                    successes=int(entry.get("successes", 0)),
                )
                self._records[record.key()] = record

    def _save_locked(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(
                json.dumps({"tiers": [asdict(r) for r in self._records.values()]}, indent=1),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
        except OSError:
            pass

    # -- registo -------------------------------------------------------------

    def _record(self, backend: str, rank: int, gib: int) -> TierRecord:
        key = f"{backend}|{rank}|{gib}"
        record = self._records.get(key)
        if record is None:
            record = TierRecord(backend=backend, rank=rank, gib=gib)
            self._records[key] = record
        return record

    def record_oom(self, backend: str, rank: int, total_mib: int | None, error: str | None = None) -> float | None:
        """Suspende o tier nesta GPU. Devolve o cooldown aplicado (``None`` sem GPU conhecida)."""
        gib = size_bucket(total_mib)
        if gib is None:
            return None
        with self._lock:
            record = self._record(backend, int(rank), gib)
            record.strikes += 1
            cooldown = min(self.max_cooldown_sec, self.base_cooldown_sec * (2 ** (record.strikes - 1)))
            now = self._clock()
            record.banned_until = now + cooldown
            record.last_oom_at = now
            record.last_error = (error or "")[:300] or None
            record.successes = 0
            self._save_locked()
        return cooldown

    def record_ok(self, backend: str, rank: int, total_mib: int | None) -> None:
        """Um job passou neste tier: depois de uma sonda bem sucedida, o histórico zera."""
        gib = size_bucket(total_mib)
        if gib is None:
            return
        with self._lock:
            record = self._records.get(f"{backend}|{int(rank)}|{gib}")
            if record is None or record.strikes == 0:
                return
            if self._clock() < record.banned_until:
                return
            record.successes += 1
            record.strikes = 0
            record.banned_until = 0.0
            self._save_locked()

    # -- consulta ------------------------------------------------------------

    def banned_ranks(self, backend: str, total_mib: int | None) -> frozenset[int]:
        """Ranks suspensos para este backend numa GPU deste tamanho, agora."""
        gib = size_bucket(total_mib)
        if gib is None:
            return frozenset()
        now = self._clock()
        with self._lock:
            return frozenset(
                r.rank for r in self._records.values() if r.backend == backend and r.gib == gib and r.banned_until > now
            )

    def reset(self, backend: str | None = None) -> int:
        with self._lock:
            keys = [k for k, r in self._records.items() if backend is None or r.backend == backend]
            for key in keys:
                del self._records[key]
            self._save_locked()
        return len(keys)

    def status(self) -> list[dict[str, Any]]:
        now = self._clock()
        with self._lock:
            rows = []
            for r in sorted(self._records.values(), key=lambda x: (x.backend, x.gib, x.rank)):
                if r.strikes == 0:
                    continue
                rows.append(
                    {
                        "backend": r.backend,
                        "rank": r.rank,
                        "gpu_gib": r.gib,
                        "strikes": r.strikes,
                        "suspended": r.banned_until > now,
                        "suspended_sec": round(max(0.0, r.banned_until - now), 1),
                        "last_error": r.last_error,
                    }
                )
            return rows
