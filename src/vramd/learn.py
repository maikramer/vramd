"""Aprendizagem contínua de picos de VRAM — o loop fechado da admissão.

O ``vramd calibrate`` mede uma vez, em condições de laboratório: GPU limpa, job
sintético, kwargs conhecidos. O que falta é saber se aquele número — ou a
estimativa que o substitui — ainda é verdade semanas depois, com tráfego real:
prompts maiores, quantização diferente, versão nova da tool com outro footprint.

O :class:`PeakTracker` observa cada job em curso (VRAM do processo worker via
NVML, ~2 Hz) e regista o pico real. :func:`analyze_drift` compara o pico
observado com o pico declarado que a admissão usou **nesse job** — não com um
número recomputado depois — e classifica o desvio. ``vramd learn --apply``
escreve a correcção como overlay YAML na config do utilizador, pelo que o loop
fecha sem editar o package::

    tráfego real → observações → drift → overlay → admissão mais certeira

Precisão sobre alcance: só backends subprocesso (worker próprio, PID próprio)
são observados. Em modo in-process a VRAM do supervisor inclui outros backends
e a observação seria uma mentira útil.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .logging import Logger

_logger = Logger()

# Veredictos de drift (campo ``verdict`` do relatório).
VERDICT_OK = "ok"
VERDICT_UNDER = "underprovisioned"  # observado > declarado → risco de OOM
VERDICT_OVER = "overprovisioned"  # declarado ≫ observado → headroom desperdiçado
VERDICT_NO_DATA = "no_data"

_DEFAULT_DIR = Path.home() / ".cache" / "vramd" / "learn"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def learn_interval_sec() -> float:
    """Intervalo de amostragem do tracker (``VRAMD_LEARN_INTERVAL_SEC``; 0 desliga).

    Clamp a >=0.05s: sem isto, um env de 0.001 mete o tracker a 1 kHz (lock da
    fila + NVML por tick) dentro do daemon — burn de CPU sustentado.
    """
    return max(0.05, _env_float("VRAMD_LEARN_INTERVAL_SEC", 0.5))


@dataclass(frozen=True)
class PeakObservation:
    """Pico de VRAM observado num job real.

    ``declared_peak_mib`` é o pico que a admissão usou no momento do job (com
    os mesmos kwargs) — é com ele, e não com um recomputado, que o drift é
    medido. ``ok=False`` marca jobs que falharam: o picoobservado é um *lower
    bound* (o job morreu, talvez por OOM) e não entra no cálculo de veredicto.
    """

    backend: str
    job_id: str
    peak_mib: int
    declared_peak_mib: int | None
    quant_mode: str
    memory_efficient: bool
    group_offload: bool
    duration_sec: float
    samples: int
    ok: bool
    state: str
    started_at: float  # wall clock (epoch) — persistido, sobrevive a restarts

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PeakObservation:
        declared = raw.get("declared_peak_mib")
        return cls(
            backend=str(raw["backend"]),
            job_id=str(raw.get("job_id", "?")),
            peak_mib=int(raw.get("peak_mib") or 0),
            declared_peak_mib=int(declared) if declared is not None else None,
            quant_mode=str(raw.get("quant_mode") or "none"),
            memory_efficient=bool(raw.get("memory_efficient")),
            group_offload=bool(raw.get("group_offload")),
            duration_sec=float(raw.get("duration_sec") or 0.0),
            samples=int(raw.get("samples") or 0),
            ok=bool(raw.get("ok", True)),
            state=str(raw.get("state") or "done"),
            started_at=float(raw.get("started_at") or 0.0),
        )


class PeakLearningStore:
    """Persistência append-only das observações, um JSONL por backend.

    Sobrevive a restarts do supervisor (o drift é um sinal de tendência, não de
    um job). Cap por backend: ao ultrapassar, reescreve o ficheiro com as
    últimas N — append-only no caso comum, truncado uma vez a cada N jobs.
    """

    def __init__(self, root: Path | None = None, *, max_per_backend: int = 200) -> None:
        self.root = Path(root) if root else _DEFAULT_DIR
        self.max_per_backend = max(1, int(max_per_backend))
        self._lock = threading.Lock()

    def _path(self, backend: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in backend)
        return self.root / f"{safe}.jsonl"

    def append(self, obs: PeakObservation) -> None:
        line = json.dumps(obs.to_dict(), ensure_ascii=False)
        with self._lock:
            path = self._path(obs.backend)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._trim(path)

    def _trim(self, path: Path) -> None:
        """Mantém o ficheiro sob o cap (conta barata: nº de linhas)."""
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return
        if len(lines) <= self.max_per_backend:
            return
        keep = lines[-self.max_per_backend :]
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(keep), encoding="utf-8")
        tmp.replace(path)

    def recent(self, backend: str, *, limit: int = 50) -> list[PeakObservation]:
        path = self._path(backend)
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return []
        out: list[PeakObservation] = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            # Cache descartável NUNCA derruba o arranque: uma linha com JSON
            # válido mas schema inválido (KeyError/ValueError do from_dict,
            # NaN que passa no json.loads) rebentava o VramdServer.__init__ e o
            # supervisor recusava arrancar até alguém apagar o ficheiro à mão.
            with contextlib.suppress(Exception):
                out.append(PeakObservation.from_dict(json.loads(line)))
        return out

    def backends(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.jsonl"))

    def reset(self, backend: str | None = None) -> int:
        """Apaga observações (todas ou de um backend). Retorna ficheiros removidos."""
        with self._lock:
            if backend:
                path = self._path(backend)
                if path.exists():
                    path.unlink()
                    return 1
                return 0
            count = 0
            if self.root.is_dir():
                for path in self.root.glob("*.jsonl"):
                    path.unlink(missing_ok=True)
                    count += 1
            return count


def _percentile(samples: list[int], p: float) -> int | None:
    if not samples:
        return None
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, round((p / 100.0) * (len(ordered) - 1))))
    return int(ordered[idx])


def _round_up_mib(value: float, granularity: int = 64) -> int:
    return int(math.ceil(max(0.0, value) / granularity) * granularity)


@dataclass(frozen=True)
class DriftReport:
    """Veredicto de drift de um backend — o que ``vramd learn`` mostra/aplica."""

    backend: str
    verdict: str
    declared_peak_mib: int | None
    observed_p95_mib: int | None
    observed_max_mib: int | None
    samples: int
    suggested_mib: int | None = None
    has_measured_block: bool = False  # ``vram:`` calibrado presente (learn não sobrepõe)
    last_seen: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DriftReport:
        """Round-trip do RPC — o CLI ``learn --apply`` recebe isto do supervisor."""
        return cls(
            backend=str(raw.get("backend") or ""),
            verdict=str(raw.get("verdict") or VERDICT_NO_DATA),
            declared_peak_mib=raw.get("declared_peak_mib"),
            observed_p95_mib=raw.get("observed_p95_mib"),
            observed_max_mib=raw.get("observed_max_mib"),
            samples=int(raw.get("samples") or 0),
            suggested_mib=raw.get("suggested_mib"),
            has_measured_block=bool(raw.get("has_measured_block")),
            last_seen=raw.get("last_seen"),
        )


def analyze_drift(
    observations: list[PeakObservation],
    *,
    declared_peak_mib: int | None = None,
    min_samples: int = 3,
) -> DriftReport:
    """Classifica o desvio entre pico observado (jobs ok) e pico declarado.

    Só observações de jobs bem-sucedidos entram no veredicto: um pico de job
    que morreu é lower bound (muitas vezes morreu *por* ultrapassar) e usá-lo
    para «declarado está OK» seria exatamente o erro que o vramd existe para
    evitar.

    Bandas: UNDER quando p95 observado > declarado (a admissão reservou menos
    do que a realidade pede). OVER quando declarado ≥ 1.4x p95 (reservar 40%
    a mais recusa backends que caberiam). O resto é OK — margens de safety
    saudáveis não são drift.
    """
    if not observations:
        return DriftReport(
            backend="",
            verdict=VERDICT_NO_DATA,
            declared_peak_mib=declared_peak_mib,
            observed_p95_mib=None,
            observed_max_mib=None,
            samples=0,
        )
    backend = observations[-1].backend
    ok_obs = [o for o in observations if o.ok and o.peak_mib > 0]
    # Declared do job mais recente (é o que a admissão usa hoje).
    if declared_peak_mib is None:
        declared_peak_mib = next((o.declared_peak_mib for o in reversed(ok_obs) if o.declared_peak_mib), None)
    if len(ok_obs) < min_samples:
        return DriftReport(
            backend=backend,
            verdict=VERDICT_NO_DATA,
            declared_peak_mib=declared_peak_mib,
            observed_p95_mib=_percentile([o.peak_mib for o in ok_obs], 95),
            observed_max_mib=max((o.peak_mib for o in ok_obs), default=None),
            samples=len(ok_obs),
            last_seen=observations[-1].started_at,
        )
    peaks = [o.peak_mib for o in ok_obs]
    p95 = _percentile(peaks, 95)
    obs_max = max(peaks)
    if declared_peak_mib is None:
        verdict = VERDICT_NO_DATA
        suggested = None
    elif p95 > declared_peak_mib:
        verdict = VERDICT_UNDER
        # Subir com folga: o p95 JÁ estourou o declarado; 1.15x dá margem para
        # a cauda que ainda não observámos.
        suggested = _round_up_mib(p95 * 1.15)
    elif declared_peak_mib >= p95 * 1.4:
        verdict = VERDICT_OVER
        # Encolher com respeito: nunca sugerir abaixo de 1.25x o observado —
        # o safety margin existe para os dias maus.
        suggested = _round_up_mib(max(obs_max, p95) * 1.25)
    else:
        verdict = VERDICT_OK
        suggested = None
    return DriftReport(
        backend=backend,
        verdict=verdict,
        declared_peak_mib=declared_peak_mib,
        observed_p95_mib=p95,
        observed_max_mib=obs_max,
        samples=len(ok_obs),
        suggested_mib=suggested,
        last_seen=observations[-1].started_at,
    )


def learn_overlay_yaml(reports: list[DriftReport], *, generated_by: str = "vramd learn --apply") -> str:
    """Emite o overlay YAML com as correcções accionáveis.

    Só entra no overlay o que muda a admissão para melhor: backends com
    veredicto UNDER/OVER, sem bloco ``vram:`` calibrado (a calibração ganha ao
    learn por design) e com sugestão concreta.
    """
    import yaml

    entries = [
        {"name": r.backend, "vram_mib": int(r.suggested_mib or 0)}
        for r in reports
        if r.verdict in (VERDICT_UNDER, VERDICT_OVER)
        and not r.has_measured_block
        and r.suggested_mib
    ]
    header = (
        f"# Gerado por {generated_by} em {time.strftime('%Y-%m-%d %H:%M:%S')} — picos observados"
        f" na produção.\n# Sobrepõe vram_mib por chave; o resto do descriptor é herdado"
        f" (ver registry.py).\n"
    )
    if not entries:
        return header + "# (nada a corrigir — sem drift accionável)\nversion: 2\nbackends: []\n"
    doc = {"version": 2, "backends": entries}
    return header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


@dataclass
class _ActiveJob:
    """Job em curso que o tracker está a amostrar."""

    job_id: str
    backend: str
    pid: int | None
    declared_peak_mib: int | None
    quant_mode: str
    memory_efficient: bool
    group_offload: bool
    started_monotonic: float
    started_wall: float
    peak_mib: int = 0
    samples: int = 0


class PeakTracker:
    """Amostra a VRAM do worker de cada job running e regista o pico real.

    Corre como thread no supervisor; nunca lança excepções para fora (um
    tracker morto não pode matar a fila). Atribuição por PID do worker
    subprocesso — ver docstring do módulo.
    """

    def __init__(
        self,
        queue: Any,
        manager: Any,
        store: PeakLearningStore,
        *,
        interval_sec: float | None = None,
        recents_per_backend: int = 50,
        on_observation: Any = None,
        on_drift: Any = None,
        sample_vram_mib: Any = None,
    ) -> None:
        self.queue = queue
        self.manager = manager
        self.store = store
        self.interval_sec = learn_interval_sec() if interval_sec is None else max(0.05, float(interval_sec))
        self._recents_per_backend = recents_per_backend
        self._on_observation = on_observation
        self._on_drift = on_drift
        self._sample_vram_mib = sample_vram_mib
        self._active: dict[str, _ActiveJob] = {}
        self._recents: dict[str, list[PeakObservation]] = {
            name: store.recent(name, limit=recents_per_backend) for name in store.backends()
        }
        self._last_verdict: dict[str, str] = {}
        self._warned_declared: set[str] = set()
        # Backoff de amostragem: com NVML em baixo, cada _sample lança um
        # nvidia-smi (20s de timeout) por job por tick — driver wedged = tick
        # de dezenas de segundos. Falhas consecutivas ⇒ pausa de 30s.
        self._sample_fail_streak = 0
        self._sample_deadline = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # -- ciclo de vida ---------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.interval_sec > 0

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="vramd-learn", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def reset(self, backend: str | None = None) -> int:
        """Apaga observações (persistidas + em memória). Retorna ficheiros removidos.

        Memória E ficheiros sob o mesmo lock, e ``_active`` também é limpo: os
        jobs que já corriam antes do reset finalizam depois e re-semeavam o
        store com observações pré-reset (o ``learn`` voltava a reportar drift
        antigo logo a seguir ao ``--reset``).
        """
        with self._lock:
            if backend:
                self._recents.pop(backend, None)
                self._last_verdict.pop(backend, None)
                for job_id in [jid for jid, a in self._active.items() if a.backend == backend]:
                    self._active.pop(job_id, None)
            else:
                self._recents.clear()
                self._last_verdict.clear()
                self._active.clear()
            return self.store.reset(backend)

    # -- amostragem --------------------------------------------------------

    def _worker_pid(self, backend: str) -> int | None:
        pool = getattr(self.manager, "_subprocess_pool", None)
        if pool is None or not hasattr(pool, "worker_pid"):
            return None
        try:
            return pool.worker_pid(backend)
        except Exception:
            return None

    def _sample(self, pid: int) -> int | None:
        if self._sample_vram_mib is not None:
            try:
                return self._sample_vram_mib(pid)
            except Exception:
                return None
        try:
            from .gpu import process_vram_mib

            return process_vram_mib(pid)
        except Exception:
            return None

    def _begin(self, job: Any) -> _ActiveJob:
        declared: int | None = None
        quant, mem_eff, group_off, _streams = "none", False, False, False
        try:
            quant, mem_eff, group_off, _streams = self.manager.resolve_peak_params(job.backend, job.request)
            declared = int(
                self.manager.peak_vram_mib(
                    job.backend,
                    quant_mode=quant,
                    memory_efficient=mem_eff,
                    group_offload=group_off,
                    footprint_key=(job.request or {}).get("footprint_key"),
                )
            )
        except Exception as e:
            # Warn-ONCE por backend: um registry persistentemente partido deixa
            # declared=None para sempre → todos os reports NO_DATA, em silêncio.
            with self._lock:
                first = job.backend not in self._warned_declared
                self._warned_declared.add(job.backend)
            if first:
                _logger.warn(f"[vramd-learn] pico declarado indisponível para {job.backend!r}: {e}")
        active = _ActiveJob(
            job_id=job.job_id,
            backend=job.backend,
            pid=self._worker_pid(job.backend),
            declared_peak_mib=declared,
            quant_mode=quant,
            memory_efficient=mem_eff,
            group_offload=group_off,
            started_monotonic=time.monotonic(),
            started_wall=time.time(),
        )
        with self._lock:
            self._active[job.job_id] = active
        return active

    def _finalize(self, job_id: str) -> None:
        with self._lock:
            active = self._active.pop(job_id, None)
        if active is None or active.samples <= 0:
            return
        job = None
        with contextlib.suppress(Exception):
            job = self.queue.get(job_id)
        state = getattr(job, "state", "done") if job is not None else "done"
        ok = state == "done"
        obs = PeakObservation(
            backend=active.backend,
            job_id=job_id,
            peak_mib=active.peak_mib,
            declared_peak_mib=active.declared_peak_mib,
            quant_mode=active.quant_mode,
            memory_efficient=active.memory_efficient,
            group_offload=active.group_offload,
            duration_sec=max(0.0, time.monotonic() - active.started_monotonic),
            samples=active.samples,
            ok=ok,
            state=state,
            started_at=active.started_wall,
        )
        try:
            self.store.append(obs)
        except Exception as e:
            _logger.warn(f"[vramd-learn] falha ao persistir observação de {active.backend}: {e}")
        with self._lock:
            bucket = self._recents.setdefault(active.backend, [])
            bucket.append(obs)
            if len(bucket) > self._recents_per_backend:
                del bucket[: len(bucket) - self._recents_per_backend]
        if self._on_observation is not None:
            with _callback_guard():
                self._on_observation(obs, job)
        self._maybe_drift(active.backend)

    def _maybe_drift(self, backend: str) -> None:
        """Dispara ``on_drift`` quando o veredicto do backend MUDA (sem spam)."""
        if self._on_drift is None:
            return
        report = self.report_for(backend)
        previous = self._last_verdict.get(backend, VERDICT_NO_DATA)
        self._last_verdict[backend] = report.verdict
        if report.verdict != previous and report.verdict in (VERDICT_UNDER, VERDICT_OVER):
            with _callback_guard():
                self._on_drift(report)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            try:
                self._tick()
            except Exception as e:  # nunca propagar — o tracker é observador
                _logger.warn(f"[vramd-learn] tick falhou (ignorado): {e}")

    def _tick(self) -> None:
        running: dict[str, Any] = {}
        try:
            running = {j.job_id: j for j in self.queue.running_jobs()}
        except Exception as e:
            # Snapshot falhou ≠ fila vazia: com ``running = {}`` todos os jobs
            # activos eram finalizados prematuramente (ok=False) e re-beginados
            # no tick seguinte — observações duplicadas e pico subestimado,
            # exactamente a métrica que este módulo existe para medir.
            _logger.warn(f"[vramd-learn] snapshot da fila falhou — tick saltado: {e}")
            return
        with self._lock:
            known = set(self._active)
        for job_id in known - set(running):
            self._finalize(job_id)
        for job_id, job in running.items():
            with self._lock:
                active = self._active.get(job_id)
            if active is None:
                active = self._begin(job)
            if active.pid is None:
                # Worker pode ter nascido depois do job entrar em running.
                active.pid = self._worker_pid(job.backend)
                if active.pid is None:
                    continue
            if time.monotonic() < self._sample_deadline:
                continue  # backoff activo (probe de VRAM em baixo)
            mib = self._sample(active.pid)
            if mib is None:
                self._sample_fail_streak += 1
                if self._sample_fail_streak >= 5:
                    self._sample_deadline = time.monotonic() + 30.0
                    self._sample_fail_streak = 0
                    _logger.warn("[vramd-learn] probe de VRAM a falhar — pausa de amostragem 30s.")
                continue
            self._sample_fail_streak = 0
            if mib > 0:
                active.samples += 1
                if mib > active.peak_mib:
                    active.peak_mib = int(mib)

    # -- consultas ---------------------------------------------------------

    def observations(self, backend: str) -> list[PeakObservation]:
        with self._lock:
            return list(self._recents.get(backend, []))

    def report_for(self, backend: str, *, min_samples: int = 3) -> DriftReport:
        report = analyze_drift(self.observations(backend), min_samples=min_samples)
        has_measured = False
        try:
            desc = self.manager._registry.descriptor(backend)
            vram = getattr(desc, "vram", None) or {}
            has_measured = vram.get("weights_gib") is not None and vram.get("activation_gib") is not None
        except Exception:
            # Fail-safe: sem conseguir ler o descriptor, ASSUMIR calibrado — o
            # overlay do learn não pode atropelar um bloco ``vram:`` medido só
            # porque o lookup falhou (a calibração ganha ao learn).
            has_measured = True
        return replace(report, backend=backend, has_measured_block=has_measured)

    def report_all(self) -> list[DriftReport]:
        names = set(self._recents)
        with contextlib.suppress(Exception):
            names |= set(self.manager.loaded_names())
        return [self.report_for(name) for name in sorted(names)]

    def status_dict(self) -> dict[str, Any]:
        """Bloco ``learn:`` do ``status`` — compacto (poll de dashboards)."""
        return {
            "enabled": self.enabled,
            "interval_sec": round(self.interval_sec, 3),
            "active": len(self._active),
            "backends": {
                r.backend: {
                    "verdict": r.verdict,
                    "observed_p95_mib": r.observed_p95_mib,
                    "declared_peak_mib": r.declared_peak_mib,
                    "samples": r.samples,
                }
                for r in self.report_all()
                if r.samples > 0 or r.verdict != VERDICT_NO_DATA
            },
        }


class _callback_guard:
    """Callbacks de learn nunca derrubam o tracker que os chamou."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc is not None:
            _logger.warn(f"[vramd-learn] callback falhou (ignorado): {exc}")
        return True
