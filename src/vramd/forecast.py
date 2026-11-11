"""Prewarm preditivo: carregar o modelo antes de o job chegar.

Num pipeline o cliente sabe o que vem a seguir: quando a wave de shapes
arranca, vão chegar N paints daqui a ~T segundos (o tempo do primeiro shape).
Sem aviso, o primeiro paint paga o load inteiro (30-60 s) no caminho crítico.

Com ``{"cmd": "forecast", "backend": "paint3d", "count": N, "eta_sec": T,
"request": {...}}`` o supervisor guarda a previsão e, quando a chegada fica a
menos de um load de distância, pré-carrega o backend **na GPU que o router
escolheria para esse job** — só se essa GPU estiver livre, para nunca atrasar
trabalho real. A rota e o tier são os do próprio pedido de exemplo.

Heurísticas (todas conservadoras):

* Não pré-carregar cedo demais: só quando ``tempo_até_chegar ≤ load x 1.25 +
  5 s`` — VRAM ocupada por um modelo à espera é VRAM que outro job podia usar.
* Um prewarm por previsão; réplicas extra nascem quando os jobs chegam e o
  router vê a fila.
* Se o backend já está quente em alguma GPU, a previsão está satisfeita.
* A GPU é reservada durante o load como se fosse um job: o dispatcher não a
  dá a outro a meio.
"""

from __future__ import annotations

import contextlib
import itertools
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from vramd.logging import Logger

from .router import ROUTE_LOCAL

_logger = Logger()
_EARLY_FACTOR = 1.25
_EARLY_SLACK_SEC = 5.0
_ids = itertools.count(1)


@dataclass
class Forecast:
    """Uma previsão de procura de um backend."""

    forecast_id: str
    backend: str
    count: int
    arrive_at: float
    expires_at: float
    request: dict[str, Any] = field(default_factory=dict)
    state: str = "pending"  # pending | warming | warm | skipped | expired
    lane: str | None = None
    note: str | None = None

    def to_dict(self, now: float) -> dict[str, Any]:
        return {
            "id": self.forecast_id,
            "backend": self.backend,
            "count": self.count,
            "arrives_in_sec": round(self.arrive_at - now, 1),
            "state": self.state,
            "lane": self.lane,
            "note": self.note,
        }


@dataclass
class _PrewarmTask:
    """Ocupa a lane durante o load (o LaneSet trata-o como um job)."""

    backend: str
    job_id: str
    progress_pct: float | None = None


class Prewarmer:
    """Guarda previsões e pré-carrega modelos quando compensa.

    Args:
        engine: :class:`~vramd.placement.RouteEngine` (decide a lane).
        preload: ``(lane, backend, request) -> None`` — carrega na lane.
        clock: Relógio monotónico.
        interval_sec: Período do loop.
    """

    def __init__(
        self,
        engine: Any,
        preload: Callable[[Any, str, Mapping[str, Any]], None],
        *,
        clock: Callable[[], float] = time.monotonic,
        interval_sec: float = 2.0,
    ) -> None:
        self.engine = engine
        self._preload = preload
        self._clock = clock
        self._interval = max(0.2, float(interval_sec))
        self._lock = threading.Lock()
        self._forecasts: dict[str, Forecast] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.prewarmed = 0

    def add(
        self,
        backend: str,
        *,
        count: int = 1,
        eta_sec: float = 0.0,
        request: Mapping[str, Any] | None = None,
        ttl_sec: float = 900.0,
    ) -> Forecast:
        now = self._clock()
        arrive = now + max(0.0, float(eta_sec))
        forecast = Forecast(
            forecast_id=f"fc{next(_ids)}",
            backend=backend,
            count=max(1, int(count)),
            arrive_at=arrive,
            expires_at=arrive + max(30.0, float(ttl_sec)),
            request={k: v for k, v in (request or {}).items() if not str(k).startswith("_")},
        )
        with self._lock:
            # Uma previsão nova para o mesmo backend substitui a anterior pendente.
            for old in self._forecasts.values():
                if old.backend == backend and old.state == "pending":
                    old.state = "skipped"
                    old.note = f"substituída por {forecast.forecast_id}"
            self._forecasts[forecast.forecast_id] = forecast
        return forecast

    def status(self) -> list[dict[str, Any]]:
        now = self._clock()
        with self._lock:
            return [f.to_dict(now) for f in self._forecasts.values() if f.state in ("pending", "warming")]

    # -- loop ----------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="vramd-prewarm", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            with contextlib.suppress(Exception):
                self.tick()

    def tick(self) -> list[str]:
        """Uma passagem: devolve os ids das previsões pré-aquecidas agora."""
        now = self._clock()
        with self._lock:
            for forecast in self._forecasts.values():
                if forecast.state == "pending" and now > forecast.expires_at:
                    forecast.state = "expired"
            # Esquecer as que já não interessam (o status só mostra as vivas).
            for key in [k for k, f in self._forecasts.items() if f.state not in ("pending", "warming")]:
                if now > self._forecasts[key].expires_at:
                    del self._forecasts[key]
            pending = [f for f in self._forecasts.values() if f.state == "pending"]
        done: list[str] = []
        for forecast in sorted(pending, key=lambda f: f.arrive_at):
            if self._try(forecast, now):
                done.append(forecast.forecast_id)
        return done

    def _try(self, forecast: Forecast, now: float) -> bool:
        lanes = self.engine.lanes
        if lanes.lanes_with(forecast.backend):
            forecast.state = "warm"
            forecast.note = "já quente"
            return False
        decision = self.engine.decide(forecast.backend, forecast.request)
        best = decision.best
        if best is None or best.kind != ROUTE_LOCAL or best.hot:
            forecast.note = "melhor rota não é uma GPU local fria" if best is not None else "sem rota"
            return False
        if forecast.arrive_at - now > best.load_sec * _EARLY_FACTOR + _EARLY_SLACK_SEC:
            forecast.note = f"cedo demais (load ~{best.load_sec:.0f}s)"
            return False
        lane = lanes.lane(best.target)
        if lane is None or lanes.is_busy(lane):
            forecast.note = "GPU de destino ocupada"
            return False
        task = _PrewarmTask(backend=forecast.backend, job_id=f"prewarm:{forecast.forecast_id}")
        if not lanes.reserve(lane, task, est_sec=best.load_sec):
            return False
        forecast.state = "warming"
        forecast.lane = lane.lane_id
        try:
            lanes.make_room(lane)
            self._preload(lane, forecast.backend, forecast.request)
            forecast.state = "warm"
            forecast.note = None
            self.prewarmed += 1
            _logger.info(f"[vramd-prewarm] {forecast.backend} pré-carregado em {lane.lane_id} ({forecast.forecast_id})")
            return True
        except Exception as exc:
            forecast.state = "skipped"
            forecast.note = f"prewarm falhou: {exc}"
            _logger.warn(f"[vramd-prewarm] {forecast.backend} em {lane.lane_id}: {exc}")
            return False
        finally:
            lanes.release(lane)
