"""Calibração de footprint VRAM — medir em vez de adivinhar.

O vramd admite jobs comparando VRAM livre com ``pesos + activação + safety``
(:mod:`vramd.vram_planner`). Esses números vivem hoje em
``data/backends.yaml`` (``vram_mib``) e em ``vramd.footprints.FOOTPRINTS``
— escritos à mão. Este package fecha o ciclo: corre um job real, mede o pico
com o driver, deriva os componentes e **emite o YAML**.

Camadas (todas testáveis sem GPU):

- :mod:`~vramd.calibrate.sampler` — amostragem de alta frequência da VRAM
  por processo (worker + descendentes) com deteção de gaps e de contaminação.
- :mod:`~vramd.calibrate.analysis` — derivação **pura** de
  contexto/pesos/activação/pico/leak + nível de confiança.
- :mod:`~vramd.calibrate.emit` — descriptor YAML v2 + relatório JSON.
- :mod:`~vramd.calibrate.compare` — medido vs declarado (drift).
- :mod:`~vramd.calibrate.runner` — orquestra o ciclo
  spawn → load → generatexN → unload → shutdown.
"""

from __future__ import annotations

from .analysis import (
    Calibration,
    PhaseStats,
    derive_calibration,
    recommend_safety_mib,
    round_up_mib,
    summarize_window,
)
from .compare import ComparisonRow, compare_to_declared, verdict_for
from .emit import calibration_to_report, calibration_to_yaml
from .runner import CalibrationRunner, CalibrationSpec, RunnerError
from .sampler import Mark, Sample, VramSampler, descendant_pids
from .serde import derive_from_report, windows_from_json, windows_to_json

__all__ = [
    "Calibration",
    "CalibrationRunner",
    "CalibrationSpec",
    "ComparisonRow",
    "Mark",
    "PhaseStats",
    "RunnerError",
    "Sample",
    "VramSampler",
    "calibration_to_report",
    "calibration_to_yaml",
    "compare_to_declared",
    "derive_calibration",
    "derive_from_report",
    "descendant_pids",
    "recommend_safety_mib",
    "round_up_mib",
    "summarize_window",
    "verdict_for",
    "windows_from_json",
    "windows_to_json",
]
