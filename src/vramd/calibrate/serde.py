"""Serialização das amostras cruas — re-derivar sem voltar a ocupar a GPU.

Medir custa minutos de GPU exclusiva; derivar custa microssegundos. Guardar só
os números derivados significa que **qualquer correção à análise obriga a
re-medir** — foi o que aconteceu na primeira calibração dos 10 backends: três
tiveram de voltar à GPU porque a análise mudou depois de os dados existirem.

Com as amostras no relatório, `vramd recalibrate` refaz a derivação com o código
atual. O formato é lista-de-listas em vez de objetos: uma corrida de 5 min a
20 Hz são ~6000 amostras, e as chaves repetidas seis vezes por amostra
triplicariam o ficheiro sem acrescentar informação.
"""

from __future__ import annotations

from typing import Any

from .analysis import PhaseWindows
from .sampler import Sample

# Ordem dos campos na forma compacta. Mudá-la parte relatórios antigos — daí a
# versão explícita no envelope.
SAMPLE_FIELDS = ("t", "self_mib", "foreign_mib", "self_pids", "tracked_pids", "gap_sec")
RAW_FORMAT_VERSION = 1


def sample_to_row(sample: Sample) -> list[float]:
    """Amostra → linha compacta (a ordem é :data:`SAMPLE_FIELDS`)."""
    return [
        round(sample.t, 4),
        sample.self_mib,
        sample.foreign_mib,
        sample.self_pids,
        sample.tracked_pids,
        round(sample.gap_sec, 4),
    ]


def row_to_sample(row: list[Any]) -> Sample:
    """Linha compacta → amostra.

    Raises:
        ValueError: Linha com menos campos que :data:`SAMPLE_FIELDS`.
    """
    if len(row) < len(SAMPLE_FIELDS):
        raise ValueError(f"linha de amostra incompleta: esperados {len(SAMPLE_FIELDS)} campos, {len(row)} dados")
    return Sample(
        t=float(row[0]),
        self_mib=int(row[1]),
        foreign_mib=int(row[2]),
        self_pids=int(row[3]),
        tracked_pids=int(row[4]),
        gap_sec=float(row[5]),
    )


def windows_to_json(windows: PhaseWindows) -> dict[str, Any]:
    """Serializa as janelas por fase."""
    return {
        "format": RAW_FORMAT_VERSION,
        "fields": list(SAMPLE_FIELDS),
        "baseline": [sample_to_row(s) for s in windows.baseline],
        "load": [sample_to_row(s) for s in windows.load],
        "loaded_settled": [sample_to_row(s) for s in windows.loaded_settled],
        "generates": [[sample_to_row(s) for s in w] for w in windows.generates],
        "settled": [[sample_to_row(s) for s in w] for w in windows.settled],
        "unloaded_settled": [sample_to_row(s) for s in windows.unloaded_settled],
        "post_shutdown": [sample_to_row(s) for s in windows.post_shutdown],
    }


def windows_from_json(data: dict[str, Any]) -> PhaseWindows:
    """Reconstrói as janelas a partir do relatório.

    Raises:
        ValueError: Formato desconhecido (relatório de uma versão futura).
    """
    version = int(data.get("format", RAW_FORMAT_VERSION))
    if version > RAW_FORMAT_VERSION:
        raise ValueError(f"formato de amostras {version} desconhecido (esta versão lê até {RAW_FORMAT_VERSION})")

    def rows(key: str) -> list[Sample]:
        return [row_to_sample(r) for r in data.get(key) or []]

    def groups(key: str) -> list[list[Sample]]:
        return [[row_to_sample(r) for r in window] for window in data.get(key) or []]

    return PhaseWindows(
        baseline=rows("baseline"),
        load=rows("load"),
        loaded_settled=rows("loaded_settled"),
        generates=groups("generates"),
        settled=groups("settled"),
        unloaded_settled=rows("unloaded_settled"),
        post_shutdown=rows("post_shutdown"),
    )


def derive_from_report(report: dict[str, Any]) -> Any:
    """Re-deriva uma :class:`~vramd.calibrate.analysis.Calibration` do relatório.

    Args:
        report: Conteúdo de um ``--report`` que inclua ``raw_samples``.

    Returns:
        Calibração recalculada com a análise **atual**.

    Raises:
        ValueError: Relatório sem amostras cruas (medido antes do M3, ou com
            ``--no-raw``) — nesse caso só resta re-medir.
    """
    from .analysis import derive_calibration

    raw = report.get("raw_samples")
    if not raw:
        raise ValueError("relatório sem 'raw_samples': foi medido com --no-raw ou antes desta versão")

    windows = windows_from_json(raw)
    timing = report.get("timing") or {}
    quality = report.get("quality") or {}
    hardware = report.get("hardware") or {}
    return derive_calibration(
        backend=str(report.get("backend") or "desconhecido"),
        tool=report.get("tool"),
        load_kwargs=report.get("load_kwargs") or {},
        quant_mode=str(report.get("quant_mode") or "none"),
        windows=windows,
        load_sec=float(timing.get("load_sec") or 0.0),
        generate_sec=list(timing.get("generate_sec") or []),
        interval_sec=float(quality.get("interval_sec") or 0.05),
        probe_errors=int(quality.get("probe_errors") or 0),
        gpu_name=hardware.get("gpu"),
        gpu_total_mib=hardware.get("gpu_total_mib"),
        driver_version=hardware.get("driver"),
        measured_at=hardware.get("measured_at"),
    )
