"""Testes da derivação pura (``vramd.calibrate.analysis``).

O contrato central: dada uma série sintética com valores conhecidos, a
decomposição tem de devolver exatamente esses valores. Os restantes testes
cobrem cada sinal de saúde e cada regra de confiança.
"""

from __future__ import annotations

import dataclasses

import pytest

from vramd.calibrate.analysis import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    MIN_SAFETY_MIB,
    Calibration,
    PhaseWindows,
    derive_calibration,
    median,
    percentile,
    recommend_safety_mib,
    round_up_mib,
    summarize_window,
)
from vramd.calibrate.sampler import Sample

INTERVAL = 0.05


def series(values, *, foreign=100, t0=0.0, gap=INTERVAL, tracked=1, self_pids=1):
    """Constrói uma janela de amostras com os MiB dados."""
    out = []
    for i, value in enumerate(values):
        out.append(
            Sample(
                t=t0 + i * gap,
                self_mib=value,
                foreign_mib=foreign,
                self_pids=self_pids if value or self_pids == 0 else 0,
                tracked_pids=tracked,
                gap_sec=gap,
            )
        )
    return out


def build_windows(
    *,
    context=320,
    weights=1200,
    activation=900,
    load_extra=400,
    repeats=3,
    fragmentation=0,
    leak=0,
    foreign=100,
    warmup_extra=0,
):
    """Janelas sintéticas com decomposição conhecida.

    As fases avançam no tempo (``t0`` crescente) como numa corrida real — a
    derivação usa a ordem cronológica para saber quando o worker apareceu pela
    primeira vez no driver.
    """
    resident = context + weights
    clock = {"t": 0.0}

    def phase(values, **kwargs):
        window = series(values, foreign=foreign, t0=clock["t"], **kwargs)
        clock["t"] = window[-1].t + INTERVAL if window else clock["t"]
        return window

    windows = PhaseWindows(
        baseline=phase([0] * 10, tracked=0, self_pids=0),
        # Transiente de carregamento: sobe, pica, assenta no residente.
        load=phase([0, resident // 2, resident + load_extra, resident]),
        loaded_settled=phase([resident] * 20),
    )
    for i in range(repeats):
        extra = warmup_extra if i == 0 else 0
        peak = resident + activation + extra + (leak * i)
        windows.generates.append(phase([resident, peak, resident + activation // 2]))
        windows.settled.append(phase([resident + fragmentation + leak * i] * 10))
    windows.unloaded_settled = phase([context] * 10)
    windows.post_shutdown = phase([0] * 5, tracked=0, self_pids=0)
    return windows


def derive(windows, **kwargs):
    """Atalho com os argumentos fixos dos testes."""
    params = {
        "backend": "fake3d",
        "tool": "fake3d",
        "load_kwargs": {"sdnq_preset": "int4"},
        "quant_mode": "sdnq-int4",
        "windows": windows,
        "load_sec": 12.0,
        "generate_sec": [8.0, 7.5, 7.6],
        "interval_sec": INTERVAL,
    }
    params.update(kwargs)
    return derive_calibration(**params)


class TestStatHelpers:
    def test_percentile_empty(self):
        assert percentile([], 95) == 0.0

    def test_percentile_bounds(self):
        values = [1, 2, 3, 4, 5]
        assert percentile(values, 0) == 1.0
        assert percentile(values, 100) == 5.0

    def test_percentile_nearest_rank(self):
        assert percentile([10, 20, 30, 40], 50) == 20.0
        assert percentile([10, 20, 30, 40], 95) == 40.0

    def test_median_even_and_odd(self):
        assert median([1, 3]) == 2.0
        assert median([1, 2, 3]) == 2.0
        assert median([]) == 0.0

    def test_round_up_mib_rounds_up(self):
        assert round_up_mib(1) == 64
        assert round_up_mib(64) == 64
        assert round_up_mib(65) == 128

    def test_round_up_mib_float_never_below_input(self):
        """Regressão: -(-round(v)//G) usava round() (banker's) e devolvia um
        múltiplo ABAIXO do input (round_up_mib(64.4) → 64) — o oposto do
        contrato "arredonda sempre para cima"."""
        assert round_up_mib(64.4) == 128
        assert round_up_mib(64.5) == 128
        assert round_up_mib(127.9) == 128
        assert round_up_mib(128.0) == 128
        assert round_up_mib(64.4, 500) == 500

    def test_round_up_mib_zero_and_negative(self):
        assert round_up_mib(0) == 0
        assert round_up_mib(-10) == 0

    def test_round_up_mib_custom_granularity(self):
        assert round_up_mib(1001, 500) == 1500

    def test_round_up_mib_granularity_zero_is_ceil(self):
        assert round_up_mib(10.2, 0) == 11
        assert round_up_mib(10.0, 0) == 10


class TestSummarizeWindow:
    def test_empty_window_is_zeroed(self):
        stats = summarize_window([], "vazia")
        assert stats.n == 0
        assert stats.max_mib == 0
        assert stats.duration_sec == 0.0

    def test_basic_statistics(self):
        stats = summarize_window(series([100, 500, 300]), "x")
        assert stats.n == 3
        assert stats.min_mib == 100
        assert stats.max_mib == 500
        assert stats.p50_mib == 300
        assert stats.foreign_max_mib == 100

    def test_counts_missed_samples(self):
        window = series([0, 0], self_pids=0)
        assert summarize_window(window, "x").missed == 2

    def test_as_dict_is_serializable(self):
        payload = summarize_window(series([1, 2]), "x").as_dict()
        assert payload["label"] == "x"
        assert set(payload) >= {"n", "min_mib", "max_mib", "p50_mib", "p95_mib"}


class TestDecomposition:
    """Recuperação da verdade conhecida — o teste que define o módulo."""

    def test_recovers_exact_ground_truth(self):
        cal = derive(build_windows(context=320, weights=1200, activation=900))
        assert cal.context_mib == 320
        assert cal.resident_loaded_mib == 1520
        assert cal.weights_mib == 1200
        assert cal.activation_mib == 900
        assert cal.generate_peak_mib == 2420
        assert cal.peak_mib == 2420

    def test_peak_is_load_bound_when_load_transient_dominates(self):
        cal = derive(build_windows(context=300, weights=1000, activation=200, load_extra=1500))
        assert cal.load_peak_mib == 2800
        assert cal.generate_peak_mib == 1500
        assert cal.peak_mib == 2800
        assert any("pico no load" in w for w in cal.warnings)

    def test_staged_load_suspected_when_activation_dwarfs_weights(self):
        cal = derive(build_windows(context=300, weights=400, activation=3000))
        assert cal.staged_load_suspected is True
        assert any("faseado" in w for w in cal.warnings)

    def test_retained_memory_without_residency_is_not_called_context(self):
        """Regressão (terrain3d real): 4996 MiB de lixo da inferência viravam 'contexto'."""
        windows = build_windows(context=0, weights=0, activation=5578, load_extra=0)
        windows.unloaded_settled = series([4996] * 10)
        cal = derive(windows)
        assert cal.context_mib == 0
        assert cal.context_gib == 0.0
        assert any("memória da inferência, não contexto CUDA" in w for w in cal.warnings)

    def test_lazy_load_inside_generate_is_flagged(self):
        """Regressão (terrain3d real): ``load`` só cria config, modelo entra no generate."""
        windows = build_windows(context=0, weights=0, activation=4660, load_extra=0)
        cal = derive(windows)
        assert cal.weights_mib == 0
        assert cal.activation_mib == 4660
        assert cal.staged_load_suspected is True
        assert any("carregado dentro da inferência" in w for w in cal.warnings)

    def test_normal_model_is_not_flagged_as_staged(self):
        cal = derive(build_windows(context=300, weights=4000, activation=1000))
        assert cal.staged_load_suspected is False

    def test_admit_peak_includes_recommended_safety(self):
        cal = derive(build_windows())
        assert cal.admit_peak_mib == cal.peak_mib + cal.recommended_safety_mib

    def test_gib_properties_round_up(self):
        cal = derive(build_windows(context=0, weights=1025, activation=0))
        assert cal.weights_gib == pytest.approx(1.01)

    def test_gib_of_zero_is_zero(self):
        cal = derive(build_windows(context=0, weights=0, activation=0, load_extra=0))
        assert cal.weights_gib == 0.0
        assert cal.context_gib == 0.0


class TestHealthSignals:
    def test_fragmentation_measured_from_first_settled_window(self):
        cal = derive(build_windows(fragmentation=128))
        assert cal.fragmentation_mib == 128

    def test_leak_slope_detected_and_warned(self):
        cal = derive(build_windows(leak=100))
        assert cal.leak_mib_per_run == pytest.approx(100.0, abs=1.0)
        assert any("fuga" in w for w in cal.warnings)

    def test_no_leak_gives_zero_slope(self):
        cal = derive(build_windows(leak=0))
        assert cal.leak_mib_per_run == 0.0

    def test_warmup_delta_isolated_from_steady_state(self):
        cal = derive(build_windows(warmup_extra=250))
        assert cal.warmup_delta_mib == 250
        assert any("warmup" in w for w in cal.warnings)

    def test_orphan_vram_after_shutdown_is_warned(self):
        windows = build_windows()
        windows.post_shutdown = series([180] * 5)
        cal = derive(windows)
        assert cal.orphan_mib == 180
        assert any("órfão" in w for w in cal.warnings)

    def test_ineffective_unload_does_not_fake_a_staged_load(self):
        """Regressão (text2icon real): unload que não liberta dava pesos=82 e 'staged'."""
        windows = build_windows(context=320, weights=4444, activation=862)
        # Unload devolve quase nada: residual ≈ residente com modelo.
        windows.unloaded_settled = series([4682] * 10)
        cal = derive(windows)
        assert cal.unload_ineffective is True
        assert cal.staged_load_suspected is False
        assert cal.weights_mib == cal.resident_loaded_mib  # atribuição conservadora
        assert cal.context_mib == 0
        assert any("NÃO liberta VRAM" in w for w in cal.warnings)

    def test_ineffective_unload_keeps_the_peak_intact(self):
        windows = build_windows(context=320, weights=4444, activation=862)
        clean = derive(windows).peak_mib
        windows.unloaded_settled = series([4682] * 10)
        assert derive(windows).peak_mib == clean

    def test_ineffective_unload_lowers_confidence(self):
        windows = build_windows(context=320, weights=4444, activation=862)
        windows.unloaded_settled = series([4682] * 10)
        assert derive(windows).confidence == CONFIDENCE_MEDIUM

    def test_effective_unload_is_not_flagged(self):
        cal = derive(build_windows(context=320, weights=4444))
        assert cal.unload_ineffective is False

    def test_residual_above_loaded_is_treated_as_ineffective_unload(self):
        """Residual > residente é o caso extremo de unload que não liberta."""
        windows = build_windows(context=300, weights=200, activation=100)
        windows.unloaded_settled = series([5000] * 5)
        cal = derive(windows)
        assert any("unload incompleto" in w for w in cal.warnings)
        assert cal.unload_ineffective is True
        assert cal.context_mib == 0
        assert cal.weights_mib == cal.resident_loaded_mib


class TestMeasurementQuality:
    def test_clean_run_is_high_confidence(self):
        cal = derive(build_windows())
        assert cal.confidence == CONFIDENCE_HIGH
        assert cal.contaminated is False

    def test_foreign_vram_growth_contaminates(self):
        windows = build_windows(foreign=100)
        windows.generates[1] = series([2000, 2400, 2000], foreign=900)
        cal = derive(windows)
        assert cal.contaminated is True
        assert cal.confidence == CONFIDENCE_LOW
        assert any("terceiros" in w for w in cal.warnings)

    def test_sampling_gap_lowers_confidence(self):
        windows = build_windows()
        windows.generates[0] = series([1520, 2420], gap=1.0)
        cal = derive(windows)
        assert cal.confidence == CONFIDENCE_LOW
        assert any("intervalo entre amostras" in w for w in cal.warnings)

    def test_driver_blindness_lowers_confidence(self):
        windows = build_windows()
        windows.loaded_settled = series([0] * 20, self_pids=0)
        cal = derive(windows)
        assert cal.missed_ratio > 0.05
        assert cal.confidence == CONFIDENCE_LOW

    def test_pre_cuda_samples_do_not_count_as_blindness(self):
        """Regressão (texture2d real): 5 s de `import torch` marcavam 19% cegos."""
        windows = build_windows()
        # Worker já existe (tracked) mas ainda não alocou CUDA: driver não o vê.
        # Entra *antes* da janela de load real, como no arranque verdadeiro.
        blind_start = windows.load[0].t - 100 * INTERVAL
        windows.load = series([0] * 100, self_pids=0, t0=blind_start) + windows.load
        cal = derive(windows)
        assert cal.missed_ratio == 0.0
        assert cal.confidence == CONFIDENCE_HIGH

    def test_blindness_after_first_sighting_still_counts(self):
        windows = build_windows()
        windows.generates[0] = series([0] * 40, self_pids=0)
        cal = derive(windows)
        assert cal.missed_ratio > 0.05
        assert cal.confidence == CONFIDENCE_LOW

    def test_single_repeat_is_medium_confidence(self):
        cal = derive(build_windows(repeats=1), generate_sec=[8.0])
        assert cal.confidence == CONFIDENCE_MEDIUM
        assert any("1 repetição" in w for w in cal.warnings)

    def test_missing_unload_window_is_medium_and_warns(self):
        windows = build_windows()
        windows.unloaded_settled = []
        cal = derive(windows)
        assert cal.context_mib == 0
        assert cal.confidence == CONFIDENCE_MEDIUM
        assert any("contexto CUDA não isolado" in w for w in cal.warnings)

    def test_unstable_peaks_lower_confidence(self):
        windows = build_windows(repeats=3)
        windows.generates[1] = series([1520, 5000, 1520])
        windows.generates[2] = series([1520, 2420, 1520])
        cal = derive(windows)
        assert cal.confidence == CONFIDENCE_MEDIUM
        assert any("variam" in w for w in cal.warnings)

    def test_probe_errors_are_reported(self):
        cal = derive(build_windows(), probe_errors=4)
        assert cal.probe_errors == 4
        assert any("probe NVML" in w for w in cal.warnings)

    def test_samples_counted_across_all_phases(self):
        cal = derive(build_windows(repeats=2), generate_sec=[8.0, 8.1])
        assert cal.samples_n > 40

    def test_hardware_metadata_is_carried(self):
        cal = derive(build_windows(), gpu_name="RTX 4050", gpu_total_mib=6141, driver_version="580.0")
        assert cal.gpu_name == "RTX 4050"
        assert cal.gpu_total_mib == 6141
        assert cal.driver_version == "580.0"

    def test_timings_are_recorded(self):
        cal = derive(build_windows(), generate_sec=[8.0, 7.0, 9.0])
        assert cal.load_sec == 12.0
        assert cal.generate_sec_median == 8.0
        assert cal.repeats == 3


class TestSafetyRecommendation:
    def test_floor_applies_for_stable_peaks(self):
        assert recommend_safety_mib([2000, 2000, 2000]) == MIN_SAFETY_MIB

    def test_spread_raises_recommendation(self):
        assert recommend_safety_mib([2000, 2900]) == round_up_mib(900)

    def test_empty_peaks_use_floor(self):
        assert recommend_safety_mib([]) == MIN_SAFETY_MIB

    def test_custom_floor(self):
        assert recommend_safety_mib([2000, 2000], floor_mib=1000) == round_up_mib(1000)

    def test_allocator_cache_does_not_inflate_safety(self):
        """Regressão (texture2d real): cache retida ≈ activação duplicava a margem."""
        cal = derive(build_windows(activation=1164, fragmentation=1164))
        assert cal.fragmentation_mib == 1164  # continua reportado como diagnóstico
        assert cal.recommended_safety_mib == MIN_SAFETY_MIB

    def test_warmup_peak_does_not_inflate_safety(self):
        """O excesso do warmup já está dentro do ``peak`` — não conta na margem."""
        cal = derive(build_windows(warmup_extra=800))
        assert cal.warmup_delta_mib == 800
        assert cal.recommended_safety_mib == MIN_SAFETY_MIB


class TestCalibrationShape:
    def test_is_frozen_dataclass(self):
        cal = derive(build_windows())
        assert isinstance(cal, Calibration)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cal.peak_mib = 1  # type: ignore[misc]

    def test_phases_include_every_window(self):
        cal = derive(build_windows(repeats=2), generate_sec=[1.0, 1.0])
        assert {"baseline", "load", "loaded_settled", "unloaded_settled", "post_shutdown"} <= set(cal.phases)
        assert "generate_1" in cal.phases
        assert "settled_2" in cal.phases

    def test_load_kwargs_are_copied_not_shared(self):
        kwargs = {"sdnq_preset": "int4"}
        cal = derive(build_windows(), load_kwargs=kwargs)
        kwargs["mutated"] = True
        assert "mutated" not in cal.load_kwargs
