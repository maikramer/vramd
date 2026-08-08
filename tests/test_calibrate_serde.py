"""Testes do M3: amostras cruas no relatório e re-derivação sem GPU.

Motivo concreto: na primeira calibração dos 10 backends, três tiveram de voltar
à GPU só porque a análise foi corrigida depois de os dados existirem. Com as
amostras guardadas, `ums recalibrate` refaz os números em microssegundos.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from vramd import cli as cli_mod
from vramd.calibrate.analysis import PhaseWindows
from vramd.calibrate.emit import calibration_to_report
from vramd.calibrate.sampler import Sample
from vramd.calibrate.serde import (
    RAW_FORMAT_VERSION,
    SAMPLE_FIELDS,
    derive_from_report,
    row_to_sample,
    sample_to_row,
    windows_from_json,
    windows_to_json,
)

from .test_calibrate_analysis import build_windows, derive


def same_samples(a: list[Sample], b: list[Sample]) -> bool:
    """Iguais a menos do arredondamento de ``t``/``gap`` a 0.1 ms.

    A serialização arredonda os tempos a 4 casas: com 50 ms de intervalo isso é
    ruído puro, e poupa bytes numa série de milhares de amostras.
    """
    if len(a) != len(b):
        return False
    return all(
        x.self_mib == y.self_mib
        and x.foreign_mib == y.foreign_mib
        and x.self_pids == y.self_pids
        and x.tracked_pids == y.tracked_pids
        and abs(x.t - y.t) < 1e-3
        and abs(x.gap_sec - y.gap_sec) < 1e-3
        for x, y in zip(a, b, strict=True)
    )


class TestSampleRoundTrip:
    def test_round_trip_preserves_the_sample(self):
        sample = Sample(t=1.25, self_mib=4070, foreign_mib=120, self_pids=1, tracked_pids=1, gap_sec=0.051)
        assert row_to_sample(sample_to_row(sample)) == sample

    def test_time_is_rounded_to_a_tenth_of_a_millisecond(self):
        sample = Sample(t=0.15000000000000002, self_mib=1, foreign_mib=0, self_pids=1, tracked_pids=1, gap_sec=0.05)
        assert row_to_sample(sample_to_row(sample)).t == 0.15

    def test_row_is_compact_and_ordered(self):
        row = sample_to_row(Sample(t=1.0, self_mib=2, foreign_mib=3, self_pids=4, tracked_pids=5, gap_sec=0.6))
        assert row == [1.0, 2, 3, 4, 5, 0.6]
        assert len(row) == len(SAMPLE_FIELDS)

    def test_truncated_row_is_rejected(self):
        with pytest.raises(ValueError, match="incompleta"):
            row_to_sample([1.0, 2, 3])


class TestWindowsRoundTrip:
    def test_all_phases_survive(self):
        original = build_windows(repeats=3)
        restored = windows_from_json(windows_to_json(original))
        assert same_samples(restored.baseline, original.baseline)
        assert same_samples(restored.load, original.load)
        assert same_samples(restored.loaded_settled, original.loaded_settled)
        assert len(restored.generates) == len(original.generates)
        assert all(same_samples(r, o) for r, o in zip(restored.generates, original.generates, strict=True))
        assert all(same_samples(r, o) for r, o in zip(restored.settled, original.settled, strict=True))
        assert same_samples(restored.unloaded_settled, original.unloaded_settled)
        assert same_samples(restored.post_shutdown, original.post_shutdown)

    def test_empty_windows_survive(self):
        restored = windows_from_json(windows_to_json(PhaseWindows()))
        assert restored.baseline == []
        assert restored.generates == []

    def test_envelope_carries_version_and_fields(self):
        payload = windows_to_json(build_windows())
        assert payload["format"] == RAW_FORMAT_VERSION
        assert payload["fields"] == list(SAMPLE_FIELDS)

    def test_future_format_is_refused(self):
        payload = windows_to_json(build_windows())
        payload["format"] = RAW_FORMAT_VERSION + 1
        with pytest.raises(ValueError, match="desconhecido"):
            windows_from_json(payload)

    def test_missing_keys_default_to_empty(self):
        restored = windows_from_json({"format": RAW_FORMAT_VERSION})
        assert restored.load == []


class TestReportWithRawSamples:
    def test_raw_samples_are_omitted_by_default(self):
        cal = derive(build_windows())
        assert "raw_samples" not in calibration_to_report(cal)

    def test_raw_samples_are_included_when_windows_are_given(self):
        windows = build_windows()
        report = calibration_to_report(derive(windows), windows=windows)
        assert report["raw_samples"]["format"] == RAW_FORMAT_VERSION

    def test_report_with_raw_samples_is_json_serializable(self):
        windows = build_windows()
        payload = json.dumps(calibration_to_report(derive(windows), windows=windows))
        assert "raw_samples" in payload


class TestDeriveFromReport:
    def test_rederivation_reproduces_the_original_numbers(self):
        windows = build_windows(context=320, weights=1200, activation=900)
        original = derive(windows)
        report = calibration_to_report(original, windows=windows)
        again = derive_from_report(report)
        assert (again.context_mib, again.weights_mib, again.activation_mib) == (
            original.context_mib,
            original.weights_mib,
            original.activation_mib,
        )
        assert again.peak_mib == original.peak_mib

    def test_metadata_survives_the_round_trip(self):
        windows = build_windows()
        original = derive(windows, gpu_name="RTX 4050", gpu_total_mib=6141, driver_version="595.84")
        again = derive_from_report(calibration_to_report(original, windows=windows))
        assert again.gpu_name == "RTX 4050"
        assert again.gpu_total_mib == 6141
        assert again.backend == original.backend
        assert again.quant_mode == original.quant_mode

    def test_timings_survive(self):
        windows = build_windows()
        original = derive(windows, generate_sec=[8.0, 7.0, 9.0], load_sec=12.0)
        again = derive_from_report(calibration_to_report(original, windows=windows))
        assert again.load_sec == 12.0
        assert again.generate_sec == original.generate_sec

    def test_report_without_raw_samples_is_refused(self):
        report = calibration_to_report(derive(build_windows()))
        with pytest.raises(ValueError, match="sem 'raw_samples'"):
            derive_from_report(report)


class TestRecalibrateCommand:
    def _report_file(self, tmp_path, **kwargs):
        windows = build_windows(**kwargs)
        report = calibration_to_report(derive(windows), windows=windows)
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_recalibrates_and_prints_the_decomposition(self, tmp_path):
        path = self._report_file(tmp_path)
        result = CliRunner().invoke(cli_mod.cli, ["recalibrate", str(path), "--no-compare"])
        assert result.exit_code == 0
        assert "pesos" in result.output

    def test_report_without_raw_samples_exits_two(self, tmp_path):
        path = tmp_path / "sem-raw.json"
        path.write_text(json.dumps(calibration_to_report(derive(build_windows()))), encoding="utf-8")
        result = CliRunner().invoke(cli_mod.cli, ["recalibrate", str(path)])
        assert result.exit_code == 2
        assert "raw_samples" in result.output

    def test_writes_yaml(self, tmp_path):
        path = self._report_file(tmp_path)
        out = tmp_path / "cal.yaml"
        result = CliRunner().invoke(cli_mod.cli, ["recalibrate", str(path), "--no-compare", "--out", str(out)])
        assert result.exit_code == 0
        assert "backends:" in out.read_text(encoding="utf-8")

    def test_new_report_stays_recalibrable(self, tmp_path):
        """Recalibrar uma vez não pode tornar o ficheiro irrecalibrável."""
        path = self._report_file(tmp_path)
        out = tmp_path / "novo.json"
        CliRunner().invoke(cli_mod.cli, ["recalibrate", str(path), "--no-compare", "--report", str(out)])
        again = json.loads(out.read_text(encoding="utf-8"))
        assert "raw_samples" in again
        assert derive_from_report(again).peak_mib > 0

    def test_json_flag_dumps_the_report(self, tmp_path):
        path = self._report_file(tmp_path)
        result = CliRunner().invoke(cli_mod.cli, ["recalibrate", str(path), "--no-compare", "--json"])
        assert result.exit_code == 0
        assert "weights" in result.output

    def test_missing_file_is_rejected_by_click(self):
        result = CliRunner().invoke(cli_mod.cli, ["recalibrate", "/nao/existe.json"])
        assert result.exit_code != 0
