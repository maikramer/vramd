"""Lint da configuração (``vramd validate``)."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from vramd.validate import LEVEL_ERROR, LEVEL_WARNING, ValidationReport, validate_config


def _write(path: Path, backends: list[dict], **top) -> str:
    path.write_text(yaml.safe_dump({"version": 2, "backends": backends, **top}), encoding="utf-8")
    return str(path)


def _good(**over) -> dict:
    base = {
        "name": "alpha",
        "adapter": "my_pkg.adapter",
        "vram_mib": 2000,
        "priority": 10,
        "runtime": {"command": [sys.executable, "-m", "my_pkg.worker"]},
    }
    base.update(over)
    return base


def _run(tmp_path: Path, backends: list[dict], **kw) -> ValidationReport:
    return validate_config(_write(tmp_path / "b.yaml", backends), check_hooks=False, **kw)


def _messages(report: ValidationReport, level: str) -> list[str]:
    return [f.message for f in report.findings if f.level == level]


class TestValidateDescriptors:
    def test_clean_config_passes(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [_good()])
        assert report.ok
        assert report.findings == []
        assert report.backends == ["alpha"]

    def test_typo_key_warns_with_suggestion(self, tmp_path: Path) -> None:
        """``prority`` era ignorado em silêncio — o backend ficava com prioridade 0."""
        report = _run(tmp_path, [_good(prority=50)])
        assert report.ok  # aviso, não erro
        (finding,) = report.warnings
        assert "prority" in finding.message
        assert finding.hint is not None and "priority" in finding.hint
        assert finding.backend == "alpha"

    def test_unknown_runtime_key_warns(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [_good(runtime={"command": [sys.executable], "load_timeout": 30})])
        (finding,) = report.warnings
        assert "runtime.load_timeout" in finding.message
        assert "load_timeout_sec" in (finding.hint or "")

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [{"name": "alpha", "priority": 1, "runtime": {"command": [sys.executable]}}])
        assert not report.ok
        assert any("falta adapter, vram_mib" in m for m in _messages(report, LEVEL_ERROR))

    def test_bad_vram_and_priority_types(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [_good(vram_mib="muito", priority="alta")])
        errors = _messages(report, LEVEL_ERROR)
        assert any("vram_mib='muito'" in m for m in errors)
        assert any("priority='alta'" in m for m in errors)

    def test_fractional_mib_is_not_truncated(self, tmp_path: Path) -> None:
        """``vram_mib: 100.5`` era aceite e truncado para 100 pelo registry."""
        errors = _messages(_run(tmp_path, [_good(vram_mib=100.5)]), LEVEL_ERROR)
        assert any("vram_mib=100.5" in m and "não é inteiro" in m for m in errors)
        runtime = {"command": [sys.executable], "max_jobs": 1.5}
        errors = _messages(_run(tmp_path, [_good(runtime=runtime)]), LEVEL_ERROR)
        assert any("max_jobs" in m for m in errors)

    def test_empty_runtime_command_is_error(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [_good(runtime={"command": []})])
        assert any("runtime.command está vazio" in m for m in _messages(report, LEVEL_ERROR))

    def test_bad_peak_mib_type_is_error(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [_good(vram={"peak_mib": "muito"})])
        assert any("vram.peak_mib" in m and "não é inteiro" in m for m in _messages(report, LEVEL_ERROR))

    def test_command_executable_missing(self, tmp_path: Path) -> None:
        """Venv apagado: só rebentava no spawn do primeiro job."""
        report = _run(tmp_path, [_good(runtime={"command": [str(tmp_path / "venv/bin/python"), "-m", "x"]})])
        assert any("executável não existe" in m for m in _messages(report, LEVEL_ERROR))

    def test_command_not_on_path(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [_good(runtime={"command": ["definitely-not-a-real-binary-xyz"]})])
        assert any("não está no PATH" in m for m in _messages(report, LEVEL_ERROR))

    def test_unset_env_reference(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("VRAMD_TEST_UNSET_PY", raising=False)
        report = _run(tmp_path, [_good(runtime={"command": ["${env:VRAMD_TEST_UNSET_PY}", "w.py"]})])
        (msg,) = _messages(report, LEVEL_ERROR)
        assert "${env:VRAMD_TEST_UNSET_PY}" in msg

    def test_env_reference_resolved(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("VRAMD_TEST_PY", sys.executable)
        report = _run(tmp_path, [_good(runtime={"command": ["${env:VRAMD_TEST_PY}", "w.py"]})])
        assert report.ok and not report.findings

    def test_command_as_string_is_error(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [_good(runtime={"command": "python -m x"})])
        assert any("deve ser uma lista" in m for m in _messages(report, LEVEL_ERROR))

    def test_missing_cwd(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [_good(runtime={"command": [sys.executable], "cwd": str(tmp_path / "nope")})])
        assert any("runtime.cwd não existe" in m for m in _messages(report, LEVEL_ERROR))

    def test_bad_timeouts_and_max_jobs(self, tmp_path: Path) -> None:
        runtime = {"command": [sys.executable], "load_timeout_sec": 0, "event_timeout_sec": "x", "max_jobs": -1}
        errors = _messages(_run(tmp_path, [_good(runtime=runtime)]), LEVEL_ERROR)
        assert any("load_timeout_sec" in m for m in errors)
        assert any("event_timeout_sec" in m for m in errors)
        assert any("max_jobs" in m for m in errors)

    def test_shape_keys_outside_load_keys(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [_good(load_keys=["device"], shape_keys=["device", "compute_type"])])
        (msg,) = _messages(report, LEVEL_WARNING)
        assert "compute_type" in msg

    def test_footprint_larger_than_gpu(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [_good(vram_mib=10000)], gpu_total_mib=6141)
        (msg,) = _messages(report, LEVEL_WARNING)
        assert "excede a VRAM total" in msg
        assert _run(tmp_path, [_good(vram_mib=10000)]).findings == []  # sem GPU não adivinha

    def test_in_process_adapter_not_importable(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [{"name": "a", "adapter": "pkg_que_nao_existe.adapter", "vram_mib": 1}])
        (msg,) = _messages(report, LEVEL_WARNING)
        assert "não é importável" in msg

    def test_duplicate_name_in_same_file(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [_good(), _good(vram_mib=3000)])
        assert any("duas vezes" in m for m in _messages(report, LEVEL_WARNING))

    def test_unknown_top_level_key(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "b.yaml", [_good()], backend=[])
        report = validate_config(path, check_hooks=False)
        assert any("chave de topo desconhecida 'backend'" in m for m in _messages(report, LEVEL_WARNING))

    def test_malformed_yaml_is_error(self, tmp_path: Path) -> None:
        path = tmp_path / "b.yaml"
        path.write_text("backends: [oops", encoding="utf-8")
        report = validate_config(str(path), check_hooks=False)
        assert not report.ok

    def test_missing_explicit_file(self, tmp_path: Path) -> None:
        report = validate_config(str(tmp_path / "nope.yaml"), check_hooks=False)
        assert not report.ok and report.sources == []

    def test_findings_sorted_errors_first(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [_good(prority=1, vram_mib="x")])
        assert [f.level for f in report.findings] == [LEVEL_ERROR, LEVEL_WARNING]

    def test_to_dict_shape(self, tmp_path: Path) -> None:
        data = _run(tmp_path, [_good(prority=1)]).to_dict()
        assert data["ok"] is True and data["warnings"] == 1 and data["errors"] == 0
        assert data["findings"][0]["level"] == LEVEL_WARNING


class TestValidateLayers:
    def test_partial_overlay_without_base_is_error(self, tmp_path: Path, monkeypatch) -> None:
        """Overlay calibrado para um backend que já não existe na base."""
        overlay_dir = tmp_path / "backends.d"
        overlay_dir.mkdir()
        _write(overlay_dir / "measured.yaml", [{"name": "fantasma", "vram_mib": 4000}])
        monkeypatch.setenv("VRAMD_BACKENDS_FILE", _write(tmp_path / "base.yaml", [_good()]))
        monkeypatch.setenv("VRAMD_BACKENDS_DIR", str(overlay_dir))
        report = validate_config(check_hooks=False)
        assert report.backends == ["alpha", "fantasma"]
        (finding,) = report.errors
        assert finding.backend == "fantasma" and "falta adapter" in finding.message
        assert finding.source is not None and finding.source.endswith("measured.yaml")

    def test_env_file_that_does_not_exist(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("VRAMD_BACKENDS_FILE", str(tmp_path / "typo.yaml"))
        report = validate_config(check_hooks=False)
        assert any("não existe" in m for m in _messages(report, LEVEL_ERROR))

    def test_corrupt_overlay_reported(self, tmp_path: Path, monkeypatch) -> None:
        """O registry salta a camada com um warning no stderr; o lint torna-o visível."""
        overlay_dir = tmp_path / "backends.d"
        overlay_dir.mkdir()
        (overlay_dir / "broken.yaml").write_text("backends: {não: lista}", encoding="utf-8")
        monkeypatch.setenv("VRAMD_BACKENDS_FILE", _write(tmp_path / "base.yaml", [_good()]))
        monkeypatch.setenv("VRAMD_BACKENDS_DIR", str(overlay_dir))
        report = validate_config(check_hooks=False)
        (finding,) = report.errors
        assert finding.source is not None and finding.source.endswith("broken.yaml")
        assert report.backends == ["alpha"]


class TestValidateHooks:
    def test_malformed_hooks_is_error(self, tmp_path: Path) -> None:
        hooks = tmp_path / "hooks.yaml"
        hooks.write_text(yaml.safe_dump({"hooks": [{"event": "on_typo", "command": ["true"]}]}), encoding="utf-8")
        report = validate_config(_write(tmp_path / "b.yaml", [_good()]), hooks_path=hooks)
        assert any("desconhecido" in m for m in _messages(report, LEVEL_ERROR))

    def test_hook_command_missing_warns(self, tmp_path: Path) -> None:
        hooks = tmp_path / "hooks.yaml"
        hooks.write_text(
            yaml.safe_dump({"hooks": [{"event": "on_job_done", "command": ["no-such-notifier-xyz"]}]}),
            encoding="utf-8",
        )
        report = validate_config(_write(tmp_path / "b.yaml", [_good()]), hooks_path=hooks)
        assert report.ok
        assert any("no-such-notifier-xyz" in m for m in _messages(report, LEVEL_WARNING))

    def test_valid_hooks_clean(self, tmp_path: Path) -> None:
        hooks = tmp_path / "hooks.yaml"
        hooks.write_text(
            yaml.safe_dump({"hooks": [{"event": "on_job_done", "command": [sys.executable, "-c", "pass"]}]}),
            encoding="utf-8",
        )
        report = validate_config(_write(tmp_path / "b.yaml", [_good()]), hooks_path=hooks)
        assert report.findings == []
