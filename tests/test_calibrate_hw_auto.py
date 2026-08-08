"""Testes do M1: kwargs de load vindos do hw-auto da tool.

Sem isto, calibrar mede um caminho que a produção não usa — os adapters não
aplicam hw-auto sozinhos, e foi assim que o paint3d OOMou na primeira
calibração (sem ``memory_efficient`` carregou em precisão cheia).
"""

from __future__ import annotations

import json
import subprocess

import pytest
from click.testing import CliRunner

from vramd import cli as cli_mod
from vramd.calibrate.hw_auto import (
    load_kwargs_from_profile,
    probe_tool_profile,
    resolve_hw_auto_kwargs,
)

# Perfis reais medidos nesta máquina (RTX 4050 6 GB, perfil cuda-1x6g).
PAINT3D_PROFILE = {
    "name": "cuda-1x6g",
    "device": "cuda",
    "gpu_ids": None,
    "total_vram_gib": 5.6,
    "memory_efficient": True,
    "max_views": 6,
    "view_resolution": 512,
    "render_size": 1536,
    "texture_size": 3072,
}
MOTION3D_PROFILE = {
    "name": "cuda-1x6g",
    "model": "full",
    "sdnq_preset": None,
    "memory_efficient": True,
    "offload_text_encoder": True,
    "staged_load": True,
    "validation_steps": 20,
    "est_peak_gib": 4.0,
}


class TestLoadKwargsFromProfile:
    def test_paint3d_profile_yields_the_kwargs_that_were_missing(self):
        kwargs = load_kwargs_from_profile(PAINT3D_PROFILE)
        # ``memory_efficient`` é o que faltava e fez o load OOMar.
        assert kwargs["memory_efficient"] is True
        assert kwargs["max_num_view"] == 6  # renomeado de max_views
        assert kwargs["view_resolution"] == 512
        assert kwargs["texture_size"] == 3072

    def test_informative_fields_are_not_passed_to_load(self):
        kwargs = load_kwargs_from_profile(PAINT3D_PROFILE)
        for noise in ("name", "device", "total_vram_gib"):
            assert noise not in kwargs

    def test_none_values_are_omitted_not_passed_as_none(self):
        """Nos adapters, "chave ausente" ≠ "chave a None"."""
        kwargs = load_kwargs_from_profile(MOTION3D_PROFILE)
        assert "sdnq_preset" not in kwargs
        assert kwargs["model"] == "full"
        assert kwargs["offload_text_encoder"] is True

    def test_half_is_renamed_to_half_precision(self):
        assert load_kwargs_from_profile({"half": True}) == {"half_precision": True}

    def test_empty_profile_gives_empty_kwargs(self):
        assert load_kwargs_from_profile({}) == {}

    def test_unknown_fields_are_ignored(self):
        assert load_kwargs_from_profile({"campo_novo": 1}) == {}


class TestProbeToolProfile:
    def test_parses_the_profile_from_stdout(self, monkeypatch):
        payload = json.dumps({"profile": PAINT3D_PROFILE})
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=f"ruido\n{payload}\n", stderr=""),
        )
        assert probe_tool_profile("paint3d", "/venv/python")["max_views"] == 6

    def test_error_payload_is_passed_through(self, monkeypatch):
        payload = json.dumps({"error": "sem modulo hardware: x"})
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=payload, stderr="")
        )
        assert "error" in probe_tool_profile("terrain3d", "/venv/python")

    def test_no_usable_output_is_an_error_not_a_crash(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="boom")
        )
        assert "sem saída utilizável" in probe_tool_profile("x", "/venv/python")["error"]

    def test_invalid_json_is_an_error(self, monkeypatch):
        monkeypatch.setattr(
            subprocess,
            "map" if False else "run",
            lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="{nao", stderr=""),
        )
        assert "JSON inválido" in probe_tool_profile("x", "/venv/python")["error"]

    def test_timeout_is_an_error(self, monkeypatch):
        def boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd="x", timeout=1)

        monkeypatch.setattr(subprocess, "run", boom)
        assert "probe falhou" in probe_tool_profile("x", "/venv/python")["error"]

    def test_missing_interpreter_is_an_error(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("não existe")

        monkeypatch.setattr(subprocess, "run", boom)
        assert "probe falhou" in probe_tool_profile("x", "/nao/existe")["error"]


class TestResolveHwAutoKwargs:
    def test_backend_without_tool_is_not_applicable(self):
        kwargs, error = resolve_hw_auto_kwargs(None)
        assert kwargs == {}
        assert "não aplicável" in error

    def test_missing_venv_reports_an_error(self, monkeypatch):
        monkeypatch.setattr("vramd.toolchain.resolve_tool_python", lambda tool: None)
        kwargs, error = resolve_hw_auto_kwargs("fantasma")
        assert kwargs == {}
        assert "venv" in error

    def test_successful_probe_returns_kwargs(self, monkeypatch):
        monkeypatch.setattr("vramd.calibrate.hw_auto.probe_tool_profile", lambda tool, python, **k: PAINT3D_PROFILE)
        kwargs, error = resolve_hw_auto_kwargs("paint3d", python="/venv/python")
        assert error is None
        assert kwargs["memory_efficient"] is True

    def test_probe_error_is_surfaced(self, monkeypatch):
        monkeypatch.setattr(
            "vramd.calibrate.hw_auto.probe_tool_profile", lambda tool, python, **k: {"error": "sem hardware"}
        )
        kwargs, error = resolve_hw_auto_kwargs("terrain3d", python="/venv/python")
        assert kwargs == {}
        assert error == "sem hardware"


class TestCliIntegration:
    @pytest.fixture
    def patched(self, monkeypatch):
        state: dict = {"spec": None, "hw": ({}, None)}

        class FakePool:
            def shutdown_all(self):
                pass

        class FakeRunner:
            last_windows = None

            def __init__(self, pool, **kwargs):
                pass

            def wait_until_drained(self, **kwargs):
                return 0

            def preflight(self, **kwargs):
                return []

            def run(self, spec):
                state["spec"] = spec
                from .test_calibrate_analysis import build_windows, derive

                return derive(build_windows(), backend="text3d", tool="text3d")

        monkeypatch.setattr("vramd.subprocess_pool.SubprocessWorkerPool", FakePool)
        monkeypatch.setattr("vramd.calibrate.CalibrationRunner", FakeRunner)
        monkeypatch.setattr("vramd.calibrate.hw_auto.resolve_hw_auto_kwargs", lambda tool, **k: state["hw"])
        return state

    def test_hw_auto_kwargs_reach_the_spec(self, patched):
        patched["hw"] = ({"memory_efficient": True, "max_num_view": 6}, None)
        result = CliRunner().invoke(cli_mod.cli, ["calibrate", "paint3d", "--no-compare"])
        assert result.exit_code == 0
        assert patched["spec"].load_kwargs["memory_efficient"] is True
        assert patched["spec"].load_kwargs["max_num_view"] == 6

    def test_explicit_kwarg_overrides_hw_auto(self, patched):
        patched["hw"] = ({"max_num_view": 6}, None)
        CliRunner().invoke(cli_mod.cli, ["calibrate", "paint3d", "--no-compare", "--load-kwarg", "max_num_view=2"])
        assert patched["spec"].load_kwargs["max_num_view"] == 2

    def test_no_hw_auto_flag_skips_the_probe(self, patched):
        patched["hw"] = ({"memory_efficient": True}, None)
        CliRunner().invoke(cli_mod.cli, ["calibrate", "paint3d", "--no-compare", "--no-hw-auto"])
        assert patched["spec"].load_kwargs == {}

    def test_hw_auto_failure_warns_and_continues(self, patched):
        patched["hw"] = ({}, "venv da tool não encontrado")
        result = CliRunner().invoke(
            cli_mod.cli, ["calibrate", "paint3d", "--no-compare", "--load-kwarg", "memory_efficient=true"]
        )
        assert result.exit_code == 0
        assert "hw-auto indisponível" in result.output
        assert patched["spec"].load_kwargs["memory_efficient"] is True
