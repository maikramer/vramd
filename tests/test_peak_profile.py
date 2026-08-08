"""Testes do F2: pico declarativo (`peak_profile:`) e footprint medido (`vram:`).

Antes, o pico dependia de `name in ("paint3d","text3d",…)` — um backend novo
caía no caso genérico e admitia mal. Agora o comportamento vem do descriptor, e
uma calibração (`ums calibrate`) substitui a estimativa.
"""

from __future__ import annotations

import pytest

from vramd.backend_manager import BackendManager, _normalize_quant
from vramd.registry import BackendDescriptor, Registry


def manager_with(**kwargs) -> BackendManager:
    desc = BackendDescriptor(
        name="d",
        adapter="a",
        vram_mib=kwargs.pop("vram_mib", 4000),
        priority=0,
        footprint_key=kwargs.pop("footprint_key", None),
        vram=kwargs.pop("vram", {}),
        peak_profile=kwargs.pop("peak_profile", {}),
    )
    return BackendManager(Registry(descriptors={"d": desc}))


class TestNormalizeQuant:
    def test_empty_variants_are_none(self):
        for value in ("", None, "none", "NULL", "false"):
            assert _normalize_quant(value) == "none"

    def test_prefix_is_stripped(self):
        assert _normalize_quant("sdnq-int4") == "int4"
        assert _normalize_quant("int4") == "int4"

    def test_case_and_spaces_ignored(self):
        assert _normalize_quant("  SDNQ-UInt8 ") == "uint8"


class TestPeakProfileDrivesHeuristics:
    def test_plain_backend_gets_no_memory_efficient_by_default(self):
        manager = manager_with()
        _, mem, go, streams = manager.resolve_peak_params("d", {"sdnq_preset": "sdnq-int4"})
        assert (mem, go, streams) == (False, False, False)

    def test_declared_profile_turns_memory_efficient_on_with_quant(self):
        manager = manager_with(peak_profile={"memory_efficient_with_quant": True})
        _, mem, _, _ = manager.resolve_peak_params("d", {"sdnq_preset": "sdnq-int4"})
        assert mem is True

    def test_declared_profile_does_nothing_without_quant(self):
        manager = manager_with(peak_profile={"memory_efficient_with_quant": True})
        _, mem, _, _ = manager.resolve_peak_params("d", {})
        assert mem is False

    def test_group_offload_follows_memory_efficient(self):
        manager = manager_with(
            peak_profile={"memory_efficient_with_quant": True, "group_offload_with_memory_efficient": True}
        )
        _, _, go, _ = manager.resolve_peak_params("d", {"sdnq_preset": "sdnq-int4"})
        assert go is True

    def test_streams_on_load_follows_memory_efficient(self):
        manager = manager_with(
            peak_profile={"memory_efficient_with_quant": True, "streams_on_load_with_memory_efficient": True}
        )
        _, _, _, streams = manager.resolve_peak_params("d", {"sdnq_preset": "sdnq-int4"})
        assert streams is True

    def test_explicit_request_still_wins_over_the_profile(self):
        manager = manager_with(peak_profile={"memory_efficient_with_quant": True})
        _, mem, _, _ = manager.resolve_peak_params("d", {"sdnq_preset": "sdnq-int4", "memory_efficient": False})
        assert mem is False

    def test_unknown_backend_uses_an_empty_profile(self):
        manager = manager_with()
        _, mem, _, _ = manager.resolve_peak_params("nao-existe", {"sdnq_preset": "sdnq-int4"})
        assert mem is False


class TestPackagedDefaultsPreserveBehaviour:
    """As heurísticas antigas viraram YAML — o comportamento tem de ser o mesmo."""

    @pytest.fixture
    def manager(self):
        return BackendManager(Registry())

    @pytest.mark.parametrize("backend", ["paint3d", "text3d", "text2d", "motion3d"])
    def test_sdnq_backends_stay_memory_efficient(self, manager, backend):
        _, mem, _, _ = manager.resolve_peak_params(backend, {"sdnq_preset": "sdnq-int4"})
        assert mem is True

    @pytest.mark.parametrize("backend", ["text3d", "motion3d"])
    def test_group_offload_backends_unchanged(self, manager, backend):
        _, _, go, _ = manager.resolve_peak_params(backend, {"sdnq_preset": "sdnq-int4"})
        assert go is True

    def test_text2d_keeps_streams_on_load(self, manager):
        _, _, _, streams = manager.resolve_peak_params("text2d", {"sdnq_preset": "sdnq-int4"})
        assert streams is True

    def test_text3d_does_not_stream_on_load(self, manager):
        """text3d carrega pesos completos e só depois faz leaf-offload."""
        _, _, _, streams = manager.resolve_peak_params("text3d", {"sdnq_preset": "sdnq-int4"})
        assert streams is False

    def test_texture2d_has_no_memory_efficient_default(self, manager):
        _, mem, _, _ = manager.resolve_peak_params("texture2d", {"sdnq_preset": "sdnq-int4"})
        assert mem is False


class TestMeasuredFootprintWins:
    def test_measured_block_replaces_the_footprint_estimate(self):
        manager = manager_with(
            footprint_key="hunyuan3d-omni",  # estimaria ~10 GiB
            vram={"weights_gib": 4.0, "activation_gib": 1.4, "context_gib": 0.12},
            peak_profile={"quant_mode": "none"},
        )
        weights, activation = manager.footprint_parts_mib("d")
        assert weights == int(4.0 * 1024) + int(0.12 * 1024)
        assert activation == int(1.4 * 1024)

    def test_context_is_folded_into_the_weights(self):
        """O contexto CUDA é VRAM que o driver cobra enquanto o backend vive."""
        manager = manager_with(vram={"weights_gib": 1.0, "activation_gib": 0.5, "context_gib": 0.25})
        weights, _ = manager.footprint_parts_mib("d")
        assert weights == 1024 + 256

    def test_measurement_is_ignored_for_a_different_quant(self):
        manager = manager_with(
            footprint_key="hunyuan3d-omni",
            vram={"weights_gib": 1.0, "activation_gib": 0.5},
            peak_profile={"quant_mode": "sdnq-int4"},
        )
        # Pedido em fp16: os pesos medidos em int4 não valem.
        weights, _ = manager.footprint_parts_mib("d", quant_mode="none")
        assert weights > 4000

    def test_measurement_applies_when_the_quant_matches_modulo_prefix(self):
        manager = manager_with(
            vram={"weights_gib": 1.0, "activation_gib": 0.5},
            peak_profile={"quant_mode": "sdnq-int4"},
        )
        weights, _ = manager.footprint_parts_mib("d", quant_mode="int4")
        assert weights == 1024

    def test_incomplete_measurement_falls_back_to_the_estimate(self):
        manager = manager_with(vram={"weights_gib": 1.0}, peak_profile={"quant_mode": "none"})
        weights, activation = manager.footprint_parts_mib("d")
        # Sem activation_gib não há medição utilizável → YAML vram_mib (4000).
        assert (weights, activation) == (3200, 800)

    def test_memory_efficient_scales_the_measured_activation(self):
        manager = manager_with(vram={"weights_gib": 1.0, "activation_gib": 2.0})
        _, plain = manager.footprint_parts_mib("d")
        _, efficient = manager.footprint_parts_mib("d", memory_efficient=True)
        assert efficient < plain

    def test_peak_uses_the_measured_parts(self):
        manager = manager_with(vram={"weights_gib": 1.0, "activation_gib": 0.5})
        peak = manager.peak_vram_mib("d")
        assert peak >= 1024 + 512

    def test_no_measurement_keeps_the_previous_path(self):
        manager = manager_with(footprint_key="stable-audio-open")
        weights, activation = manager.footprint_parts_mib("d")
        assert weights > 0 and activation > 0
