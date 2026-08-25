"""Admit/refuse por pico pesos+activação — sem OOM silencioso."""

from __future__ import annotations

import pytest

from vramd import protocol as P
from vramd.backend_manager import BackendManager, InsufficientVramError
from vramd.registry import Registry


@pytest.fixture(autouse=True)
def _fast_admit_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse tests não devem esperar 8s de admit wait."""
    monkeypatch.setattr(P, "VRAM_ADMIT_WAIT_SEC", 0.0)
    monkeypatch.setattr(P, "VRAM_ADMIT_POLL_SEC", 0.05)


class TestEnsureLoadedAdmitsPeak:
    def test_refuses_when_free_below_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VRAMD_VRAM_SAFETY_MIB", "384")
        registry = Registry()
        # 6GB livre — text3d fp16 peak ~8GB+
        mgr = BackendManager(registry, query_free_mib=lambda: 5657, clear_vram=lambda: None)
        with pytest.raises(InsufficientVramError) as ei:
            mgr.ensure_loaded("text3d", sdnq_preset="none")
        err = ei.value
        assert err.peak_mib > 5657
        assert err.activation_mib > 0
        assert err.quant_mode == "none"

    def test_admit_waits_until_free_recovers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VRAMD_VRAM_SAFETY_MIB", "384")
        # Override autouse: aqui queremos espera activa.
        monkeypatch.setattr(P, "VRAM_ADMIT_WAIT_SEC", 2.0)
        monkeypatch.setattr(P, "VRAM_ADMIT_POLL_SEC", 0.05)
        registry = Registry()
        free = {"v": 4000}
        ticks = {"n": 0}

        def _free() -> int:
            ticks["n"] += 1
            # Após alguns polls, VRAM "liberta" (processo externo saiu).
            if ticks["n"] >= 3:
                free["v"] = 5657
            return free["v"]

        class _FakeAdapter:
            def load(self, **kwargs):
                return object()

            def unload(self, model):
                pass

        mgr = BackendManager(registry, query_free_mib=_free, clear_vram=lambda: None)
        monkeypatch.setattr(mgr._registry, "adapter", lambda name: _FakeAdapter())
        model = mgr.ensure_loaded("text3d", sdnq_preset="sdnq-int4")
        assert model is not None
        assert ticks["n"] >= 3

    def test_int4_admitted_on_6gb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VRAMD_VRAM_SAFETY_MIB", "384")
        registry = Registry()
        loaded: dict[str, object] = {}

        class _FakeAdapter:
            def load(self, **kwargs):
                loaded["ok"] = True
                return object()

            def unload(self, model):
                pass

        mgr = BackendManager(registry, query_free_mib=lambda: 5657, clear_vram=lambda: None)
        # Patch só o adapter text3d — evita torch.
        monkeypatch.setattr(mgr._registry, "adapter", lambda name: _FakeAdapter())
        model = mgr.ensure_loaded("text3d", sdnq_preset="sdnq-int4")
        assert model is not None
        assert loaded.get("ok") is True

    def test_ensure_vram_uses_backend_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VRAMD_VRAM_SAFETY_MIB", "384")
        registry = Registry()
        free = {"v": 5657}
        mgr = BackendManager(registry, query_free_mib=lambda: free["v"], clear_vram=lambda: None)
        # Pedido cliente 5000 < peak text3d none — sem backends loaded, não consegue libertar.
        ok = mgr.ensure_vram(5000, backend="text3d", quant_mode="none")
        assert ok is False


class TestPaint3dMemoryEfficientAdmit:
    """paint3d envia memory_efficient sem sdnq_preset — não pode assumir fp16 peak ~8 GiB."""

    def test_resolve_quant_from_memory_efficient(self) -> None:
        assert BackendManager.resolve_quant_mode({"memory_efficient": True}) == "sdnq-uint8"
        assert BackendManager.resolve_quant_mode({"memory_efficient": False}) == "none"
        assert BackendManager.resolve_quant_mode({"sdnq_preset": "none", "memory_efficient": True}) == "none"
        assert BackendManager.resolve_quant_mode({"sdnq_preset": "sdnq-int4"}) == "sdnq-int4"

    def test_paint_mem_eff_admitted_on_6gb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VRAMD_VRAM_SAFETY_MIB", "384")
        registry = Registry()
        loaded: dict[str, object] = {}

        class _FakeAdapter:
            def load(self, **kwargs):
                loaded["kwargs"] = dict(kwargs)
                return object()

            def unload(self, model):
                pass

        mgr = BackendManager(registry, query_free_mib=lambda: 5657, clear_vram=lambda: None)
        monkeypatch.setattr(mgr._registry, "adapter", lambda name: _FakeAdapter())
        # Sem memory_efficient → fp16 peak ~8576 → refuse
        with pytest.raises(InsufficientVramError) as ei:
            mgr.ensure_loaded("paint3d")
        assert ei.value.quant_mode == "none"
        assert ei.value.peak_mib > 5657

        model = mgr.ensure_loaded("paint3d", memory_efficient=True)
        assert model is not None
        assert loaded["kwargs"].get("memory_efficient") is True
        peak = mgr.peak_vram_mib("paint3d", quant_mode="sdnq-uint8", memory_efficient=True)
        assert peak <= 5657


class TestText2dStreamsOnLoadAdmit:
    """Regressão 0.3.0 (RTX 4050 6 GB, catálogo backends-6g.yaml): text2d
    calibrado em sdnq-int4 com load streaming (diffusers offload) era recusado
    com peak=6117 MiB contra ~5736 MiB livres — ``streams_on_load`` dobrado
    como ``group_offload`` desligava o desconto 0.65 da activação medida."""

    @staticmethod
    def _registry() -> Registry:
        from vramd.registry import BackendDescriptor

        return Registry(
            descriptors={
                "text2d": BackendDescriptor(
                    name="text2d",
                    adapter="a",
                    vram_mib=5760,
                    priority=25,
                    vram={"weights_gib": 0.19, "activation_gib": 5.18, "context_gib": 0.23},
                    peak_profile={
                        "quant_mode": "sdnq-int4",
                        "memory_efficient_with_quant": True,
                        "streams_on_load_with_memory_efficient": True,
                    },
                )
            }
        )

    @pytest.fixture(autouse=True)
    def _no_admit_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(P, "VRAM_ADMIT_WAIT_SEC", 0.0)

    def test_calibrated_text2d_admits_on_6gb_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VRAMD_VRAM_SAFETY_MIB", "384")
        loaded: dict[str, object] = {}

        class _FakeAdapter:
            def load(self, **kwargs):
                loaded["ok"] = True
                return object()

            def unload(self, model):
                pass

        mgr = BackendManager(self._registry(), query_free_mib=lambda: 5736, clear_vram=lambda: None)
        monkeypatch.setattr(mgr._registry, "adapter", lambda name: _FakeAdapter())
        model = mgr.ensure_loaded("text2d", sdnq_preset="sdnq-int4", memory_efficient=True)
        assert model is not None
        assert loaded.get("ok") is True

    def test_still_refuses_when_free_below_discounted_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VRAMD_VRAM_SAFETY_MIB", "384")

        class _FakeAdapter:
            def load(self, **kwargs):
                return object()

            def unload(self, model):
                pass

        mgr = BackendManager(self._registry(), query_free_mib=lambda: 4000, clear_vram=lambda: None)
        monkeypatch.setattr(mgr._registry, "adapter", lambda name: _FakeAdapter())
        with pytest.raises(InsufficientVramError) as ei:
            mgr.ensure_loaded("text2d", sdnq_preset="sdnq-int4", memory_efficient=True)
        # Descontado (429+3447+384=4260) mas ainda acima de 4000 → o admit guarda.
        assert ei.value.peak_mib == 4260
        assert ei.value.activation_mib == 3447
