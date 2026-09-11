"""Admit/refuse por pico pesos+activação — sem OOM silencioso."""

from __future__ import annotations

from typing import Any

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


class TestText2dGroupOffloadAdmit:
    """Regressão 0.3.7: um request com ``allow_group_offload`` não pode ser
    admitido pela medição do caminho clássico — a "activação" medida (5.18 GiB)
    incluía o warmup não-GO (quantização runtime na GPU + colocação). Com GO os
    pesos streamam por grupos: o pico é o footprint (maior módulo + activação),
    ~2.6 GiB para o flux-klein-4b int4 — caber (e admitir) onde o 6117 recusava."""

    @staticmethod
    def _descriptor(go_measured: bool = False) -> Any:
        from vramd.registry import BackendDescriptor

        load_kwargs: dict[str, Any] = {"memory_efficient": True}
        if go_measured:
            load_kwargs["allow_group_offload"] = True
        return BackendDescriptor(
            name="text2d",
            adapter="a",
            vram_mib=5760,
            priority=25,
            footprint_key="flux-klein-4b",
            vram={
                "weights_gib": 0.19,
                "activation_gib": 5.18,
                "context_gib": 0.23,
                "peak_mib": 5760,
                "safety_mib": 384,
                "admit_peak_mib": 6144,
            },
            peak_profile={
                "quant_mode": "sdnq-int4",
                "memory_efficient_with_quant": True,
                "streams_on_load_with_memory_efficient": True,
                "load_kwargs": load_kwargs,
            },
        )

    @pytest.fixture(autouse=True)
    def _no_admit_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(P, "VRAM_ADMIT_WAIT_SEC", 0.0)

    def test_go_request_ignores_classic_measurement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GO no request + medição clássica → footprint GO (2558), não 6117."""
        monkeypatch.setenv("VRAMD_VRAM_SAFETY_MIB", "384")

        class _FakeAdapter:
            def load(self, **kwargs):
                return object()

            def unload(self, model):
                pass

        mgr = BackendManager(
            Registry(descriptors={"text2d": self._descriptor()}),
            query_free_mib=lambda: 3600,
            clear_vram=lambda: None,
        )
        monkeypatch.setattr(mgr._registry, "adapter", lambda name: _FakeAdapter())
        # 3600 MiB livres: medição clássica (6117, ou 4260 descontada) recusava;
        # footprint GO = largest(int4) 1638 + act 1536 + safety 384 = 3558.
        model = mgr.ensure_loaded(
            "text2d",
            sdnq_preset="sdnq-int4",
            memory_efficient=True,
            allow_group_offload=True,
            footprint_key="flux-klein-4b",
        )
        assert model is not None
        peak = mgr.peak_vram_mib(
            "text2d",
            quant_mode="sdnq-int4",
            memory_efficient=True,
            group_offload=True,
            footprint_key="flux-klein-4b",
        )
        assert peak == 3558

    def test_go_measurement_used_for_go_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calibração medida COM GO + request GO → pico medido vale (5760)."""
        monkeypatch.setenv("VRAMD_VRAM_SAFETY_MIB", "384")
        mgr = BackendManager(
            Registry(descriptors={"text2d": self._descriptor(go_measured=True)}),
            query_free_mib=lambda: 6000,
            clear_vram=lambda: None,
        )
        peak = mgr.peak_vram_mib(
            "text2d",
            quant_mode="sdnq-int4",
            memory_efficient=True,
            group_offload=True,
            footprint_key="flux-klein-4b",
        )
        assert peak == 5760  # peak_mib medido (sem safety duplicada)

    def test_go_measurement_not_used_for_classic_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Medição feita COM GO não descreve request clássico (subestimaria):
        volta ao footprint completo — pesos int4 (4587) + act descontada 998
        (mem-eff clássico) + safety 384 = 5969."""
        monkeypatch.setenv("VRAMD_VRAM_SAFETY_MIB", "384")
        mgr = BackendManager(
            Registry(descriptors={"text2d": self._descriptor(go_measured=True)}),
            query_free_mib=lambda: 6000,
            clear_vram=lambda: None,
        )
        peak = mgr.peak_vram_mib(
            "text2d",
            quant_mode="sdnq-int4",
            memory_efficient=True,
            group_offload=False,
            footprint_key="flux-klein-4b",
        )
        assert peak == 4587 + 998 + 384


class TestEnsureLoadedUsesMeasuredPeak:
    """Regressão 0.3.9: o admit de LOAD honra ``vram.peak_mib`` medido (mesmo
    quant+modo) em vez de somar safety por cima — text2d GO 6g: 5568 medido vs
    5922 somado, numa GPU com ~5727 livres (o somado recusava SEMPRE)."""

    def test_measured_peak_admits_where_summed_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VRAMD_VRAM_SAFETY_MIB", "384")

        class _FakeAdapter:
            def load(self, **kwargs):
                return object()

            def unload(self, model):
                pass

        from vramd.registry import BackendDescriptor

        desc = BackendDescriptor(
            name="text2d",
            adapter="a",
            vram_mib=5568,
            priority=25,
            footprint_key="flux-klein-4b",
            vram={
                "weights_gib": 0.19,
                "activation_gib": 4.99,
                "context_gib": 0.23,
                "peak_mib": 5568,
                "safety_mib": 384,
                "admit_peak_mib": 5952,
            },
            peak_profile={
                "quant_mode": "sdnq-int4",
                "load_kwargs": {"memory_efficient": True, "allow_group_offload": True},
            },
        )
        mgr = BackendManager(
            Registry(descriptors={"text2d": desc}),
            query_free_mib=lambda: 5727,  # livre máximo real duma 4050 6 GB
            clear_vram=lambda: None,
        )
        monkeypatch.setattr(mgr._registry, "adapter", lambda name: _FakeAdapter())
        # Somado (429+5108+384=5921) recusaria; o medido 5568 admite.
        model = mgr.ensure_loaded(
            "text2d",
            sdnq_preset="sdnq-int4",
            memory_efficient=True,
            allow_group_offload=True,
            footprint_key="flux-klein-4b",
        )
        assert model is not None

    def test_summed_peak_still_used_without_measurement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sem medição aplicável (modo difere), vale a soma footprint+safety."""
        monkeypatch.setenv("VRAMD_VRAM_SAFETY_MIB", "384")
        from vramd.registry import BackendDescriptor

        desc = BackendDescriptor(
            name="text2d",
            adapter="a",
            vram_mib=5568,
            priority=25,
            footprint_key="flux-klein-4b",
            vram={
                "weights_gib": 0.19,
                "activation_gib": 4.99,
                "context_gib": 0.23,
                "peak_mib": 5568,
                "safety_mib": 384,
            },
            peak_profile={
                "quant_mode": "sdnq-int4",
                # medição feita COM GO — request abaixo pede o clássico.
                "load_kwargs": {"allow_group_offload": True},
            },
        )

        class _FakeAdapter:
            def load(self, **kwargs):
                return object()

            def unload(self, model):
                pass

        mgr = BackendManager(
            Registry(descriptors={"text2d": desc}),
            query_free_mib=lambda: 5800,
            clear_vram=lambda: None,
        )
        monkeypatch.setattr(mgr._registry, "adapter", lambda name: _FakeAdapter())
        with pytest.raises(InsufficientVramError) as ei:
            mgr.ensure_loaded(
                "text2d",
                sdnq_preset="sdnq-int4",
                memory_efficient=True,
                allow_group_offload=False,  # clássico: medição GO não vale
                footprint_key="flux-klein-4b",
            )
        # footprint clássico: 4587 pesos + act descontada 998 + safety 384 = 5969 > 5800.
        assert ei.value.peak_mib == 5969
