"""Testes do M2: backends cujo ``unload`` não devolve VRAM ao driver.

Medido em text2icon: o `unload` devolveu 82 MiB de 4764 residentes. Sem
tratamento, «evictar» esse backend é um no-op que destrói o modelo quente e
deixa a VRAM presa — e o plano de evicção conta com MiB que nunca chegam.
"""

from __future__ import annotations

from vramd.backend_manager import BackendManager
from vramd.registry import BackendDescriptor, Registry
from vramd.vram_planner import LoadedBackend, plan_eviction


def loaded(name, vram, *, priority=10, ref=0, used=0.0, frees=True) -> LoadedBackend:
    return LoadedBackend(name=name, vram_mib=vram, priority=priority, ref_count=ref, last_used=used, frees_vram=frees)


class TestPlannerOrdering:
    def test_frees_vram_defaults_true(self):
        assert LoadedBackend("a", 100, 10, 0, 0.0).frees_vram is True

    def test_backend_that_frees_is_preferred(self):
        plan = plan_eviction([loaded("preso", 4000, frees=False), loaded("normal", 4000)], 3000, 0)
        assert plan[0] == "normal"

    def test_stuck_backend_is_still_evictable_as_last_resort(self):
        """Excluí-lo faria a placa ficar presa para sempre."""
        plan = plan_eviction([loaded("preso", 4000, frees=False)], 3000, 0)
        assert plan == ["preso"]

    def test_stuck_backend_only_used_when_the_others_are_not_enough(self):
        plan = plan_eviction([loaded("preso", 4000, frees=False), loaded("normal", 1000)], 3000, 0)
        assert plan == ["normal", "preso"]

    def test_ordering_within_the_free_group_is_unchanged(self):
        plan = plan_eviction([loaded("leve", 1000, priority=1), loaded("pesado", 4000, priority=40)], 5000, 0)
        # efficiency = vram/priority: o leve rende mais por prioridade perdida.
        assert plan[0] == "leve"

    def test_active_backends_are_never_evicted_even_if_stuck(self):
        assert plan_eviction([loaded("preso", 4000, ref=1, frees=False)], 3000, 0) == []


def manager_with(*, frees: bool, pool) -> BackendManager:
    desc = BackendDescriptor(
        name="d",
        adapter="a",
        vram_mib=4000,
        priority=10,
        tool="d",
        peak_profile={} if frees else {"unload_frees_vram": False},
    )
    return BackendManager(Registry(descriptors={"d": desc}), subprocess_pool=pool)


class FakePool:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def unload(self, backend):
        self.calls.append(f"unload:{backend}")
        return True

    def shutdown(self, backend):
        self.calls.append(f"shutdown:{backend}")
        return True

    def is_alive(self, backend):
        return True

    def vram_mib(self, backend):
        return 4000


class TestEvictionEscalation:
    def _prepare(self, frees: bool):
        pool = FakePool()
        manager = manager_with(frees=frees, pool=pool)
        from vramd.backend_manager import _LoadedState

        state = _LoadedState()
        state.subprocess_loaded = True
        state.ref_count = 0
        manager._states["d"] = state
        return manager, pool

    def test_normal_backend_is_unloaded(self):
        manager, pool = self._prepare(frees=True)
        assert manager.evict("d") is True
        assert pool.calls == ["unload:d"]

    def test_stuck_backend_has_its_worker_terminated(self):
        manager, pool = self._prepare(frees=False)
        assert manager.evict("d") is True
        # ``unload`` seria um no-op: a VRAM só volta quando o processo morre.
        assert pool.calls == ["shutdown:d"]

    def test_state_is_marked_unloaded_either_way(self):
        for frees in (True, False):
            manager, _ = self._prepare(frees=frees)
            manager.evict("d")
            assert manager._states["d"].is_loaded() is False

    def test_shutdown_failure_does_not_break_the_eviction(self):
        manager, pool = self._prepare(frees=False)

        def boom(backend):
            pool.calls.append("shutdown-falhou")
            raise RuntimeError("worker não responde")

        pool.shutdown = boom
        assert manager.evict("d") is True

    def test_helper_defaults_to_true_for_unknown_backend(self):
        manager, _ = self._prepare(frees=True)
        assert manager._frees_vram_on_unload("nao-existe") is True

    def test_helper_reads_the_descriptor(self):
        manager, _ = self._prepare(frees=False)
        assert manager._frees_vram_on_unload("d") is False


class TestSnapshotCarriesTheFlag:
    def test_loaded_backend_snapshot_reflects_the_descriptor(self):
        pool = FakePool()
        manager = manager_with(frees=False, pool=pool)
        from vramd.backend_manager import _LoadedState

        state = _LoadedState()
        state.subprocess_loaded = True
        manager._states["d"] = state
        snapshot = manager._snapshot("d")
        assert snapshot is not None
        assert snapshot.frees_vram is False
