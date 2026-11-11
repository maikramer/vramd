"""Lanes multi-GPU e despacho roteado: réplicas num batch, split, GPUs fixadas."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from vramd import protocol as P
from vramd.dispatcher import WorkerPool
from vramd.job_queue import JobQueue
from vramd.lanes import KIND_SPAN, Lane, LaneSet, lane_env, lane_id_for, parse_gpu_list, resolve_gpu_devices
from vramd.placement import RouteEngine, parse_pinned_devices
from vramd.router import RoutePolicy


class _Desc:
    def __init__(self, name: str, vram_mib: int, **placement: Any) -> None:
        self.name = name
        self.vram_mib = vram_mib
        self.unload_frees_vram = True
        self.placement = placement


class _Registry:
    def __init__(self, *descs: _Desc) -> None:
        self._descs = {d.name: d for d in descs}

    @property
    def names(self) -> list[str]:
        return list(self._descs)

    def descriptor(self, name: str) -> _Desc:
        return self._descs[name]


class _Manager:
    """Gestor de uma lane: regista o que correu e quando."""

    def __init__(self, *, delay: float = 0.0, result: dict[str, Any] | None = None) -> None:
        self.delay = delay
        self.result = result
        self.loaded: set[str] = set()
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.spans: list[tuple[float, float]] = []
        self.evict_all_calls = 0
        self.stats = None

    def loaded_names(self) -> list[str]:
        return sorted(self.loaded)

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def shape_matches_loaded(self, name: str, request: dict[str, Any] | None = None) -> bool:
        return name in self.loaded

    def evict(self, name: str) -> bool:
        had = name in self.loaded
        self.loaded.discard(name)
        return had

    def evict_all(self) -> int:
        self.evict_all_calls += 1
        count = len(self.loaded)
        self.loaded.clear()
        return count

    def generate(self, name: str, request: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        self.calls.append((name, {k: v for k, v in request.items() if not k.startswith("_")}))
        if self.result is not None:
            return dict(self.result)
        self.loaded.add(name)
        if self.delay:
            time.sleep(self.delay)
        self.spans.append((started, time.monotonic()))
        return {"status": P.STATUS_OK, "output": name}


def _lane(index: int, manager: _Manager, *, free: int = 6000, total: int = 6000, pinned: bool = True) -> Lane:
    return Lane(
        lane_id=f"gpu{index}",
        devices=(index,),
        manager=manager,
        pinned=pinned,
        query_free_mib=lambda: free,
        query_total_mib=lambda: total,
    )


class _Rig:
    """Duas GPUs falsas, fila real, router real."""

    def __init__(self, registry: _Registry, *, delay: float = 0.0, gpus: int = 2, margin: float = 0.0) -> None:
        self.managers = [_Manager(delay=delay) for _ in range(gpus)]
        self.span_managers: dict[tuple[int, ...], _Manager] = {}

        def span_factory(devices: tuple[int, ...]) -> Lane:
            mgr = _Manager(delay=delay)
            self.span_managers[devices] = mgr
            return Lane(
                lane_id=lane_id_for(devices),
                devices=devices,
                manager=mgr,
                kind=KIND_SPAN,
                pinned=True,
                query_free_mib=lambda: 6000 * len(devices),
                query_total_mib=lambda: 6000 * len(devices),
            )

        self.lanes = LaneSet(
            [_lane(i, m) for i, m in enumerate(self.managers)], registry=registry, span_factory=span_factory
        )
        self.engine = RouteEngine(
            registry,
            self.lanes,
            None,
            policy=RoutePolicy(replica_margin_sec=margin, remote_penalty_sec=0.0, split_penalty_sec=0.0),
        )
        self.queue = JobQueue(max_depth=64)
        self.pool = WorkerPool(self.queue, self.managers[0], engine=self.engine)

    def run(self, jobs: list[tuple[str, dict[str, Any]]], *, timeout: float = 10.0) -> list[Any]:
        enqueued = [self.queue.enqueue(backend, request) for backend, request in jobs]
        self.pool.start()
        try:
            deadline = time.monotonic() + timeout
            for job in enqueued:
                assert job.done_event.wait(max(0.1, deadline - time.monotonic())), f"job {job.job_id} não terminou"
        finally:
            self.pool.stop()
        return enqueued


class TestGpuSelection:
    def test_parse_gpu_list(self) -> None:
        assert parse_gpu_list("0, 2,2") == [0, 2]
        with pytest.raises(ValueError):
            parse_gpu_list("0,x")
        with pytest.raises(ValueError):
            parse_gpu_list("-1")

    def test_auto_needs_more_than_one_gpu(self, monkeypatch) -> None:
        """Uma GPU: nada muda (sem lanes fixadas, sem CUDA_VISIBLE_DEVICES)."""
        monkeypatch.delenv("VRAMD_GPUS", raising=False)
        assert resolve_gpu_devices(lambda: [0]) is None
        assert resolve_gpu_devices(lambda: None) is None
        assert resolve_gpu_devices(lambda: [0, 1]) == [0, 1]

    def test_explicit_list_and_legacy(self, monkeypatch) -> None:
        monkeypatch.setenv("VRAMD_GPUS", "1")
        assert resolve_gpu_devices(lambda: [0, 1]) == [1]  # fixa o supervisor na GPU 1
        monkeypatch.setenv("VRAMD_GPUS", "legacy")
        assert resolve_gpu_devices(lambda: [0, 1]) is None
        monkeypatch.setenv("VRAMD_GPUS", "zero,um")
        assert resolve_gpu_devices(lambda: [0, 1]) is None

    def test_lane_env_pins_workers(self) -> None:
        env = lane_env((2,))
        assert env["CUDA_VISIBLE_DEVICES"] == "2" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert lane_env((0, 1))["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert lane_id_for((0, 1)) == "split:gpu0+gpu1"

    def test_pool_env_overrides_reach_the_worker(self) -> None:
        from vramd.subprocess_pool import SubprocessWorkerPool

        pool = SubprocessWorkerPool(env_overrides=lane_env((1,)))
        env = pool._worker_env("qualquer")
        assert env is not None and env["CUDA_VISIBLE_DEVICES"] == "1"
        assert SubprocessWorkerPool()._worker_env("qualquer") is None  # sem lanes: herda, como sempre

    def test_parse_pinned_devices(self) -> None:
        assert parse_pinned_devices({"gpu_ids": "1,0"}) == (0, 1)
        assert parse_pinned_devices({"gpu_ids": [1]}) == (1,)
        assert parse_pinned_devices({"gpu_ids": []}) is None
        assert parse_pinned_devices({"gpu_ids": "x"}) is None
        assert parse_pinned_devices({}) is None


class TestLaneSet:
    def _set(self) -> tuple[LaneSet, list[_Manager]]:
        managers = [_Manager(), _Manager()]
        registry = _Registry(_Desc("a", 3000), _Desc("b", 2000))
        span_mgr = _Manager()
        lanes = LaneSet(
            [_lane(0, managers[0]), _lane(1, managers[1])],
            registry=registry,
            span_factory=lambda devs: Lane(lane_id_for(devs), devs, span_mgr, kind=KIND_SPAN, pinned=True),
        )
        return lanes, [*managers, span_mgr]

    def test_reserve_is_exclusive(self) -> None:
        lanes, _ = self._set()
        gpu0 = lanes.lane("gpu0")
        assert lanes.reserve(gpu0, object(), est_sec=5.0) is True
        assert lanes.reserve(gpu0, object(), est_sec=5.0) is False
        lanes.release(gpu0)
        assert lanes.reserve(gpu0, object(), est_sec=5.0) is True

    def test_split_blocks_its_gpus_and_vice_versa(self) -> None:
        lanes, _ = self._set()
        span = lanes.span_for((1, 0))
        assert span.lane_id == "split:gpu0+gpu1" and lanes.span_for((0, 1)) is span
        assert lanes.reserve(span, object(), est_sec=10.0)
        assert lanes.is_busy(lanes.lane("gpu0")) and lanes.is_busy(lanes.lane("gpu1"))
        assert lanes.reserve(lanes.lane("gpu1"), object(), est_sec=1.0) is False
        lanes.release(span)
        assert lanes.reserve(lanes.lane("gpu1"), object(), est_sec=1.0)
        assert lanes.reserve(span, object(), est_sec=1.0) is False  # uma das GPUs está ocupada

    def test_view_reports_hot_and_evictable(self) -> None:
        lanes, managers = self._set()
        managers[0].loaded.update({"a", "b"})
        view = lanes.view(lanes.lane("gpu0"), "a")
        assert view.hot and view.loaded
        assert view.evictable_mib == 2000  # só o "b": o próprio backend não se evicta a si
        cold = lanes.view(lanes.lane("gpu1"), "a")
        assert not cold.hot and cold.evictable_mib == 0 and cold.busy_sec is None

    def test_idle_split_model_counts_as_evictable_on_its_gpus(self) -> None:
        lanes, managers = self._set()
        lanes.span_for((0, 1))
        managers[2].loaded.add("a")  # modelo dividido ocioso: 3000 MiB por 2 GPUs
        assert lanes.view(lanes.lane("gpu0"), "b").evictable_mib == 1500

    def test_make_room_evicts_idle_split_for_a_gpu_job(self) -> None:
        lanes, managers = self._set()
        lanes.span_for((0, 1))
        managers[2].loaded.add("a")
        gpu0 = lanes.lane("gpu0")
        lanes.reserve(gpu0, object(), est_sec=1.0)
        assert lanes.make_room(gpu0) == ["split:gpu0+gpu1"]
        assert managers[2].loaded == set()

    def test_make_room_for_split_drains_member_gpus_only_if_idle(self) -> None:
        lanes, managers = self._set()
        span = lanes.span_for((0, 1))
        managers[0].loaded.add("a")
        managers[1].loaded.add("b")
        lanes.reserve(span, object(), est_sec=1.0)
        assert sorted(lanes.make_room(span)) == ["gpu0", "gpu1"]
        assert managers[0].loaded == set() and managers[1].loaded == set()

    def test_device_lanes_never_evict_each_other(self) -> None:
        lanes, managers = self._set()
        managers[1].loaded.add("b")
        gpu0 = lanes.lane("gpu0")
        lanes.reserve(gpu0, object(), est_sec=1.0)
        assert lanes.make_room(gpu0) == []
        assert managers[1].loaded == {"b"}

    def test_remaining_prefers_reported_progress(self) -> None:
        now = [100.0]
        mgr = _Manager()
        lanes = LaneSet([_lane(0, mgr)], clock=lambda: now[0])

        class _Job:
            progress_pct = None

        job = _Job()
        lane = lanes.primary
        lanes.reserve(lane, job, est_sec=40.0)
        now[0] = 110.0
        assert lanes.remaining_sec(lane) == 30.0  # estimativa - decorrido
        job.progress_pct = 0.25
        assert lanes.remaining_sec(lane) == 30.0  # 10s para 25% → faltam 30s
        job.progress_pct = 0.5
        assert lanes.remaining_sec(lane) == 10.0
        job.progress_pct = None
        now[0] = 200.0
        assert lanes.remaining_sec(lane) == 8.0  # passou da estimativa: nunca "já"
        lanes.release(lane)
        assert lanes.remaining_sec(lane) == 0.0

    def test_single_unpinned_lane_is_not_routed(self) -> None:
        assert LaneSet([_lane(0, _Manager(), pinned=False)]).routed is False
        assert LaneSet([_lane(1, _Manager(), pinned=True)]).routed is True


class TestRoutedDispatch:
    def test_batch_opens_a_replica_on_the_idle_gpu(self) -> None:
        """4 jobs do mesmo modelo, load barato: as duas GPUs trabalham em paralelo."""
        rig = _Rig(_Registry(_Desc("m", 4000, load_sec=0.05, gen_sec=0.4)), delay=0.4)
        started = time.monotonic()
        jobs = rig.run([("m", {"prompt": str(i)}) for i in range(4)])
        elapsed = time.monotonic() - started
        assert all(j.state == P.JOB_DONE for j in jobs)
        assert len(rig.managers[0].calls) >= 1 and len(rig.managers[1].calls) >= 1
        assert elapsed < 1.45  # em série seriam 1.6s
        targets = {j.route["target"] for j in jobs}
        assert targets == {"gpu0", "gpu1"}
        assert any(j.route.get("replica") for j in jobs)
        assert rig.engine.counts["replica"] >= 1

    def test_expensive_load_keeps_the_batch_on_the_hot_gpu(self) -> None:
        """Load de 30s contra uma fila de décimos de segundo: nunca abre réplica."""
        rig = _Rig(_Registry(_Desc("m", 4000, load_sec=30.0, gen_sec=0.1)), delay=0.1, margin=5.0)
        jobs = rig.run([("m", {"prompt": str(i)}) for i in range(4)])
        assert all(j.state == P.JOB_DONE for j in jobs)
        assert len(rig.managers[0].calls) == 4 and rig.managers[1].calls == []
        assert rig.engine.counts["replica"] == 0

    def test_max_replicas_one_never_fans_out(self) -> None:
        rig = _Rig(_Registry(_Desc("m", 4000, load_sec=0.01, gen_sec=0.2, max_replicas=1)), delay=0.2)
        rig.run([("m", {}) for _ in range(3)])
        assert rig.managers[1].calls == []

    def test_different_models_use_both_gpus_at_once(self) -> None:
        rig = _Rig(_Registry(_Desc("a", 4000, gen_sec=0.3), _Desc("b", 4000, gen_sec=0.3)), delay=0.3)
        rig.run([("a", {}), ("b", {})])
        (a_start, a_end), (b_start, b_end) = rig.managers[0].spans[0], rig.managers[1].spans[0]
        assert a_start < b_end and b_start < a_end  # sobrepostos no tempo

    def test_model_too_big_for_one_gpu_is_split(self) -> None:
        rig = _Rig(_Registry(_Desc("big", 10000, split=True, gen_sec=0.1)), delay=0.1)
        (job,) = rig.run([("big", {"prompt": "x"})])
        assert job.state == P.JOB_DONE
        assert job.route["kind"] == "split" and job.route["devices"] == [0, 1]
        (backend, request) = rig.span_managers[(0, 1)].calls[0]
        assert backend == "big"
        # O worker vê as GPUs como 0..k-1 e recebe a quota de cada uma.
        assert request["gpu_ids"] == [0, 1]
        assert request["max_memory_mib"] == {"0": 5500, "1": 5500}
        assert rig.managers[0].calls == [] and rig.managers[1].calls == []
        assert rig.engine.counts["split"] == 1

    def test_split_evicts_what_the_member_gpus_had_loaded(self) -> None:
        rig = _Rig(_Registry(_Desc("big", 10000, split=True, gen_sec=0.05), _Desc("small", 1000)), delay=0.05)
        rig.managers[0].loaded.add("small")
        rig.run([("big", {})])
        assert rig.managers[0].evict_all_calls == 1 and rig.managers[0].loaded == set()

    def test_no_split_without_opt_in_fails_with_the_managers_error(self) -> None:
        """Sem rota possível o job não desaparece: corre e a admissão explica."""
        rig = _Rig(_Registry(_Desc("big", 10000)))
        refusal = {
            "status": P.STATUS_ERROR,
            "error": "VRAM insuficiente para big",
            "error_code": P.ERR_VRAM_INSUFFICIENT,
            "peak_mib": 10000,
        }
        rig.managers[0].result = refusal
        (job,) = rig.run([("big", {})])
        assert job.state == P.JOB_FAILED
        assert job.result["error_code"] == P.ERR_VRAM_INSUFFICIENT and "hint" in job.result
        assert rig.engine.counts["reject"] == 1

    def test_explicit_gpu_ids_are_honoured_and_rewritten(self) -> None:
        rig = _Rig(_Registry(_Desc("m", 4000, gen_sec=0.05)), delay=0.05)
        (job,) = rig.run([("m", {"gpu_ids": [1]})])
        assert job.route["target"] == "gpu1" and rig.managers[0].calls == []
        # A GPU física 1 é a GPU 0 do worker fixado nela.
        assert rig.managers[1].calls[0][1]["gpu_ids"] == [0]

    def test_rejected_job_fallback_respects_explicit_gpu(self) -> None:
        """Sem rota o job ainda corre — mas nunca numa GPU que o cliente não fixou."""
        rig = _Rig(_Registry(_Desc("big", 10000)))
        (job,) = rig.run([("big", {"gpu_ids": [1]})])
        assert job.state == P.JOB_DONE
        assert rig.managers[0].calls == []
        assert rig.managers[1].calls[0][1]["gpu_ids"] == [0]

    def test_rejected_pinned_pair_falls_back_to_a_span(self) -> None:
        rig = _Rig(_Registry(_Desc("big", 10000)))
        (job,) = rig.run([("big", {"gpu_ids": [0, 1]})])
        assert job.state == P.JOB_DONE
        assert job.route["kind"] == "split" and job.route["devices"] == [0, 1]
        assert rig.span_managers[(0, 1)].calls[0][1]["gpu_ids"] == [0, 1]

    def test_interactive_job_takes_the_free_gpu_ahead_of_batch(self) -> None:
        rig = _Rig(_Registry(_Desc("m", 4000, load_sec=0.01, gen_sec=0.2)), delay=0.2, gpus=1)
        batch = rig.queue.enqueue("m", {"n": "batch"}, priority=P.PRIORITY_BATCH)
        inter = rig.queue.enqueue("m", {"n": "inter"}, priority=P.PRIORITY_INTERACTIVE)
        rig.pool.start()
        try:
            assert batch.done_event.wait(5.0) and inter.done_event.wait(5.0)
        finally:
            rig.pool.stop()
        assert [c[1]["n"] for c in rig.managers[0].calls] == ["inter", "batch"]

    def test_lane_is_released_when_the_job_raises(self) -> None:
        rig = _Rig(_Registry(_Desc("m", 4000)), gpus=1)

        def boom(name: str, request: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("rebentou")

        rig.managers[0].generate = boom  # type: ignore[method-assign]
        (job,) = rig.run([("m", {})])
        assert job.state == P.JOB_FAILED
        assert rig.lanes.is_busy(rig.lanes.primary) is False

    def test_capacity_is_one_job_per_gpu(self) -> None:
        rig = _Rig(_Registry(_Desc("m", 4000)), gpus=3)
        assert rig.pool.routed and rig.pool.capacity() == 3

    def test_concurrent_claims_never_double_book_a_gpu(self) -> None:
        """Muitas threads de dispatch a disputar 2 GPUs: nunca 2 jobs na mesma."""
        rig = _Rig(_Registry(_Desc("m", 4000, load_sec=0.0, gen_sec=0.02)), delay=0.02)
        active: dict[int, int] = {0: 0, 1: 0}
        worst = [0]
        lock = threading.Lock()
        for index, mgr in enumerate(rig.managers):
            original = mgr.generate

            def wrapped(name: str, request: dict[str, Any], _i: int = index, _orig: Any = original) -> dict[str, Any]:
                with lock:
                    active[_i] += 1
                    worst[0] = max(worst[0], active[_i])
                try:
                    return _orig(name, request)
                finally:
                    with lock:
                        active[_i] -= 1

            mgr.generate = wrapped  # type: ignore[method-assign]
        rig.run([("m", {"i": i}) for i in range(24)])
        assert worst[0] == 1
