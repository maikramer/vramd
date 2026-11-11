"""hw_tiers (qualidade pela GPU de destino), criticidade e ``topology``."""

from __future__ import annotations

import json
import time
from typing import Any, ClassVar

import pytest
from click.testing import CliRunner

from vramd import cli as cli_mod
from vramd import protocol as P
from vramd.dispatcher import WorkerPool
from vramd.job_queue import JobQueue
from vramd.lanes import Lane, LaneSet
from vramd.placement import HwTierError, RouteEngine, apply_hw_tiers, parse_hw_tiers
from vramd.router import (
    ACTION_RUN,
    ACTION_WAIT,
    JobProfile,
    LaneView,
    PeerView,
    RoutePolicy,
    Tier,
    normalize_criticality,
    plan_routes,
    select_tier,
)
from vramd.tier_health import TierHealth

POLICY = RoutePolicy(
    replica_margin_sec=0.0,
    remote_penalty_sec=0.0,
    split_penalty_sec=0.0,
    quality_high_sec=600.0,
    quality_normal_sec=0.0,
    reserve_low_sec=30.0,
    reserve_low_pending_sec=300.0,
)

# O perfil hw-auto do text3d para dois tamanhos de GPU: fp16 com mais steps
# na de 24 GB, int4 na de 6 GB.
TIERS = (
    Tier(label="int4", min_vram_mib=0, rank=0, peak_mib=4500, headroom_mib=900, gen_sec=60.0),
    Tier(label="hq", min_vram_mib=10240, rank=1, peak_mib=11000, headroom_mib=2000, gen_sec=90.0),
)


def lane(idx: int, total: int, **over: Any) -> LaneView:
    return LaneView(lane_id=f"gpu{idx}", devices=(idx,), free_mib=total, total_mib=total, **over)


def job(**over: Any) -> JobProfile:
    base: dict[str, Any] = {"backend": "text3d", "peak_mib": 4500, "load_sec": 30.0, "gen_sec": 60.0, "tiers": TIERS}
    base.update(over)
    return JobProfile(**base)


SMALL, BIG = lane(0, 6144), lane(1, 24576)


class TestSelectTier:
    def test_picks_best_tier_the_gpu_satisfies(self) -> None:
        assert select_tier(TIERS, 6144).label == "int4"
        assert select_tier(TIERS, 24576).label == "hq"
        assert select_tier(TIERS, 10240).label == "hq"

    def test_unknown_vram_gets_the_modest_tier(self) -> None:
        """Sem leitura de VRAM só o tier mais modesto é seguro de tentar."""
        assert select_tier(TIERS, None).label == "int4"
        assert select_tier((TIERS[1],), 2048).label == "hq"  # nenhum cabe: o único que há


class TestTiersInRoutes:
    def test_each_gpu_routes_with_its_own_tier(self) -> None:
        d = plan_routes(job(), [SMALL, BIG], policy=POLICY)
        by = {r.target: r for r in d.routes}
        assert (by["gpu0"].tier, by["gpu0"].gen_sec) == ("int4", 60.0)
        assert (by["gpu1"].tier, by["gpu1"].gen_sec) == ("hq", 90.0)

    def test_normal_asset_takes_the_fastest_route(self) -> None:
        """Normal: só o tempo conta — o int4 acaba antes, vai para a pequena."""
        d = plan_routes(job(), [SMALL, BIG], policy=POLICY)
        assert d.best.target == "gpu0" and d.best.quality_sec == 0

    def test_critical_asset_goes_to_the_big_gpu(self) -> None:
        d = plan_routes(job(criticality="high"), [SMALL, BIG], policy=POLICY)
        assert d.best.target == "gpu1" and d.best.tier == "hq"
        small = next(r for r in d.routes if r.target == "gpu0")
        assert small.quality_sec == 600.0
        assert "preferida pela qualidade" in d.explanation

    def test_critical_asset_waits_for_the_big_gpu(self) -> None:
        """Esperar 5 min pela GPU grande vale mais que um tier abaixo já."""
        busy_big = lane(1, 24576, busy_sec=300.0)
        d = plan_routes(job(criticality="high"), [SMALL, busy_big], policy=POLICY)
        assert d.action == ACTION_WAIT and d.best.target == "gpu1"

    def test_critical_does_not_wait_forever(self) -> None:
        """Mais de 600s por tier: aceita a GPU pequena."""
        busy_big = lane(1, 24576, busy_sec=2000.0)
        d = plan_routes(job(criticality="high"), [SMALL, busy_big], policy=POLICY)
        assert d.action == ACTION_RUN and d.best.target == "gpu0"

    def test_filler_avoids_the_big_gpu_even_if_slightly_slower(self) -> None:
        # Mesmo tempo nos dois tiers: o filler fica na pequena para não ocupar a grande.
        tiers = (TIERS[0], Tier("hq", 10240, 1, 11000, 2000, 60.0))
        d = plan_routes(job(tiers=tiers, criticality="low"), [SMALL, BIG], policy=POLICY)
        assert d.best.target == "gpu0"
        big = next(r for r in d.routes if r.target == "gpu1")
        assert big.quality_sec == 30.0

    def test_filler_waits_harder_when_criticals_are_queued(self) -> None:
        tiers = (TIERS[0], Tier("hq", 10240, 1, 11000, 2000, 60.0))
        small_busy = lane(0, 6144, busy_sec=100.0)
        relaxed = plan_routes(job(tiers=tiers, criticality="low"), [small_busy, BIG], policy=POLICY)
        assert relaxed.best.target == "gpu1"  # 100s de espera > 30s de reserva
        guarded = plan_routes(
            job(tiers=tiers, criticality="low", reserve_for_critical=True), [small_busy, BIG], policy=POLICY
        )
        assert guarded.action == ACTION_WAIT and guarded.best.target == "gpu0"

    def test_same_sized_gpus_have_no_quality_term(self) -> None:
        d = plan_routes(job(criticality="high"), [lane(0, 6144), lane(1, 6144, busy_sec=10.0)], policy=POLICY)
        assert d.best.target == "gpu0" and all(r.quality_sec == 0 for r in d.routes)

    def test_big_tier_peak_is_checked_against_the_big_gpu(self) -> None:
        """O tier hq tem pico de 11 GB: numa GPU de 24 GB com 8 GB livres não entra."""
        crowded = lane(1, 24576)
        crowded = LaneView(**{**crowded.__dict__, "free_mib": 8000})
        d = plan_routes(job(), [crowded], policy=POLICY)
        assert d.action == "reject" and "precisa de 11000" in d.explanation

    def test_split_uses_the_tier_of_the_summed_vram(self) -> None:
        tiers = (Tier("int4", 0, 0, 7000, 900, 60.0), Tier("hq", 10240, 1, 11000, 2000, 90.0))
        two_small = [lane(0, 6000), lane(1, 6000)]
        d = plan_routes(job(tiers=tiers, allow_split=True, peak_mib=7000), two_small, policy=POLICY)
        assert d.best.kind == "split"
        # 12000 MiB somados chegam ao tier hq, mas 11000 x 1.1 = 12100 não cabe: desce para int4.
        assert d.best.tier == "int4"

    def test_peer_tier_comes_from_its_gpus(self) -> None:
        peer = PeerView(name="box", gpu_totals_mib=(24576,), hot=True)
        d = plan_routes(job(criticality="high"), [lane(0, 6144, busy_sec=50.0)], [peer], policy=POLICY)
        assert d.best.target == "peer:box" and d.best.tier == "hq"

    def test_without_tiers_nothing_changes(self) -> None:
        d = plan_routes(job(tiers=(), criticality="high"), [SMALL, BIG], policy=POLICY)
        assert d.best.target == "gpu0" and d.best.tier is None


class TestCriticality:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("hero", "high"),
            ("CRITICAL", "high"),
            ("filler", "low"),
            ("background", "low"),
            (None, "normal"),
            ("x", "normal"),
        ],
    )
    def test_aliases(self, raw: Any, expected: str) -> None:
        assert normalize_criticality(raw) == expected


class TestParseAndApply:
    RAW: ClassVar[list[dict[str, Any]]] = [
        {"min_vram_mib": 10240, "label": "hq", "set": {"sdnq_preset": "none", "memory_efficient": False}},
        {"min_vram_mib": 0, "label": "int4", "set": {"sdnq_preset": "sdnq-int4", "memory_efficient": True}},
    ]

    def test_default_rank_follows_vram(self) -> None:
        tiers = parse_hw_tiers(self.RAW)
        assert [(t.label, t.rank) for t in tiers] == [("int4", 0), ("hq", 1)]

    def test_apply_sets_tier_values_and_strips_routing_keys(self) -> None:
        req = {"prompt": "x", "sdnq_preset": "sdnq-int4", "hw_tiers": self.RAW, "criticality": "high"}
        out, tier = apply_hw_tiers(req, 24576)
        assert tier.label == "hq"
        assert out == {"prompt": "x", "sdnq_preset": "none", "memory_efficient": False}
        small, tier_small = apply_hw_tiers(req, 6144)
        assert tier_small.label == "int4" and small["sdnq_preset"] == "sdnq-int4"

    def test_none_in_set_removes_the_key(self) -> None:
        raw = [{"min_vram_mib": 0, "set": {}}, {"min_vram_mib": 10240, "set": {"sdnq_preset": None}}]
        out, _ = apply_hw_tiers({"sdnq_preset": "sdnq-int4", "hw_tiers": raw}, 24576)
        assert out == {}
        out_small, _ = apply_hw_tiers({"sdnq_preset": "sdnq-int4", "hw_tiers": raw}, 6144)
        assert out_small == {"sdnq_preset": "sdnq-int4"}

    def test_apply_without_tiers_only_strips(self) -> None:
        out, tier = apply_hw_tiers({"prompt": "x", "criticality": "low"}, 6000)
        assert tier is None and out == {"prompt": "x"}

    @pytest.mark.parametrize(
        "raw",
        [
            "hq",
            [1, 2],
            [{"min_vram_mib": "muito"}],
            [{"min_vram_mib": -1}],
            [{"set": "x"}],
            [{"set": {"cmd": "shutdown"}}],
            [{"set": {"_abort": True}}],
            [{"gen_factor": 0}],
            [{}] * 17,
        ],
    )
    def test_malformed(self, raw: Any) -> None:
        with pytest.raises(HwTierError):
            parse_hw_tiers(raw)


class _Desc:
    def __init__(self, name: str, vram_mib: int, **placement: Any) -> None:
        self.name = name
        self.vram_mib = vram_mib
        self.unload_frees_vram = True
        self.placement = placement


class _Registry:
    def __init__(self, *descs: _Desc) -> None:
        self._d = {d.name: d for d in descs}

    @property
    def names(self) -> list[str]:
        return list(self._d)

    def descriptor(self, name: str) -> _Desc:
        return self._d[name]


class _Manager:
    """Gestor falso cujo pico depende do quant pedido, como o real."""

    def __init__(self, delay: float = 0.05) -> None:
        self.loaded: set[str] = set()
        self.calls: list[dict[str, Any]] = []
        self.delay = delay
        self.stats = None

    def loaded_names(self) -> list[str]:
        return sorted(self.loaded)

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def shape_matches_loaded(self, name: str, request: Any = None) -> bool:
        return name in self.loaded

    def evict_all(self) -> int:
        n = len(self.loaded)
        self.loaded.clear()
        return n

    def evict(self, name: str) -> bool:
        return bool(self.loaded.discard(name))

    def resolve_peak_params(self, name: str, req: dict[str, Any]) -> tuple[str, bool, bool, bool]:
        return str(req.get("sdnq_preset") or "none"), False, False, False

    def peak_vram_mib(self, name: str, *, quant_mode: str = "none", **_: Any) -> int:
        return 4500 if quant_mode.startswith("sdnq") else 11000

    def activation_headroom_mib(self, name: str, **_: Any) -> int:
        return 900

    def generate(self, name: str, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict((k, v) for k, v in request.items() if not k.startswith("_")))
        self.loaded.add(name)
        time.sleep(self.delay)
        return {"status": P.STATUS_OK, "output": request.get("sdnq_preset")}


def _rig(totals: tuple[int, int] = (6144, 24576)) -> tuple[WorkerPool, JobQueue, list[_Manager], RouteEngine]:
    managers = [_Manager(), _Manager()]
    lanes = LaneSet(
        [
            Lane(
                lane_id=f"gpu{i}",
                devices=(i,),
                manager=m,
                pinned=True,
                query_free_mib=lambda t=totals[i]: t,
                query_total_mib=lambda t=totals[i]: t,
            )
            for i, m in enumerate(managers)
        ],
        registry=_Registry(_Desc("text3d", 4500, load_sec=0.01, gen_sec=0.05)),
    )
    engine = RouteEngine(_Registry(_Desc("text3d", 4500, load_sec=0.01, gen_sec=0.05)), lanes, None, policy=POLICY)
    queue = JobQueue(max_depth=32)
    return WorkerPool(queue, managers[0], engine=engine), queue, managers, engine


HW = [
    {"min_vram_mib": 0, "label": "int4", "set": {"sdnq_preset": "sdnq-int4"}},
    {"min_vram_mib": 10240, "label": "hq", "set": {"sdnq_preset": "none"}, "gen_factor": 1.5},
]


class TestEndToEndDispatch:
    def _run(self, pool: WorkerPool, queue: JobQueue, requests: list[dict[str, Any]]) -> list[Any]:
        jobs = [queue.enqueue("text3d", r) for r in requests]
        pool.start()
        try:
            for j in jobs:
                assert j.done_event.wait(5.0)
        finally:
            pool.stop()
        return jobs

    def test_hero_lands_on_big_gpu_in_fp16_and_filler_on_small_in_int4(self) -> None:
        pool, queue, managers, _ = _rig()
        hero, filler = self._run(
            pool,
            queue,
            [
                {"prompt": "boss", "hw_tiers": HW, "criticality": "hero", "sdnq_preset": "sdnq-int4"},
                {"prompt": "rock", "hw_tiers": HW, "criticality": "filler", "sdnq_preset": "sdnq-int4"},
            ],
        )
        assert hero.route["target"] == "gpu1" and hero.route["tier"] == "hq" and hero.route["criticality"] == "high"
        assert filler.route["target"] == "gpu0" and filler.route["tier"] == "int4"
        # O worker recebeu o quant do tier da SUA GPU, sem os campos de routing.
        assert managers[1].calls[0] == {"prompt": "boss", "sdnq_preset": "none"}
        assert managers[0].calls[0] == {"prompt": "rock", "sdnq_preset": "sdnq-int4"}

    def test_hot_runs_teach_the_tier_timing(self) -> None:
        pool, queue, _managers, engine = _rig()
        self._run(pool, queue, [{"prompt": str(i), "hw_tiers": HW, "criticality": "hero"} for i in range(3)])
        assert (("text3d", 1) in engine._tier_gen) and engine._tier_gen[("text3d", 1)] > 0

    def test_classic_single_gpu_path_resolves_tiers_by_its_gpu(self, monkeypatch) -> None:
        """Um portátil submete a um servidor de 24 GB: vale a placa do servidor."""
        manager = _Manager()
        queue = JobQueue(max_depth=8)
        pool = WorkerPool(queue, manager)
        monkeypatch.setattr(pool, "_total_mib", lambda: 24576)
        (j,) = self._run(pool, queue, [{"prompt": "x", "hw_tiers": HW, "criticality": "low"}])
        assert j.state == P.JOB_DONE
        assert manager.calls[0] == {"prompt": "x", "sdnq_preset": "none"}


class TestServer:
    @pytest.fixture
    def srv(self, tmp_path):
        from vramd.registry import BackendDescriptor, Registry
        from vramd.server import VramdServer

        registry = Registry(
            descriptors={"text3d": BackendDescriptor(name="text3d", adapter="_x", vram_mib=4500, priority=10)}
        )
        return VramdServer(registry=registry, socket_path=tmp_path / "t.sock", query_free_mib=lambda: 6000)

    def test_malformed_hw_tiers_rejected_at_enqueue(self, srv) -> None:
        resp = srv._dispatch({"cmd": "submit", "backend": "text3d", "hw_tiers": "hq"})
        assert resp["error_code"] == P.ERR_INVALID_REQUEST and "hw_tiers" in resp["error"]
        assert srv.queue.depth == 0

    def test_topology(self, srv) -> None:
        resp = srv._dispatch({"cmd": P.CMD_TOPOLOGY})
        assert resp["status"] == P.STATUS_OK
        assert resp["capacity"] == 1 and resp["routed"] is False
        assert [lane["lane"] for lane in resp["lanes"]] == ["gpu0"]
        assert {"routing", "hw_tiers", "criticality"} <= set(resp["features"])

    def test_route_shows_tiers_and_criticality(self, srv, monkeypatch) -> None:
        resp = srv._dispatch({"cmd": P.CMD_ROUTE, "backend": "text3d", "hw_tiers": HW, "criticality": "hero"})
        assert resp["profile"]["criticality"] == "high"
        assert [t["label"] for t in resp["profile"]["tiers"]] == ["int4", "hq"]
        monkeypatch.setattr(cli_mod, "_send", lambda request, *, timeout=30.0: resp)
        res = CliRunner().invoke(cli_mod.cli, ["route", "text3d", "-p", "criticality=hero"], env={"COLUMNS": "220"})
        assert res.exit_code == 0, res.output
        assert "criticidade high" in res.output and "Tier" in res.output
        assert json.loads(json.dumps(resp))["status"] == "ok"


class TestTierHealthCooldown:
    def test_negative_base_cooldown_does_not_ban(self) -> None:
        """Cooldown negativo não podia deixar o tier elegível com strikes a subir."""
        now = [1000.0]
        health = TierHealth(clock=lambda: now[0], base_cooldown_sec=-10)
        assert health.record_oom("alpha", 0, 12288) == 0.0
        assert health.banned_ranks("alpha", 12288) == frozenset()
        assert health.status()[0]["suspended"] is False

    def test_non_finite_env_cooldown_is_zero(self, monkeypatch) -> None:
        monkeypatch.setenv("VRAMD_TIER_COOLDOWN_SEC", "nan")
        health = TierHealth(clock=lambda: 1.0)
        assert health.record_oom("alpha", 0, 12288) == 0.0
        assert health.banned_ranks("alpha", 12288) == frozenset()

    def test_positive_cooldown_expires(self) -> None:
        now = [1000.0]
        health = TierHealth(clock=lambda: now[0], base_cooldown_sec=30)
        assert health.record_oom("alpha", 0, 12288) == 30
        assert health.banned_ranks("alpha", 12288) == {0}
        now[0] += 31
        assert health.banned_ranks("alpha", 12288) == frozenset()
