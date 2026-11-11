"""Router: escolha de rota por custo estimado (GPU local, réplica, split, rede)."""

from __future__ import annotations

import pytest

from vramd.router import (
    ACTION_REJECT,
    ACTION_RUN,
    ACTION_WAIT,
    ROUTE_LOCAL,
    ROUTE_REMOTE,
    ROUTE_SPLIT,
    JobProfile,
    LaneView,
    PeerView,
    RoutePolicy,
    Tier,
    plan_routes,
    plan_split,
)

POLICY = RoutePolicy(replica_margin_sec=5.0, remote_penalty_sec=2.0, split_penalty_sec=5.0)


def job(**over) -> JobProfile:
    base = {"backend": "m", "peak_mib": 5000, "headroom_mib": 1000, "load_sec": 40.0, "gen_sec": 20.0}
    base.update(over)
    return JobProfile(**base)


def lane(idx: int, *, free: int | None = 6000, total: int | None = 6000, **over) -> LaneView:
    return LaneView(lane_id=f"gpu{idx}", devices=(idx,), free_mib=free, total_mib=total, **over)


def decide(profile, lanes, peers=()):
    return plan_routes(profile, lanes, peers, policy=POLICY)


class TestSingleGpu:
    def test_cold_load_on_free_gpu(self) -> None:
        d = decide(job(), [lane(0)])
        assert d.action == ACTION_RUN
        assert d.best.target == "gpu0" and not d.best.hot
        assert d.best.total_sec == 60.0  # load 40 + gen 20

    def test_hot_skips_load_and_needs_only_headroom(self) -> None:
        d = decide(job(), [lane(0, free=1200, hot=True, loaded=True)])
        assert d.action == ACTION_RUN and d.best.hot
        assert d.best.total_sec == 20.0

    def test_peak_larger_than_total_is_rejected_with_reason(self) -> None:
        d = decide(job(peak_mib=8000), [lane(0)])
        assert d.action == ACTION_REJECT and d.best is None
        assert "8000 MiB > VRAM total 6000" in d.explanation

    def test_eviction_makes_room_and_is_charged(self) -> None:
        d = decide(job(), [lane(0, free=2000, evictable_mib=4000, evict_cost_sec=30.0)])
        assert d.action == ACTION_RUN
        # Défice 3000 de 4000 evictáveis → 75% do custo de recarga.
        assert d.best.evict_sec == pytest.approx(22.5)

    def test_not_enough_even_after_eviction(self) -> None:
        d = decide(job(), [lane(0, free=2000, evictable_mib=1000)])
        assert d.action == ACTION_REJECT
        assert "livre 2000 + evictável 1000" in d.explanation

    def test_unknown_free_does_not_block(self) -> None:
        """Sem leitura de VRAM (NVML em baixo) a admissão decide — o router não inventa."""
        assert decide(job(), [lane(0, free=None, total=None)]).action == ACTION_RUN

    def test_busy_lane_means_wait(self) -> None:
        d = decide(job(), [lane(0, hot=True, loaded=True, busy_sec=12.0)])
        assert d.action == ACTION_WAIT
        assert d.best.wait_sec == 12.0 and d.best.total_sec == 32.0


class TestReplicas:
    """O mesmo modelo em várias GPUs — só quando a fila amortiza o load."""

    def test_single_job_waits_for_the_hot_gpu(self) -> None:
        """Faltam 10s à GPU quente; carregar uma réplica custaria 40s."""
        lanes = [lane(0, hot=True, loaded=True, busy_sec=10.0), lane(1)]
        d = decide(job(), lanes)
        assert d.action == ACTION_WAIT and d.best.target == "gpu0"

    def test_backlog_spawns_a_replica(self) -> None:
        """Batch: 3 jobs à frente na GPU quente (70s) > load numa GPU livre (40s+margem)."""
        lanes = [lane(0, hot=True, loaded=True, busy_sec=70.0), lane(1)]
        d = decide(job(), lanes)
        assert d.action == ACTION_RUN
        assert d.best.target == "gpu1" and d.best.replica
        assert d.best.total_sec == 65.0  # 40 load + 20 gen + 5 margem

    def test_margin_prevents_marginal_replica(self) -> None:
        """Poupar 2s não justifica um load e uma GPU ocupada."""
        lanes = [lane(0, hot=True, loaded=True, busy_sec=42.0), lane(1)]
        assert decide(job(), lanes).best.target == "gpu0"

    def test_two_hot_replicas_pick_the_idle_one(self) -> None:
        lanes = [lane(0, hot=True, loaded=True, busy_sec=15.0), lane(1, hot=True, loaded=True)]
        d = decide(job(), lanes)
        assert d.action == ACTION_RUN and d.best.target == "gpu1" and not d.best.replica

    def test_max_replicas_caps_fanout(self) -> None:
        lanes = [lane(0, hot=True, loaded=True, busy_sec=500.0), lane(1)]
        d = decide(job(max_replicas=1), lanes)
        assert d.action == ACTION_WAIT and d.best.target == "gpu0"
        blocked = next(r for r in d.routes if r.target == "gpu1")
        assert not blocked.feasible and "teto de réplicas" in blocked.reason

    def test_first_load_is_not_a_replica(self) -> None:
        d = decide(job(), [lane(0), lane(1)])
        assert d.best.target == "gpu0" and not d.best.replica and d.best.penalty_sec == 0.0

    def test_prefers_gpu_that_avoids_eviction(self) -> None:
        lanes = [lane(0, free=1000, evictable_mib=5000, evict_cost_sec=30.0), lane(1)]
        assert decide(job(), lanes).best.target == "gpu1"

    def test_heterogeneous_gpus_route_to_the_one_that_fits(self) -> None:
        lanes = [lane(0, free=6000, total=6000), lane(1, free=24000, total=24000)]
        d = decide(job(peak_mib=11000), lanes)
        assert d.action == ACTION_RUN and d.best.target == "gpu1"


class TestSplit:
    """Modelo dividido por várias GPUs — só quando não cabe numa."""

    def test_never_split_when_one_gpu_fits(self) -> None:
        d = decide(job(allow_split=True), [lane(0), lane(1)])
        assert d.best.kind == ROUTE_LOCAL
        assert all(r.kind != ROUTE_SPLIT for r in d.routes)

    def test_split_when_no_single_gpu_fits(self) -> None:
        d = decide(job(peak_mib=10000, allow_split=True), [lane(0), lane(1)])
        assert d.action == ACTION_RUN and d.best.kind == ROUTE_SPLIT
        assert d.best.target == "split:gpu0+gpu1" and d.best.devices == (0, 1)
        # 10000 x 1.10 de overhead, repartido por igual.
        assert d.best.shares_mib == {0: 5500, 1: 5500}
        assert d.best.gen_sec == pytest.approx(26.0)  # 20 x 1.3

    def test_split_requires_backend_opt_in(self) -> None:
        d = decide(job(peak_mib=10000), [lane(0), lane(1)])
        assert d.action == ACTION_REJECT

    def test_split_waits_for_every_member(self) -> None:
        lanes = [lane(0, busy_sec=30.0), lane(1)]
        d = decide(job(peak_mib=10000, allow_split=True), lanes)
        assert d.action == ACTION_WAIT and d.best.wait_sec == 30.0

    def test_split_uses_fewest_gpus(self) -> None:
        lanes = [lane(0), lane(1, free=24000, total=24000), lane(2)]
        plan = plan_split(job(peak_mib=26000), lanes)
        assert plan is not None
        members, shares = plan
        assert [m.lane_id for m in members] == ["gpu1", "gpu0"]
        assert sum(shares.values()) <= 28600 and shares[1] > shares[0]

    def test_split_infeasible_when_sum_is_short(self) -> None:
        d = decide(job(peak_mib=20000, allow_split=True), [lane(0), lane(1)])
        assert d.action == ACTION_REJECT
        assert "VRAM somada" in d.explanation

    def test_split_counts_evictable_vram(self) -> None:
        lanes = [lane(0, free=1000, evictable_mib=5000, evict_cost_sec=20.0), lane(1)]
        d = decide(job(peak_mib=10000, allow_split=True), lanes)
        assert d.best.kind == ROUTE_SPLIT and d.best.evict_sec > 0

    def test_split_prefers_free_gpus_over_a_busy_big_one(self) -> None:
        """Duas GPUs pequenas livres ganham a uma maior ocupada 100 s."""
        lanes = [lane(0), lane(1), lane(2, free=9000, total=9000, busy_sec=100.0)]
        d = decide(job(peak_mib=10000, allow_split=True), lanes)
        assert d.action == ACTION_RUN
        assert d.best.kind == ROUTE_SPLIT
        assert set(d.best.devices) == {0, 1} and d.best.wait_sec == 0.0

    def test_split_does_not_use_a_tier_banned_on_its_gpus(self) -> None:
        tiers = (
            Tier(label="hq", min_vram_mib=0, rank=2, peak_mib=10000),
            Tier(label="low", min_vram_mib=0, rank=1, peak_mib=9000),
        )
        lanes = [lane(0, tier_bans=frozenset({2})), lane(1)]
        d = decide(job(peak_mib=10000, allow_split=True, tiers=tiers), lanes)
        assert d.best.kind == ROUTE_SPLIT
        assert d.best.rank == 1 and d.best.tier == "low"


class TestRemote:
    """Outra máquina na rede — latência + transferência + fila do outro lado."""

    def test_local_wins_a_tie(self) -> None:
        d = decide(job(), [lane(0)], [PeerView(name="box")])
        assert d.best.kind == ROUTE_LOCAL

    def test_busy_local_goes_remote(self) -> None:
        d = decide(job(), [lane(0, hot=True, loaded=True, busy_sec=300.0)], [PeerView(name="box", hot=True)])
        assert d.action == ACTION_RUN and d.best.target == "peer:box"

    def test_model_too_big_locally_goes_remote(self) -> None:
        d = decide(job(peak_mib=20000), [lane(0)], [PeerView(name="box", fits=True)])
        assert d.action == ACTION_RUN and d.best.kind == ROUTE_REMOTE

    def test_big_files_on_slow_link_stay_local(self) -> None:
        """2 GiB a 10 MiB/s = 205s de rede: mais vale esperar 60s pela GPU local."""
        peer = PeerView(name="box", hot=True, bandwidth_mib_s=10.0)
        lanes = [lane(0, hot=True, loaded=True, busy_sec=60.0)]
        d = decide(job(input_bytes=2 * 1024**3), lanes, [peer])
        assert d.action == ACTION_WAIT and d.best.kind == ROUTE_LOCAL
        remote = next(r for r in d.routes if r.kind == ROUTE_REMOTE)
        assert remote.transfer_sec == pytest.approx(204.8)

    def test_same_files_on_fast_link_go_remote(self) -> None:
        peer = PeerView(name="box", hot=True, bandwidth_mib_s=1000.0)
        lanes = [lane(0, hot=True, loaded=True, busy_sec=60.0)]
        assert decide(job(input_bytes=2 * 1024**3), lanes, [peer]).best.kind == ROUTE_REMOTE

    def test_peer_queue_is_part_of_the_cost(self) -> None:
        peers = [PeerView(name="busy", hot=True, wait_sec=200.0), PeerView(name="idle")]
        d = decide(job(peak_mib=20000), [lane(0)], peers)
        assert d.best.target == "peer:idle"  # frio mas livre ganha ao quente com fila

    def test_peer_estimates_override_local_ones(self) -> None:
        d = decide(job(peak_mib=20000), [lane(0)], [PeerView(name="a100", hot=True, gen_sec=4.0)])
        assert d.best.gen_sec == 4.0

    @pytest.mark.parametrize(
        ("peer", "reason"),
        [
            (PeerView(name="p", healthy=False, reason="timeout"), "timeout"),
            (PeerView(name="p", has_backend=False), "não tem este backend"),
            (PeerView(name="p", fits=False), "não cabe"),
        ],
    )
    def test_unusable_peers_are_explained(self, peer: PeerView, reason: str) -> None:
        d = decide(job(peak_mib=20000), [lane(0)], [peer])
        assert d.action == ACTION_REJECT and reason in d.explanation

    def test_forwarded_job_is_never_forwarded_again(self) -> None:
        """Dois nós que se têm como peers não podem fazer ping-pong com um job."""
        d = decide(job(peak_mib=20000, local_only=True), [lane(0)], [PeerView(name="box")])
        assert d.action == ACTION_REJECT and "já foi encaminhado" in d.explanation

    def test_backend_can_opt_out_of_remote(self) -> None:
        lanes = [lane(0, hot=True, loaded=True, busy_sec=900.0)]
        d = decide(job(allow_remote=False), lanes, [PeerView(name="box", hot=True)])
        assert d.action == ACTION_WAIT and d.best.kind == ROUTE_LOCAL

    def test_no_free_forward_slot_is_a_wait(self) -> None:
        d = decide(job(peak_mib=20000), [lane(0)], [PeerView(name="box", slot_free=False)])
        assert d.action == ACTION_WAIT


class TestPinning:
    def test_explicit_gpu_is_respected(self) -> None:
        d = decide(job(pinned_devices=(1,)), [lane(0), lane(1, busy_sec=50.0)], [PeerView(name="box")])
        assert d.action == ACTION_WAIT and d.best.target == "gpu1"

    def test_pinned_pair_becomes_a_split(self) -> None:
        d = decide(job(pinned_devices=(0, 1), allow_split=True, peak_mib=4000), [lane(0), lane(1), lane(2)])
        assert d.best.kind == ROUTE_SPLIT and d.best.devices == (0, 1)


class TestDecisionShape:
    def test_routes_sorted_feasible_first_and_serializable(self) -> None:
        d = decide(job(), [lane(0, busy_sec=5.0), lane(1, total=2000, free=2000)], [PeerView(name="box")])
        data = d.to_dict()
        assert [r["feasible"] for r in data["routes"]] == [True, True, False]
        assert data["best"]["target"] == data["routes"][0]["target"]
        assert "total_sec" in data["routes"][0] and "reason" in data["routes"][-1]

    def test_explanation_names_the_runner_up(self) -> None:
        d = decide(job(), [lane(0), lane(1, hot=True, loaded=True, busy_sec=100.0)])
        assert "gpu0" in d.explanation and "alternativa gpu1" in d.explanation

    def test_no_resources_at_all(self) -> None:
        d = decide(job(), [])
        assert d.action == ACTION_REJECT and "sem GPUs nem peers" in d.explanation

    def test_policy_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("VRAMD_ROUTE_REPLICA_MARGIN_SEC", "0")
        monkeypatch.setenv("VRAMD_ROUTE_REMOTE_PENALTY_SEC", "lixo")
        policy = RoutePolicy.from_env()
        assert policy.replica_margin_sec == 0.0 and policy.remote_penalty_sec == 2.0
