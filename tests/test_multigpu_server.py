"""Supervisor com várias GPUs: lanes no status, comando ``route``, comandos por lane, CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from vramd import cli as cli_mod
from vramd import cli_ops
from vramd import protocol as P
from vramd.cli import cli
from vramd.lanes import KIND_SPAN, Lane, lane_id_for
from vramd.metrics import render_prometheus
from vramd.registry import BackendDescriptor, Registry
from vramd.server import VramdServer
from vramd.validate import LEVEL_ERROR, LEVEL_WARNING, validate_config


class _LaneManager:
    """Gestor de uma lane extra, sem GPU."""

    def __init__(self) -> None:
        self.loaded: set[str] = set()
        self.stats: Any = None
        self.zeroed = 0
        self.respawned: list[str] = []
        self._subprocess_pool = None

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
        count = len(self.loaded)
        self.loaded.clear()
        return count

    def zero_vram(self) -> dict[str, Any]:
        self.zeroed += 1
        self.loaded.clear()
        return {"workers_killed": 1}

    def respawn(self, name: str, *, lazy: bool = True) -> dict[str, Any]:
        self.respawned.append(name)
        return {"backend": name, "respawned": True}

    def respawn_all(self, *, lazy: bool = True) -> list[dict[str, Any]]:
        return [self.respawn(n, lazy=lazy) for n in sorted(self.loaded)]


def _registry() -> Registry:
    descs = {
        "small": BackendDescriptor(name="small", adapter="_x", vram_mib=2000, priority=10),
        "big": BackendDescriptor(
            name="big", adapter="_x", vram_mib=10000, priority=10, placement={"split": True, "gen_sec": 8.0}
        ),
        "nosplit": BackendDescriptor(name="nosplit", adapter="_x", vram_mib=10000, priority=10),
    }
    return Registry(descriptors=descs)


@pytest.fixture
def rig(tmp_path: Path):
    managers: dict[tuple[int, ...], _LaneManager] = {}

    def lane_factory(devices: tuple[int, ...]) -> Lane:
        mgr = _LaneManager()
        managers[devices] = mgr
        size = 6000 * len(devices)
        return Lane(
            lane_id=lane_id_for(devices),
            devices=devices,
            manager=mgr,
            kind=KIND_SPAN if len(devices) > 1 else "device",
            pinned=True,
            query_free_mib=lambda: size,
            query_total_mib=lambda: size,
        )

    srv = VramdServer(
        registry=_registry(),
        socket_path=tmp_path / "ums-mgpu.sock",
        query_free_mib=lambda: 6000,
        clear_vram=lambda: None,
        gpu_devices=[0, 1],
        lane_factory=lane_factory,
    )
    srv.lanes.primary.query_total_mib = lambda: 6000
    return srv, managers


class TestServerLanes:
    def test_default_server_is_single_lane_and_unrouted(self, tmp_path: Path) -> None:
        """Sem ``gpu_devices`` nada muda: uma lane não fixada, despacho clássico."""
        srv = VramdServer(registry=_registry(), socket_path=tmp_path / "a.sock", query_free_mib=lambda: 6000)
        assert [lane.lane_id for lane in srv.lanes.device_lanes] == ["gpu0"]
        assert srv.lanes.primary.pinned is False
        assert srv.router.active is False and srv.workers.routed is False
        assert srv.workers.capacity() == srv.workers.max_inflight

    def test_two_gpus_make_two_pinned_lanes(self, rig) -> None:
        srv, managers = rig
        assert [lane.lane_id for lane in srv.lanes.device_lanes] == ["gpu0", "gpu1"]
        assert all(lane.pinned for lane in srv.lanes.device_lanes)
        assert srv.lanes.primary.manager is srv.manager
        assert srv.workers.routed and srv.workers.capacity() == 2
        # Estatísticas partilhadas entre réplicas (a lane real herda as do gestor primário).
        assert (1,) in managers

    def test_real_extra_lane_pins_its_workers(self, tmp_path: Path) -> None:
        """Sem factory: a lane da GPU 1 tem pool próprio com CUDA_VISIBLE_DEVICES=1."""
        srv = VramdServer(registry=_registry(), socket_path=tmp_path / "b.sock", gpu_devices=[0, 1])
        primary_env = srv.manager._subprocess_pool._worker_env("small")
        extra = srv.lanes.lane("gpu1")
        extra_env = extra.manager._subprocess_pool._worker_env("small")
        assert primary_env["CUDA_VISIBLE_DEVICES"] == "0" and extra_env["CUDA_VISIBLE_DEVICES"] == "1"
        assert extra.manager.stats is srv.manager.stats
        log = extra.manager._subprocess_pool._log_path_fn("small")
        assert log.name == "vramd-worker-small@gpu1.log"
        span = srv.lanes.span_for((0, 1))
        assert span.manager._subprocess_pool._worker_env("big")["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert ":" not in span.manager._subprocess_pool._log_path_fn("big").name  # nome válido no Windows

    def test_status_reports_lanes_and_routing(self, rig) -> None:
        srv, managers = rig
        managers[(1,)].loaded.add("small")
        status = srv._dispatch({"cmd": P.CMD_STATUS})
        lanes = {row["lane"]: row for row in status["lanes"]}
        assert set(lanes) == {"gpu0", "gpu1"}
        assert lanes["gpu1"]["loaded"] == ["small"] and lanes["gpu1"]["total_mib"] == 6000
        assert status["routing"]["active"] is True
        assert [g["index"] for g in status["routing"]["gpus"]] == [0, 1]
        assert status["routing"]["estimates"]["big"]["gen_sec"] == 8.0
        assert status["peers"] == []
        assert "small" in status["debug"]["loaded_backends"]

    def test_list_backends_sees_any_lane(self, rig) -> None:
        srv, managers = rig
        managers[(1,)].loaded.add("small")
        rows = {r["name"]: r for r in srv._dispatch({"cmd": P.CMD_LIST_BACKENDS})["backends"]}
        assert rows["small"]["loaded"] is True and rows["big"]["loaded"] is False

    def test_evict_and_zero_reach_every_lane(self, rig) -> None:
        srv, managers = rig
        managers[(1,)].loaded.add("small")
        resp = srv._dispatch({"cmd": P.CMD_RELEASE, "backend": "small"})
        assert resp["status"] == P.STATUS_OK and managers[(1,)].loaded == set()
        managers[(1,)].loaded.add("small")
        zero = srv._dispatch({"cmd": P.CMD_ZERO})
        assert zero["status"] == P.STATUS_OK
        assert managers[(1,)].zeroed == 1 and zero["lanes"]["gpu1"] == {"workers_killed": 1}
        assert zero["loaded_backends"] == []

    def test_respawn_reaches_every_lane(self, rig) -> None:
        srv, managers = rig
        resp = srv._dispatch({"cmd": P.CMD_RESPAWN, "backend": "small"})
        assert resp["status"] == P.STATUS_OK
        assert managers[(1,)].respawned == ["small"]
        assert any(r.get("lane") == "gpu1" for r in resp["results"])

    def test_cleanup_drains_extra_lanes(self, rig) -> None:
        srv, managers = rig
        managers[(1,)].loaded.add("small")
        srv._cleanup()
        assert managers[(1,)].loaded == set()


class TestRouteCommand:
    def test_small_model_runs_on_a_gpu(self, rig) -> None:
        srv, _ = rig
        resp = srv._dispatch({"cmd": P.CMD_ROUTE, "backend": "small"})
        assert resp["status"] == P.STATUS_OK and resp["routed"] is True
        assert resp["action"] == "run" and resp["best"]["kind"] == "local"
        assert {r["target"] for r in resp["routes"]} == {"gpu0", "gpu1"}
        assert resp["profile"]["peak_mib"] >= 2000

    def test_big_model_is_split_across_both(self, rig) -> None:
        srv, _ = rig
        resp = srv._dispatch({"cmd": P.CMD_ROUTE, "backend": "big"})
        assert resp["best"]["kind"] == "split" and resp["best"]["devices"] == [0, 1]
        assert set(resp["best"]["shares_mib"]) == {"0", "1"}
        assert "dividido" in resp["explanation"]

    def test_big_model_without_opt_in_is_rejected_with_reasons(self, rig) -> None:
        srv, _ = rig
        resp = srv._dispatch({"cmd": P.CMD_ROUTE, "backend": "nosplit"})
        assert resp["action"] == "reject" and resp["best"] is None
        assert "VRAM total" in resp["explanation"]

    def test_pinned_gpu(self, rig) -> None:
        srv, _ = rig
        resp = srv._dispatch({"cmd": P.CMD_ROUTE, "backend": "small", "gpu_ids": [1]})
        assert resp["best"]["target"] == "gpu1" and resp["profile"]["pinned_devices"] == [1]

    def test_route_is_a_dry_run(self, rig) -> None:
        srv, _ = rig
        srv._dispatch({"cmd": P.CMD_ROUTE, "backend": "small"})
        assert srv.queue.depth == 0 and not srv.lanes.is_busy(srv.lanes.primary)
        assert sum(srv.router.counts.values()) == 0

    def test_unknown_backend(self, rig) -> None:
        srv, _ = rig
        assert srv._dispatch({"cmd": P.CMD_ROUTE, "backend": "nope"})["error_code"] == P.ERR_BACKEND_UNKNOWN

    def test_forwarded_flag_from_route_metadata(self, rig) -> None:
        srv, _ = rig
        job = srv._enqueue_from_request({"cmd": "submit", "backend": "small", "_route": {"hops": 1, "origin": "a"}})
        assert job.forwarded is True and "_route" not in job.request
        plain = srv._enqueue_from_request({"cmd": "submit", "backend": "small"})
        assert plain.forwarded is False
        assert plain.to_public_dict()["route"] is None


class TestRoutingCli:
    def _patch(self, monkeypatch, responses: dict[str, Any]) -> list[dict]:
        sent: list[dict] = []

        def fake_send(request: dict, *, timeout: float = 30.0) -> Any:
            sent.append(request)
            return responses.get(request["cmd"])

        monkeypatch.setattr(cli_mod, "_send", fake_send)
        return sent

    def test_route_renders_table_and_marks_best(self, rig, monkeypatch) -> None:
        srv, _ = rig
        sent = self._patch(monkeypatch, {"route": srv._dispatch({"cmd": P.CMD_ROUTE, "backend": "big"})})
        res = CliRunner().invoke(cli, ["route", "big", "-p", "steps=4"], env={"COLUMNS": "200"})
        assert res.exit_code == 0, res.output
        assert "split:gpu0+gpu1" in res.output and "run" in res.output
        assert "VRAM total" in res.output  # as rotas inviáveis e porquê
        assert sent[0]["cmd"] == "route" and sent[0]["steps"] == 4 and "priority" not in sent[0]

    def test_route_json_and_down(self, rig, monkeypatch) -> None:
        srv, _ = rig
        self._patch(monkeypatch, {"route": srv._dispatch({"cmd": P.CMD_ROUTE, "backend": "small"})})
        res = CliRunner().invoke(cli, ["route", "small", "--json"])
        assert json.loads(res.output)["best"]["kind"] == "local"
        self._patch(monkeypatch, {})
        assert CliRunner().invoke(cli, ["route", "small"]).exit_code == 1

    def test_devices(self, rig, monkeypatch) -> None:
        srv, managers = rig
        managers[(1,)].loaded.add("small")
        self._patch(monkeypatch, {"status": srv._dispatch({"cmd": P.CMD_STATUS})})
        res = CliRunner().invoke(cli, ["devices"], env={"COLUMNS": "160"})
        assert res.exit_code == 0 and "gpu0" in res.output and "gpu1" in res.output and "small" in res.output
        data = json.loads(CliRunner().invoke(cli, ["devices", "--json"]).output)
        assert len(data["lanes"]) == 2

    def test_peers_empty_and_listed(self, monkeypatch) -> None:
        self._patch(monkeypatch, {"status": {"status": "status", "peers": []}})
        assert "Sem peers configurados" in CliRunner().invoke(cli, ["peers"]).output
        row = {
            "name": "box",
            "address": "10.0.0.2:8790",
            "healthy": False,
            "last_error": "connection refused [Errno 111]",
            "rtt_ms": None,
            "bandwidth_mib_s": 50.0,
            "bandwidth_measured": False,
            "inflight": 0,
            "queue_depth": 0,
            "loaded": [],
            "forwarded_total": 3,
            "failed_total": 1,
        }
        self._patch(monkeypatch, {"status": {"status": "status", "peers": [row]}})
        res = CliRunner().invoke(cli, ["peers"], env={"COLUMNS": "200"})
        assert res.exit_code == 0 and "box" in res.output and "connection refused [Errno 111]" in res.output
        assert "não medida" in res.output

    def test_commands_registered(self) -> None:
        for name in ("route", "devices", "peers"):
            assert name in cli.commands
        assert cli_ops.register_routing is not None


class TestMetricsAndValidate:
    def test_metrics_expose_lanes_peers_and_decisions(self, rig) -> None:
        srv, _ = rig
        status = srv._dispatch({"cmd": P.CMD_STATUS})
        status["peers"] = [
            {"name": "box", "healthy": True, "rtt_ms": 12.0, "bandwidth_mib_s": 80.0, "forwarded_total": 2}
        ]
        status["routing"]["decisions"] = {"local": 5, "replica": 2}
        text = render_prometheus(status)
        assert 'vramd_lane_total_mib{lane="gpu1",kind="device"} 6000' in text
        assert 'vramd_lane_busy{lane="gpu0",kind="device"} 0' in text
        assert 'vramd_peer_rtt_seconds{peer="box"} 0.012' in text
        assert 'vramd_peer_forwarded_total{peer="box"} 2' in text
        assert 'vramd_route_decisions_total{route="replica"} 2' in text

    def _validate(self, tmp_path: Path, **over: Any):
        entry = {
            "name": "m",
            "adapter": "pkg.adapter",
            "vram_mib": 100,
            "runtime": {"command": [sys.executable, "w.py"]},
            **over,
        }
        path = tmp_path / "b.yaml"
        path.write_text(yaml.safe_dump({"backends": [entry]}), encoding="utf-8")
        return validate_config(str(path), check_hooks=False)

    def test_valid_placement_block(self, tmp_path: Path) -> None:
        placement = {
            "split": True,
            "split_overhead": 0.15,
            "max_replicas": 2,
            "remote": False,
            "input_keys": ["mesh_path"],
            "output_keys": ["output"],
            "gen_sec": 40,
        }
        report = self._validate(tmp_path, placement=placement, load_keys=["gpu_ids", "max_memory_mib"])
        assert report.findings == []

    def test_placement_typos_and_types(self, tmp_path: Path) -> None:
        placement = {"spit": True, "max_replicas": -1, "split": "sim", "input_keys": "mesh_path", "gen_sec": 0}
        report = self._validate(tmp_path, placement=placement)
        errors = [f.message for f in report.findings if f.level == LEVEL_ERROR]
        assert any("max_replicas" in m for m in errors) and any("placement.split=" in m for m in errors)
        assert any("input_keys" in m for m in errors) and any("gen_sec" in m for m in errors)
        (typo,) = [f for f in report.findings if "placement.spit" in f.message]
        assert typo.level == LEVEL_WARNING and "split" in (typo.hint or "")

    def test_split_without_gpu_ids_in_load_keys_warns(self, tmp_path: Path) -> None:
        """O router dividia o modelo e o load nunca sabia em que GPUs."""
        report = self._validate(tmp_path, placement={"split": True}, load_keys=["device"])
        assert any("gpu_ids" in f.message for f in report.warnings)

    def test_malformed_peers_file_is_an_error(self, tmp_path: Path, monkeypatch) -> None:
        peers = tmp_path / "peers.yaml"
        peers.write_text("peers: [{name: box}]", encoding="utf-8")
        monkeypatch.setenv("VRAMD_PEERS_FILE", str(peers))
        monkeypatch.setenv("VRAMD_HOOKS_FILE", str(tmp_path / "sem-hooks.yaml"))
        entry = {"name": "m", "adapter": "a", "vram_mib": 1, "runtime": {"command": [sys.executable]}}
        path = tmp_path / "b.yaml"
        path.write_text(yaml.safe_dump({"backends": [entry]}), encoding="utf-8")
        report = validate_config(str(path))
        assert any("falta 'address'" in f.message for f in report.errors)

    def test_peer_without_token_warns(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("VRAMD_PEERS", "box=10.0.0.2:8790")
        monkeypatch.delenv("VRAMD_PEERS_TOKEN", raising=False)
        monkeypatch.delenv("VRAMD_REMOTE_TOKEN", raising=False)
        monkeypatch.setenv("VRAMD_PEERS_FILE", str(tmp_path / "nope.yaml"))
        monkeypatch.setenv("VRAMD_HOOKS_FILE", str(tmp_path / "sem-hooks.yaml"))
        entry = {"name": "m", "adapter": "a", "vram_mib": 1, "runtime": {"command": [sys.executable]}}
        path = tmp_path / "b.yaml"
        path.write_text(yaml.safe_dump({"backends": [entry]}), encoding="utf-8")
        report = validate_config(str(path))
        assert report.ok and any("sem token" in f.message for f in report.warnings)
