"""Peers: configuração, monitor, e encaminhamento real entre dois supervisores."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from vramd import protocol as P
from vramd import transport
from vramd.adapter import BackendAdapter
from vramd.peers import (
    PeerSet,
    PeerSpec,
    PeerUnreachable,
    RemoteExecutor,
    load_peers,
    parse_peers_doc,
    peer_request,
)
from vramd.registry import BackendDescriptor, Registry

from .test_server import _send_request, _start_ums, _stop_ums

SPEC = PeerSpec(name="box", host="127.0.0.1", port=9, token="tok")


def _status(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "status": "status",
        "eta_sec": None,
        "queue": {"queue_depth": 0, "inflight": 0},
        "backends": [{"name": "m", "loaded": True, "peak_mib": 5000}, {"name": "other", "loaded": False}],
        "routing": {
            "version": "9.9.9",
            "gpus": [{"index": 0, "total_mib": 24000, "free_mib": 20000}],
            "estimates": {"m": {"load_sec": 12.0, "gen_sec": 3.0}},
        },
    }
    base.update(over)
    return base


class TestPeerConfig:
    def test_env_list(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("VRAMD_PEERS", "box=10.0.0.2:8790, 10.0.0.3:8790, lixo")
        monkeypatch.setenv("VRAMD_PEERS_TOKEN", "segredo")
        specs = load_peers(tmp_path / "nao-existe.yaml")
        assert [(s.name, s.address, s.token) for s in specs] == [
            ("box", "10.0.0.2:8790", "segredo"),
            ("10.0.0.3", "10.0.0.3:8790", "segredo"),
        ]

    def test_yaml_overrides_env_and_reads_token_file(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("VRAMD_PEERS", "box=10.0.0.2:8790")
        token = tmp_path / "box.token"
        token.write_text("do-ficheiro\n", encoding="utf-8")
        path = tmp_path / "peers.yaml"
        doc = {
            "peers": [
                {
                    "name": "box",
                    "address": "192.168.1.9:8790",
                    "token_file": str(token),
                    "bandwidth_mib_s": 110,
                    "max_inflight": 2,
                }
            ]
        }
        path.write_text(yaml.safe_dump(doc), encoding="utf-8")
        (spec,) = load_peers(path)
        assert (spec.address, spec.token, spec.bandwidth_mib_s, spec.max_inflight) == (
            "192.168.1.9:8790",
            "do-ficheiro",
            110.0,
            2,
        )

    @pytest.mark.parametrize(
        "doc",
        [
            {"peers": "box"},
            {"peers": [{"name": "x"}]},
            {"peers": [{"address": "sem-porta"}]},
            {"peers": [{"address": "h:1", "bandwidth_mib_s": 0}]},
            {"peers": [{"name": "a", "address": "h:1"}, {"name": "a", "address": "h:2"}]},
            ["lista"],
        ],
    )
    def test_malformed_doc_raises(self, doc: Any) -> None:
        """Encaminhar jobs para o sítio errado em silêncio é pior que não arrancar."""
        with pytest.raises(ValueError):
            parse_peers_doc(doc)

    def test_no_config_means_no_peers(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.delenv("VRAMD_PEERS", raising=False)
        assert load_peers(tmp_path / "nope.yaml") == []
        assert not PeerSet([])


class TestPeerSet:
    def _set(self, responses: list[Any]) -> tuple[PeerSet, list[float]]:
        now = [1000.0]

        def requester(spec: PeerSpec, request: dict[str, Any], *, timeout_sec: float = 5.0) -> dict[str, Any]:
            now[0] += 0.02  # 20 ms de ida e volta
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        return PeerSet([SPEC], requester=requester, clock=lambda: now[0]), now

    def test_poll_learns_state_and_rtt(self) -> None:
        peers, _ = self._set([_status()])
        peers.poll_once()
        state = peers.get("box")
        assert state.healthy and state.last_error is None
        assert state.rtt_sec == pytest.approx(0.02)
        assert state.gpu_totals_mib == [24000] and state.version == "9.9.9"
        (view,) = peers.views("m", 5000)
        assert view.healthy and view.has_backend and view.hot and view.fits is True
        assert (view.load_sec, view.gen_sec) == (12.0, 3.0)

    def test_unpolled_peer_is_not_routable(self) -> None:
        peers, _ = self._set([])
        (view,) = peers.views("m", 5000)
        assert not view.healthy and "sem poll" in view.reason

    def test_unreachable_then_recovers(self) -> None:
        peers, _ = self._set([_status(), PeerUnreachable("connection refused"), _status()])
        peers.poll_once()
        peers.poll_once()
        assert peers.get("box").healthy is False and "refused" in peers.get("box").last_error
        peers.poll_once()
        assert peers.get("box").healthy is True

    def test_wrong_token_is_reported(self) -> None:
        peers, _ = self._set([{"status": "error", "error_code": P.ERR_UNAUTHORIZED}])
        peers.poll_once()
        assert "token recusado" in peers.get("box").last_error

    def test_backend_missing_and_too_big(self) -> None:
        peers, _ = self._set([_status()])
        peers.poll_once()
        assert peers.views("desconhecido", 100)[0].has_backend is False
        # O peer declara 5000 MiB para "m": cabe lá, diga o nosso descriptor o que disser.
        assert peers.views("m", 40000)[0].fits is True
        assert peers.views("other", 40000)[0].fits is False  # sem pico declarado: vale o nosso

    def test_multi_gpu_peer_may_split_so_fit_is_unknown(self) -> None:
        gpus = [{"index": 0, "total_mib": 24000}, {"index": 1, "total_mib": 24000}]
        peers, _ = self._set([_status(routing={"gpus": gpus})])
        peers.poll_once()
        assert peers.views("other", 40000)[0].fits is None

    def test_old_peer_without_routing_block(self) -> None:
        """Peer numa versão anterior: sem GPUs nem estimativas — rota possível, sem certezas."""
        status = _status()
        del status["routing"]
        peers, _ = self._set([status])
        peers.poll_once()
        (view,) = peers.views("m", 5000)
        assert view.healthy and view.fits is None and view.gen_sec is None

    def test_busy_peer_without_eta_is_not_free(self) -> None:
        peers, _ = self._set([_status(queue={"queue_depth": 0, "inflight": 1})])
        peers.poll_once()
        assert peers.views("m", 5000)[0].wait_sec == 3.0  # a geração estimada dele

    def test_queue_eta_and_backlog_add_up(self) -> None:
        peers, _ = self._set([_status(eta_sec=40.0, queue={"queue_depth": 2, "inflight": 1})])
        peers.poll_once()
        assert peers.views("m", 5000, extra_wait={"box": 10.0})[0].wait_sec == 50.0

    def test_forward_slots_are_bounded(self) -> None:
        peers, _ = self._set([_status()])
        peers.poll_once()
        assert peers.reserve("box") is True
        assert peers.reserve("box") is False
        assert peers.views("m", 5000)[0].slot_free is False
        peers.release("box")
        assert peers.reserve("box") is True and peers.get("box").forwarded_total == 1

    def test_failure_mid_job_cools_down_then_returns(self) -> None:
        peers, now = self._set([_status()])
        peers.poll_once()
        peers.mark_failed("box", "caiu a meio")
        assert peers.views("m", 5000)[0].healthy is False
        peers.get("box").healthy = True  # o poll seguinte volta a vê-lo vivo…
        view = peers.views("m", 5000)[0]
        assert view.healthy is False and "cooldown" in view.reason  # …mas ainda não se confia
        now[0] += 60.0
        assert peers.views("m", 5000)[0].healthy is True

    def test_excluded_peer_for_a_job(self) -> None:
        peers, _ = self._set([_status()])
        peers.poll_once()
        assert peers.views("m", 5000, exclude={"box"})[0].healthy is False

    def test_measured_bandwidth_beats_declared(self) -> None:
        spec = PeerSpec(name="box", host="h", port=1, bandwidth_mib_s=10.0)
        peers = PeerSet([spec])
        assert peers.get("box").effective_bandwidth() == 10.0
        peers.record_bandwidth("box", 0.2, 0.001)  # minúscula: mede latência, ignorada
        assert peers.get("box").effective_bandwidth() == 10.0
        peers.record_bandwidth("box", 200.0, 2.0)
        assert peers.get("box").effective_bandwidth() == 100.0
        assert peers.status()[0]["bandwidth_measured"] is True


class _CopyAdapter(BackendAdapter):
    """Backend do peer: copia o input para o output em maiúsculas."""

    name = "copy"

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def load(self, **kwargs: Any) -> Any:
        return {}

    def generate(self, model: Any, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append({k: v for k, v in request.items() if not k.startswith("_")})
        if request.get("fail"):
            return {"status": "error", "error": "modelo rebentou"}
        text = Path(request["src"]).read_text(encoding="utf-8") if request.get("src") else str(request.get("prompt"))
        if request.get("output"):
            Path(request["output"]).write_text(text.upper(), encoding="utf-8")
        return {"status": "ok", "output": request.get("output") or text.upper()}

    def unload(self, model: Any) -> None:
        return None


def _sandboxes(home: Path) -> list[str]:
    """Sandboxes de job que ficaram no peer (o cache de blobs não conta)."""
    return [p.name for p in (home / "remote").iterdir() if p.name != "blobs"]


def _registry(adapter: BackendAdapter | None, *, vram_mib: int = 1000, **placement: Any) -> Registry:
    desc = BackendDescriptor(name="copy", adapter="_copy", vram_mib=vram_mib, priority=10, placement=placement)
    registry = Registry(descriptors={"copy": desc})
    if adapter is not None:
        registry._adapter_instances["copy"] = adapter
    return registry


@pytest.fixture
def peer_node(tmp_path: Path):
    """Um supervisor real a escutar em TCP loopback — o "outro nó"."""
    adapter = _CopyAdapter()
    home = tmp_path / "peer"
    home.mkdir()
    srv, sock, thread = _start_ums(home, registry=_registry(adapter), listen=("127.0.0.1", 0), token="tok-peer")
    info = transport.read_endpoint(sock)
    assert info is not None and info.port
    spec = PeerSpec(name="box", host="127.0.0.1", port=int(info.port), token="tok-peer")
    yield srv, spec, adapter, home
    _stop_ums(srv, sock, thread)


class TestRemoteExecutor:
    def test_request_helper_and_auth(self, peer_node) -> None:
        _srv, spec, _adapter, _home = peer_node
        assert peer_request(spec, {"cmd": P.CMD_STATUS})["status"] == P.STATUS_STATUS
        denied = peer_request(PeerSpec("box", spec.host, spec.port, token="errado"), {"cmd": P.CMD_STATUS})
        assert denied["error_code"] == P.ERR_UNAUTHORIZED

    def test_dead_peer_raises_unreachable(self) -> None:
        with pytest.raises(PeerUnreachable):
            peer_request(PeerSpec("morto", "127.0.0.1", 1), {"cmd": P.CMD_STATUS}, timeout_sec=1.0)

    def test_forward_with_files_round_trip(self, peer_node, tmp_path: Path) -> None:
        """Input sobe para o sandbox do peer, corre lá, o output volta ao path local."""
        _srv, spec, adapter, home = peer_node
        src = tmp_path / "in.txt"
        src.write_text("olá mundo " * 300_000, encoding="utf-8")  # ~3 MiB: vários chunks
        out = tmp_path / "resultado" / "out.txt"
        peers = PeerSet([spec])
        peers.poll_once()
        progress: list[tuple[Any, Any]] = []
        result = RemoteExecutor(peers, node_name="frente").run(
            peers.get("box"),
            "copy",
            {"src": str(src), "output": str(out), "prompt": "p", "_progress": "lixo-interno"},
            input_keys=("src",),
            output_keys=("output",),
            on_progress=lambda pct, msg: progress.append((pct, msg)),
        )
        assert result["status"] == P.STATUS_OK, result
        assert out.read_text(encoding="utf-8") == src.read_text(encoding="utf-8").upper()
        # O resultado fala em paths DESTE nó, não nos do sandbox remoto.
        assert result["output"] == str(out)
        assert result["route"]["target"] == "peer:box" and result["route"]["remote_job_id"]
        # O peer recebeu paths do sandbox dele e nenhuma chave interna.
        (seen,) = adapter.requests
        assert seen["src"].startswith(str(home)) and seen["output"].startswith(str(home))
        assert "_progress" not in seen and "_route" not in seen and "auth" not in seen
        # Sandbox limpo no fim; largura de banda medida com a transferência real.
        assert _sandboxes(home) == []
        assert peers.get("box").bandwidth_mib_s is not None
        assert any("a enviar in.txt" in str(msg) for _pct, msg in progress)

    def test_forward_without_files_needs_no_sandbox(self, peer_node) -> None:
        _srv, spec, _adapter, home = peer_node
        peers = PeerSet([spec])
        result = RemoteExecutor(peers).run(peers.get("box"), "copy", {"prompt": "abc"})
        assert result["status"] == P.STATUS_OK and result["output"] == "ABC"
        assert _sandboxes(home) == []

    def test_remote_job_failure_is_a_result_not_a_transport_error(self, peer_node) -> None:
        _srv, spec, _adapter, _home = peer_node
        peers = PeerSet([spec])
        result = RemoteExecutor(peers).run(peers.get("box"), "copy", {"fail": True})
        assert result["status"] == P.STATUS_ERROR and "modelo rebentou" in result["error"]

    def test_forwarded_job_is_marked_and_recorded_on_the_peer(self, peer_node) -> None:
        srv, spec, _adapter, _home = peer_node
        seen: list[bool] = []
        original = srv._on_job_finished
        srv.queue._on_finish = lambda job: (seen.append(job.forwarded), original(job))[1]
        peers = PeerSet([spec])
        RemoteExecutor(peers).run(peers.get("box"), "copy", {"prompt": "x"})
        deadline = time.monotonic() + 3.0
        while not seen and time.monotonic() < deadline:
            time.sleep(0.02)
        assert seen == [True]

    def test_unreachable_before_accept_is_safe_to_retry(self) -> None:
        peers = PeerSet([PeerSpec("morto", "127.0.0.1", 1)])
        with pytest.raises(PeerUnreachable):
            RemoteExecutor(peers).run(peers.get("morto"), "copy", {"prompt": "x"})


class TestSupervisorForwards:
    """Dois supervisores: o da frente não tem VRAM, decide sozinho mandar para o peer."""

    def test_job_is_routed_over_the_network(self, peer_node, tmp_path: Path) -> None:
        _peer_srv, spec, adapter, _home = peer_node
        front_home = tmp_path / "front"
        front_home.mkdir()
        # Cá o modelo não cabe agora: 100 MiB livres para um pico de ~3 GiB.
        registry = _registry(None, vram_mib=3000, input_keys=["src"], output_keys=["output"])
        src = tmp_path / "a.txt"
        src.write_text("via rede", encoding="utf-8")
        out = tmp_path / "b.txt"
        srv, sock, thread = _start_ums(front_home, registry=registry, peers=[spec], query_free_mib=lambda: 100)
        try:
            deadline = time.monotonic() + 5.0
            while not srv.peers.get("box").healthy and time.monotonic() < deadline:
                time.sleep(0.05)
            assert srv.peers.get("box").healthy

            plan = _send_request(sock, {"cmd": P.CMD_ROUTE, "backend": "copy", "src": str(src)})
            assert plan["status"] == P.STATUS_OK and plan["routed"] is True
            assert plan["action"] == "run" and plan["best"]["target"] == "peer:box"
            assert any(not r["feasible"] and r["kind"] == "local" for r in plan["routes"])

            resp = _send_request(
                sock, {"cmd": P.CMD_GENERATE, "backend": "copy", "src": str(src), "output": str(out)}, timeout=30.0
            )
            assert resp["status"] == P.STATUS_OK, resp
            assert out.read_text(encoding="utf-8") == "VIA REDE"
            assert resp["route"]["target"] == "peer:box"
            assert len(adapter.requests) == 1

            status = _send_request(sock, {"cmd": P.CMD_STATUS})
            assert status["peers"][0]["forwarded_total"] == 1
            assert status["routing"]["decisions"]["remote"] == 1
        finally:
            _stop_ums(srv, sock, thread)

    def test_peer_down_falls_back_to_local(self, tmp_path: Path) -> None:
        """Peer morto entre o poll e o envio: o job volta à fila e corre cá."""
        local = _CopyAdapter()
        registry = _registry(local, vram_mib=1000)
        spec = PeerSpec("fantasma", "127.0.0.1", 1)
        srv, sock, thread = _start_ums(tmp_path, registry=registry, peers=[spec])
        try:
            # Enganar o router: peer "saudável e quente", GPU local "ocupada 1h".
            state = srv.peers.get("fantasma")
            srv.peers.stop()
            srv.peers._apply_status(state, {"backends": [{"name": "copy", "loaded": True}], "queue": {}}, rtt_sec=0.001)
            blocker = object()
            assert srv.lanes.reserve(srv.lanes.primary, blocker, est_sec=3600.0)
            job_resp = _send_request(sock, {"cmd": P.CMD_SUBMIT, "backend": "copy", "prompt": "x"})
            job_id = job_resp["job_id"]
            deadline = time.monotonic() + 5.0
            while "fantasma" not in srv.queue.get(job_id).route_excluded and time.monotonic() < deadline:
                time.sleep(0.02)
            assert "fantasma" in srv.queue.get(job_id).route_excluded
            assert srv.peers.get("fantasma").healthy is False
            srv.lanes.release(srv.lanes.primary)
            srv.queue.notify()
            final = _send_request(sock, {"cmd": P.CMD_WAIT, "job_id": job_id, "timeout_sec": 10}, timeout=15.0)
            assert final["status"] == P.STATUS_OK and final["output"] == "X"
            assert len(local.requests) == 1
        finally:
            _stop_ums(srv, sock, thread)
