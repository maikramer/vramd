"""Peers: configuração, monitor, e encaminhamento real entre dois supervisores."""

from __future__ import annotations

import base64
import hashlib
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from vramd import peers as peers_mod
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


class _FakeSock:
    """Socket mínimo para exercitar os loops de leitura sem rede real."""

    def __init__(self, chunks: list[bytes] | None = None, *, drip: bytes | None = None, delay: float = 0.0) -> None:
        self._chunks = list(chunks or [])
        self._drip = drip
        self._delay = delay
        self.sent = b""
        self.closed = False

    def __enter__(self) -> _FakeSock:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def sendall(self, data: bytes) -> None:
        self.sent += data

    def settimeout(self, _timeout: float | None) -> None:
        return None

    def recv(self, _size: int) -> bytes:
        if self._delay:
            time.sleep(self._delay)
        if self._drip is not None:
            return self._drip
        return self._chunks.pop(0) if self._chunks else b""

    def close(self) -> None:
        self.closed = True


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

    def test_malformed_status_does_not_kill_the_monitor(self) -> None:
        """Um peer (não confiável) com ``queue_depth`` do tipo errado não pode
        matar a thread de poll — o peer fica em baixo e o poll seguinte recupera."""
        peers, _ = self._set([_status(queue={"queue_depth": "muitos", "inflight": 0}), _status()])
        peers.poll_once()
        state = peers.get("box")
        assert state.healthy is False and "status inválido" in (state.last_error or "")
        peers.poll_once()
        assert state.healthy is True and state.last_error is None

    def test_stop_does_not_orphan_a_slow_poll_thread(self) -> None:
        """``stop`` com join curto não pode esquecer a thread: ``start`` depois
        criava uma segunda e as duas ficavam a fazer poll para sempre."""
        peers = PeerSet([SPEC])

        class _AliveThread:
            def join(self, timeout: float | None = None) -> None:
                return None

            def is_alive(self) -> bool:
                return True

        fake = _AliveThread()
        peers._thread = fake  # type: ignore[assignment]
        peers.stop()
        assert peers._stop.is_set()
        assert peers._thread is fake  # ainda viva: referência mantida
        peers.start()
        assert peers._thread is fake  # não arranca uma segunda thread
        assert peers._stop.is_set()  # e não ressuscita a antiga limpando o stop

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


class TestPeerTransportLimits:
    """Limites contra peers lentos/maliciosos (o peer não é de confiança)."""

    def test_peer_stream_rejects_an_unbounded_line(self, monkeypatch) -> None:
        monkeypatch.setattr(peers_mod, "_MAX_LINE_BYTES", 32)
        sock = _FakeSock([b"x" * 64, b"y" * 64])
        monkeypatch.setattr(peers_mod.transport, "connect_remote", lambda *_a, **_k: (sock, None))
        with pytest.raises(PeerUnreachable, match="excede"):
            list(peers_mod.peer_stream(SPEC, {"cmd": P.CMD_STATUS}, timeout_sec=1.0))
        assert sock.closed

    def test_peer_request_enforces_a_total_deadline(self, monkeypatch) -> None:
        """Bytes a conta-gotas dentro do timeout por-recv não podem prender o poll."""
        sock = _FakeSock([b" "] * 20, delay=0.02)
        monkeypatch.setattr(peers_mod.transport, "connect_remote", lambda *_a, **_k: (sock, None))
        with pytest.raises(PeerUnreachable, match="timeout"):
            peer_request(SPEC, {"cmd": P.CMD_STATUS}, timeout_sec=0.1)
        assert sock.closed


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

    def test_upload_failure_cleans_the_remote_sandbox(self, peer_node, tmp_path: Path, monkeypatch) -> None:
        """Sandbox criado no peer antes do upload não pode ficar órfão até ao TTL."""
        _srv, spec, _adapter, home = peer_node
        src = tmp_path / "in.txt"
        src.write_text("x", encoding="utf-8")
        peers = PeerSet([spec])
        executor = RemoteExecutor(peers)

        def _fail(*_args: Any, **_kwargs: Any) -> None:
            raise PeerUnreachable("upload morreu")

        monkeypatch.setattr(executor, "_upload", _fail)
        with pytest.raises(PeerUnreachable, match="upload morreu"):
            executor.run(peers.get("box"), "copy", {"src": str(src)}, input_keys=("src",))
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


class TestRemoteTransferGuards:
    """Guardas de tamanho do upload/download entre supervisores."""

    @staticmethod
    def _executor(responses: list[Any]) -> tuple[RemoteExecutor, list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []

        def requester(spec: PeerSpec, request: dict[str, Any], *, timeout_sec: float = 60.0) -> dict[str, Any]:
            calls.append(dict(request))
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        return RemoteExecutor(PeerSet([]), requester=requester), calls

    def test_upload_sends_only_the_declared_size(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / "cresceu.bin"
        path.write_bytes(b"abcdef")
        frozen = type("_FrozenStat", (), {"st_size": 3})()
        real_stat = Path.stat

        def stat_before_growth(self: Path, *args: Any, **kwargs: Any) -> Any:
            return frozen if self == path else real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", stat_before_growth)
        executor, calls = self._executor([{"status": P.STATUS_OK}])
        executor._upload(SPEC, "sandbox", path, "in/f.bin")
        assert len(calls) == 1
        assert base64.b64decode(calls[0]["data"]) == b"abc"
        assert calls[0]["sha256"] == hashlib.sha256(b"abc").hexdigest()
        assert calls[0]["eof"] is True and calls[0]["offset"] == 0

    def test_upload_of_a_file_that_shrinks_fails_instead_of_looping(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / "encolheu.bin"
        path.write_bytes(b"abc")
        frozen = type("_FrozenStat", (), {"st_size": 6})()
        real_stat = Path.stat

        def stat_before_shrink(self: Path, *args: Any, **kwargs: Any) -> Any:
            return frozen if self == path else real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", stat_before_shrink)
        executor, _calls = self._executor([{"status": P.STATUS_OK}])
        with pytest.raises(PeerUnreachable, match="encolheu"):
            executor._upload(SPEC, "sandbox", path, "in/f.bin")

    def test_empty_file_is_uploaded_with_a_final_empty_put(self, tmp_path: Path) -> None:
        path = tmp_path / "vazio.bin"
        path.write_bytes(b"")
        executor, calls = self._executor([{"status": P.STATUS_OK}])
        executor._upload(SPEC, "sandbox", path, "in/f.bin")
        assert len(calls) == 1
        assert calls[0]["data"] == "" and calls[0]["eof"] is True
        assert calls[0]["sha256"] == hashlib.sha256(b"").hexdigest()

    def test_download_rejects_more_bytes_than_the_declared_size(self, tmp_path: Path) -> None:
        chunk = base64.b64encode(b"12345678").decode()
        executor, _calls = self._executor(
            [{"status": P.STATUS_OK, "size": 4, "offset": 0, "data": chunk, "eof": False}]
        )
        target = tmp_path / "extra.bin"
        with pytest.raises(PeerUnreachable, match="excede"):
            executor._download(SPEC, "a" * 12, "out/x", target)
        assert not target.exists()
        assert not list(target.parent.glob(target.name + ".*"))

    def test_download_rejects_a_size_change_mid_stream(self, tmp_path: Path) -> None:
        executor, _calls = self._executor(
            [
                {"status": P.STATUS_OK, "size": 4, "offset": 0, "data": base64.b64encode(b"12").decode(), "eof": False},
                {"status": P.STATUS_OK, "size": 8, "offset": 2, "data": base64.b64encode(b"34").decode(), "eof": True},
            ]
        )
        with pytest.raises(PeerUnreachable, match="mudou"):
            executor._download(SPEC, "a" * 12, "out/x", tmp_path / "mudou.bin")

    def test_download_retries_the_same_offset_after_a_transport_error(self, tmp_path: Path) -> None:
        calls = {"n": 0}

        def requester(spec: PeerSpec, request: dict[str, Any], *, timeout_sec: float = 60.0) -> dict[str, Any]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise PeerUnreachable("queda simulada")
            return {
                "status": P.STATUS_OK,
                "size": 3,
                "offset": int(request["offset"]),
                "data": base64.b64encode(b"abc").decode(),
                "eof": True,
            }

        executor = RemoteExecutor(PeerSet([]), requester=requester)
        target = tmp_path / "retomado.bin"
        assert executor._download(SPEC, "a" * 12, "out/x", target) == 3
        assert target.read_bytes() == b"abc"
        assert calls["n"] == 2


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
