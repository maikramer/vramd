"""Testes do servidor MCP — JSON-RPC, tools e guards de confirmação."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from vramd import mcp_server as M


def _msg(method: str, params: dict | None = None, *, msg_id: Any = 1) -> dict:
    out: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        out["params"] = params
    return out


class TestProtocol:
    def test_initialize(self) -> None:
        resp = M.handle_message(_msg("initialize"))
        result = resp["result"]
        assert result["protocolVersion"] == M.PROTOCOL_VERSION
        assert result["serverInfo"]["name"] == "vramd"
        assert "tools" in result["capabilities"]

    def test_notifications_get_no_response(self) -> None:
        assert M.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    def test_ping(self) -> None:
        assert M.handle_message(_msg("ping"))["result"] == {}

    def test_unknown_method(self) -> None:
        resp = M.handle_message(_msg("resources/list"))
        assert resp["error"]["code"] == -32601

    def test_tools_list_has_expected_tools(self) -> None:
        tools = M.handle_message(_msg("tools/list"))["result"]["tools"]
        names = {t["name"] for t in tools}
        assert {
            "vramd_status",
            "vramd_queue",
            "vramd_backends",
            "vramd_stats",
            "vramd_learn",
            "vramd_submit",
            "vramd_wait",
            "vramd_cancel",
            "vramd_preload",
            "vramd_evict",
            "vramd_zero",
            "vramd_doctor",
        } <= names
        # Schemas JSON válidos para os clientes MCP.
        for t in tools:
            assert t["inputSchema"]["type"] == "object"


def _call(name: str, args: dict | None = None) -> dict:
    return M.handle_message(_msg("tools/call", {"name": name, "arguments": args or {}}))


class TestToolCalls:
    def test_unknown_tool_is_invalid_params(self) -> None:
        resp = _call("vramd_nada")
        assert resp["error"]["code"] == -32602

    @pytest.fixture
    def fake_ums(self, monkeypatch: pytest.MonkeyPatch):
        """Intercepta as chamadas ao supervisor (dict resposta por cmd)."""
        sent: list[dict] = []
        responses: dict[str, Any] = {}

        def fake_send(request, *, timeout_sec=300.0, auto_start=False):
            sent.append(request)
            return responses.get(str(request.get("cmd")))

        monkeypatch.setattr("vramd.client.send_to_ums", fake_send)
        return sent, responses

    def test_status_passes_through(self, fake_ums) -> None:
        sent, responses = fake_ums
        responses["status"] = {"status": "status", "pid": 42, "backends": []}
        result = _call("vramd_status")["result"]
        assert result["isError"] is False
        assert json.loads(result["content"][0]["text"])["pid"] == 42
        assert sent[0]["cmd"] == "status"

    def test_daemon_down_is_error_with_hint(self, fake_ums) -> None:
        result = _call("vramd_queue")["result"]
        assert result["isError"] is True
        assert "vramd start" in result["content"][0]["text"]

    @pytest.mark.parametrize("tool", ["vramd_evict", "vramd_zero"])
    def test_destructive_tools_require_confirm(self, fake_ums, tool: str) -> None:
        result = _call(tool)["result"]
        assert result["isError"] is True
        assert "confirm" in result["content"][0]["text"]

    def test_zero_with_confirm_calls_ums(self, fake_ums) -> None:
        sent, responses = fake_ums
        responses["zero"] = {"status": "ok", "workers_killed": 1}
        result = _call("vramd_zero", {"confirm": True})["result"]
        assert result["isError"] is False
        assert sent[0]["cmd"] == "zero"

    def test_submit_requires_backend(self, fake_ums) -> None:
        result = _call("vramd_submit", {"request": {}})["result"]
        assert result["isError"] is True

    def test_submit_without_wait_returns_job_id(self, fake_ums) -> None:
        sent, responses = fake_ums
        responses["submit"] = {"status": "ok", "job_id": "abc", "backend": "alpha"}
        result = _call("vramd_submit", {"backend": "alpha", "wait": False})["result"]
        assert not result["isError"]
        assert sent[0]["backend"] == "alpha"

    def test_submit_with_wait_waits_for_result(self, fake_ums, monkeypatch: pytest.MonkeyPatch) -> None:
        _sent, responses = fake_ums
        responses["submit"] = {"status": "ok", "job_id": "abc", "backend": "alpha"}
        monkeypatch.setattr(
            "vramd.client.wait_ums_job",
            lambda job_id, timeout_sec=600.0: {"status": "ok", "output": "/tmp/x.png"},
        )
        result = _call("vramd_submit", {"backend": "alpha", "request": {"prompt": "oi"}})["result"]
        assert not result["isError"]
        payload = json.loads(result["content"][0]["text"])
        assert payload["output"] == "/tmp/x.png"


class TestStdioLoop:
    def test_end_to_end_initialize_and_eof(self) -> None:
        stdin = io.StringIO(json.dumps(_msg("initialize")) + "\n")
        stdout = io.StringIO()
        rc = M.serve_stdio(stdin=stdin, stdout=stdout, stderr=io.StringIO())
        assert rc == 0
        resp = json.loads(stdout.getvalue().strip())
        assert resp["result"]["serverInfo"]["name"] == "vramd"

    def test_invalid_json_line_gets_parse_error(self) -> None:
        stdin = io.StringIO("{isto não é json}\n")
        stdout = io.StringIO()
        M.serve_stdio(stdin=stdin, stdout=stdout, stderr=io.StringIO())
        resp = json.loads(stdout.getvalue().strip())
        assert resp["error"]["code"] == -32700

    def test_handler_exception_becomes_internal_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(_msg: dict) -> dict:
            raise RuntimeError("kaboom")

        monkeypatch.setattr(M, "handle_message", boom)
        resp = M.handle_message_safe(_msg("ping"))
        assert resp["error"]["code"] == -32603
