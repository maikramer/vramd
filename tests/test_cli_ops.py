"""CLI de operação: submit (params/JSON/stream), history, logs, metrics, validate, completion."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

import click
import pytest
import yaml
from click.testing import CliRunner

from vramd import cli as cli_mod
from vramd import cli_ops
from vramd.cli import build_submit_request, cli
from vramd.history import JobHistory


class TestBuildSubmitRequest:
    def test_no_params_keeps_smoke_test(self) -> None:
        """`vramd submit <backend> --wait` sem mais nada continua a ser o teste de fumo."""
        req = build_submit_request("echo")
        assert req == {
            "cmd": "submit",
            "backend": "echo",
            "prompt": "smoke",
            "output": "/tmp/ums-smoke-out.bin",
            "priority": "interactive",
        }

    def test_params_are_coerced(self) -> None:
        req = build_submit_request("m", params=("steps=8", "cfg=1.5", "tiled=true", "seed=none", "name=a=b"))
        assert req["steps"] == 8 and req["cfg"] == 1.5 and req["tiled"] is True
        assert req["seed"] is None and req["name"] == "a=b"
        assert "prompt" not in req  # com parâmetros reais não injecta o smoke

    def test_precedence_json_then_param_then_flag(self) -> None:
        req = build_submit_request(
            "m",
            request_json='{"prompt": "do json", "steps": 4, "priority": "batch"}',
            params=("steps=8",),
            prompt="da flag",
        )
        assert req["prompt"] == "da flag" and req["steps"] == 8
        assert req["priority"] == "batch"  # do JSON, porque --priority não foi dado

    def test_explicit_priority_beats_json(self) -> None:
        req = build_submit_request("m", request_json='{"priority": "batch"}', priority="interactive")
        assert req["priority"] == "interactive"

    def test_json_from_file(self, tmp_path: Path) -> None:
        path = tmp_path / "job.json"
        path.write_text('{"mesh_path": "a.glb"}', encoding="utf-8")
        assert build_submit_request("m", request_json=f"@{path}")["mesh_path"] == "a.glb"

    @pytest.mark.parametrize("bad", ["{não json", "[1, 2]", "@/nao/existe.json"])
    def test_bad_json_is_usage_error(self, bad: str) -> None:
        with pytest.raises(click.BadParameter):
            build_submit_request("m", request_json=bad)

    @pytest.mark.parametrize("key", ["cmd", "backend", "stream", "auth", "_abort"])
    def test_reserved_keys_rejected(self, key: str) -> None:
        """`-p cmd=shutdown` não pode transformar um submit noutro comando."""
        with pytest.raises(click.BadParameter):
            build_submit_request("m", params=(f"{key}=shutdown",))

    def test_backend_cannot_be_overridden_by_json(self) -> None:
        with pytest.raises(click.BadParameter):
            build_submit_request("m", request_json='{"backend": "outro"}')


class TestSubmitCmd:
    def _patch(self, monkeypatch, *, submit: dict | None, events: list[dict] | None = None) -> list[dict]:
        sent: list[dict] = []

        def fake_send(request: dict, *, timeout: float = 30.0) -> dict | None:
            sent.append(request)
            return submit

        def fake_stream(request: dict, socket_path: Any = None, *, timeout_sec: float = 600.0):
            sent.append(request)
            yield from events or []

        monkeypatch.setattr(cli_mod, "_send", fake_send)
        monkeypatch.setattr(cli_mod, "send_request_stream", fake_stream)
        return sent

    def test_supervisor_down(self, monkeypatch) -> None:
        self._patch(monkeypatch, submit=None)
        res = CliRunner().invoke(cli, ["submit", "echo"])
        assert res.exit_code == 1 and "não está ativo" in res.output

    def test_submit_sends_params(self, monkeypatch) -> None:
        sent = self._patch(monkeypatch, submit={"status": "ok", "job_id": "abcdef123456", "priority": "batch"})
        res = CliRunner().invoke(cli, ["submit", "echo", "-p", "steps=8", "--prompt", "olá", "--priority", "batch"])
        assert res.exit_code == 0, res.output
        assert sent == [{"cmd": "submit", "backend": "echo", "steps": 8, "prompt": "olá", "priority": "batch"}]

    def test_wait_streams_until_done(self, monkeypatch) -> None:
        events = [
            {"event": "queued", "queue_position": 2},
            {"event": "started"},
            {"event": "progress", "pct": 0.5, "msg": "denoise"},
            {"status": "ok", "job_id": "abcdef123456", "output": "/tmp/x.png", "ums_debug": {"generate_sec": 3.2}},
        ]
        sent = self._patch(monkeypatch, submit={"status": "ok", "job_id": "abcdef123456"}, events=events)
        res = CliRunner().invoke(cli, ["submit", "echo", "--wait", "--timeout", "42"])
        assert res.exit_code == 0, res.output
        assert "done" in res.output and "/tmp/x.png" in res.output and "3.2s" in res.output
        assert sent[1] == {"cmd": "wait", "job_id": "abcdef123456", "stream": True, "timeout_sec": 42.0}

    def test_wait_failure_exits_nonzero_with_code(self, monkeypatch) -> None:
        events = [{"status": "error", "error": "CUDA out of memory", "error_code": "GENERATE_FAILED"}]
        self._patch(monkeypatch, submit={"status": "ok", "job_id": "abc"}, events=events)
        res = CliRunner().invoke(cli, ["submit", "echo", "--wait"])
        assert res.exit_code == 1
        assert "GENERATE_FAILED" in res.output and "CUDA out of memory" in res.output

    def test_wait_json_prints_only_final_result(self, monkeypatch) -> None:
        events = [{"event": "started"}, {"status": "ok", "job_id": "abc", "output": "o"}]
        self._patch(monkeypatch, submit={"status": "ok", "job_id": "abc"}, events=events)
        res = CliRunner().invoke(cli, ["submit", "echo", "--wait", "--json"])
        assert res.exit_code == 0
        assert json.loads(res.output) == {"status": "ok", "job_id": "abc", "output": "o"}

    def test_connection_lost_mid_wait(self, monkeypatch) -> None:
        """Stream acaba sem linha final: antes ficava a pollar 600s às escuras."""
        self._patch(monkeypatch, submit={"status": "ok", "job_id": "abc"}, events=[{"event": "started"}])
        res = CliRunner().invoke(cli, ["submit", "echo", "--wait"])
        assert res.exit_code == 1 and "caiu durante wait" in res.output

    def test_rejected_submit_shows_error(self, monkeypatch) -> None:
        self._patch(
            monkeypatch, submit={"status": "error", "error": "backend desconhecido", "error_code": "BACKEND_UNKNOWN"}
        )
        res = CliRunner().invoke(cli, ["submit", "nope"])
        assert res.exit_code == 1 and "BACKEND_UNKNOWN" in res.output

    def test_reserved_param_is_usage_error(self, monkeypatch) -> None:
        sent = self._patch(monkeypatch, submit={"status": "ok", "job_id": "abc"})
        res = CliRunner().invoke(cli, ["submit", "echo", "-p", "cmd=shutdown"])
        assert res.exit_code == 2 and sent == []


class TestHistoryCmd:
    @pytest.fixture
    def hist(self, tmp_path: Path, monkeypatch) -> JobHistory:
        path = tmp_path / "vramd-history.jsonl"
        monkeypatch.setattr("vramd.history.default_history_path", lambda: path)
        hist = JobHistory(path, max_entries=100)
        hist.record({"ts": 1000.0, "job_id": "aaaaaaaa1", "backend": "alpha", "state": "done", "generate_sec": 2.0})
        hist.record(
            {
                "ts": 1001.0,
                "job_id": "bbbbbbbb2",
                "backend": "beta",
                "state": "failed",
                "generate_sec": 0.4,
                "error": "CUDA out of memory [red]",
                "error_code": "GENERATE_FAILED",
            }
        )
        return hist

    def test_lists_newest_first(self, hist: JobHistory) -> None:
        res = CliRunner().invoke(cli, ["history", "--json"])
        assert res.exit_code == 0, res.output
        assert [j["job_id"] for j in json.loads(res.output)["jobs"]] == ["bbbbbbbb2", "aaaaaaaa1"]

    def test_table_shows_error(self, hist: JobHistory) -> None:
        res = CliRunner().invoke(cli, ["history"], env={"COLUMNS": "200"})
        assert res.exit_code == 0
        assert "bbbbbbbb" in res.output and "GENERATE_FAILED" in res.output
        # `[red]` no erro de um modelo é texto, não markup do rich.
        assert "CUDA out of memory [red]" in res.output

    def test_failed_filter(self, hist: JobHistory) -> None:
        res = CliRunner().invoke(cli, ["history", "--failed", "--json"])
        assert [j["backend"] for j in json.loads(res.output)["jobs"]] == ["beta"]

    def test_summary_ignores_limit(self, hist: JobHistory) -> None:
        res = CliRunner().invoke(cli, ["history", "--summary", "-n", "1", "--json"])
        data = json.loads(res.output)
        assert data["jobs"] == 2 and set(data["backends"]) == {"alpha", "beta"}
        assert data["backends"]["beta"]["success_rate"] == 0.0

    def test_clear(self, hist: JobHistory) -> None:
        res = CliRunner().invoke(cli, ["history", "--clear"])
        assert res.exit_code == 0 and "2 entradas" in res.output
        assert hist.read() == []

    def test_empty_history_is_not_an_error(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("vramd.history.default_history_path", lambda: tmp_path / "none.jsonl")
        res = CliRunner().invoke(cli, ["history"])
        assert res.exit_code == 0 and "Sem histórico" in res.output

    def test_bad_since(self, hist: JobHistory) -> None:
        assert CliRunner().invoke(cli, ["history", "--since", "ontem"]).exit_code == 2


class TestParseDuration:
    @pytest.mark.parametrize(("text", "expected"), [("90", 90.0), ("15m", 900.0), ("2h", 7200.0), ("1d", 86400.0)])
    def test_units(self, text: str, expected: float) -> None:
        assert cli_ops.parse_duration(text) == expected

    @pytest.mark.parametrize("text", ["", "abc", "0", "-5m", "m"])
    def test_invalid(self, text: str) -> None:
        with pytest.raises(click.BadParameter):
            cli_ops.parse_duration(text)


class TestLogs:
    @pytest.fixture
    def cache(self, tmp_path: Path, monkeypatch) -> Path:
        monkeypatch.setenv("VRAMD_CACHE_DIR", str(tmp_path))
        monkeypatch.delenv("VRAMD_LOG_DIR", raising=False)
        monkeypatch.delenv("VRAMD_LOG_FILE", raising=False)
        monkeypatch.delenv("VRAMD_AUTO_START_LOG", raising=False)
        return tmp_path

    def test_tail_lines_across_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "x.log"
        path.write_text("".join(f"linha {i}\n" for i in range(1000)), encoding="utf-8")
        assert cli_ops.tail_lines(path, 3, chunk_size=64) == ["linha 997", "linha 998", "linha 999"]
        assert len(cli_ops.tail_lines(path, 5000, chunk_size=64)) == 1000
        assert cli_ops.tail_lines(path, 0) == []

    def test_worker_log_tail(self, cache: Path) -> None:
        (cache / "vramd-worker-alpha.log").write_text("a\nb\nc\n", encoding="utf-8")
        res = CliRunner().invoke(cli, ["logs", "alpha", "-n", "2"])
        assert res.exit_code == 0 and res.output.splitlines() == ["b", "c"]

    def test_supervisor_log_picks_newest(self, cache: Path) -> None:
        import os

        logs = cache / "logs"
        logs.mkdir()
        old, new = logs / "vramd-2026-01-01.log", logs / "vramd-2026-01-02.log"
        old.write_text("velho\n", encoding="utf-8")
        new.write_text("novo\n", encoding="utf-8")
        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))
        res = CliRunner().invoke(cli, ["logs"])
        assert res.exit_code == 0 and res.output.strip() == "novo"
        assert CliRunner().invoke(cli, ["logs", "--path"]).output.strip() == str(new)

    def test_missing_log_exits_nonzero_and_lists_known(self, cache: Path) -> None:
        (cache / "vramd-worker-alpha.log").write_text("x\n", encoding="utf-8")
        res = CliRunner().invoke(cli, ["logs", "beta"])
        assert res.exit_code == 1 and "worker/alpha" in res.output

    def test_list(self, cache: Path) -> None:
        (cache / "vramd-worker-alpha.log").write_text("x\n", encoding="utf-8")
        res = CliRunner().invoke(cli, ["logs", "--list"])
        assert res.exit_code == 0 and "worker/alpha" in res.output

    def test_follow_emits_appends_and_survives_truncation(self, tmp_path: Path) -> None:
        path = tmp_path / "f.log"
        path.write_text("antigo\n", encoding="utf-8")
        got: list[str] = []
        stop = threading.Event()
        seen = threading.Event()

        def out(text: str) -> None:
            got.append(text)
            seen.set()

        thread = threading.Thread(
            target=cli_ops.follow, args=(path,), kwargs={"poll_sec": 0.01, "stop": stop.is_set, "out": out}
        )
        thread.start()
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write("novo\n")
            assert seen.wait(5.0)
            seen.clear()
            path.write_text("x\n", encoding="utf-8")  # truncado (rotação)
            assert seen.wait(5.0)
        finally:
            stop.set()
            thread.join(5.0)
        # O texto mode do Windows traduz ``\n`` para ``\r\n`` na escrita.
        assert "".join(got).replace("\r\n", "\n") == "novo\nx\n"  # o conteúdo anterior ao follow não é repetido


class TestMetricsCmd:
    def test_down_prints_up_zero_and_fails(self, monkeypatch) -> None:
        monkeypatch.setattr(cli_mod, "_send", lambda request, *, timeout=30.0: None)
        monkeypatch.setattr(cli_ops, "_local_gpu", lambda: None)
        res = CliRunner().invoke(cli, ["metrics"])
        assert res.exit_code == 1 and "vramd_up 0" in res.output

    def test_textfile_written_atomically(self, tmp_path: Path, monkeypatch) -> None:
        def fake_send(request: dict, *, timeout: float = 30.0) -> dict:
            if request["cmd"] == "status":
                return {"status": "status", "queue": {"queue_depth": 3, "inflight": 1}}
            return {"status": "ok", "backends": {"alpha": {"generate_count": 7}}}

        monkeypatch.setattr(cli_mod, "_send", fake_send)
        monkeypatch.setattr(cli_ops, "_local_gpu", lambda: {"free_mib": 10, "total_mib": 20})
        out = tmp_path / "node" / "vramd.prom"
        res = CliRunner().invoke(cli, ["metrics", "--textfile", str(out)])
        assert res.exit_code == 0 and res.output == ""
        text = out.read_text(encoding="utf-8")
        assert "vramd_up 1" in text and "vramd_queue_depth 3" in text
        assert 'vramd_backend_generations_total{backend="alpha"} 7' in text
        assert [p.name for p in out.parent.iterdir()] == ["vramd.prom"]  # sem .tmp esquecido


class TestValidateCmd:
    def _yaml(self, tmp_path: Path, **over: Any) -> str:
        entry = {
            "name": "alpha",
            "adapter": "pkg.adapter",
            "vram_mib": 100,
            "runtime": {"command": [sys.executable, "w.py"]},
            **over,
        }
        path = tmp_path / "b.yaml"
        path.write_text(yaml.safe_dump({"backends": [entry]}), encoding="utf-8")
        return str(path)

    @pytest.fixture(autouse=True)
    def _no_gpu_no_hooks(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(cli_ops, "_local_gpu", lambda: None)
        monkeypatch.setenv("VRAMD_HOOKS_FILE", str(tmp_path / "sem-hooks.yaml"))

    def test_clean(self, tmp_path: Path) -> None:
        res = CliRunner().invoke(cli, ["validate", self._yaml(tmp_path)])
        assert res.exit_code == 0, res.output
        assert "1 backend(s), 0 erro(s), 0 aviso(s)" in res.output

    def test_error_exits_nonzero(self, tmp_path: Path) -> None:
        res = CliRunner().invoke(cli, ["validate", self._yaml(tmp_path, vram_mib="x")])
        assert res.exit_code == 1 and "não é inteiro" in res.output

    def test_warning_only_fails_under_strict(self, tmp_path: Path) -> None:
        path = self._yaml(tmp_path, prority=5)
        assert CliRunner().invoke(cli, ["validate", path]).exit_code == 0
        res = CliRunner().invoke(cli, ["validate", path, "--strict"])
        assert res.exit_code == 1 and "priority" in res.output

    def test_json(self, tmp_path: Path) -> None:
        res = CliRunner().invoke(cli, ["validate", self._yaml(tmp_path, prority=5), "--json"])
        data = json.loads(res.output)
        assert data["ok"] is True and data["warnings"] == 1 and data["backends"] == ["alpha"]


class TestCompletion:
    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_script_for_each_shell(self, shell: str) -> None:
        res = CliRunner().invoke(cli, ["completion", shell])
        assert res.exit_code == 0, res.output
        assert "_VRAMD_COMPLETE" in res.output and "vramd" in res.output

    def test_detects_shell_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("SHELL", "/usr/bin/zsh")
        res = CliRunner().invoke(cli, ["completion"])
        assert res.exit_code == 0 and "zsh_complete" in res.output

    def test_unknown_shell_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("SHELL", "/bin/tcsh")
        assert CliRunner().invoke(cli, ["completion"]).exit_code == 2

    def test_backend_names_complete_from_registry(self) -> None:
        assert "text2icon" in cli_ops.complete_backend(None, None, "text2")
        assert cli_ops.complete_backend(None, None, "zzz") == []

    def test_new_commands_are_registered(self) -> None:
        for name in ("history", "logs", "metrics", "validate", "completion"):
            assert name in cli.commands
