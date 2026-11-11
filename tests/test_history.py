"""Histórico persistente de jobs terminados (``vramd history``)."""

from __future__ import annotations

import json
from pathlib import Path

from vramd.history import HISTORY_FILENAME, JobHistory, entry_from_job, read_history, summarize
from vramd.registry import Registry
from vramd.server import VramdServer


def _entry(i: int, *, backend: str = "alpha", state: str = "done", gen: float = 1.0, ts: float = 1000.0) -> dict:
    return {"ts": ts + i, "job_id": f"job-{i}", "backend": backend, "state": state, "generate_sec": gen}


class TestJobHistory:
    def test_newest_first_and_limit(self, tmp_path: Path) -> None:
        hist = JobHistory(tmp_path / "h.jsonl", max_entries=100)
        for i in range(5):
            assert hist.record(_entry(i))
        got = hist.read(limit=3)
        assert [e["job_id"] for e in got] == ["job-4", "job-3", "job-2"]

    def test_filters(self, tmp_path: Path) -> None:
        hist = JobHistory(tmp_path / "h.jsonl", max_entries=100)
        hist.record(_entry(0, backend="alpha"))
        hist.record(_entry(1, backend="beta", state="failed"))
        hist.record(_entry(2, backend="alpha", state="failed"))
        assert [e["job_id"] for e in hist.read(backend="alpha")] == ["job-2", "job-0"]
        assert [e["job_id"] for e in hist.read(state="failed")] == ["job-2", "job-1"]
        assert [e["job_id"] for e in hist.read(backend="beta", state="done")] == []

    def test_since_window(self, tmp_path: Path) -> None:
        path = tmp_path / "h.jsonl"
        hist = JobHistory(path, max_entries=100)
        hist.record(_entry(0, ts=1000.0))
        hist.record(_entry(1, ts=5000.0))
        got = read_history(path, since_sec=600.0, now=5100.0)
        assert [e["job_id"] for e in got] == ["job-1"]

    def test_compaction_keeps_newest_within_cap(self, tmp_path: Path) -> None:
        """Sem teto o ficheiro crescia para sempre num supervisor de longa duração."""
        path = tmp_path / "h.jsonl"
        hist = JobHistory(path, max_entries=20)
        for i in range(200):
            hist.record(_entry(i))
        lines = path.read_text(encoding="utf-8").splitlines()
        assert 20 <= len(lines) <= 20 + 16  # teto + folga de compactação
        assert json.loads(lines[-1])["job_id"] == "job-199"
        assert hist.read(limit=1)[0]["job_id"] == "job-199"

    def test_count_survives_restart(self, tmp_path: Path) -> None:
        """Instância nova conta as linhas existentes — o teto vale entre arranques."""
        path = tmp_path / "h.jsonl"
        first = JobHistory(path, max_entries=20)
        for i in range(30):
            first.record(_entry(i))
        second = JobHistory(path, max_entries=20)
        for i in range(30, 60):
            second.record(_entry(i))
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 36

    def test_corrupt_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "h.jsonl"
        path.write_text('{"job_id": "a", "backend": "x"}\nnot json\n[1, 2]\n\n{"job_id": "b"}\n', encoding="utf-8")
        assert [e["job_id"] for e in read_history(path)] == ["b", "a"]

    def test_disabled_writes_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "h.jsonl"
        hist = JobHistory(path, max_entries=0)
        assert hist.record(_entry(0)) is False
        assert not path.exists()

    def test_env_disables(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("VRAMD_HISTORY_MAX", "0")
        assert JobHistory(tmp_path / "h.jsonl").enabled is False

    def test_record_never_raises_on_unwritable_path(self, tmp_path: Path) -> None:
        """O histórico é observabilidade: disco cheio/sem permissão não pode partir a fila."""
        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        hist = JobHistory(blocker / "sub" / "h.jsonl", max_entries=10)
        assert hist.record(_entry(0)) is False

    def test_clear(self, tmp_path: Path) -> None:
        hist = JobHistory(tmp_path / "h.jsonl", max_entries=10)
        hist.record(_entry(0))
        hist.record(_entry(1))
        assert hist.clear() == 2
        assert hist.read() == []
        assert hist.record(_entry(2))
        assert len(hist.read()) == 1

    def test_missing_file_reads_empty(self, tmp_path: Path) -> None:
        assert read_history(tmp_path / "nope.jsonl") == []


class TestSummarize:
    def test_per_backend_rates_and_timings(self) -> None:
        entries = [
            _entry(0, gen=2.0),
            _entry(1, gen=4.0),
            _entry(2, state="failed", gen=0.1),
            _entry(3, state="cancelled"),
            _entry(4, backend="beta", gen=10.0),
        ]
        rows = summarize(entries)
        alpha = rows["alpha"]
        assert (alpha["jobs"], alpha["done"], alpha["failed"], alpha["cancelled"]) == (4, 2, 1, 1)
        # Falhados não entram na média: 0.1s puxaria o backend para "rápido".
        assert alpha["generate_avg_sec"] == 3.0
        assert alpha["generate_total_sec"] == 6.0
        # Cancelados não contam para a taxa de sucesso (não são culpa do backend).
        assert alpha["success_rate"] == round(2 / 3, 4)
        assert rows["beta"]["success_rate"] == 1.0

    def test_no_finished_jobs_has_no_rate(self) -> None:
        rows = summarize([_entry(0, state="cancelled")])
        assert rows["alpha"]["success_rate"] is None
        assert rows["alpha"]["generate_avg_sec"] is None


class TestServerRecordsHistory:
    def _server(self, tmp_path: Path) -> VramdServer:
        return VramdServer(registry=Registry(), socket_path=tmp_path / "ums-hist.sock")

    def test_finished_job_lands_in_history(self, tmp_path: Path) -> None:
        srv = self._server(tmp_path)
        job = srv.queue.enqueue("text2icon", {"prompt": "segredo do utilizador"})
        assert srv.queue.take(job.job_id) is not None
        job.mark_started()  # o dispatcher marca o início antes do generate
        srv.queue.finish(job, {"status": "ok", "output": "/tmp/out.png"})

        path = tmp_path / HISTORY_FILENAME
        entries = read_history(path)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["job_id"] == job.job_id
        assert entry["backend"] == "text2icon"
        assert entry["state"] == "done"
        assert entry["output"] == "/tmp/out.png"
        assert entry["generate_sec"] is not None
        # O request não é persistido: prompts podem ser privados.
        assert "segredo" not in path.read_text(encoding="utf-8")

    def test_failed_job_keeps_error_and_code(self, tmp_path: Path) -> None:
        srv = self._server(tmp_path)
        job = srv.queue.enqueue("text2icon", {})
        srv.queue.take(job.job_id)
        srv.queue.finish(job, {"status": "error", "error": "CUDA out of memory", "error_code": "GENERATE_FAILED"})
        entry = read_history(tmp_path / HISTORY_FILENAME)[0]
        assert entry["state"] == "failed"
        assert entry["error"] == "CUDA out of memory"
        assert entry["error_code"] == "GENERATE_FAILED"

    def test_cancelled_queued_job_is_recorded(self, tmp_path: Path) -> None:
        srv = self._server(tmp_path)
        job = srv.queue.enqueue("text2icon", {})
        srv.queue.cancel(job.job_id)
        entries = read_history(tmp_path / HISTORY_FILENAME)
        assert [e["state"] for e in entries] == ["cancelled"]


class TestEntryFromJob:
    def test_long_error_is_truncated(self) -> None:
        class _Job:
            job_id = "j"
            backend = "alpha"
            state = "failed"
            priority = "batch"
            result = {"error": "x" * 5000}  # noqa: RUF012

        entry = entry_from_job(_Job(), now=42.0)
        assert entry["ts"] == 42.0
        assert len(entry["error"]) == 400
        assert entry["queue_wait_sec"] is None


class TestMcpHistoryTool:
    @staticmethod
    def _call(args: dict | None = None) -> dict:
        from vramd import mcp_server as M

        msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call"}
        msg["params"] = {"name": "vramd_history", "arguments": args or {}}
        return M.handle_message(msg)["result"]

    @staticmethod
    def _payload(result: dict) -> dict:
        text = result["content"][0]["text"]
        return json.loads(text[text.index("{") :])

    def _seed(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / HISTORY_FILENAME
        monkeypatch.setattr("vramd.history.default_history_path", lambda: path)
        hist = JobHistory(path, max_entries=100)
        hist.record(_entry(0))
        hist.record(_entry(1, backend="beta", state="failed"))

    def test_works_without_a_running_supervisor(self, tmp_path: Path, monkeypatch) -> None:
        """Lê o ficheiro: responde ao "o que falhou esta noite?" com o daemon parado."""
        self._seed(tmp_path, monkeypatch)
        result = self._call({"state": "failed"})
        assert result["isError"] is False
        assert [j["job_id"] for j in self._payload(result)["jobs"]] == ["job-1"]

    def test_summary(self, tmp_path: Path, monkeypatch) -> None:
        self._seed(tmp_path, monkeypatch)
        data = self._payload(self._call({"summary": True, "limit": 1}))
        assert data["jobs"] == 2 and set(data["backends"]) == {"alpha", "beta"}

    def test_invalid_state_is_error(self, tmp_path: Path, monkeypatch) -> None:
        self._seed(tmp_path, monkeypatch)
        assert self._call({"state": "running"})["isError"] is True

    def test_tool_is_listed(self) -> None:
        from vramd import mcp_server as M

        names = {t["name"] for t in M._tool_definitions()}
        assert "vramd_history" in names
        assert names == set(M._TOOL_HANDLERS)
