"""Testes dos hooks de eventos — parse, dispatch, substituição, throttle."""

from __future__ import annotations

import json

import pytest
import yaml

from vramd.hooks import (
    EVENT_DRIFT,
    EVENT_JOB_DONE,
    EVENT_JOB_FAILED,
    HookRunner,
    HookSpec,
    load_hooks,
    parse_hooks,
)


class TestParseHooks:
    def test_single_event_and_list_command(self) -> None:
        specs = parse_hooks({"hooks": [{"event": EVENT_JOB_DONE, "command": ["notify-send", "vramd"]}]})
        assert len(specs) == 1
        assert specs[0].events == {EVENT_JOB_DONE}
        assert specs[0].command == ("notify-send", "vramd")

    def test_string_command_is_shlex_split(self) -> None:
        specs = parse_hooks({"hooks": [{"event": EVENT_JOB_DONE, "command": "sh -c 'echo hi'"}]})
        assert specs[0].command == ("sh", "-c", "echo hi")

    def test_multiple_events(self) -> None:
        specs = parse_hooks(
            {
                "hooks": [
                    {"events": [EVENT_JOB_DONE, EVENT_JOB_FAILED], "command": ["x"]},
                ]
            }
        )
        assert specs[0].matches(EVENT_JOB_DONE)
        assert specs[0].matches(EVENT_JOB_FAILED)
        assert not specs[0].matches(EVENT_DRIFT)

    @pytest.mark.parametrize(
        "entry",
        [
            {},  # sem evento nem comando
            {"event": EVENT_JOB_DONE},  # sem comando
            {"event": EVENT_JOB_DONE, "command": []},
            {"event": "on_nada", "command": ["x"]},  # evento desconhecido
            {"event": EVENT_JOB_DONE, "command": ["x"], "timeout_sec": "rápido"},
            {"event": EVENT_JOB_DONE, "command": {"a": 1}},  # command não str/lista
        ],
    )
    def test_invalid_entries_raise(self, entry: dict) -> None:
        with pytest.raises(ValueError):
            parse_hooks({"hooks": [entry]})

    def test_non_list_hooks_raise(self) -> None:
        with pytest.raises(ValueError):
            parse_hooks({"hooks": "x"})


class TestLoadHooks:
    def test_missing_file_is_empty(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("VRAMD_HOOKS_FILE", str(tmp_path / "não-existe.yaml"))
        assert load_hooks() == []

    def test_env_override(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / "hooks.yaml"
        doc = yaml.safe_dump({"hooks": [{"event": EVENT_DRIFT, "command": ["reboot-world"]}]})
        path.write_text(doc, encoding="utf-8")
        monkeypatch.setenv("VRAMD_HOOKS_FILE", str(path))
        specs = load_hooks()
        assert specs[0].command == ("reboot-world",)

    def test_malformed_doc_raises(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / "hooks.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")  # lista, não mapa
        monkeypatch.setenv("VRAMD_HOOKS_FILE", str(path))
        with pytest.raises(ValueError):
            load_hooks()


class TestHookRunner:
    def _recording_runner(self, calls: list) -> object:
        def run(argv, timeout_sec, env, input_text):
            calls.append({"argv": argv, "timeout": timeout_sec, "env": env, "input": input_text})
            return 0

        return run

    def test_dispatch_sync_runs_matching_hook(self) -> None:
        calls: list = []
        runner = HookRunner(
            [HookSpec(events=frozenset({EVENT_JOB_DONE}), command=("echo", "${backend}"))],
            runner=self._recording_runner(calls),
        )
        launched = runner.dispatch_sync(EVENT_JOB_DONE, {"backend": "text3d"})
        assert launched == 1
        assert calls[0]["argv"] == ["echo", "text3d"]
        assert calls[0]["env"]["VRAMD_EVENT"] == EVENT_JOB_DONE
        payload = json.loads(calls[0]["input"])
        assert payload["backend"] == "text3d"
        assert runner.stats.to_dict()["succeeded"] == 1

    def test_unknown_field_substitutes_empty(self) -> None:
        calls: list = []
        runner = HookRunner(
            [HookSpec(events=frozenset({EVENT_JOB_FAILED}), command=("echo", "${erro_que_nao_existe}"))],
            runner=self._recording_runner(calls),
        )
        runner.dispatch_sync(EVENT_JOB_FAILED, {"backend": "x"})
        assert calls[0]["argv"] == ["echo", ""]

    def test_non_matching_event_no_run(self) -> None:
        calls: list = []
        runner = HookRunner(
            [HookSpec(events=frozenset({EVENT_DRIFT}), command=("x",))],
            runner=self._recording_runner(calls),
        )
        assert runner.dispatch_sync(EVENT_JOB_DONE, {}) == 0
        assert calls == []

    def test_failure_counts_and_continues(self) -> None:
        def boom(argv, timeout_sec, env, input_text):
            raise RuntimeError("curl morto")

        runner = HookRunner(
            [HookSpec(events=frozenset({EVENT_JOB_DONE}), command=("curl",))],
            runner=boom,
        )
        runner.dispatch_sync(EVENT_JOB_DONE, {})
        assert runner.stats.to_dict()["failed"] == 1

    def test_timeout_exit_code_recorded(self) -> None:
        runner = HookRunner(
            [HookSpec(events=frozenset({EVENT_JOB_DONE}), command=("sleep",))],
            runner=lambda *a: -9,
        )
        runner.dispatch_sync(EVENT_JOB_DONE, {})
        assert runner.stats.to_dict()["last_error"] and "-9" in runner.stats.to_dict()["last_error"]

    def test_dispatch_throttles_bursts(self) -> None:
        """Rajada do mesmo evento → 1 lançamento + throttled (threads, mas o
        throttle é decidido de forma síncrona antes do spawn)."""
        calls: list = []
        runner = HookRunner(
            [HookSpec(events=frozenset({EVENT_JOB_DONE}), command=("echo",))],
            min_interval_sec=60.0,
            runner=self._recording_runner(calls),
        )
        first = runner.dispatch(EVENT_JOB_DONE, {})
        second = runner.dispatch(EVENT_JOB_DONE, {})
        assert first == 1
        assert second == 0
        assert runner.stats.to_dict()["throttled"] == 1

    def test_status_dict(self) -> None:
        runner = HookRunner([HookSpec(events=frozenset({EVENT_JOB_DONE, EVENT_DRIFT}), command=("x",))])
        block = runner.status_dict()
        assert block["configured"] == 1
        assert block["events"] == ["on_drift", "on_job_done"]


def test_end_to_end_real_subprocess(tmp_path):
    """O runner default executa um comando real com o payload no stdin."""
    out = tmp_path / "payload.json"
    runner = HookRunner(
        [HookSpec(events=frozenset({EVENT_JOB_DONE}), command=("tee", str(out)))],
    )
    runner.dispatch_sync(EVENT_JOB_DONE, {"backend": "text3d", "job_id": "j1"})
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == {"event": EVENT_JOB_DONE, "hook": "tee", "backend": "text3d", "job_id": "j1"}
