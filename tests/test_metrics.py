"""Exposição Prometheus (``vramd metrics``) — função pura sobre status/stats."""

from __future__ import annotations

import re
from pathlib import Path

from vramd.metrics import render_prometheus
from vramd.registry import Registry
from vramd.server import VramdServer

_SAMPLE_RE = re.compile(r'^[a-z_][a-z0-9_]*(\{[a-z_]+="(?:[^"\\]|\\.)*"(,[a-z_]+="(?:[^"\\]|\\.)*")*\})? -?[0-9.e+-]+$')


def _samples(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            key, _, value = line.rpartition(" ")
            out[key] = float(value)
    return out


class TestRenderPrometheus:
    def test_down_emits_only_up_zero(self) -> None:
        """Supervisor em baixo: séries antigas não podem ficar congeladas."""
        text = render_prometheus(None)
        assert _samples(text) == {"vramd_up": 0.0}

    def test_down_still_reports_gpu(self) -> None:
        got = _samples(render_prometheus(None, gpu={"free_mib": 100, "total_mib": 6141}))
        assert got == {"vramd_up": 0.0, "vramd_gpu_free_mib": 100.0, "vramd_gpu_total_mib": 6141.0}

    def test_real_server_status_is_valid_exposition(self, tmp_path: Path) -> None:
        srv = VramdServer(registry=Registry(), socket_path=tmp_path / "ums-metrics.sock")
        srv.manager.stats.record_load("text2icon", 3.0)
        srv.manager.stats.record_generate("text2icon", 1.5)
        srv.manager.stats.record_error("text2icon", "boom")
        srv.queue.enqueue("text2icon", {})
        text = render_prometheus(srv._dispatch({"cmd": "status"}), srv._dispatch({"cmd": "stats"}))

        assert text.endswith("\n")
        for line in text.splitlines():
            if line.startswith("#"):
                assert line.startswith(("# HELP vramd_", "# TYPE vramd_")), line
            else:
                assert _SAMPLE_RE.match(line), line
        got = _samples(text)
        assert got["vramd_up"] == 1.0
        assert got["vramd_queue_depth"] == 1.0
        assert got["vramd_jobs_enqueued_total"] == 1.0
        assert got['vramd_backend_loaded{backend="text2icon"}'] == 0.0
        assert got['vramd_backend_generations_total{backend="text2icon"}'] == 1.0
        assert got['vramd_backend_errors_total{backend="text2icon"}'] == 1.0
        assert got['vramd_backend_generate_seconds_avg{backend="text2icon"}'] == 1.5

    def test_help_and_type_emitted_once_per_metric(self) -> None:
        status = {
            "backends": [
                {"name": "a", "loaded": True, "vram_mib": 100},
                {"name": "b", "loaded": False, "vram_mib": 200},
            ]
        }
        text = render_prometheus(status)
        assert text.count("# TYPE vramd_backend_loaded gauge") == 1
        assert text.count("vramd_backend_loaded{") == 2

    def test_missing_values_are_omitted_not_zeroed(self) -> None:
        """Um 0 inventado é indistinguível de uma medição real num gráfico."""
        text = render_prometheus({"eta_sec": None, "worker_vram_mib": None, "queue": {}})
        assert "vramd_queue_eta_seconds" not in text
        assert "vramd_worker_vram_mib" not in text
        assert "vramd_queue_depth" not in text

    def test_nan_and_non_numeric_are_dropped(self) -> None:
        text = render_prometheus({"eta_sec": float("nan"), "requests_served": "muitos"})
        assert "nan" not in text.lower()
        assert "vramd_requests_served_total" not in text

    def test_label_values_are_escaped(self) -> None:
        text = render_prometheus({"backends": [{"name": 'we"ird\\na\nme', "loaded": True}]})
        assert 'backend="we\\"ird\\\\na\\nme"' in text

    def test_learn_and_worker_blocks(self) -> None:
        status = {
            "learn": {"backends": {"a": {"verdict": "ok", "observed_p95_mib": 4100, "samples": 7}}},
            "debug": {"workers": {"a": {"alive": True, "spawn_count": 3, "recycle_count": 2, "jobs_served": 5}}},
            "hooks": {"configured": 2, "fired": 9, "failed": 1, "throttled": 0},
        }
        got = _samples(render_prometheus(status))
        assert got['vramd_learn_observed_p95_mib{backend="a"}'] == 4100.0
        assert got['vramd_learn_verdict{backend="a",verdict="ok"}'] == 1.0
        assert got['vramd_worker_spawns_total{backend="a"}'] == 3.0
        assert got['vramd_worker_alive{backend="a"}'] == 1.0
        assert got["vramd_hooks_failed_total"] == 1.0
