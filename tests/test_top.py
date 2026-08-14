"""Testes do dashboard ``vramd top`` — renderização com dados fabricados (sem GPU)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from rich.console import Console

from vramd.top import _fmt_sec, _mib, _short_job_id, render_dashboard, run_top


def _render_text(data: dict[str, Any], width: int = 160) -> str:
    console = Console(width=width, force_terminal=False, record=True)
    console.print(render_dashboard(data))
    return console.export_text()


class TestFormatters:
    def test_short_job_id(self) -> None:
        assert _short_job_id("abcdef1234567890", n=8) == "abcdef12…"
        assert _short_job_id("curto") == "curto"
        assert _short_job_id(None) == "?"

    def test_mib(self) -> None:
        assert _mib(4096) == "4 096 MiB"
        assert _mib(None) == "—"

    def test_fmt_sec(self) -> None:
        assert _fmt_sec(45.2) == "45s"
        assert _fmt_sec(125) == "2m05s"
        assert _fmt_sec(7200) == "2h00m"
        assert _fmt_sec(None) == "—"


class TestRender:
    def test_down_renders_waiting_panel(self) -> None:
        text = _render_text({"running": False})
        assert "vramd" in text and "não está ativo" in text

    def test_full_frame_renders_all_sections(self) -> None:
        data = {
            "running": True,
            "status": {
                "pid": 42,
                "socket": "/tmp/vramd.sock",
                "requests_served": 7,
                "eta_sec": 90,
                "queue": {"queue_depth": 1, "inflight": 1},
                "idle_evict_timeout_sec": 120.0,
                "backends": [
                    {
                        "name": "alpha",
                        "loaded": True,
                        "peak_mib": 1200,
                        "ref_count": 1,
                        "last_used": 0.0,
                    },
                    {"name": "beta", "loaded": False, "peak_mib": 3000, "ref_count": 0, "last_used": 0.0},
                ],
                "learn": {
                    "backends": {
                        "alpha": {"verdict": "underprovisioned", "observed_p95_mib": 1800, "samples": 4}
                    }
                },
            },
            "queue": {
                "running": [
                    {
                        "job_id": "j123456789abc",
                        "backend": "alpha",
                        "priority": "interactive",
                        "progress_pct": 0.42,
                        "progress_msg": "diffusion",
                        "generate_sec": 12.5,
                    }
                ],
                "queued": [
                    {"job_id": "j987654321def", "backend": "beta", "priority": "batch", "queue_wait_sec": 30.0}
                ],
                "eta_sec": 90,
            },
            "gpu": SimpleNamespace(name="RTX 4050", total_mib=6144, free_mib=2048),
            "procs": [(42, 512), (99, 128)],
        }
        text = _render_text(data)
        assert "alpha" in text and "beta" in text
        assert "running" in text and "queued" in text
        assert "1 800 MiB" in text  # p95 observado do learn
        assert "RTX 4050" in text

    def test_learn_marker_under(self) -> None:
        """Veredicto underprovisioned marca o backend com risco visível."""
        data = {
            "running": True,
            "status": {
                "pid": 1,
                "backends": [{"name": "alpha", "loaded": True, "peak_mib": 1, "ref_count": 0, "last_used": 0.0}],
                "learn": {"backends": {"alpha": {"verdict": "underprovisioned", "observed_p95_mib": 99}}},
            },
            "queue": {"running": [], "queued": []},
        }
        text = _render_text(data)
        assert "!" in text


class TestRunOnce:
    def test_run_top_once_prints_frame(self, capsys: Any) -> None:
        class FixedSource:
            def fetch(self) -> dict[str, Any]:
                return {"running": False}

        rc = run_top(once=True, source=FixedSource())
        assert rc == 0
        assert "não está ativo" in capsys.readouterr().out
