"""Testes do learn — loop fechado de picos observados vs declarados.

Sem GPU: o tracker recebe um sampler injectado e um pool fake com worker_pid.
"""

from __future__ import annotations

from typing import Any

import pytest

from vramd.learn import (
    VERDICT_NO_DATA,
    VERDICT_OK,
    VERDICT_OVER,
    VERDICT_UNDER,
    PeakLearningStore,
    PeakObservation,
    PeakTracker,
    analyze_drift,
    learn_overlay_yaml,
)


def _obs(backend: str = "alpha", peak: int = 1000, declared: int = 1200, ok: bool = True, **kw: Any) -> PeakObservation:
    return PeakObservation(
        backend=backend,
        job_id=kw.get("job_id", "j1"),
        peak_mib=peak,
        declared_peak_mib=declared,
        quant_mode=kw.get("quant_mode", "none"),
        memory_efficient=False,
        group_offload=False,
        duration_sec=1.0,
        samples=5,
        ok=ok,
        state=kw.get("state", "done"),
        started_at=kw.get("started_at", 1000.0),
    )


class TestAnalyzeDrift:
    def test_no_observations_is_no_data(self) -> None:
        report = analyze_drift([], declared_peak_mib=1000)
        assert report.verdict == VERDICT_NO_DATA
        assert report.samples == 0

    def test_few_samples_is_no_data(self) -> None:
        reports = analyze_drift([_obs(), _obs()], declared_peak_mib=1000)
        assert reports.verdict == VERDICT_NO_DATA
        assert reports.samples == 2

    def test_p95_above_declared_is_under(self) -> None:
        observations = [_obs(peak=1500 + i, declared=1400) for i in range(5)]
        report = analyze_drift(observations)
        assert report.verdict == VERDICT_UNDER
        assert report.suggested_mib is not None
        assert report.suggested_mib >= 1504 * 1.15 * 0.99  # p95*1.15 arredondado

    def test_declared_far_above_p95_is_over(self) -> None:
        observations = [_obs(peak=1000 + i, declared=4000) for i in range(5)]
        report = analyze_drift(observations)
        assert report.verdict == VERDICT_OVER
        # Nunca sugerir abaixo do máximo observado * 1.25 (safety sagrado).
        assert report.suggested_mib is not None
        assert report.suggested_mib >= 1004 * 1.25

    def test_healthy_margin_is_ok(self) -> None:
        observations = [_obs(peak=1000 + i, declared=1300) for i in range(5)]
        assert analyze_drift(observations).verdict == VERDICT_OK

    def test_failed_jobs_do_not_count_for_verdict(self) -> None:
        """Pico de job que morreu é lower bound — não valida o declarado."""
        ok_obs = [_obs(peak=500, declared=1200) for _ in range(2)]  # < min_samples
        failed = [_obs(peak=9000, declared=1200, ok=False, state="failed") for _ in range(5)]
        report = analyze_drift(ok_obs + failed)
        assert report.verdict == VERDICT_NO_DATA  # só 2 observações ok

    def test_declared_defaults_to_latest_observation(self) -> None:
        observations = [_obs(peak=1000, declared=None), _obs(peak=1010, declared=1300, job_id="j2")]
        report = analyze_drift(observations)
        assert report.declared_peak_mib == 1300


class TestStore:
    def test_roundtrip_and_recent(self, tmp_path) -> None:
        store = PeakLearningStore(root=tmp_path)
        store.append(_obs(job_id="j1"))
        store.append(_obs(job_id="j2", started_at=1001.0))
        recent = store.recent("alpha")
        assert [o.job_id for o in recent] == ["j1", "j2"]

    def test_backends_lists_files(self, tmp_path) -> None:
        store = PeakLearningStore(root=tmp_path)
        store.append(_obs(backend="alpha"))
        store.append(_obs(backend="beta"))
        assert store.backends() == ["alpha", "beta"]

    def test_cap_trims_oldest(self, tmp_path) -> None:
        store = PeakLearningStore(root=tmp_path, max_per_backend=3)
        for i in range(6):
            store.append(_obs(job_id=f"j{i}", started_at=1000.0 + i))
        recent = store.recent("alpha")
        assert [o.job_id for o in recent] == ["j3", "j4", "j5"]

    def test_corrupt_lines_skipped(self, tmp_path) -> None:
        store = PeakLearningStore(root=tmp_path)
        store.append(_obs())
        (tmp_path / "alpha.jsonl").write_text('{"backend": "alpha"\n', encoding="utf-8")
        assert store.recent("alpha") == []

    def test_reset(self, tmp_path) -> None:
        store = PeakLearningStore(root=tmp_path)
        store.append(_obs(backend="alpha"))
        store.append(_obs(backend="beta"))
        assert store.reset("alpha") == 1
        assert store.backends() == ["beta"]
        assert store.reset() == 1
        assert store.backends() == []


class _FakeJob:
    def __init__(self, job_id: str, backend: str, state: str = "running") -> None:
        self.job_id = job_id
        self.backend = backend
        self.request: dict[str, Any] = {}
        self.state = state


class _FakeQueue:
    def __init__(self, jobs: list[_FakeJob]) -> None:
        self._jobs = {j.job_id: j for j in jobs}

    def running_jobs(self) -> list[_FakeJob]:
        return [j for j in self._jobs.values() if j.state == "running"]

    def get(self, job_id: str) -> _FakeJob | None:
        return self._jobs.get(job_id)


class _FakePool:
    def __init__(self, pids: dict[str, int]) -> None:
        self._pids = pids

    def worker_pid(self, backend: str) -> int | None:
        return self._pids.get(backend)


class _FakeManager:
    def __init__(self, pids: dict[str, int] | None = None, declared: int = 1200) -> None:
        self._subprocess_pool = _FakePool(pids or {})
        self._declared = declared
        self.loaded_names = lambda: ["alpha"]

    def resolve_peak_params(self, name: str, request: dict) -> tuple[str, bool, bool, bool]:
        return ("none", False, False, False)

    def peak_vram_mib(self, name: str, **kw: Any) -> int:
        return self._declared

    class _registry:  # descriptor() levanta KeyError → has_measured False
        @staticmethod
        def descriptor(name: str) -> Any:
            raise KeyError(name)

    _registry = _registry()


class TestPeakTracker:
    def _tracker(
        self,
        tmp_path,
        samples: dict[int, list[int]],
        jobs: list[_FakeJob],
        manager: _FakeManager | None = None,
        **kw: Any,
    ) -> tuple[PeakTracker, list[tuple[PeakObservation, str]]]:
        seen: list[tuple[PeakObservation, str]] = []
        tracker = PeakTracker(
            _FakeQueue(jobs),
            manager or _FakeManager(pids={"alpha": 42}, declared=1500),
            store=PeakLearningStore(root=tmp_path),
            interval_sec=kw.pop("interval_sec", 999.0),  # thread nunca corre; ticks manuais
            on_observation=lambda obs, job: seen.append((obs, getattr(job, "state", "?"))),
            **kw,
        )
        calls = {pid: iter(vals) for pid, vals in samples.items()}
        tracker._sample_vram_mib = lambda pid: next(calls[pid], None)
        return tracker, seen

    def test_job_lifecycle_produces_observation(self, tmp_path) -> None:
        job = _FakeJob("j1", "alpha", state="running")
        tracker, seen = self._tracker(tmp_path, samples={42: [100, 900, 500]}, jobs=[job])

        tracker._tick()  # begin + sample 100
        tracker._tick()  # sample 900
        tracker._tick()  # sample 500
        assert tracker._active["j1"].peak_mib == 900

        job.state = "done"
        tracker._tick()  # finaliza
        assert "j1" not in tracker._active
        assert len(seen) == 1
        obs, state = seen[0]
        assert state == "done"
        assert obs.peak_mib == 900
        assert obs.declared_peak_mib == 1500
        assert obs.samples == 3
        # Persistiu: um tracker NOVO (restart) vê a observação.
        assert PeakLearningStore(root=tmp_path).recent("alpha")[0].job_id == "j1"

    def test_failed_job_recorded_but_not_counted(self, tmp_path) -> None:
        job = _FakeJob("j1", "alpha", state="running")
        tracker, seen = self._tracker(tmp_path, samples={42: [700]}, jobs=[job])
        tracker._tick()
        job.state = "failed"
        tracker._tick()
        assert seen[0][0].ok is False
        report = tracker.report_for("alpha")
        assert report.verdict == VERDICT_NO_DATA  # falhou → sem veredicto

    def test_no_worker_pid_no_observation(self, tmp_path) -> None:
        """Backend in-process (sem PID próprio): não aprender números mentirosos."""
        job = _FakeJob("j1", "alpha", state="running")
        tracker, seen = self._tracker(
            tmp_path, samples={}, jobs=[job], manager=_FakeManager(pids={}, declared=1500)
        )
        tracker._tick()
        job.state = "done"
        tracker._tick()
        assert seen == []

    def test_drift_callback_only_on_change(self, tmp_path) -> None:
        drifts: list[Any] = []

        def make_with_recents(observations_peaks: list[int]) -> PeakTracker:
            job = _FakeJob("j1", "alpha", state="running")
            tracker, _ = self._tracker(
                tmp_path,
                samples={},
                jobs=[job],
                on_drift=drifts.append,
            )
            # Ingestão directa de observações (o tick já cobrimos noutro teste).
            tracker._recents["alpha"] = [
                _obs(peak=p, declared=1000, job_id=f"j{i}") for i, p in enumerate(observations_peaks)
            ]
            return tracker

        t = make_with_recents([1500, 1600, 1700])
        t._maybe_drift("alpha")
        assert len(drifts) == 1 and drifts[0].verdict == VERDICT_UNDER
        t._maybe_drift("alpha")  # mesmo veredicto → sem spam
        assert len(drifts) == 1

    def test_status_dict_compact(self, tmp_path) -> None:
        job = _FakeJob("j1", "alpha", state="running")
        tracker, _ = self._tracker(tmp_path, samples={42: [800]}, jobs=[job])
        tracker._tick()
        job.state = "done"
        tracker._tick()
        block = tracker.status_dict()
        assert block["enabled"] is True
        assert block["backends"]["alpha"]["samples"] == 1

    def test_reset_clears_memory_and_disk(self, tmp_path) -> None:
        job = _FakeJob("j1", "alpha", state="running")
        tracker, _ = self._tracker(tmp_path, samples={42: [800]}, jobs=[job])
        tracker._tick()
        job.state = "done"
        tracker._tick()
        assert tracker.reset() == 1
        assert tracker.observations("alpha") == []


class TestOverlay:
    def test_only_actionable_without_measured(self) -> None:
        from vramd.learn import DriftReport

        reports = [
            DriftReport(backend="under", verdict=VERDICT_UNDER, declared_peak_mib=1000,
                        observed_p95_mib=1500, observed_max_mib=1600, samples=5, suggested_mib=1728),
            DriftReport(backend="ok", verdict=VERDICT_OK, declared_peak_mib=1000,
                        observed_p95_mib=900, observed_max_mib=950, samples=5, suggested_mib=None),
            DriftReport(backend="calibrado", verdict=VERDICT_UNDER, declared_peak_mib=1000,
                        observed_p95_mib=1500, observed_max_mib=1600, samples=5, suggested_mib=1728,
                        has_measured_block=True),
        ]
        import yaml

        doc = yaml.safe_load(learn_overlay_yaml(reports))
        names = {e["name"]: e["vram_mib"] for e in doc["backends"]}
        assert names == {"under": 1728}  # ok não entra; calibrado tem measured block

    def test_empty_is_valid_yaml(self) -> None:
        import yaml

        doc = yaml.safe_load(learn_overlay_yaml([]))
        assert doc == {"version": 2, "backends": []}


@pytest.mark.parametrize(
    ("peaks", "declared", "expected"),
    [
        ([1000, 1000, 1000], 1200, VERDICT_OK),
        ([1500, 1600, 1700], 1200, VERDICT_UNDER),
        ([1000, 1000, 1100], 3000, VERDICT_OVER),
    ],
)
def test_drift_matrix(peaks: list[int], declared: int, expected: str) -> None:
    observations = [_obs(peak=p, declared=declared) for p in peaks]
    assert analyze_drift(observations).verdict == expected
