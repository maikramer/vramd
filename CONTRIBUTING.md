# Contributing

## Getting started

```bash
git clone https://github.com/maikramer/vramd
cd vramd
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q            # 775 tests, ~27 s, no GPU
```

The suite is **CPU-only on purpose**. The supervisor doesn't import torch, and
the test workers are stand-ins that speak the real protocol. If one of your
tests needs a GPU, it's probably testing the model, not `vramd`.

## Before opening a PR

```bash
ruff check . && ruff format --check .
pytest -q
```

## What this project values

**Measure instead of estimate.** The whole reason the calibrator exists is
that hand-written footprints are wrong — over ten real models, between −3154
and +22448 MiB. If you change admission numbers, say how you verified them. A
`vramd calibrate` run before/after is worth more than an argument.

**Fail early and explain.** Refusing a job in 0.2 s with an actionable message
is better than accepting it and dying with an OOM at 80%. When something can't
be trusted, the code lowers confidence and says why — it doesn't round to a
plausible number.

**One test per bug, named after the real case.** This repo's regression tests
cite what revealed them (`texture2d marked 19% "no data"`, `text2icon: 82 MiB
of 4764`). That makes it obvious, two years later, why the condition exists.

**Comments explain the why, not the what.** The code already says what.

## Style

- English for comments and docstrings (the project's language).
- Google-style docstrings; `from __future__ import annotations` first.
- 120 columns, double quotes — all applied by `ruff` (`ruff.toml`).
- Types in new code. `Any` is acceptable for model objects.

## Structure

```
src/vramd/
  server.py cli.py             supervisor and CLI
  job_queue.py scheduler.py    queue: priority + VRAM affinity
  backend_manager.py           load, admission, eviction
  vram_planner.py              eviction plan (pure, no GPU)
  subprocess_pool.py           persistent workers (JSONL over stdin/stdout)
  registry.py                  descriptors + configuration layers
  calibrate/                   footprint measurement
  worker/                      model-side SDK
  client.py                    submit/wait/cancel
```

`vram_planner.py`, `job_queue.py` and `calibrate/analysis.py` are **pure** —
no GPU, no sockets, no threads. Keeping them that way is what makes the suite
fast.

## Releases

Semantic versioning. Publishing is: update `CHANGELOG.md` and the version in
`pyproject.toml`, and create the tag `vX.Y.Z` — the workflow handles the rest
(build, PyPI via Trusted Publishing, GitHub release).
