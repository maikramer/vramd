# vramd

[![CI](https://github.com/maikramer/vramd/actions/workflows/ci.yml/badge.svg)](https://github.com/maikramer/vramd/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/vramd.svg)](https://pypi.org/project/vramd/)
[![Python](https://img.shields.io/pypi/pyversions/vramd.svg)](https://pypi.org/project/vramd/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**VRAM admission control for generative inference on consumer GPUs.**

One process holds the GPU and decides who gets in. It admits by the **real
peak** — weights + activation + margin, not just weights — queues with
priority and affinity, evicts by weight+LRU, and runs each model in its own
process and venv.

Built for inference that lasts **seconds to minutes** on a card that can't fit
everything. It's not an LLM server: it doesn't optimize token throughput, it
optimizes *fitting*.

```bash
pip install vramd          # 9 MB — the supervisor doesn't import torch
vramd start &
vramd submit my-model --prompt "…" --wait
```

## The problem

You have 6 GB of VRAM and five models that, together, ask for 40. Each runs
fine on its own. Together, the second job starts in the middle of the first
and both die with `CUDA out of memory` — after the weights were already
loaded.

The usual solutions assume something that isn't true here: that the model fits
(vLLM, TGI), that the environment is homogeneous (Ray Serve, Triton), or that
the unit of work is a token rather than a two-minute job.

## What vramd does differently

**Admits by the peak, not the weights.** The question isn't "does the model
fit" — it's "does the inference peak fit". It's the difference between
refusing in 0.2 s and dying at 80% of the job with the weights already
loaded.

**Each model in its own venv.** A backend is a process with its own
interpreter. Models with incompatible dependencies — torch 2.x vs 2.y,
different CUDA wheels — coexist without seeing each other.

**Affinity in the queue.** If the head of the queue needs a cold model and a
job further back needs one already in VRAM, the scheduler skips the head (up
to 3 times, then it forces). Where a load costs 60 s, this turns a 40-minute
batch into a 10-minute one.

**Cooperative cancellation.** Long jobs report progress per phase and stop
*between* phases — no CUDA kernels killed mid-flight.

**Measures instead of guessing.** `vramd calibrate` runs a real job, samples
VRAM per process at 20 Hz, and writes the measured footprint. Over ten real
models on an RTX 4050, hand-written values were off by between −3154 and
+22448 MiB.

**Keeps measuring.** Calibration is a lab snapshot; production is where the
numbers pay rent. The supervisor samples the real peak of every job it runs
and `vramd learn` compares it against what admission actually reserved —
`--apply` closes the loop by writing a corrected overlay, no GPU time spent
(see [Continuous learning](#continuous-learning)).

**Reacts and integrates.** Events (`on_job_done`, `on_evict`, `on_drift`, …)
fire shell hooks with a JSON payload, and `vramd mcp` exposes the queue to
AI agents over the Model Context Protocol — the same busy-guards apply.

## Integrating a model

Three methods:

```python
from vramd.worker import WorkerAdapter, run_worker_loop


class Adapter(WorkerAdapter):
    name = "my-model"

    def load(self, **kw):
        import torch, mylib

        return mylib.load(device=kw.get("device", "cuda"))

    def generate(self, model, request):
        if self.should_abort(request):
            return self.cancelled_response()
        self.report_progress(request, 0.0, "generating")
        return {"status": "ok", "output": model(request["prompt"])}

    def unload(self, model):
        del model


if __name__ == "__main__":
    run_worker_loop(Adapter, backend_name="my-model")
```

And register it — without touching vramd's code:

```yaml
# ~/.config/vramd/backends.d/my-model.yaml
version: 2
backends:
  - name: my-model
    adapter: my_package.adapter
    vram_mib: 4200
    priority: 20
    runtime:
      command: ["/opt/my-model/venv/bin/python", "-m", "my_package.worker"]
      env: { HF_HOME: ~/hf-cache }
    load_keys: [device, compute_type]
    shape_keys: [device]
```

Full runnable example: [`examples/echo-backend/`](examples/echo-backend/).

## Calibration

The friction of any such system is the "what numbers do I put in the
descriptor?" question. vramd's answer: none — you measure.

```bash
vramd calibrate my-model --repeats 3 --out ~/.config/vramd/backends.d/measured.yaml
```

It runs the job, splits **CUDA context / weights / activation** at phase
boundaries, and writes the descriptor. What it catches, that an estimate
can't:

| Signal | Why it matters |
|---|---|
| peak at **load** above the inference peak | loading fp16 and quantizing afterwards OOMs before generating |
| activation ≫ weights | the model loads another model inside `generate` |
| nothing resident after load | lazy loading: there's nothing to evict |
| `unload` that doesn't return VRAM | evicting this backend frees nothing — the eviction plan would be fiction |
| leak on repetition | the resident footprint grows with every job |
| warmup on 1st inference | calibrating with `--repeats 1` inflates the number |

Each measurement keeps the raw samples: `vramd recalibrate report.json`
recomputes the numbers when the analysis improves, without re-occupying the
GPU.

**Calibration works out of the box for backends that need inputs.** A
descriptor can declare the generation request and the load kwargs that
calibration should use by default, so `vramd calibrate <backend>` works
without flags even for backends that require inputs (`mesh_path`/`output`) or
specific formats:

```yaml
backends:
  - name: text3d
    calibrate_request: { mesh_path: test-mesh.glb, output: /tmp/out.glb }
    calibrate_load_kwargs: { compute_type: fp16 }
```

Short names that match a file bundled with the package (`test-mesh.glb`,
`test-image.png`) are resolved to the packaged path — no test model needed.
Load-kwargs precedence: hw-auto < descriptor < explicit.

## Continuous learning

Calibration measures once, under lab conditions. Production drifts: prompts
grow, a tool updates, someone switches the quant preset. While jobs run, the
supervisor samples each worker's VRAM (~2 Hz) and records the real peak:

```bash
vramd learn               # declared vs observed p95, per backend
vramd learn --apply       # writes ~/.config/vramd/backends.d/learned.yaml
```

What the verdicts mean:

| Verdict | Meaning | Action |
|---|---|---|
| subdimensionado | observed p95 exceeded what admission reserved — OOM risk | `--apply` raises `vram_mib` (p95 × 1.15) |
| sobredimensionado | declared ≥ 1.4× observed — admission refuses work that would fit | `--apply` trims (never below max × 1.25) |
| ok | healthy margin | nothing |

Rules the loop obeys: only subprocess backends are observed (a shared PID
would lie), failed jobs don't count toward verdicts (their peak is a lower
bound), a calibrated `vram:` block always wins over learned numbers, and
observations persist across restarts. `VRAMD_LEARN_INTERVAL_SEC=0` turns it
off.

## Hooks

`~/.config/vramd/hooks.yaml` — the daemon reacts without editing its code:

```yaml
hooks:
  - event: on_job_failed
    command: ["notify-send", "-u", "critical", "vramd", "${backend}: ${error_code}"]
  - event: on_drift
    command: ["curl", "-sS", "-XPOST", "https://hooks.example/vramd", "-d@-"]
    timeout_sec: 5
```

Events: `on_job_done`, `on_job_failed`, `on_job_cancelled`, `on_evict`,
`on_zero`, `on_drift`, `on_shutdown`. Payload arrives as JSON on stdin (plus
`VRAMD_EVENT`/`VRAMD_HOOK` env); `${field}` interpolates into argv. Hooks run
in daemon threads with timeouts and throttling — they can never stall the
queue.

## Agents (MCP)

`vramd mcp` speaks the Model Context Protocol over stdio — any MCP client
(Claude Desktop, agent frameworks) can drive the GPU queue:

```json
{"mcpServers": {"vramd": {"command": "vramd", "args": ["mcp"]}}}
```

Twelve tools, read-heavy by design: `vramd_status`, `vramd_queue`,
`vramd_learn`, `vramd_doctor`, … Mutations that free VRAM (`vramd_evict`,
`vramd_zero`, `vramd_preload`) require an explicit `confirm: true` argument
and remain subject to the supervisor's busy-guards. An agent that's told
"never kill GPU processes" is enforced by the same rules as a human.

## Commands

```
start stop status queue wait cancel flush backends preload evict reap
respawn zero stats top learn debug bench doctor calibrate recalibrate mcp
```

- `vramd status` / `queue` — who has the GPU and what's waiting
- `vramd top` — live dashboard (read-only, batch-safe): GPU bar, per-process
  VRAM, job progress, evict countdowns, drift verdicts
- `vramd zero` — frees all idle VRAM without stopping the supervisor
- `vramd respawn <backend>` — restarts a single worker (new code) without stopping the queue
- `vramd doctor` — environment diagnostics

**You never need `kill`.** Killing GPU processes works against the queue and
kills the wrong workload.

## Configuration

```
data/backends.yaml (example)  →  $VRAMD_BACKENDS_FILE  →  ~/.config/vramd/backends.d/*.yaml
```

Per-key overlay: a file with `{name: x, vram_mib: 5632}` fixes only that field
and inherits the rest. That's how a calibrated descriptor takes effect without
editing the package.

Variables: `VRAMD_BACKENDS_FILE`, `VRAMD_BACKENDS_DIR`, `VRAMD_TOOLS_ROOT`,
`VRAMD_MAX_INFLIGHT`, `VRAMD_MAX_QUEUE_DEPTH`, `VRAMD_VRAM_SAFETY_MIB`,
`VRAMD_PRIORITY`.

## Known limitations

Worth knowing before adopting:

- **`MAX_INFLIGHT=1` by default** — one generation at a time. The right choice
  for 6 GB, and it underuses an A100. There is support for >1 with VRAM
  checks, but no real *packing* yet.
- **Multi-GPU without central placement.** `gpu_ids` is passed to the worker;
  the supervisor neither decides placement nor accounts per device.
- **POSIX.** Pipe reads use `select`/`O_NONBLOCK`. Windows needs a different
  IO layer.
- **No authentication.** Unix socket with user permissions. Local, not shared.
- **Calibration isn't magic.** It measures what your pipeline does. A model
  that loads everything in fp16 at once doesn't start fitting because you
  measured it — you just learn that it doesn't fit, in 0.2 s instead of
  mid-job.

## Origin

Extracted from the [AiGameKit](https://github.com/maikramer), where it was
born to have ten generative models (text→image, →3D, →audio, →motion) share a
6 GB RTX 4050 without manual intervention. The numbers in this README are
measurements from that card.

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) — getting started, style, and what this
project values. The suite runs in ~27 s with no GPU.

## License

MIT — see [LICENSE](LICENSE).
