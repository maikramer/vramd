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
export VRAMD_BACKENDS_FILE=/path/to/backends.yaml   # your registry
vramd validate            # lint the descriptors before the first job
vramd start &             # refuses to start with zero backends configured
vramd submit my-model --prompt "…" -p steps=8 --wait
```

Windows uses the same CLI (`vramd start` / `status` / `stop`); the endpoint and
paths change — see [Platforms](#platforms).

## Share GPUs on the network

One machine can expose its GPUs to the LAN and run jobs whose **code and
dependencies come from the client**:

```bash
# GPU server — only Python, vramd, uv and the NVIDIA driver
pip install vramd
vramd config          # wizard: listener, token, cache, HF_HOME, systemd
```

The wizard writes `config.yml`, creates the 0600 token, optionally migrates an
existing Hugging Face cache **without re-downloading**, and installs a systemd
unit. No Docker, no project checkout, no tool venvs on the server.

```bash
# client
export VRAMD_REMOTE=host:8790
export VRAMD_REMOTE_TOKEN=…        # /etc/vramd/token on the server
vramd status
vramd devices                      # local lanes + remote peers
vramd models list
```

A client opens a **session** with a content-addressed code bundle (sources +
vendor) and one exact dependency lock per tool. vramd verifies and extracts the
bundle, builds or reuses `<cache>/envs/<hash>` with `uv`, and starts workers
with `PYTHONPATH` pointing at the bundle. Changing code creates a new bundle
hash; changing a lock creates a new env; nothing is silently executed from an
older copy. The server only needs `--allow-client-code` (and
`--allow-client-env` for `uv` builds) when you trust the client with code
execution.

Jobs carry inputs as SHA-256 blobs (`blob-has` skips uploads the server already
has) and return outputs through binary frames, not base64. `vramd sessions
list/close` and `vramd models gc` cover the lifecycle; `vramd devices` and
`vramd route` include remote peers when `VRAMD_PEERS` is configured.

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

Thirteen tools, read-heavy by design: `vramd_status`, `vramd_queue`,
`vramd_learn`, `vramd_history`, `vramd_doctor`, … Mutations that free VRAM (`vramd_evict`,
`vramd_zero`, `vramd_preload`) require an explicit `confirm: true` argument
and remain subject to the supervisor's busy-guards. An agent that's told
"never kill GPU processes" is enforced by the same rules as a human.

## Routing: several GPUs, several machines

With one GPU the only question is "does it fit?". With more hardware the
question becomes "where does it finish first?" — and vramd answers it per job.
For every candidate route it estimates, in seconds:

```
total = wait + eviction + load + generation + transfer
```

and takes the smallest. Three strategies fall out of that one model:

| Route | When it wins |
|---|---|
| **Replica** — the same model on several GPUs | a batch is queued behind the warm GPU and that backlog is longer than a cold load on a free one. A single job never pays a second load. |
| **Split** — one model across GPUs | only when the peak fits on no single GPU and the backend declares `placement.split`. It pays a slowdown, so it never beats a GPU where the model fits whole. |
| **Network** — another vramd | the local GPUs are busy or too small, and latency + file transfer at the *measured* bandwidth + the peer's queue still finishes sooner. |

```
$ vramd route text3d -p mesh_path=in.glb
text3d: pico 5632 MiB, load ~41s, geração ~96s, inputs 38.2 MiB

      Rota        Tipo        Total   Espera   Evict   Load   Geração   Rede   Nota
  →   gpu1        GPU local    137s     0.0s    0.0s    41s       96s   0.0s   a frio
      gpu0        GPU local    171s      75s    0.0s   0.0s       96s   0.0s   quente
      peer:box    rede         204s      60s    0.0s    41s       96s   4.8s   a frio

run: GPU local gpu1 (réplica nova), pronto em ~137s; alternativa gpu0 em ~171s
```

`vramd route` is a dry run — it shows every route considered, including the
impossible ones and why. `vramd devices` and `vramd peers` show what the router
sees; `vramd history` and the job's `route` field show what it chose.

**Several GPUs.** Each GPU gets a *lane*: its own backend manager and its own
workers, pinned with `CUDA_VISIBLE_DEVICES`. Every lane believes it has one
GPU, so peak admission, eviction and calibration work unchanged per device.
Detected automatically when there is more than one GPU; `VRAMD_GPUS=0,2` picks
a subset (a single index pins the supervisor to that GPU), `VRAMD_GPUS=legacy`
restores the single unpinned lane. An explicit `gpu_ids` in a request is
honoured as a pin.

**Other machines.** Peers are other supervisors started with `--listen`:

```yaml
# ~/.config/vramd/peers.yaml      (or VRAMD_PEERS=box=10.0.0.2:8790 + VRAMD_PEERS_TOKEN)
peers:
  - name: box
    address: 10.0.0.2:8790
    token_file: ~/.config/vramd/box.token
    bandwidth_mib_s: 110        # optional prior; replaced by measurements
```

The supervisor polls each peer (queue, warm models, GPU sizes), measures
round-trip time and learns bandwidth from real transfers. A forwarded job
uploads its input files to the peer's sandbox, streams progress back, downloads
the outputs to the paths you asked for, and is never forwarded a second time.
If the peer dies before accepting the job, it goes back to the queue and takes
another route.

**What a backend declares** — all optional:

```yaml
backends:
  - name: text3d
    placement:
      split: true                  # the adapter can spread over several GPUs
      max_replicas: 2              # cap copies loaded at once (0 = no cap)
      remote: false                # never leave this machine
      input_keys: [mesh_path]      # request fields that are files to upload
      output_keys: [output]        # request fields that are files to bring back
      load_sec: 40                 # priors until real timings exist
      gen_sec: 90
```

A split worker sees its GPUs as `0..k-1` and receives `gpu_ids` plus
`max_memory_mib` (the per-GPU quota the router computed) — put both in
`load_keys`. Peers carry your prompts and files in the clear and a peer's token
is full control over it: trusted network or WireGuard only (see
[`SECURITY.md`](SECURITY.md)).

**Quality follows the GPU, criticality decides who gets the big one.** A
client's hw-auto used to pick quantization for the GPU of the machine that
*submitted* — wrong as soon as jobs land on another card. Instead a request can
carry `hw_tiers`: the tool's own hw-auto profile for every GPU size the router
might use (`{"cmd": "topology"}` lists them):

```json
{"cmd": "submit", "backend": "text3d", "sdnq_preset": "sdnq-int4", "criticality": "hero",
 "hw_tiers": [{"min_vram_mib": 0, "label": "int4", "set": {}},
              {"min_vram_mib": 12288, "label": "fp16", "set": {"sdnq_preset": null, "memory_efficient": false}}]}
```

The router picks the tier per route — peak, admission and timing come from it —
and the worker receives the request with that tier applied (`null` removes a
key). `criticality` turns tier differences into seconds: `high`/`hero` waits
for the GPU with the best tier (up to `VRAMD_ROUTE_QUALITY_HIGH_SEC` per tier),
`low`/`filler` stays on the smallest GPU that fits — more firmly when critical
jobs are queued — and `normal` just takes the fastest route. Generation time is
learned per tier, so an fp16 job on the 24 GB card isn't estimated with the
int4 average.

## Commands

```
start stop status submit queue wait cancel flush backends preload evict reap
respawn zero stats top learn debug bench doctor calibrate recalibrate mcp
history logs metrics validate completion config route devices peers models env
sessions
```

- `vramd config` — wizard/flags, `config.yml`, token, HF cache migration and
  systemd unit for a GPU server
- `vramd devices` — local GPUs/lanes and remote peers: VRAM, loaded models,
  occupancy
- `vramd models list/ensure/gc` — the normal Hugging Face cache, pinned while a
  session is open
- `vramd env list/build` — content-addressed Python environments built with
  `uv` from a client lock
- `vramd sessions list/close` — client code sessions (`<backend>@<session>`
  workers with bundle, env and model pins)
- `vramd submit <backend>` — queue a job: `-p K=V` (repeatable),
  `--request-json '{…}'` / `@job.json` / `-` (stdin); `--wait` streams queue
  position and per-phase progress, and exits non-zero if the job fails
- `vramd status` / `queue` — who has the GPU and what's waiting
- `vramd history` — what ran, how long it took, what failed
- `vramd logs [backend]` — supervisor log, or a worker's stderr (`-f` follows)
- `vramd top` — live dashboard (read-only, batch-safe): GPU bar, per-process
  VRAM, job progress, evict countdowns, drift verdicts
- `vramd zero` — frees all idle VRAM without stopping the supervisor
- `vramd respawn <backend>` — restarts a single worker (new code) without stopping the queue
- `vramd doctor` — environment diagnostics

**You never need `kill`.** Killing GPU processes works against the queue and
kills the wrong workload.

## Operating it

**Check the config before starting.** The registry is forgiving on purpose — a
corrupt overlay is skipped so the supervisor still boots — which means a typo
is silently ignored. `vramd validate` reads the same layers and says what's
wrong, with no GPU and without spawning anything:

```
$ vramd validate
✗ erro   text3d: worker não arranca — executável não existe: /opt/text3d/venv/bin/python
! aviso  my-model: chave desconhecida 'prority' — é ignorada pelo supervisor
         → querias dizer 'priority'?
! aviso  sky: vram_mib=7000 excede a VRAM total da GPU (6141 MiB) — nunca será admitido
```

Exit code 1 on errors (`--strict`: on warnings too), `--json` for CI.

**After a batch.** Finished jobs are appended to `vramd-history.jsonl` next to
the socket (capped by `VRAMD_HISTORY_MAX`, default 2000; requests are *not*
stored — prompts can be private). It's a file, so it answers even when the
supervisor is down:

```bash
vramd history --failed --since 12h     # what broke overnight, with error codes
vramd history --summary                # per backend: success rate, avg/p95, GPU time
vramd logs text3d -n 100               # the worker's stderr for the one that failed
```

**Monitoring.** `vramd metrics` prints the Prometheus text format — queue
depth, per-backend loads/generations/errors, worker recycles, learn drift, GPU
free/total. No HTTP server to secure: point node_exporter's textfile collector
at a file refreshed by a timer.

```bash
vramd metrics --textfile /var/lib/node_exporter/textfile/vramd.prom
```

With the supervisor down it emits `vramd_up 0` and exits 1, so stale series
don't freeze at their last value.

**Shell completion** (commands, flags, backend names):

```bash
eval "$(vramd completion bash)"        # or zsh; fish: vramd completion fish > ~/.config/fish/completions/vramd.fish
```

## Configuration

```
data/backends.yaml (empty)  →  $VRAMD_BACKENDS_FILE  →  ~/.config/vramd/backends.d/*.yaml
```

On Windows the user layer is `%APPDATA%\vramd\backends.d\*.yaml`. The packaged
registry is **empty on purpose**: `vramd start` refuses to serve fake example
backends (they'd grab the socket and real delegation would fail with opaque
errors). Descriptor examples live in
[`examples/backends.yaml`](examples/backends.yaml); `vramd start --allow-empty`
escapes the guard for tests.

Per-key overlay: a file with `{name: x, vram_mib: 5632}` fixes only that field
and inherits the rest. That's how a calibrated descriptor takes effect without
editing the package.

Variables: `VRAMD_BACKENDS_FILE`, `VRAMD_BACKENDS_DIR`, `VRAMD_TOOLS_ROOT`,
`VRAMD_MAX_INFLIGHT`, `VRAMD_MAX_QUEUE_DEPTH`, `VRAMD_VRAM_SAFETY_MIB`,
`VRAMD_PRIORITY`, `VRAMD_CACHE_DIR`, `VRAMD_TRANSPORT` (`unix`/`tcp`),
`VRAMD_WORKER_MAX_JOBS` (default 12). A descriptor can override the recycle
limit per backend with `runtime.max_jobs` — useful for fast, stable models that
should not pay a reload every dozen jobs. `VRAMD_HISTORY_MAX` caps the job
history (default 2000; `0` disables it).

## Diagnosing the queue

`vramd status --json` / `vramd debug` expose the supervisor's own view of each
persistent worker: PID, alive/loaded, jobs served vs. recycle limit, spawn and
recycle counts, last exit code, last abort reason, last progress (`pct` + stage),
age of the last event and the worker log path. The `learn` block also reports
the VRAM probe (`last_reason`, error, consecutive failures, backoff) and the
active sampled jobs, so a blind tracker is visible instead of looking like
"no data". Failed jobs carry a `worker` block and the tail of the worker stderr
in `worker_log_tail`.

## Platforms

**Linux/macOS** listen on a Unix domain socket at
`~/.cache/vramd/vramd.sock`; the socket's file permissions are the access
control.

**Windows** uses a loopback TCP endpoint. `%LOCALAPPDATA%\vramd\vramd.sock` is
a small JSON descriptor with the ephemeral port and a per-process token;
requests without that token are refused (a Unix socket restricts by file owner,
a loopback port does not). Worker stdout is drained by a reader thread — the
Windows `select()` only works on sockets, not pipes — the singleton lock uses
`msvcrt.locking`, and process discovery/kill go through `psutil` + NVML because
there is no `/proc`.

```powershell
pip install vramd
$env:VRAMD_BACKENDS_FILE = "C:\path\to\backends.yaml"
vramd start
vramd status
vramd stop
```

Worker commands must point at the tool venv's interpreter
(`...\Scripts\python.exe`) or at an executable on `PATH`; `runtime.command`
works the same as on Linux. `nvidia-smi` is found in `System32`/`NVSMI` even
when it is not on `PATH`. `vramd doctor` reports the endpoint transport, the
NVIDIA driver/GPU and each backend's imports.

`VRAMD_TRANSPORT=tcp` forces the TCP endpoint on Linux too — useful for CI and
for reproducing Windows behaviour without a Windows machine.

## Known limitations

Worth knowing before adopting:

- **One generation per GPU.** Each GPU runs one job at a time (`MAX_INFLIGHT`
  only applies to the single-GPU mode). The right choice for 6 GB, and it
  underuses an A100 — there is no *packing* of several small jobs on one card.
- **Routing estimates are estimates.** Load and generation times come from
  measured averages (or the `placement` priors before any exist); the first
  jobs of a new backend are routed on guesses. Split is opt-in and only as
  good as the adapter's own multi-GPU support. `vramd learn` observes the
  primary GPU's workers only.
- **One supervisor per host.** Start-up reaps other vramd supervisors on the
  same machine as orphans — peers are meant to be other machines
  (`VRAMD_REAP_ON_START=0` if you really want two on one box).
- **No authentication on Unix sockets.** Access is the socket's file
  permissions (the Windows TCP endpoint adds a token, see *Platforms*).
- **Windows process trees are best-effort.** There are no POSIX process groups;
  a worker's descendants are terminated explicitly through `psutil` on abort,
  but a grandchild that fully detaches can survive. Prefer cooperative abort or
  `vramd zero` over killing processes.
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
project values. The suite runs in about a minute with no GPU.

## License

MIT — see [LICENSE](LICENSE).
