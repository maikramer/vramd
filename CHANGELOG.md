# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/1.1.0/).
Versioning: [SemVer](https://semver.org/).

## [0.2.3] — 2026-08-08

### Adicionado

- **Assets de teste empacotados** (`data/test-mesh.glb`, `data/test-image.png`):
  o calibrador resolve nomes curtos no `calibrate_request` (ex.:
  `mesh_path: test-mesh.glb`) para o package data do vramd — backends que
  exigem um mesh/imagem de entrada calibram sem o utilizador ter de fornecer
  nada.

## [0.2.2] — 2026-08-08

### Added

- **`calibrate_request` / `calibrate_load_kwargs` on the descriptor**: the YAML
  can declare the generation request and the load kwargs that calibration
  should use by default — `vramd calibrate <backend>` works without flags even
  for backends that require inputs (mesh_path/output) or specific formats.
  Load-kwargs precedence: hw-auto < descriptor < explicit.

## [0.2.1] — 2026-08-08

### Fixed

- **camelCase venv discovery** (`toolchain._candidate_dirs`):
  `text2icon` couldn't find the `Text2Icon` folder (nor `skymap2d`→`Skymap2D`,
  `paint3d`→`Paint3D`…) because `str.capitalize()` only capitalizes the first
  letter. The AiGameKit layout (folder capitalized per segment) is now covered
  by `_camel_title`. Without this the tool workers never spawned and clients
  fell back to the in-process fallback.

## [0.2.0] — 2026-08-08

### Fixed

- **Ref-count leaked on cancel-after-load**: `generate` with a cancel firing
  after `ensure_loaded` left the backend pinned (`ref_count > 0`) forever —
  never evictable, and invisible to the IdleEvictor/`ensure_vram`.
- **Idle self-shutdown no longer kills in-flight generates**: auto-shutdown
  after `idle_timeout` didn't consult the queue; a job > 30 min with a client
  in `wait`/stream made the supervisor kill the worker mid-way.
- **`max_inflight` validated atomically in `take`**: the check outside the
  lock allowed exceeding the cap by 1 with `max_inflight > 1` (with a cap of 2,
  3 jobs could run).
- **Memory-efficient factor applied once in the headroom**:
  `activation_headroom_mib` re-applied the 0.65 that `footprint_parts_mib` had
  already applied (0.42× effective) — the free-VRAM check passed with less
  than intended.
- **`vramd calibrate --out` preserves the `runtime:` block** (command/cwd/env/
  timeouts) and `load_keys`/`shape_keys` of the descriptor — it used to
  regenerate `monorepo_tool` and lose the startup configuration of external
  backends.
- **Cooperative abort no longer lost in the worker queue**: resetting the flag
  on dequeue wiped an abort arriving while generate waited; now the job
  doesn't even start (answers `cancelled before start`) and the flag is
  consumed at the end of generate.
- **EOF on worker stdin does `unload` before exiting** (adapter cleanup), as
  the loop docstring always promised.
- **`round_up_mib` always rounds up** — `round()` (banker's) returned
  multiples below the input (e.g.: `64.4` → `64`).

### Configuration

- Overlay directory aligned with the documentation:
  `~/.config/vramd/backends.d` (was `~/.config/ums/backends.d`, an AiGameKit
  leftover — whoever followed the README put files that were never read).

### Internal

- WAL lock order fixed (`_lock` → `_wal_lock` everywhere):
  `_rewrite_wal_from_queue` inverted the order and the comment claimed the
  opposite — a latent ABBA deadlock if the call graph changed.
- `footprints.py`: removed duplicate block (the second definition shadowed the
  first — dead code that could diverge).
- `${monorepo:python}` reachable again (the generic `${monorepo:*}` branch
  swallowed it); user-facing strings and docs "UMS" → "vramd"; `doctor`
  flags low free VRAM with loaded models again.
- Subprocess tests inherit `PYTHONPATH` with the repo's `src` — the suite runs
  without `pip install -e .` (before: 7 local failures that passed in CI).

## [0.1.0] — 2026-08-08

First public release. Extracted from the [AiGameKit][origin], where it was
born to have ten generative models share a 6 GB RTX 4050.

### Supervisor

- Admission by the **real peak** (weights + activation + margin), not just
  weights.
- Priority queue (`interactive` > `batch`) with **VRAM affinity**: skips the
  head up to 3 times if a later job uses an already-hot backend.
- Weight+LRU eviction, escalating to terminating the worker when calibration
  proved that backend's `unload` doesn't return VRAM.
- Cooperative cancellation between phases — no CUDA kernels killed mid-flight.
- Job WAL, `zero`, `respawn`, `reap`, `doctor`.

### Isolation

- Each backend runs in its own process and venv, speaking JSONL over
  stdin/stdout. Models with incompatible dependencies coexist.
- Worker SDK (`vramd.worker`): three methods to wrap any model.

### Calibration

- `vramd calibrate` measures the footprint with per-process VRAM sampling at
  20 Hz and splits CUDA context / weights / activation at phase boundaries.
- Detects: load peak above inference peak, lazy loading, `unload` that doesn't
  free, leak by repetition, 1st-inference warmup, contamination by neighboring
  processes, and sampler blindness.
- Raw samples kept in the report: `vramd recalibrate` recomputes the numbers
  when the analysis improves, without re-occupying the GPU.

### Configuration

- `backends.yaml` v2 with per-key overlay:
  `data/backends.yaml` → `$VRAMD_BACKENDS_FILE` → `~/.config/vramd/backends.d/*.yaml`.
- `runtime:` block (command/cwd/env/timeouts) with `${env:VAR}`; `load_keys`
  and `shape_keys` per backend; declarative `peak_profile:`.
- The measured `vram:` block **wins** over the estimate at admit, as long as
  the requested quantization is the one it was measured under.

### Packaging

- Torch-free core: the supervisor installs in ~9 MB and never creates a CUDA
  context.
- 760 tests, no GPU, on Python 3.11 / 3.12 / 3.13.

[origin]: https://github.com/maikramer
[Unreleased]: https://github.com/maikramer/vramd/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/maikramer/vramd/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/maikramer/vramd/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/maikramer/vramd/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/maikramer/vramd/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/maikramer/vramd/releases/tag/v0.1.0
