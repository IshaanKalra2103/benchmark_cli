# Journal - 2026-04-06 (America/New_York)

## 1) Scope and Goals

Primary goals tackled today:
- Run benchmark jobs through the CLI + Slurm, storing outputs in scratch.
- Compare base model runs across requested datasets.
- Use specific GPU/node constraints (initially `clip04`, then L40S nodes).
- Stabilize TUI behavior (status logic, progress, manifest visibility, GPU discovery UX).
- Move repeated runtime settings to `.env` and remove hardcoded operational values.

Secondary goals:
- Diagnose repeated failures (OOM, wrong GPU architecture, bad Slurm constraint/nodelist specs).
- Improve reproducibility and postmortem traceability.

---

## 2) Environment and Config Changes

### 2.1 `.env`/config work
Implemented env-driven config loading and override path:
- Added `.env` + `.env.local` auto-load in `cli/config.py`.
- Added env override mapping for:
  - Paths: `BENCHMARK_DATASET_ROOT`, `BENCHMARK_RESULTS_ROOT`, `BENCHMARK_CACHE_DIR`, `BENCHMARK_REPOEVAL_DATASET_PATH`
  - Slurm: `BENCHMARK_SLURM_PARTITION`, `BENCHMARK_SLURM_ACCOUNT`, `BENCHMARK_SLURM_QOS`, `BENCHMARK_SLURM_TIME`, `BENCHMARK_SLURM_MEM`, `BENCHMARK_SLURM_CONSTRAINT`, `BENCHMARK_SLURM_NODELIST`
  - Integer knobs: `BENCHMARK_SLURM_GPUS`, `BENCHMARK_SLURM_CPUS_PER_TASK`, `BENCHMARK_SLURM_POLL_INTERVAL_SEC`, `BENCHMARK_BATCH_SIZE`

Files changed:
- `cli/config.py`
- `.env.example` (added)
- `.env` (added, local)
- `.gitignore` (added `.env`, `.env.local`)
- `config/local.json` reset to `{}` to prevent config drift/hardcoding.
- `README.md` updated with env usage and precedence.

### 2.2 Slurm defaults used
Configured in env:
- `BENCHMARK_SLURM_PARTITION=scavenger`
- `BENCHMARK_SLURM_ACCOUNT=scavenger`
- `BENCHMARK_SLURM_QOS=scavenger`
- `BENCHMARK_BATCH_SIZE=16`
- `BENCHMARK_SCRATCH_WORKSPACE=/fs/cml-scratch/ishaank/benchmark_cli`
- scratch path overrides for datasets/results/cache.

Later adjustments:
- `BENCHMARK_SLURM_CONSTRAINT=l40s` (failed at submit time due invalid feature spec; reverted to empty)
- `BENCHMARK_SLURM_NODELIST=csd00,gammagpu18` (failed due node count specification invalid)
- `BENCHMARK_SLURM_NODELIST=csd00` (worked)

---

## 3) TUI / CLI Functional Changes Made Today

### 3.1 Completion correctness and empty summary handling
- Added logic so Todo completion requires a valid non-empty `summary.json` (not just file presence).
- Added automatic cleanup of empty `summary.json` files during Todo refresh.
- Cleanup run today removed `0` files (none existed at cleanup time).

### 3.2 Manifest visibility in Todo
- Added manifest details panel in Todo tab.
- Selecting a completed row now shows manifest metadata from `run_manifest.json`:
  - run_id, benchmark, dataset_group, model block, slurm flag, dataset result entry, manifest path, summary path.

### 3.3 Preflight dataset path behavior
- TUI now reloads config from `.env/.env.local` before preflight and before queue execution.
- Preflight errors now explicitly point to missing/misconfigured env variables.
- Auto-repair dataset symlink behavior is disabled when dataset path is explicitly set via env (env is source-of-truth).

### 3.4 OOM resilience in evaluator
- `benchmarks/eval_repo_bench_retriever.py` patched with automatic CUDA OOM backoff:
  - retry embedding encode with halved batch size until success (down to 1), clearing CUDA cache between retries.

---

## 4) Failures, Mistakes, and Root Causes

### 4.1 Initial `clip04` simple jobs failed OOM
Affected:
- `6539029` (`apps`)
- `6539031` (`stackoverflow-qa`)

Error type:
- `torch.OutOfMemoryError` during doc embedding encode path.

Root cause:
- Workload size + model memory footprint exceeded available GPU memory on that run context.

Fix attempted:
- Added OOM backoff in evaluator (batch-size halving retries).

### 4.2 Later requeues failed on legacy GPUs (wrong architecture)
Affected:
- `6539094`, `6539095`, `6539096`

Error type:
- `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`

Observed hardware:
- `NVIDIA GeForce GTX TITAN X`, CC 5.2 (`legacygpu06/07`).

Root cause:
- Jobs were queued without compatible GPU constraint/node pin; scheduler placed jobs on legacy GPUs unsupported by installed PyTorch CUDA build.

Fix attempted:
- Move to L40S nodes by explicit routing.

### 4.3 L40S routing mistakes
1. Attempted `--constraint l40s`:
- Submission failed: `Invalid feature specification`.
- Root cause: cluster feature names do not accept `l40s` as a constraint token.

2. Attempted nodelist with comma-separated nodes (`csd00,gammagpu18`):
- Submission failed: `Node count specification invalid`.
- Root cause: this cluster/partition rejected multi-node allowlist in that submission form.

3. Final working approach:
- Pin single node `--nodelist csd00` for L40S jobs.

### 4.4 Repoeval metadata preflight issue
Error shown:
- `RepoEval metadata missing from BENCHMARK_REPOEVAL_DATASET_PATH` at configured scratch path.

Observed state:
- Metadata file `function_level_completion_2k_context_codex.test.clean.jsonl` was not found under expected roots during checks.

Impact:
- TUI preflight correctly blocks `repoeval` when metadata file is absent.

---

## 5) GPU/Node Selection Notes

L40S inventory observed via `sinfo`:
- `cml34` (mix, l40s:8)
- `csd00` (scavenger, mix, l40s:4)
- `gammagpu18` (scavenger, idle, l40s:4)
- `gammagpu20/21` mixed/occupied

Key scheduling outcomes:
- `clip04` runs started quickly but had instability (OOM for some datasets).
- unconstrained scavenger placements led to legacy GPU assignment (`legacygpu06/07`) and immediate architecture failures.
- explicit L40S node pinning to `csd00` worked.
- explicit `gammagpu18` pin for swe-bench-lite successfully queued and started.

---

## 6) Jobs Queued Today (Detailed)

### 6.1 `clip04` wave
- `6539029` - `apps` (base) on `clip04` - **FAILED** (OOM)
- `6539031` - `stackoverflow-qa` (base) on `clip04` - **FAILED** (OOM)
- `6539033` - `codefeedback-st` (base) on `clip04` - **RUNNING**
- `6539034` - `codefeedback-mt` (base) on `clip04` - **CANCELLED by user flow**
- `6539047` - `repoeval` base forced on `clip04` - **COMPLETED**

### 6.2 Unconstrained requeue wave (mistake: landed on legacy GPUs)
- `6539094` - `apps` - node `legacygpu06` - **FAILED** (`no kernel image`)
- `6539095` - `stackoverflow-qa` - node `legacygpu07` - **FAILED** (`no kernel image`)
- `6539096` - `codefeedback-mt` - node `legacygpu07` - **FAILED** (`no kernel image`)

### 6.3 L40S corrected wave (`csd00` pin)
- `6539335` - `apps` - node `csd00` - **COMPLETED**
- `6539336` - `stackoverflow-qa` - node `csd00` - **COMPLETED**
- `6539337` - `codefeedback-mt` - node `csd00` - **RUNNING**

### 6.4 L40S swe-bench-lite on other available node
- `6539397` - `swe-bench-lite` base pinned to `gammagpu18` - **RUNNING**

Status snapshot source:
- `sacct` queried for jobs: `6539029,6539031,6539033,6539034,6539047,6539094,6539095,6539096,6539335,6539336,6539337,6539397`.

---

## 7) Explicit Mistake Log

1. Queued jobs without explicit compatible GPU routing after user requested specific GPU class intent.
- Result: jobs landed on legacy Titan nodes and failed.

2. Assumed `--constraint l40s` would be valid feature syntax.
- Result: `Invalid feature specification`.

3. Used comma-separated multi-node nodelist for single-node job submissions.
- Result: `Node count specification invalid`.

4. `repoeval` submission was initially skipped due existing results (without force).
- Corrected by forcing repoeval run.

5. RepoEval metadata path in env pointed to a file that did not exist on disk.
- Preflight correctly flagged, but dataset artifact still needs restoration at configured path.

---

## 8) Operational Practices Adopted by End of Day

- Use env-backed config everywhere (`.env/.env.local`) instead of ad-hoc command exports.
- Prefer explicit node pinning when GPU architecture compatibility matters.
- Keep TUI completion logic strict (valid summary required).
- Include run manifests in TUI for completed runs for traceability.
- Use OOM backoff in evaluator to reduce hard-fail frequency on smaller-memory GPUs.

---

## 9) Remaining Open Items

1. Restore `repoeval` metadata file to:
- `/fs/cml-scratch/ishaank/benchmark_cli/benchmark_cli/datasets/repoeval_metadata/function_level_completion_2k_context_codex.test.clean.jsonl`

2. Decide whether to keep `BENCHMARK_SLURM_NODELIST=csd00` globally in `.env` (stable) or switch per-run from TUI/CLI.

3. Continue monitoring active runs:
- `6539033`, `6539337`, `6539397`.

