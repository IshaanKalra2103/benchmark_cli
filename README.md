# benchmark_cli

TUI + CLI benchmark runner for dense retrieval and reranking evaluation.

Key capabilities:
- Baseline runs for Qwen 0.6B profiles (8B profiles hidden by default).
- Continuous checkpoint runs for `aysinghal/ide-code-retrieval-qwen3-0.6b`.
- Main benchmark coverage: `repoeval`, `swe-bench-lite`, `coir` dataset groups.
- Slurm integration: GPU discovery, queue submission, live job status, interactive smoke checks.
- Manual analysis mode for positive/failure cases and raw retrieval outputs.

This repository contains local benchmark scripts under `benchmarks/`; it does not execute scripts from `Code_Retrieval`.

## Install

```bash
cd /Users/ishaankalra/Documents/Dev/benchmark_cli
uv sync
```

## Initialize config

```bash
uv run benchmark config init
uv run benchmark config show
```

Default config files:
- `config/defaults.json` (auto-created)
- `config/local.json` (your editable overrides)

## CLI usage

List HF checkpoints and recommended schedule:

```bash
uv run benchmark checkpoints list
```

Run baseline locally:

```bash
uv run benchmark run --benchmark repoeval --model-profile qwen3_embed_0_6b
```

Run checkpoint through Slurm:

```bash
uv run benchmark run --benchmark swe-bench-lite --checkpoint-step 3000 --slurm --slurm-gpus 1
```

COIR group run:

```bash
uv run benchmark run --benchmark coir --dataset-group text-to-code --checkpoint-step 3000 --slurm
```

Smoke test locally:

```bash
uv run benchmark run --benchmark repoeval --model-profile qwen3_embed_0_6b --smoke
```

Interactive Slurm smoke:

```bash
uv run benchmark run --benchmark repoeval --checkpoint-step 3000 --interactive --smoke
```

GPU/status helpers:

```bash
uv run benchmark slurm gpus
uv run benchmark slurm status
```

Manual analysis:

```bash
uv run benchmark analyze \
  --dataset repoeval \
  --runs base=/abs/path/base/raw_results.jsonl ckpt=/abs/path/ckpt/raw_results.jsonl
```

Dataset download/create (sharded):

```bash
# local shard execution
uv run benchmark dataset download --dataset swe-bench-lite --num-shards 8 --shard-id 0

# slurm array execution (0..num_shards-1)
uv run benchmark dataset download --dataset swe-bench-lite --num-shards 8 --slurm
uv run benchmark dataset download --dataset repoeval --num-shards 6 --slurm
uv run benchmark dataset download --dataset coir --num-shards 10 --slurm
```

## TUI

```bash
uv run benchmark tui
```

TUI features:
- Select benchmark/model/checkpoint.
- Refresh HF checkpoints.
- Inspect Slurm GPU table from `scontrol/sinfo`.
- Click a GPU row to auto-fill Slurm `nodelist`.
- Queue local or Slurm jobs.
- Launch interactive `srun --pty` smoke run.
- Live job panel from `squeue` polling.
- Manual analysis trigger.

## Control guide

### TUI controls

Keyboard:
- `q`: quit the TUI.
- `r`: refresh checkpoints + Slurm GPU/job panels.

Form fields:
- `Benchmark`: choose `repoeval`, `swe-bench-lite`, or `coir`.
- `COIR Group`: optional group filter when benchmark is `coir`.
- `Model Profile`: baseline model profile key from config.
- `Checkpoint Step`: optional fine-tune checkpoint step, e.g. `3000`.
- `Slurm GPUs`: GPU count override for queued Slurm jobs.
- `Slurm Constraint`: optional Slurm constraint string.
- `Slurm Nodelist`: optional node pinning. Clicking a GPU table row auto-fills this.
- `Smoke`: run reduced-size sanity execution.
- `Force`: rerun baseline even if previous results already exist.

Buttons:
- `Execute Local`: run immediately in current environment.
- `Execute Slurm`: submit via `sbatch`.
- `Interactive Slurm Smoke`: launch `srun --pty` using smoke settings.
- `Refresh Checkpoints`: refresh HF checkpoint list from configured repo.
- `Refresh GPUs/Jobs`: refresh Slurm node and queue tables.
- `Run Analysis`: run positive/failure case analysis using `NAME=/path/raw_results.jsonl` specs.

Tables/panels:
- `GPU Availability`: node inventory from Slurm (`scontrol`/`sinfo` fallback).
- `Slurm Queue`: live job state from `squeue`.
- `Log`: run submissions, errors, analysis output location, and status messages.

### CLI controls

Discover commands:

```bash
uv run benchmark --help
uv run benchmark run --help
uv run benchmark dataset download --help
```

Core command groups:
- `uv run benchmark config ...`: initialize/show config.
- `uv run benchmark checkpoints list`: list available + recommended checkpoints.
- `uv run benchmark run ...`: execute benchmark runs (local, slurm, interactive).
- `uv run benchmark slurm gpus|status`: inspect cluster resources and queue.
- `uv run benchmark analyze ...`: run positive/failure analysis on raw results.
- `uv run benchmark dataset download ...`: sharded dataset preparation, local or Slurm array.

Recommended day-to-day sequence:
1. `uv run benchmark config show`
2. `uv run benchmark dataset download --dataset swe-bench-lite --num-shards 8 --slurm`
3. `uv run benchmark checkpoints list`
4. `uv run benchmark run --benchmark repoeval --checkpoint-step 3000 --slurm`
5. `uv run benchmark analyze --dataset repoeval --runs base=... ckpt=...`

## Data layout assumptions

Datasets should be arranged in BEIR style under `datasets/` with directories prefixed by dataset key:
- `datasets/repoeval_*`
- `datasets/swe-bench-lite_*`
- COIR datasets per configured names (e.g. `datasets/apps_*`, `datasets/codesearchnet-ccr_*`, etc.).

## Output artifacts

Runs are stored under `results/<run_id>/...`.
Each dataset/model run stores:
- `summary.json`
- `retrieval_results.jsonl`
- `raw_results.jsonl`
- `per_query_metrics.jsonl`
- `run.log`

Run metadata:
- `results/<run_id>/run_manifest.json`

Analysis output:
- `summary.json`
- `positive_cases.jsonl`
- `failure_cases.jsonl`
- `report.md`
- `analysis.log`
