---
name: benchmark-cli-operator
description: Use this skill when operating benchmark_cli for retrieval benchmarking, especially to compare base vs finetuned models with identical evaluation settings, diagnose regressions, and fix CLI/evaluator bugs before rerunning.
metadata:
  short-description: Operate and repair benchmark_cli for fair base-vs-finetuned comparisons
---

# Benchmark CLI Operator

## When To Use

Use this skill when the task involves:

- Running `benchmark_cli` for `repoeval`, `swe-bench-lite`, or `coir`.
- Comparing base vs finetuned model quality.
- Verifying comparison fairness (same evaluation setup).
- Debugging or fixing CLI/evaluator issues and then rerunning.

## Core Rule: Fair Comparison First

For base vs finetuned comparisons, all evaluation controls must match unless a user explicitly requests otherwise.

Required parity:

- Dataset and split scope
- `top_k`, `candidate_k`
- `normalize_embeddings`
- `reranker_model` and reranker settings
- Query/document formatting strategy (`query_prefix`, `doc_prefix`, `query_prompt_name`, `doc_prompt_name`)
- Cache isolation (model + prompt settings must not collide)

Allowed to differ:

- `--model` value
- Run IDs / output paths
- Batch size only when needed for OOM mitigation (document when changed)

## Standard Workflow

1. Load effective config:
   - `uv run benchmark config show`
2. Preview commands before running:
   - `uv run benchmark run --benchmark <dataset> --model-profile <profile> --smoke`
   - Use `preview_commands` behavior from the CLI to verify generated args.
3. Verify parity between base and finetuned commands:
   - Normalize away `--model`, output/log paths, and run IDs.
   - If non-model args differ unexpectedly, do not trust results yet.
4. Fix bugs first, then rerun:
   - Patch `cli/runner.py`, `cli/config.py`, `benchmarks/eval_repo_bench_retriever.py`, or `benchmarks/run_repo_bench_grid.py` as needed.
   - Validate with syntax checks and command preview.
5. Run paired evaluations.
6. Compare with the same method:
   - `uv run benchmark analyze --dataset <dataset> --runs base=<raw_base> finetuned=<raw_ft>`
   - Inspect `summary.json`, `report.md`, and case files.

## Prompting Policy

Default comparison policy in this repo should be:

- `query_prefix: ""`
- `doc_prefix: ""`
- `query_prompt_name: "query"`
- `doc_prompt_name: null`

Rationale:

- Keeps base and finetuned on the same prompt strategy.
- Uses model-native query prompt templates where available.
- Avoids mixing prompt templates with legacy hardcoded prefixes unless intentionally testing that variant.

If reproducing legacy experiments, explicitly set:

- `query_prefix: "query: "`
- `doc_prefix: "passage: "`
- `query_prompt_name: null`
- `doc_prompt_name: null`

and apply the same settings to both models.

## Bug-Fix Expectations

If a bug is found while operating the CLI:

1. Identify root cause from command generation, logs, and evaluator behavior.
2. Patch the minimal set of files.
3. Re-validate command parity.
4. Re-run the failed comparison path.
5. Report both:
   - what was broken and fixed
   - whether conclusion about base vs finetuned changed after fix

## Useful Commands

```bash
# Show effective config
uv run benchmark config show

# Run baseline
uv run benchmark run --benchmark repoeval --model-profile qwen3_embed_0_6b

# Run latest finetuned profile
uv run benchmark run --benchmark repoeval --model-profile qwen3_embed_0_6b_finetuned_latest

# Analyze paired raw results
uv run benchmark analyze \
  --dataset repoeval \
  --runs base=/abs/path/base/raw_results.jsonl finetuned=/abs/path/ft/raw_results.jsonl
```

## Output Requirements

When reporting comparison results:

- State exact runs used (run IDs and paths).
- Confirm parity checks passed (or list any intentional differences).
- Provide per-dataset metric deltas.
- Highlight top regressions and top improvements at query level.
