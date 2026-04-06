import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ALLOWED_DATASETS = {"repoeval", "swe-bench-lite"}
SUPPORTED_TAGS = {"open-source", "closed-source"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a model grid for RepoEval/SWE-Bench retrieval and auto-generate analysis + leaderboard."
    )
    parser.add_argument("--grid_config", type=str, required=True, help="Path to JSON config for model grid.")
    parser.add_argument("--output_dir", type=str, default="results/repo_bench_grid")
    parser.add_argument("--datasets", type=str, nargs="*", default=None, help="Optional dataset override.")
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    parser.add_argument("--analysis_k", type=int, default=10)
    parser.add_argument("--analysis_num_cases", type=int, default=10)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument(
        "--tag",
        type=str,
        default="end2end",
        choices=["open-source", "closed-source", "end2end"],
        help="Model subset to run: open-source, closed-source, or both (end2end).",
    )
    parser.add_argument("--run_tag", type=str, default=None, help="Optional suffix for run directory.")
    return parser.parse_args()


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in value)


def normalize_tag(value: Any, default: str = "open-source") -> str:
    if value is None:
        return default
    tag = str(value).strip().lower()
    if tag not in SUPPORTED_TAGS:
        raise ValueError(f"Unsupported tag '{value}'. Supported tags: {sorted(SUPPORTED_TAGS)}.")
    return tag


def tag_matches(selected: str, item_tag: str) -> bool:
    if selected == "end2end":
        return True
    return selected == item_tag


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def ensure_fields(model_cfg: dict[str, Any], index: int) -> None:
    if "name" not in model_cfg:
        raise ValueError(f"Model config at index {index} is missing required field 'name'.")
    if "model" not in model_cfg:
        raise ValueError(f"Model config at index {index} is missing required field 'model'.")


def load_grid_config(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    cfg = read_json(path)
    if "models" not in cfg or not isinstance(cfg["models"], list) or not cfg["models"]:
        raise ValueError("Grid config must contain a non-empty 'models' list.")

    models: list[dict[str, Any]] = cfg["models"]
    for i, model_cfg in enumerate(models):
        ensure_fields(model_cfg, i)

    defaults = cfg.get("defaults", {})
    datasets = cfg.get("datasets", ["repoeval", "swe-bench-lite"])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("'datasets' in grid config must be a non-empty list.")
    unknown = sorted(set(datasets) - ALLOWED_DATASETS)
    if unknown:
        raise ValueError(f"Unsupported dataset(s) in config: {unknown}. Supported: {sorted(ALLOWED_DATASETS)}.")

    return defaults, models, datasets


def append_if(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def append_opt(cmd: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([key, str(value)])


def merged_value(model_cfg: dict[str, Any], defaults: dict[str, Any], key: str, fallback: Any = None) -> Any:
    if key in model_cfg:
        return model_cfg[key]
    if key in defaults:
        return defaults[key]
    return fallback


def run_command(cmd: list[str], log_file: Path) -> tuple[bool, int]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.write(proc.stdout or "")
    return proc.returncode == 0, proc.returncode


def build_eval_cmd(
    python_executable: str,
    dataset: str,
    dataset_root: str,
    defaults: dict[str, Any],
    model_cfg: dict[str, Any],
    out_summary: Path,
    out_results: Path,
    out_raw: Path,
    out_per_query: Path,
) -> list[str]:
    cmd = [python_executable, "eval_repo_bench_retriever.py"]
    cmd.extend(["--dataset", dataset, "--dataset_root", dataset_root])

    append_opt(cmd, "--model", merged_value(model_cfg, defaults, "model"))
    append_opt(cmd, "--batch_size", merged_value(model_cfg, defaults, "batch_size", 32))
    append_opt(cmd, "--query_prefix", merged_value(model_cfg, defaults, "query_prefix", ""))
    append_opt(cmd, "--doc_prefix", merged_value(model_cfg, defaults, "doc_prefix", ""))
    append_opt(cmd, "--device", merged_value(model_cfg, defaults, "device", None))
    append_opt(cmd, "--reranker_model", merged_value(model_cfg, defaults, "reranker_model", None))
    append_opt(cmd, "--reranker_batch_size", merged_value(model_cfg, defaults, "reranker_batch_size", 8))
    append_opt(cmd, "--reranker_max_length", merged_value(model_cfg, defaults, "reranker_max_length", 512))
    append_opt(cmd, "--top_k", merged_value(model_cfg, defaults, "top_k", 10))
    append_opt(cmd, "--candidate_k", merged_value(model_cfg, defaults, "candidate_k", 100))
    append_opt(cmd, "--cache_dir", merged_value(model_cfg, defaults, "cache_dir", "cache/retrieval_embeddings"))
    append_opt(cmd, "--max_instances", merged_value(model_cfg, defaults, "max_instances", None))
    append_opt(cmd, "--instance_regex", merged_value(model_cfg, defaults, "instance_regex", None))
    append_opt(
        cmd,
        "--repoeval_dataset_path",
        merged_value(
            model_cfg,
            defaults,
            "repoeval_dataset_path",
            "output/repoeval/datasets/function_level_completion_2k_context_codex.test.clean.jsonl",
        ),
    )

    append_if(cmd, "--normalize_embeddings", bool(merged_value(model_cfg, defaults, "normalize_embeddings", False)))
    append_if(cmd, "--trust_remote_code", bool(merged_value(model_cfg, defaults, "trust_remote_code", False)))
    append_if(cmd, "--no_cache", bool(merged_value(model_cfg, defaults, "no_cache", False)))

    cmd.extend(
        [
            "--output_file",
            str(out_summary),
            "--results_file",
            str(out_results),
            "--raw_results_file",
            str(out_raw),
            "--per_query_metrics_file",
            str(out_per_query),
        ]
    )
    return cmd


def load_summary(path: Path) -> dict[str, Any]:
    return read_json(path)


def extract_scores(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "NDCG@10": float(summary.get("ndcg", {}).get("NDCG@10", 0.0)),
        "MRR@10": float(summary.get("mrr", {}).get("MRR@10", 0.0)),
        "Recall@10": float(summary.get("recall", {}).get("Recall@10", 0.0)),
        "P@10": float(summary.get("precision", {}).get("P@10", 0.0)),
    }


def write_leaderboard(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "leaderboard.json"
    write_json(json_path, rows)

    csv_path = out_dir / "leaderboard.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "tag",
                "run_name",
                "model",
                "reranker_model",
                "NDCG@10",
                "MRR@10",
                "Recall@10",
                "P@10",
                "num_queries",
                "num_instances",
                "time_sec",
                "summary_file",
                "raw_results_file",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = out_dir / "leaderboard.md"
    lines = ["# RepoRAG-Bench Retrieval Leaderboard", ""]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(row)
    for dataset, group_rows in grouped.items():
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| rank | tag | run | MRR@10 | Recall@10 | NDCG@10 | P@10 | reranker |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---|")
        ranked = sorted(group_rows, key=lambda r: (r["MRR@10"], r["Recall@10"]), reverse=True)
        for i, row in enumerate(ranked, start=1):
            lines.append(
                f"| {i} | `{row.get('tag','')}` | `{row['run_name']}` | {row['MRR@10']:.4f} | {row['Recall@10']:.4f} | {row['NDCG@10']:.4f} | {row['P@10']:.4f} | `{row['reranker_model'] or ''}` |"
            )
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.grid_config).resolve()
    defaults, models, cfg_datasets = load_grid_config(cfg_path)
    selected_models: list[dict[str, Any]] = []
    for model_cfg in models:
        model_tag = normalize_tag(model_cfg.get("tag", "open-source"))
        if tag_matches(args.tag, model_tag):
            selected_models.append(model_cfg)
    if not selected_models:
        raise ValueError(f"No models matched --tag={args.tag}.")

    datasets = args.datasets if args.datasets else cfg_datasets
    unknown = sorted(set(datasets) - ALLOWED_DATASETS)
    if unknown:
        raise ValueError(f"Unsupported dataset override(s): {unknown}.")

    run_id = args.run_tag if args.run_tag else now_tag()
    run_root = Path(args.output_dir).resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(
        run_root / "used_grid_config.json",
        {"defaults": defaults, "models": models, "selected_tag": args.tag, "datasets": datasets},
    )

    successful_runs_by_dataset: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in datasets}
    failed_runs: list[dict[str, Any]] = []

    retrieval_script = Path(__file__).resolve().parent / "eval_repo_bench_retriever.py"
    analysis_script = Path(__file__).resolve().parent / "analyze_repo_bench_cases.py"

    for dataset in datasets:
        dataset_dir = run_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for model_cfg in selected_models:
            run_name = safe_name(str(model_cfg["name"]))
            model_tag = normalize_tag(model_cfg.get("tag", "open-source"))
            run_dir = dataset_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            out_summary = run_dir / "summary.json"
            out_results = run_dir / "retrieval_results.jsonl"
            out_raw = run_dir / "raw_results.jsonl"
            out_per_query = run_dir / "per_query_metrics.jsonl"
            log_path = run_dir / "run.log"

            if args.skip_existing and out_summary.exists() and out_raw.exists():
                successful_runs_by_dataset[dataset].append(
                    {
                        "dataset": dataset,
                        "tag": model_tag,
                        "run_name": run_name,
                        "model": model_cfg.get("model"),
                        "reranker_model": model_cfg.get("reranker_model"),
                        "summary_file": str(out_summary),
                        "raw_results_file": str(out_raw),
                    }
                )
                continue

            cmd = build_eval_cmd(
                python_executable=args.python_executable,
                dataset=dataset,
                dataset_root=args.dataset_root,
                defaults=defaults,
                model_cfg=model_cfg,
                out_summary=out_summary,
                out_results=out_results,
                out_raw=out_raw,
                out_per_query=out_per_query,
            )
            cmd[1] = str(retrieval_script)
            ok, code = run_command(cmd, log_path)
            if ok:
                successful_runs_by_dataset[dataset].append(
                    {
                        "dataset": dataset,
                        "tag": model_tag,
                        "run_name": run_name,
                        "model": model_cfg.get("model"),
                        "reranker_model": model_cfg.get("reranker_model"),
                        "summary_file": str(out_summary),
                        "raw_results_file": str(out_raw),
                    }
                )
            else:
                failed_runs.append(
                    {
                        "dataset": dataset,
                        "run_name": run_name,
                        "return_code": code,
                        "log_file": str(log_path),
                    }
                )
                if not args.continue_on_error:
                    write_json(run_root / "failed_runs.json", failed_runs)
                    raise RuntimeError(f"Grid run failed at {dataset}/{run_name}. See {log_path}")

        # Run analysis on successful models for this dataset.
        dataset_success = successful_runs_by_dataset[dataset]
        if len(dataset_success) >= 2:
            analysis_dir = dataset_dir / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            analysis_log = analysis_dir / "analysis.log"
            analysis_cmd = [
                args.python_executable,
                str(analysis_script),
                "--dataset",
                dataset,
                "--dataset_root",
                args.dataset_root,
                "--k",
                str(args.analysis_k),
                "--num_cases",
                str(args.analysis_num_cases),
                "--output_dir",
                str(analysis_dir),
                "--runs",
            ]
            for run in dataset_success:
                analysis_cmd.append(f"{run['run_name']}={run['raw_results_file']}")
            ok, code = run_command(analysis_cmd, analysis_log)
            if not ok:
                failed_runs.append(
                    {
                        "dataset": dataset,
                        "run_name": "__analysis__",
                        "return_code": code,
                        "log_file": str(analysis_log),
                    }
                )
                if not args.continue_on_error:
                    write_json(run_root / "failed_runs.json", failed_runs)
                    raise RuntimeError(f"Analysis failed for dataset {dataset}. See {analysis_log}")

    leaderboard_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for run in successful_runs_by_dataset[dataset]:
            summary_file = Path(run["summary_file"])
            if not summary_file.exists():
                continue
            summary = load_summary(summary_file)
            scores = extract_scores(summary)
            leaderboard_rows.append(
                {
                    "dataset": dataset,
                    "tag": run.get("tag", ""),
                    "run_name": run["run_name"],
                    "model": run.get("model"),
                    "reranker_model": run.get("reranker_model"),
                    "NDCG@10": scores["NDCG@10"],
                    "MRR@10": scores["MRR@10"],
                    "Recall@10": scores["Recall@10"],
                    "P@10": scores["P@10"],
                    "num_queries": int(summary.get("num_queries", 0)),
                    "num_instances": int(summary.get("num_instances", 0)),
                    "time_sec": float(summary.get("time", 0.0)),
                    "summary_file": run["summary_file"],
                    "raw_results_file": run["raw_results_file"],
                }
            )

    write_leaderboard(leaderboard_rows, run_root)
    write_json(run_root / "failed_runs.json", failed_runs)
    write_json(run_root / "run_manifest.json", {"run_root": str(run_root), "datasets": datasets})

    print(json.dumps({"run_root": str(run_root), "num_successful_runs": len(leaderboard_rows), "num_failures": len(failed_runs)}, indent=2))


if __name__ == "__main__":
    main()
