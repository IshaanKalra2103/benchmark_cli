from __future__ import annotations

import datetime as dt
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .slurm import submit_sbatch


@dataclass
class ModelSpec:
    key: str
    name: str
    model: str
    reranker_model: str | None
    source: str


@dataclass
class RunSpec:
    run_id: str
    dataset: str
    model: ModelSpec
    run_dir: Path
    summary_file: Path
    results_file: Path
    raw_results_file: Path
    per_query_metrics_file: Path
    log_file: Path


@dataclass
class RunExecutionResult:
    dataset: str
    run_name: str
    status: str
    return_code: int | None
    log_file: str
    job_id: str | None = None
    script_file: str | None = None


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def visible_model_profiles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    show_8b = bool(config["models"].get("show_8b", False))
    out: dict[str, dict[str, Any]] = {}
    for key, profile in config["models"]["profiles"].items():
        if "8b" in key and not show_8b:
            continue
        if not profile.get("enabled", True):
            continue
        out[key] = profile
    return out


def resolve_model_spec(
    config: dict[str, Any],
    model_profile_key: str,
    checkpoint_step: int | None,
) -> ModelSpec:
    if checkpoint_step is not None:
        repo_id = config["models"]["finetune"]["hf_repo"]
        embed_only = bool(config["models"]["finetune"].get("embed_only", True))
        reranker = None if embed_only else config["models"]["finetune"].get("default_reranker_model")
        return ModelSpec(
            key=f"checkpoint_{checkpoint_step}",
            name=f"checkpoint-{checkpoint_step}",
            model=f"{repo_id}/checkpoint-{checkpoint_step}",
            reranker_model=reranker,
            source="checkpoint",
        )

    profiles = visible_model_profiles(config)
    if model_profile_key not in profiles:
        raise ValueError(f"Unknown model profile '{model_profile_key}'.")
    profile = profiles[model_profile_key]
    return ModelSpec(
        key=model_profile_key,
        name=profile["name"],
        model=profile["model"],
        reranker_model=profile.get("reranker_model"),
        source="baseline",
    )


def _expand_datasets(config: dict[str, Any], benchmark: str, dataset_group: str | None) -> list[str]:
    if benchmark == "coir":
        groups = config["benchmarks"]["coir"]["groups"]
        if dataset_group:
            if dataset_group not in groups:
                raise ValueError(f"Unknown coir dataset group '{dataset_group}'.")
            return list(groups[dataset_group])
        datasets: list[str] = []
        for values in groups.values():
            datasets.extend(values)
        return datasets

    if benchmark not in config["benchmarks"]:
        raise ValueError(f"Unknown benchmark '{benchmark}'.")
    return [config["benchmarks"][benchmark]["dataset"]]


def _build_run_spec(run_root: Path, dataset: str, model: ModelSpec, run_id: str) -> RunSpec:
    run_name = _safe_name(model.name)
    run_dir = run_root / dataset / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunSpec(
        run_id=run_id,
        dataset=dataset,
        model=model,
        run_dir=run_dir,
        summary_file=run_dir / "summary.json",
        results_file=run_dir / "retrieval_results.jsonl",
        raw_results_file=run_dir / "raw_results.jsonl",
        per_query_metrics_file=run_dir / "per_query_metrics.jsonl",
        log_file=run_dir / "run.log",
    )


def _build_eval_command(config: dict[str, Any], spec: RunSpec, smoke: bool) -> list[str]:
    defaults = config["models"]["defaults"]
    paths = config["paths"]
    script = PROJECT_ROOT / "benchmarks" / "eval_repo_bench_retriever.py"

    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        spec.dataset,
        "--dataset_root",
        str(paths["dataset_root"]),
        "--model",
        spec.model.model,
        "--batch_size",
        str(defaults.get("batch_size", 32)),
        "--query_prefix",
        str(defaults.get("query_prefix", "")),
        "--doc_prefix",
        str(defaults.get("doc_prefix", "")),
        "--top_k",
        str(defaults.get("top_k", 10)),
        "--candidate_k",
        str(defaults.get("candidate_k", 100)),
        "--cache_dir",
        str(paths["cache_dir"]),
        "--reranker_batch_size",
        str(defaults.get("reranker_batch_size", 8)),
        "--reranker_max_length",
        str(defaults.get("reranker_max_length", 512)),
        "--repoeval_dataset_path",
        str(paths["repoeval_dataset_path"]),
        "--output_file",
        str(spec.summary_file),
        "--results_file",
        str(spec.results_file),
        "--raw_results_file",
        str(spec.raw_results_file),
        "--per_query_metrics_file",
        str(spec.per_query_metrics_file),
    ]

    query_prompt_name = defaults.get("query_prompt_name")
    if query_prompt_name:
        cmd.extend(["--query_prompt_name", str(query_prompt_name)])
    doc_prompt_name = defaults.get("doc_prompt_name")
    if doc_prompt_name:
        cmd.extend(["--doc_prompt_name", str(doc_prompt_name)])

    if defaults.get("normalize_embeddings", False):
        cmd.append("--normalize_embeddings")
    if defaults.get("trust_remote_code", False):
        cmd.append("--trust_remote_code")
    if spec.model.reranker_model:
        cmd.extend(["--reranker_model", spec.model.reranker_model])

    if smoke:
        cmd.extend(["--max_instances", "1", "--top_k", "5", "--candidate_k", "20"])

    return cmd


def preview_commands(
    *,
    config: dict[str, Any],
    benchmark: str,
    model_profile_key: str,
    checkpoint_step: int | None,
    dataset_group: str | None,
    smoke: bool,
    run_id: str | None = None,
) -> list[tuple[str, list[str]]]:
    run_id = run_id or _now_tag()
    results_root = Path(config["paths"]["results_root"]).resolve()
    run_root = results_root / run_id
    model = resolve_model_spec(config, model_profile_key=model_profile_key, checkpoint_step=checkpoint_step)
    datasets = _expand_datasets(config, benchmark=benchmark, dataset_group=dataset_group)

    out: list[tuple[str, list[str]]] = []
    for dataset in datasets:
        spec = _build_run_spec(run_root=run_root, dataset=dataset, model=model, run_id=run_id)
        out.append((dataset, _build_eval_command(config=config, spec=spec, smoke=smoke)))
    return out


def _already_has_results(spec: RunSpec) -> bool:
    return spec.summary_file.exists() and spec.raw_results_file.exists()


def _find_previous_results(results_root: Path, dataset: str, model_name: str) -> Path | None:
    run_name = _safe_name(model_name)
    pattern = f"*/{dataset}/{run_name}/summary.json"
    matches = sorted(results_root.glob(pattern))
    for summary_path in reversed(matches):
        raw_path = summary_path.parent / "raw_results.jsonl"
        if raw_path.exists():
            return summary_path.parent
    return None


def _write_run_manifest(run_root: Path, payload: dict[str, Any]) -> None:
    (run_root / "run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def execute_runs(
    *,
    config: dict[str, Any],
    benchmark: str,
    model_profile_key: str,
    checkpoint_step: int | None,
    dataset_group: str | None,
    smoke: bool,
    force: bool,
    run_id: str | None,
    use_slurm: bool,
    slurm_gpus: int | None,
    slurm_partition: str | None,
    slurm_constraint: str | None,
    slurm_nodelist: str | None,
) -> tuple[Path, list[RunExecutionResult]]:
    run_id = run_id or _now_tag()
    results_root = Path(config["paths"]["results_root"]).resolve()
    run_root = results_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    model = resolve_model_spec(config, model_profile_key=model_profile_key, checkpoint_step=checkpoint_step)
    datasets = _expand_datasets(config, benchmark=benchmark, dataset_group=dataset_group)

    results: list[RunExecutionResult] = []
    for dataset in datasets:
        spec = _build_run_spec(run_root=run_root, dataset=dataset, model=model, run_id=run_id)
        cmd = _build_eval_command(config=config, spec=spec, smoke=smoke)

        if not force and model.source == "baseline":
            previous_dir = _find_previous_results(results_root, dataset=dataset, model_name=spec.model.name)
            if previous_dir is not None:
                results.append(
                    RunExecutionResult(
                        dataset=dataset,
                        run_name=spec.model.name,
                        status="skipped_existing",
                        return_code=0,
                        log_file=str(previous_dir / "run.log"),
                    )
                )
                continue

        if not force and model.source == "baseline" and _already_has_results(spec):
            results.append(
                RunExecutionResult(
                    dataset=dataset,
                    run_name=spec.model.name,
                    status="skipped_existing",
                    return_code=0,
                    log_file=str(spec.log_file),
                )
            )
            continue

        if use_slurm:
            slurm_job_name = f"{_safe_name(run_id)}__bench_{_safe_name(dataset)}_{_safe_name(spec.model.name)}"
            if len(slurm_job_name) > 120:
                slurm_job_name = slurm_job_name[:120]
            job_id, output, script_path = submit_sbatch(
                command=cmd,
                job_name=slurm_job_name,
                log_dir=spec.run_dir,
                slurm_cfg=config["slurm"],
                gpus=slurm_gpus,
                partition=slurm_partition,
                constraint=slurm_constraint,
                nodelist=slurm_nodelist,
            )
            spec.log_file.write_text(output + "\n", encoding="utf-8")
            status = "submitted" if job_id else "submit_failed"
            results.append(
                RunExecutionResult(
                    dataset=dataset,
                    run_name=spec.model.name,
                    status=status,
                    return_code=0 if job_id else 1,
                    log_file=str(spec.log_file),
                    job_id=job_id,
                    script_file=str(script_path),
                )
            )
            continue

        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        spec.log_file.write_text("$ " + shlex.join(cmd) + "\n\n" + (proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
        results.append(
            RunExecutionResult(
                dataset=dataset,
                run_name=spec.model.name,
                status="ok" if proc.returncode == 0 else "failed",
                return_code=proc.returncode,
                log_file=str(spec.log_file),
            )
        )

    _write_run_manifest(
        run_root,
        {
            "run_id": run_id,
            "benchmark": benchmark,
            "dataset_group": dataset_group,
            "model": {
                "key": model.key,
                "name": model.name,
                "model": model.model,
                "reranker_model": model.reranker_model,
                "source": model.source,
            },
            "smoke": smoke,
            "slurm": use_slurm,
            "results": [r.__dict__ for r in results],
        },
    )
    return run_root, results


def run_analysis(
    *,
    config: dict[str, Any],
    dataset: str,
    run_specs: list[str],
    output_dir: Path,
    k: int,
    num_cases: int,
) -> tuple[int, str]:
    script = PROJECT_ROOT / "benchmarks" / "analyze_repo_bench_cases.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        dataset,
        "--dataset_root",
        str(config["paths"]["dataset_root"]),
        "--k",
        str(k),
        "--num_cases",
        str(num_cases),
        "--output_dir",
        str(output_dir),
        "--runs",
    ]
    cmd.extend(run_specs)
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    output = "$ " + shlex.join(cmd) + "\n\n" + (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output
