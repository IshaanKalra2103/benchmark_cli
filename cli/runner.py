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
from .slurm import submit_sbatch, submit_sbatch_array_wrap


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


def _list_swebench_instance_dirs(dataset_root: Path) -> list[str]:
    dirs: list[str] = []
    for path in dataset_root.glob("swe-bench-lite_*"):
        if not path.is_dir():
            continue
        if not (path / "corpus.jsonl").exists():
            continue
        if not (path / "queries.jsonl").exists():
            continue
        if not (path / "qrels" / "test.tsv").exists():
            continue
        dirs.append(path.name)
    return sorted(dirs)


def _split_shards(items: list[str], num_shards: int) -> list[list[str]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be > 0")
    if not items:
        raise ValueError("No SWE-Bench instances found to shard.")
    if num_shards > len(items):
        raise ValueError(f"num_shards ({num_shards}) cannot exceed instance count ({len(items)}).")
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    for idx, item in enumerate(items):
        shards[idx % num_shards].append(item)
    return shards


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{value}\n" for value in values)
    path.write_text(body, encoding="utf-8")


def _eval_base_command(config: dict[str, Any], model: ModelSpec, smoke: bool) -> list[str]:
    defaults = config["models"]["defaults"]
    paths = config["paths"]
    script = PROJECT_ROOT / "benchmarks" / "eval_repo_bench_retriever.py"
    cmd = [
        "uv",
        "run",
        "python",
        str(script),
        "--dataset",
        "swe-bench-lite",
        "--dataset_root",
        str(paths["dataset_root"]),
        "--model",
        model.model,
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
    if model.reranker_model:
        cmd.extend(["--reranker_model", model.reranker_model])
    if smoke:
        cmd.extend(["--max_instances", "1", "--top_k", "5", "--candidate_k", "20"])
    return cmd


def _build_swebench_array_wrap_command(
    *,
    config: dict[str, Any],
    model: ModelSpec,
    model_tag: str,
    run_root: Path,
    smoke: bool,
) -> str:
    token = "__SLURM_ARRAY_TASK_ID__"
    shard_list = run_root / "swebench_dual" / "shard_lists" / model_tag / f"shard_{token}.txt"
    shard_dir = run_root / "swebench_dual" / "shards" / model_tag / f"shard_{token}"
    cmd = _eval_base_command(config=config, model=model, smoke=smoke)
    cmd.extend(
        [
            "--instance_list_file",
            str(shard_list),
            "--output_file",
            str(shard_dir / "summary.json"),
            "--results_file",
            str(shard_dir / "retrieval_results.jsonl"),
            "--raw_results_file",
            str(shard_dir / "raw_results.jsonl"),
            "--per_query_metrics_file",
            str(shard_dir / "per_query_metrics.jsonl"),
        ]
    )
    quoted = shlex.join(cmd)
    quoted_token = shlex.quote(token)
    return f"cd {shlex.quote(str(PROJECT_ROOT))} && {quoted}".replace(quoted_token, "$SLURM_ARRAY_TASK_ID")


def submit_swebench_dual_sharded_slurm(
    *,
    config: dict[str, Any],
    num_shards: int,
    base_model_profile: str,
    finetuned_model_profile: str,
    run_id: str | None,
    smoke: bool,
    slurm_gpus: int | None,
    slurm_partition: str | None,
    slurm_constraint: str | None,
    slurm_nodelist: str | None,
) -> dict[str, Any]:
    run_id = run_id or _now_tag()
    results_root = Path(config["paths"]["results_root"]).resolve()
    run_root = results_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(config["paths"]["dataset_root"]).resolve()
    instance_dirs = _list_swebench_instance_dirs(dataset_root=dataset_root)
    shards = _split_shards(instance_dirs, num_shards=num_shards)
    shard_sizes = [len(s) for s in shards]

    # Persist shard-to-instance mapping once; both models consume same mapping.
    mapping_root = run_root / "swebench_dual" / "shard_lists"
    _write_json(
        run_root / "swebench_dual" / "shard_mapping.json",
        {
            "run_id": run_id,
            "benchmark": "swe-bench-lite",
            "num_shards": num_shards,
            "num_instances": len(instance_dirs),
            "shard_sizes": shard_sizes,
        },
    )

    for model_tag in ("base", "finetuned"):
        for shard_id, shard_instances in enumerate(shards):
            shard_file = mapping_root / model_tag / f"shard_{shard_id}.txt"
            _write_lines(shard_file, shard_instances)

    model_specs = {
        "base": resolve_model_spec(config=config, model_profile_key=base_model_profile, checkpoint_step=None),
        "finetuned": resolve_model_spec(config=config, model_profile_key=finetuned_model_profile, checkpoint_step=None),
    }

    jobs: dict[str, dict[str, Any]] = {}
    for model_tag, model in model_specs.items():
        wrap_command = _build_swebench_array_wrap_command(
            config=config,
            model=model,
            model_tag=model_tag,
            run_root=run_root,
            smoke=smoke,
        )
        log_dir = run_root / "swebench_dual" / "logs" / model_tag
        job_name = f"{_safe_name(run_id)}__swedual_{model_tag}"
        if len(job_name) > 120:
            job_name = job_name[:120]
        job_id, output = submit_sbatch_array_wrap(
            wrap_command=wrap_command,
            job_name=job_name,
            log_dir=log_dir,
            slurm_cfg=config["slurm"],
            array=f"0-{num_shards - 1}",
            gpus=slurm_gpus,
            partition=slurm_partition,
            constraint=slurm_constraint,
            nodelist=slurm_nodelist,
        )
        jobs[model_tag] = {
            "job_id": job_id,
            "job_name": job_name,
            "status": "submitted" if job_id else "submit_failed",
            "output": output,
            "log_dir": str(log_dir),
            "model_profile": model.key,
            "model_name": model.name,
            "model_id": model.model,
            "reranker_model": model.reranker_model,
        }

    submit_manifest = {
        "run_id": run_id,
        "benchmark": "swe-bench-lite",
        "mode": "dual_sharded_slurm",
        "num_shards": num_shards,
        "num_instances": len(instance_dirs),
        "shard_sizes": shard_sizes,
        "smoke": smoke,
        "run_root": str(run_root),
        "dataset_root": str(dataset_root),
        "slurm_overrides": {
            "gpus": slurm_gpus,
            "partition": slurm_partition,
            "constraint": slurm_constraint,
            "nodelist": slurm_nodelist,
        },
        "jobs": jobs,
        "paths": {
            "shard_mapping": str(run_root / "swebench_dual" / "shard_mapping.json"),
            "submit_manifest": str(run_root / "swebench_dual" / "manifest.submit.json"),
        },
    }
    _write_json(run_root / "swebench_dual" / "manifest.submit.json", submit_manifest)
    return submit_manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _merge_summary_files(summary_files: list[Path]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = [json.loads(path.read_text(encoding="utf-8")) for path in summary_files]
    if not summaries:
        raise ValueError("No shard summaries found to merge.")

    total_queries = sum(int(s.get("num_queries", 0)) for s in summaries)
    merged: dict[str, Any] = {}
    for metric_name in ("ndcg", "mrr", "recall", "precision"):
        metric_keys: set[str] = set()
        for summary in summaries:
            metric_keys.update((summary.get(metric_name) or {}).keys())
        merged_metric: dict[str, float] = {}
        for key in sorted(metric_keys):
            numerator = 0.0
            for summary in summaries:
                weight = float(summary.get("num_queries", 0))
                numerator += weight * float((summary.get(metric_name) or {}).get(key, 0.0))
            merged_metric[key] = (numerator / total_queries) if total_queries > 0 else 0.0
        merged[metric_name] = merged_metric

    merged["time"] = sum(float(s.get("time", 0.0)) for s in summaries)
    merged["num_queries"] = total_queries
    merged["num_instances"] = sum(int(s.get("num_instances", 0)) for s in summaries)
    merged["model"] = summaries[0].get("model")
    merged["reranker_model"] = summaries[0].get("reranker_model")

    per_instance: dict[str, Any] = {}
    for summary in summaries:
        for instance, metrics in (summary.get("per_instance") or {}).items():
            per_instance[instance] = metrics
    merged["per_instance"] = per_instance
    merged["num_shards_merged"] = len(summaries)
    return merged


def merge_swebench_dual_sharded_outputs(
    *,
    config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    results_root = Path(config["paths"]["results_root"]).resolve()
    run_root = results_root / run_id
    submit_manifest_path = run_root / "swebench_dual" / "manifest.submit.json"
    if not submit_manifest_path.exists():
        raise ValueError(f"Submit manifest not found: {submit_manifest_path}")

    submit_manifest = json.loads(submit_manifest_path.read_text(encoding="utf-8"))
    num_shards = int(submit_manifest["num_shards"])
    models = ("base", "finetuned")

    model_outputs: dict[str, dict[str, Any]] = {}
    for model_tag in models:
        shard_root = run_root / "swebench_dual" / "shards" / model_tag
        summary_files: list[Path] = []
        merged_retrieval: list[dict[str, Any]] = []
        merged_raw: list[dict[str, Any]] = []
        merged_per_query: list[dict[str, Any]] = []
        have_per_query = True

        for shard_id in range(num_shards):
            shard_dir = shard_root / f"shard_{shard_id}"
            summary_file = shard_dir / "summary.json"
            retrieval_file = shard_dir / "retrieval_results.jsonl"
            raw_file = shard_dir / "raw_results.jsonl"
            per_query_file = shard_dir / "per_query_metrics.jsonl"
            for required in (summary_file, retrieval_file, raw_file):
                if not required.exists():
                    raise ValueError(f"Missing shard artifact: {required}")
            summary_files.append(summary_file)
            merged_retrieval.extend(_read_jsonl(retrieval_file))
            merged_raw.extend(_read_jsonl(raw_file))
            if have_per_query and per_query_file.exists():
                merged_per_query.extend(_read_jsonl(per_query_file))
            else:
                have_per_query = False

        final_dir = run_root / "swebench_dual" / "final" / model_tag
        final_summary = final_dir / "summary.json"
        final_retrieval = final_dir / "retrieval_results.jsonl"
        final_raw = final_dir / "raw_results.jsonl"
        final_per_query = final_dir / "per_query_metrics.jsonl"

        _write_json(final_summary, _merge_summary_files(summary_files))
        _write_jsonl(final_retrieval, merged_retrieval)
        _write_jsonl(final_raw, merged_raw)
        if have_per_query:
            _write_jsonl(final_per_query, merged_per_query)

        model_outputs[model_tag] = {
            "final_dir": str(final_dir),
            "summary_file": str(final_summary),
            "retrieval_results_file": str(final_retrieval),
            "raw_results_file": str(final_raw),
            "per_query_metrics_file": str(final_per_query) if have_per_query else None,
            "num_retrieval_rows": len(merged_retrieval),
            "num_raw_rows": len(merged_raw),
            "num_per_query_rows": len(merged_per_query) if have_per_query else None,
        }

    final_manifest = {
        "run_id": run_id,
        "benchmark": "swe-bench-lite",
        "mode": "dual_sharded_slurm",
        "submit_manifest": str(submit_manifest_path),
        "final_outputs": model_outputs,
    }
    final_manifest_path = run_root / "swebench_dual" / "manifest.final.json"
    _write_json(final_manifest_path, final_manifest)
    final_manifest["final_manifest"] = str(final_manifest_path)
    return final_manifest
