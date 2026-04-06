from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from .checkpoints import list_hf_checkpoints
from .config import USER_CONFIG_PATH, load_config, write_default_config
from .datasets import run_dataset_download, submit_dataset_download_slurm
from .runner import execute_runs, preview_commands, run_analysis
from .slurm import launch_interactive_srun, list_gpu_nodes, list_jobs


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2))


def _cmd_config(args: argparse.Namespace) -> int:
    if args.config_action == "init":
        path = write_default_config(force=args.force)
        print(path)
        return 0
    if args.config_action == "show":
        _print_json(load_config())
        return 0
    if args.config_action == "path":
        print(USER_CONFIG_PATH)
        return 0
    raise ValueError(f"Unknown config action: {args.config_action}")


def _cmd_checkpoints(_: argparse.Namespace) -> int:
    cfg = load_config()
    try:
        state = list_hf_checkpoints(cfg)
        _print_json(
            {
                "repo_id": state.repo_id,
                "latest_step": state.latest_step,
                "available_steps": state.available_steps,
                "recommended_steps": state.recommended_steps,
                "inferred_latest_step": state.inferred_latest_step,
                "inferred_schedule": state.inferred_schedule,
                "notes": state.notes,
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"checkpoint discovery failed: {exc}", file=sys.stderr)
        return 1


def _cmd_slurm_gpus(_: argparse.Namespace) -> int:
    nodes = list_gpu_nodes()
    _print_json([n.__dict__ for n in nodes])
    return 0


def _cmd_slurm_status(args: argparse.Namespace) -> int:
    jobs = list_jobs(user=args.user)
    _print_json([j.__dict__ for j in jobs])
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config()
    if args.interactive:
        previews = preview_commands(
            config=cfg,
            benchmark=args.benchmark,
            model_profile_key=args.model_profile,
            checkpoint_step=args.checkpoint_step,
            dataset_group=args.dataset_group,
            smoke=args.smoke,
            run_id=args.run_id,
        )
        if not previews:
            print("No commands resolved.", file=sys.stderr)
            return 1
        if len(previews) > 1:
            print("Interactive mode supports one dataset at a time. Use --dataset-group or benchmark with one dataset.", file=sys.stderr)
            return 1
        dataset, command = previews[0]
        print(f"Launching interactive Slurm run for dataset={dataset}")
        print("$ " + shlex.join(command))
        code = launch_interactive_srun(
            command=command,
            slurm_cfg=cfg["slurm"],
            gpus=args.slurm_gpus,
            partition=args.slurm_partition,
            constraint=args.slurm_constraint,
            nodelist=args.slurm_nodelist,
        )
        return code

    run_root, results = execute_runs(
        config=cfg,
        benchmark=args.benchmark,
        model_profile_key=args.model_profile,
        checkpoint_step=args.checkpoint_step,
        dataset_group=args.dataset_group,
        smoke=args.smoke,
        force=args.force,
        run_id=args.run_id,
        use_slurm=args.slurm,
        slurm_gpus=args.slurm_gpus,
        slurm_partition=args.slurm_partition,
        slurm_constraint=args.slurm_constraint,
        slurm_nodelist=args.slurm_nodelist,
    )

    payload = {
        "run_root": str(run_root),
        "results": [r.__dict__ for r in results],
        "failures": [r.__dict__ for r in results if r.return_code not in (0, None)],
    }
    _print_json(payload)
    return 0 if not payload["failures"] else 1


def _cmd_analyze(args: argparse.Namespace) -> int:
    cfg = load_config()
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["paths"]["results_root"]) / "analysis" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    code, output = run_analysis(
        config=cfg,
        dataset=args.dataset,
        run_specs=args.runs,
        output_dir=output_dir,
        k=args.k,
        num_cases=args.num_cases,
    )
    (output_dir / "analysis.log").write_text(output, encoding="utf-8")
    _print_json({"output_dir": str(output_dir), "return_code": code, "log_file": str(output_dir / "analysis.log")})
    return code


def _cmd_tui(_: argparse.Namespace) -> int:
    from .tui import BenchmarkTuiApp

    app = BenchmarkTuiApp()
    app.run()
    return 0


def _cmd_dataset_download(args: argparse.Namespace) -> int:
    cfg = load_config()
    if args.slurm:
        job_id, output, log_dir = submit_dataset_download_slurm(
            config=cfg,
            dataset=args.dataset,
            num_shards=args.num_shards,
            slurm_gpus=args.slurm_gpus,
            slurm_constraint=args.slurm_constraint,
            slurm_nodelist=args.slurm_nodelist,
        )
        payload = {
            "dataset": args.dataset,
            "mode": "slurm_array",
            "num_shards": args.num_shards,
            "job_id": job_id,
            "log_dir": log_dir,
            "output": output,
        }
        _print_json(payload)
        return 0 if job_id else 1

    code, output = run_dataset_download(
        config=cfg,
        dataset=args.dataset,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    print(output)
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark CLI for retrieval evaluation and Slurm orchestration.")
    sub = parser.add_subparsers(dest="command", required=True)

    tui = sub.add_parser("tui", help="Launch Textual TUI.")
    tui.set_defaults(func=_cmd_tui)

    config = sub.add_parser("config", help="Manage local config.")
    config_sub = config.add_subparsers(dest="config_action", required=True)
    config_init = config_sub.add_parser("init", help="Create local config from defaults.")
    config_init.add_argument("--force", action="store_true", help="Overwrite existing local config.")
    config_init.set_defaults(func=_cmd_config)
    config_show = config_sub.add_parser("show", help="Show merged config.")
    config_show.set_defaults(func=_cmd_config)
    config_path = config_sub.add_parser("path", help="Show local config path.")
    config_path.set_defaults(func=_cmd_config)

    checkpoints = sub.add_parser("checkpoints", help="Checkpoint operations.")
    checkpoints_sub = checkpoints.add_subparsers(dest="checkpoints_action", required=True)
    checkpoints_list = checkpoints_sub.add_parser("list", help="List checkpoints from configured HF repo.")
    checkpoints_list.set_defaults(func=_cmd_checkpoints)

    slurm = sub.add_parser("slurm", help="Slurm status helpers.")
    slurm_sub = slurm.add_subparsers(dest="slurm_action", required=True)
    slurm_gpus = slurm_sub.add_parser("gpus", help="List GPUs from Slurm nodes.")
    slurm_gpus.set_defaults(func=_cmd_slurm_gpus)
    slurm_status = slurm_sub.add_parser("status", help="List Slurm jobs from squeue.")
    slurm_status.add_argument("--user", default=os.environ.get("USER", ""), help="Filter by user.")
    slurm_status.set_defaults(func=_cmd_slurm_status)

    run = sub.add_parser("run", help="Run benchmark jobs.")
    run.add_argument("--benchmark", default="repoeval", choices=["repoeval", "swe-bench-lite", "coir"], help="Benchmark key.")
    run.add_argument("--dataset-group", default=None, help="COIR group key when benchmark=coir.")
    run.add_argument("--model-profile", default="qwen3_embed_0_6b", help="Model profile key from config.")
    run.add_argument("--checkpoint-step", type=int, default=None, help="Evaluate finetune checkpoint step instead of baseline profile.")
    run.add_argument("--smoke", action="store_true", help="Run with tiny settings for quick validation.")
    run.add_argument("--force", action="store_true", help="Force baseline rerun even if results already exist.")
    run.add_argument("--run-id", default=None, help="Optional explicit run id.")
    run.add_argument("--slurm", action="store_true", help="Submit with sbatch instead of local execution.")
    run.add_argument("--interactive", action="store_true", help="Run via interactive srun --pty.")
    run.add_argument("--slurm-gpus", type=int, default=None, help="Override GPU count for Slurm jobs.")
    run.add_argument("--slurm-partition", default=None, help="Override Slurm partition.")
    run.add_argument("--slurm-constraint", default=None, help="Override Slurm constraint string.")
    run.add_argument("--slurm-nodelist", default=None, help="Override Slurm nodelist (comma-separated nodes).")
    run.set_defaults(func=_cmd_run)

    analyze = sub.add_parser("analyze", help="Analyze positive/negative cases from raw results.")
    analyze.add_argument("--dataset", required=True, help="Dataset key matching run raw outputs.")
    analyze.add_argument("--runs", nargs="+", required=True, help="Run specs NAME=/path/to/raw_results.jsonl")
    analyze.add_argument("--k", type=int, default=10)
    analyze.add_argument("--num-cases", type=int, default=8)
    analyze.add_argument("--output-dir", default=None, help="Output directory for analysis artifacts.")
    analyze.set_defaults(func=_cmd_analyze)

    dataset = sub.add_parser("dataset", help="Dataset preparation/download commands.")
    dataset_sub = dataset.add_subparsers(dest="dataset_action", required=True)
    dataset_download = dataset_sub.add_parser("download", help="Download/create datasets with optional sharding.")
    dataset_download.add_argument("--dataset", required=True, choices=["swe-bench-lite", "repoeval", "coir"])
    dataset_download.add_argument("--num-shards", type=int, default=1, help="Total shards.")
    dataset_download.add_argument("--shard-id", type=int, default=0, help="Shard index for local mode.")
    dataset_download.add_argument("--slurm", action="store_true", help="Submit as Slurm job array.")
    dataset_download.add_argument("--slurm-gpus", type=int, default=None, help="Override Slurm GPU count.")
    dataset_download.add_argument("--slurm-constraint", default=None, help="Override Slurm constraint.")
    dataset_download.add_argument("--slurm-nodelist", default=None, help="Override Slurm nodelist.")
    dataset_download.set_defaults(func=_cmd_dataset_download)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
