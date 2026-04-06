from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .slurm import submit_sbatch_array_wrap


def _base_python_module_cmd(module_name: str) -> list[str]:
    return [sys.executable, "-m", module_name]


def _dataset_command(
    *,
    config: dict[str, Any],
    dataset: str,
    shard_id: int,
    num_shards: int,
) -> list[str]:
    paths = config["paths"]
    dataset_root = str(paths["dataset_root"])
    results_root = str(paths["results_root"])

    if dataset == "swe-bench-lite":
        cmd = _base_python_module_cmd("create.swebench_repo")
        cmd.extend(
            [
                "--dataset_name",
                "princeton-nlp/SWE-bench_Lite",
                "--cache_dir",
                str(Path(dataset_root) / "hf_cache"),
                "--tmp_dir",
                str(Path(dataset_root) / "tmp"),
                "--output_dir",
                dataset_root,
                "--skip_existing",
                "--num_shards",
                str(num_shards),
                "--shard_id",
                str(shard_id),
            ]
        )
        return cmd

    if dataset == "repoeval":
        cmd = _base_python_module_cmd("create.repoeval_repo")
        cmd.extend(
            [
                "--output_dir",
                dataset_root,
                "--results_dir",
                str(Path(results_root) / "repoeval_gt"),
                "--split",
                "function",
                "--context_length",
                "2k",
                "--data_cache_dir",
                str(Path(dataset_root) / "repoeval_cache"),
                "--num_shards",
                str(num_shards),
                "--shard_id",
                str(shard_id),
            ]
        )
        return cmd

    if dataset == "coir":
        cmd = _base_python_module_cmd("create.coir_download")
        cmd.extend(
            [
                "--dataset_root",
                dataset_root,
                "--num_shards",
                str(num_shards),
                "--shard_id",
                str(shard_id),
            ]
        )
        return cmd

    raise ValueError(f"Unsupported dataset '{dataset}'.")


def run_dataset_download(
    *,
    config: dict[str, Any],
    dataset: str,
    shard_id: int,
    num_shards: int,
) -> tuple[int, str]:
    cmd = _dataset_command(config=config, dataset=dataset, shard_id=shard_id, num_shards=num_shards)
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    output = "$ " + shlex.join(cmd) + "\n\n" + (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output


def submit_dataset_download_slurm(
    *,
    config: dict[str, Any],
    dataset: str,
    num_shards: int,
    slurm_gpus: int | None,
    slurm_constraint: str | None,
    slurm_nodelist: str | None,
) -> tuple[str | None, str, str]:
    if num_shards <= 0:
        raise ValueError("num_shards must be > 0")

    dataset_root = Path(config["paths"]["dataset_root"])
    log_dir = Path(config["paths"]["results_root"]) / "dataset_download_logs" / dataset
    log_dir.mkdir(parents=True, exist_ok=True)

    module_map = {
        "swe-bench-lite": "create.swebench_repo",
        "repoeval": "create.repoeval_repo",
        "coir": "create.coir_download",
    }
    if dataset not in module_map:
        raise ValueError(f"Unsupported dataset '{dataset}'.")

    # Always run from project root so relative outputs match the benchmark_cli workspace.
    base = ["cd", str(PROJECT_ROOT), "&&", "uv", "run", "python", "-m", module_map[dataset]]

    if dataset == "swe-bench-lite":
        args = [
            "--dataset_name", "princeton-nlp/SWE-bench_Lite",
            "--cache_dir", str(dataset_root / "hf_cache"),
            "--tmp_dir", str(dataset_root / "tmp"),
            "--output_dir", str(dataset_root),
            "--skip_existing",
            "--num_shards", str(num_shards),
            "--shard_id", "$SLURM_ARRAY_TASK_ID",
        ]
    elif dataset == "repoeval":
        args = [
            "--output_dir", str(dataset_root),
            "--results_dir", str(Path(config["paths"]["results_root"]) / "repoeval_gt"),
            "--split", "function",
            "--context_length", "2k",
            "--data_cache_dir", str(dataset_root / "repoeval_cache"),
            "--num_shards", str(num_shards),
            "--shard_id", "$SLURM_ARRAY_TASK_ID",
        ]
    else:
        args = [
            "--dataset_root", str(dataset_root),
            "--num_shards", str(num_shards),
            "--shard_id", "$SLURM_ARRAY_TASK_ID",
        ]

    wrap_command = " ".join(shlex.quote(part) for part in base + args)
    array_spec = f"0-{num_shards - 1}"
    job_id, output = submit_sbatch_array_wrap(
        wrap_command=wrap_command,
        job_name=f"dataset_{dataset}",
        log_dir=log_dir,
        slurm_cfg=config["slurm"],
        array=array_spec,
        gpus=slurm_gpus,
        constraint=slurm_constraint,
        nodelist=slurm_nodelist,
    )
    return job_id, output, str(log_dir)
