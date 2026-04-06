from __future__ import annotations

import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GpuNode:
    node: str
    partition: str
    total_gpus: int
    allocated_gpus: int
    available_gpus: int
    state: str
    gres: str


@dataclass
class SlurmJob:
    job_id: str
    name: str
    state: str
    runtime: str
    nodes: str
    reason: str


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(args=cmd, returncode=127, stdout="", stderr=str(exc))


def _parse_tres_value(tres: str, key: str) -> int:
    for chunk in tres.split(","):
        if chunk.startswith(f"{key}="):
            raw = chunk.split("=", 1)[1]
            try:
                return int(raw)
            except ValueError:
                return 0
    return 0


def slurm_tool_status() -> dict[str, bool]:
    return {
        "scontrol": shutil.which("scontrol") is not None,
        "sinfo": shutil.which("sinfo") is not None,
        "squeue": shutil.which("squeue") is not None,
        "sbatch": shutil.which("sbatch") is not None,
        "srun": shutil.which("srun") is not None,
    }


def _local_gpu_fallback() -> list[GpuNode]:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return []

    count = int(torch.cuda.device_count())
    if count <= 0:
        return [
            GpuNode(
                node="(unavailable)",
                partition="local",
                total_gpus=0,
                allocated_gpus=0,
                available_gpus=0,
                state="unavailable",
                gres="no-slurm-tools-and-no-local-cuda",
            )
        ]
    return [
        GpuNode(
            node=socket.gethostname(),
            partition="local",
            total_gpus=count,
            allocated_gpus=0,
            available_gpus=count,
            state="local",
            gres=f"gpu:{count}",
        )
    ]


def _diagnostic_gpu_row(reason: str) -> list[GpuNode]:
    short_reason = reason.strip().replace("\n", " ")
    if len(short_reason) > 140:
        short_reason = short_reason[:137] + "..."
    return [
        GpuNode(
            node="(unavailable)",
            partition="slurm",
            total_gpus=0,
            allocated_gpus=0,
            available_gpus=0,
            state="unavailable",
            gres=short_reason or "no-gpu-data-reported",
        )
    ]


def list_gpu_nodes() -> list[GpuNode]:
    tools = slurm_tool_status()
    if not tools["scontrol"] and not tools["sinfo"]:
        return _local_gpu_fallback()

    proc = _run(["scontrol", "show", "nodes", "-o"])
    nodes: list[GpuNode] = []
    if proc.returncode == 0 and proc.stdout.strip():
        for line in proc.stdout.splitlines():
            parts = dict(item.split("=", 1) for item in line.split() if "=" in item)
            node = parts.get("NodeName", "")
            partition = parts.get("Partitions", "")
            state = parts.get("State", "unknown")
            gres = parts.get("Gres", "")
            cfg_tres = parts.get("CfgTRES", "")
            alloc_tres = parts.get("AllocTRES", "")
            total = _parse_tres_value(cfg_tres, "gres/gpu")
            allocated = _parse_tres_value(alloc_tres, "gres/gpu")
            if total <= 0 and "gpu:" in gres:
                match = re.search(r"gpu(?::[^:]+)?:([0-9]+)", gres)
                if match:
                    total = int(match.group(1))
            if total <= 0:
                continue
            nodes.append(
                GpuNode(
                    node=node,
                    partition=partition,
                    total_gpus=total,
                    allocated_gpus=allocated,
                    available_gpus=max(total - allocated, 0),
                    state=state,
                    gres=gres,
                )
            )
        if nodes:
            return sorted(nodes, key=lambda x: (x.partition, x.node))

    # fallback to sinfo
    fallback = _run(["sinfo", "-N", "-h", "-o", "%N|%P|%G|%t"])
    if fallback.returncode != 0:
        local_fallback = _local_gpu_fallback()
        if local_fallback:
            return local_fallback
        reasons = []
        if proc.stderr.strip():
            reasons.append(f"scontrol: {proc.stderr.strip().splitlines()[0]}")
        if fallback.stderr.strip():
            reasons.append(f"sinfo: {fallback.stderr.strip().splitlines()[0]}")
        return _diagnostic_gpu_row("; ".join(reasons))
    for row in fallback.stdout.splitlines():
        node, partition, gres, state = row.split("|", 3)
        match = re.search(r"gpu(?::[^:]+)?:([0-9]+)", gres)
        if not match:
            continue
        total = int(match.group(1))
        nodes.append(
            GpuNode(
                node=node,
                partition=partition,
                total_gpus=total,
                allocated_gpus=0,
                available_gpus=total,
                state=state,
                gres=gres,
            )
        )
    if nodes:
        return sorted(nodes, key=lambda x: (x.partition, x.node))
    return _diagnostic_gpu_row("Slurm returned no GPU nodes")


def list_jobs(user: str | None = None) -> list[SlurmJob]:
    cmd = ["squeue", "-h", "-o", "%i|%j|%T|%M|%D|%R"]
    if user:
        cmd.extend(["-u", user])
    proc = _run(cmd)
    if proc.returncode != 0:
        return []

    jobs: list[SlurmJob] = []
    for row in proc.stdout.splitlines():
        fields = row.split("|")
        if len(fields) != 6:
            continue
        jobs.append(
            SlurmJob(
                job_id=fields[0],
                name=fields[1],
                state=fields[2],
                runtime=fields[3],
                nodes=fields[4],
                reason=fields[5],
            )
        )
    return jobs


def build_sbatch_command(
    *,
    command: list[str],
    job_name: str,
    log_dir: Path,
    slurm_cfg: dict[str, Any],
    gpus: int | None = None,
    constraint: str | None = None,
    nodelist: str | None = None,
) -> tuple[list[str], Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    script_body = "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n"
    fd, script_path_str = tempfile.mkstemp(prefix=f"{job_name}_", suffix=".sbatch.sh", dir=log_dir.as_posix())
    os.close(fd)
    script_path = Path(script_path_str)
    script_path.write_text(script_body, encoding="utf-8")

    cmd = ["sbatch", "--job-name", job_name]
    if slurm_cfg.get("partition"):
        cmd.extend(["--partition", str(slurm_cfg["partition"])])
    if slurm_cfg.get("account"):
        cmd.extend(["--account", str(slurm_cfg["account"])])
    if slurm_cfg.get("qos"):
        cmd.extend(["--qos", str(slurm_cfg["qos"])])
    if slurm_cfg.get("time"):
        cmd.extend(["--time", str(slurm_cfg["time"])])
    if slurm_cfg.get("mem"):
        cmd.extend(["--mem", str(slurm_cfg["mem"])])
    if slurm_cfg.get("cpus_per_task"):
        cmd.extend(["--cpus-per-task", str(slurm_cfg["cpus_per_task"])])

    gpu_count = int(gpus if gpus is not None else slurm_cfg.get("gpus", 1))
    if gpu_count > 0:
        cmd.extend(["--gpus", str(gpu_count)])

    final_constraint = constraint if constraint is not None else slurm_cfg.get("constraint", "")
    if final_constraint:
        cmd.extend(["--constraint", str(final_constraint)])
    final_nodelist = nodelist if nodelist is not None else slurm_cfg.get("nodelist", "")
    if final_nodelist:
        cmd.extend(["--nodelist", str(final_nodelist)])

    extra_args = slurm_cfg.get("extra_args", [])
    if isinstance(extra_args, list):
        cmd.extend([str(a) for a in extra_args])

    cmd.extend(["--output", str(log_dir / f"{job_name}-%j.out")])
    cmd.extend(["--error", str(log_dir / f"{job_name}-%j.err")])
    cmd.append(str(script_path))
    return cmd, script_path


def submit_sbatch(
    *,
    command: list[str],
    job_name: str,
    log_dir: Path,
    slurm_cfg: dict[str, Any],
    gpus: int | None = None,
    constraint: str | None = None,
    nodelist: str | None = None,
) -> tuple[str | None, str, Path]:
    sbatch_cmd, script_path = build_sbatch_command(
        command=command,
        job_name=job_name,
        log_dir=log_dir,
        slurm_cfg=slurm_cfg,
        gpus=gpus,
        constraint=constraint,
        nodelist=nodelist,
    )
    proc = _run(sbatch_cmd)
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return None, output.strip(), script_path

    match = re.search(r"Submitted batch job (\\d+)", proc.stdout or "")
    job_id = match.group(1) if match else None
    return job_id, output.strip(), script_path


def submit_sbatch_array_wrap(
    *,
    wrap_command: str,
    job_name: str,
    log_dir: Path,
    slurm_cfg: dict[str, Any],
    array: str,
    gpus: int | None = None,
    constraint: str | None = None,
    nodelist: str | None = None,
) -> tuple[str | None, str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["sbatch", "--job-name", job_name, "--array", array]
    if slurm_cfg.get("partition"):
        cmd.extend(["--partition", str(slurm_cfg["partition"])])
    if slurm_cfg.get("account"):
        cmd.extend(["--account", str(slurm_cfg["account"])])
    if slurm_cfg.get("qos"):
        cmd.extend(["--qos", str(slurm_cfg["qos"])])
    if slurm_cfg.get("time"):
        cmd.extend(["--time", str(slurm_cfg["time"])])
    if slurm_cfg.get("mem"):
        cmd.extend(["--mem", str(slurm_cfg["mem"])])
    if slurm_cfg.get("cpus_per_task"):
        cmd.extend(["--cpus-per-task", str(slurm_cfg["cpus_per_task"])])

    gpu_count = int(gpus if gpus is not None else slurm_cfg.get("gpus", 0))
    if gpu_count > 0:
        cmd.extend(["--gpus", str(gpu_count)])

    final_constraint = constraint if constraint is not None else slurm_cfg.get("constraint", "")
    if final_constraint:
        cmd.extend(["--constraint", str(final_constraint)])

    final_nodelist = nodelist if nodelist is not None else slurm_cfg.get("nodelist", "")
    if final_nodelist:
        cmd.extend(["--nodelist", str(final_nodelist)])

    extra_args = slurm_cfg.get("extra_args", [])
    if isinstance(extra_args, list):
        cmd.extend([str(a) for a in extra_args])

    cmd.extend(["--output", str(log_dir / f"{job_name}_%A_%a.out")])
    cmd.extend(["--error", str(log_dir / f"{job_name}_%A_%a.err")])
    cmd.extend(["--wrap", wrap_command])

    proc = _run(cmd)
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return None, output.strip()

    match = re.search(r"Submitted batch job (\d+)", proc.stdout or "")
    job_id = match.group(1) if match else None
    return job_id, output.strip()


def launch_interactive_srun(
    *,
    command: list[str],
    slurm_cfg: dict[str, Any],
    gpus: int | None = None,
    constraint: str | None = None,
    nodelist: str | None = None,
) -> int:
    cmd = ["srun", "--pty"]
    if slurm_cfg.get("partition"):
        cmd.extend(["--partition", str(slurm_cfg["partition"])])
    if slurm_cfg.get("account"):
        cmd.extend(["--account", str(slurm_cfg["account"])])
    if slurm_cfg.get("qos"):
        cmd.extend(["--qos", str(slurm_cfg["qos"])])
    if slurm_cfg.get("time"):
        cmd.extend(["--time", str(slurm_cfg["time"])])
    if slurm_cfg.get("mem"):
        cmd.extend(["--mem", str(slurm_cfg["mem"])])
    if slurm_cfg.get("cpus_per_task"):
        cmd.extend(["--cpus-per-task", str(slurm_cfg["cpus_per_task"])])

    gpu_count = int(gpus if gpus is not None else slurm_cfg.get("gpus", 1))
    if gpu_count > 0:
        cmd.extend(["--gpus", str(gpu_count)])

    final_constraint = constraint if constraint is not None else slurm_cfg.get("constraint", "")
    if final_constraint:
        cmd.extend(["--constraint", str(final_constraint)])
    final_nodelist = nodelist if nodelist is not None else slurm_cfg.get("nodelist", "")
    if final_nodelist:
        cmd.extend(["--nodelist", str(final_nodelist)])

    extra_args = slurm_cfg.get("extra_args", [])
    if isinstance(extra_args, list):
        cmd.extend([str(a) for a in extra_args])

    cmd.extend(command)
    try:
        proc = subprocess.run(cmd, check=False)
        return proc.returncode
    except FileNotFoundError:
        return 127
