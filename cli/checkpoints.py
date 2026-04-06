from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class CheckpointState:
    repo_id: str
    available_steps: list[int]
    recommended_steps: list[int]
    latest_step: int | None
    inferred_latest_step: int | None
    inferred_schedule: list[int]
    notes: list[str]


def _parse_checkpoint_steps(paths: list[str], pattern: str = "checkpoint-") -> list[int]:
    steps: set[int] = set()
    regex = re.compile(rf"{re.escape(pattern)}(\\d+)")
    for path in paths:
        match = regex.search(path)
        if not match:
            continue
        steps.add(int(match.group(1)))
    return sorted(steps)


def build_checkpoint_schedule(
    available_steps: list[int],
    include_first: bool,
    anchor_step: int,
    interval: int,
) -> list[int]:
    if not available_steps:
        return []

    selected: set[int] = set()
    if include_first:
        selected.add(available_steps[0])

    if anchor_step in available_steps:
        selected.add(anchor_step)

    latest = available_steps[-1]
    if interval > 0:
        step = anchor_step
        while step <= latest:
            if step in available_steps:
                selected.add(step)
            step += interval

    return sorted(selected)


def list_hf_checkpoints(config: dict[str, Any]) -> CheckpointState:
    repo_id = config["models"]["finetune"]["hf_repo"]
    pattern = config["models"]["finetune"].get("checkpoint_pattern", "checkpoint-")
    policy = config["checkpoint_policy"]

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for checkpoint discovery. Install project dependencies first."
        ) from exc

    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    steps = _parse_checkpoint_steps(files, pattern=pattern)
    notes: list[str] = []

    inferred_latest_step: int | None = None
    if "trainer_state.json" in files:
        try:
            from huggingface_hub import hf_hub_download

            trainer_state_path = hf_hub_download(repo_id=repo_id, filename="trainer_state.json", repo_type="model")
            trainer_state = json.loads(open(trainer_state_path, "r", encoding="utf-8").read())
            global_step = trainer_state.get("global_step")
            if isinstance(global_step, int):
                inferred_latest_step = global_step
        except Exception as exc:  # noqa: BLE001
            notes.append(f"could not parse trainer_state.json: {exc}")

    inferred_schedule: list[int] = []
    if inferred_latest_step is not None:
        anchor = int(policy.get("anchor_step", 3000))
        interval = max(int(policy.get("interval", 1000)), 1)
        start = anchor if inferred_latest_step >= anchor else inferred_latest_step
        inferred_schedule = list(range(start, inferred_latest_step + 1, interval))
        if inferred_schedule and inferred_schedule[-1] != inferred_latest_step:
            inferred_schedule.append(inferred_latest_step)
        if not inferred_schedule and inferred_latest_step > 0:
            inferred_schedule = [inferred_latest_step]

    schedule = build_checkpoint_schedule(
        available_steps=steps,
        include_first=bool(policy.get("include_first", True)),
        anchor_step=int(policy.get("anchor_step", 3000)),
        interval=int(policy.get("interval", 1000)),
    )
    if not steps:
        notes.append("no checkpoint-* directories/files found in repo")

    return CheckpointState(
        repo_id=repo_id,
        available_steps=steps,
        recommended_steps=schedule,
        latest_step=steps[-1] if steps else inferred_latest_step,
        inferred_latest_step=inferred_latest_step,
        inferred_schedule=inferred_schedule,
        notes=notes,
    )
