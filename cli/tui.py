from __future__ import annotations

import datetime as dt
import json
import os
import re
from copy import deepcopy
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from .checkpoints import list_hf_checkpoints
from .config import load_config
from .runner import execute_runs, run_analysis, visible_model_profiles
from .slurm import launch_interactive_srun, list_gpu_nodes, list_jobs, slurm_tool_status


class BenchmarkTuiApp(App[None]):
    CSS = """
    Screen {
      layout: vertical;
      background: rgb(16, 18, 22);
      color: rgb(236, 238, 242);
    }

    Header, Footer {
      background: rgb(24, 26, 31);
    }

    TabbedContent, TabPane {
      background: rgb(16, 18, 22);
    }

    #benchmark_main {
      height: 1fr;
      background: rgb(16, 18, 22);
    }

    #controls {
      width: 50;
      padding: 1;
      border: solid $primary;
      overflow-y: auto;
      background: rgb(20, 22, 27);
    }

    #status {
      width: 1fr;
      padding: 1;
      border: solid $secondary;
      background: rgb(20, 22, 27);
    }

    #slurm_status {
      width: 1fr;
      padding: 1;
      border: solid $secondary;
      background: rgb(20, 22, 27);
    }

    .spacer {
      height: 1;
    }

    #jobs_table {
      height: 1fr;
      background: rgb(20, 22, 27);
    }

    #log {
      height: 1fr;
      border: solid $accent;
      background: rgb(20, 22, 27);
    }

    DataTable, RichLog {
      background: rgb(20, 22, 27);
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_all", "Refresh"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.cfg = load_config()
        self._warned_no_squeue = False
        self._todo_manifest_details: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        profiles = visible_model_profiles(self.cfg)
        benchmark_options = [("repoeval", "repoeval"), ("swe-bench-lite", "swe-bench-lite"), ("coir", "coir")]
        model_options = [(v["name"], k) for k, v in profiles.items()]
        if not model_options:
            model_options = [("(no enabled model profiles)", "__none__")]
        coir_groups = list(self.cfg["benchmarks"]["coir"]["groups"].keys())
        group_options = [("(all)", "all")] + [(g, g) for g in coir_groups]
        dataset_options = self._common_dataset_options()
        common_commands = [
            ("Queue Base On Slurm", "queue_base_slurm"),
            ("Queue Finetuned On Slurm", "queue_finetuned_slurm"),
            ("Queue Base Locally", "queue_base_local"),
        ]

        yield Header(show_clock=True)
        with TabbedContent(initial="benchmark_tab"):
            with TabPane("Benchmark", id="benchmark_tab"):
                with Horizontal(id="benchmark_main"):
                    with Vertical(id="controls"):
                        yield Label("Benchmark")
                        yield Select(options=benchmark_options, value="repoeval", id="benchmark")
                        yield Label("COIR Group")
                        yield Select(options=group_options, value="all", id="dataset_group")
                        yield Label("Model Profile")
                        default_profile = model_options[0][1]
                        yield Select(options=model_options, value=default_profile, id="model_profile")
                        yield Label("Checkpoint Step (optional)")
                        yield Input(placeholder="e.g. 3000", id="checkpoint_step")
                        yield Label("Task Name (optional)")
                        yield Input(placeholder="e.g. coir_t2c_base_rtxa6000", id="task_name")
                        yield Label("Slurm GPUs")
                        yield Input(value=str(self.cfg["slurm"].get("gpus", 1)), id="slurm_gpus")
                        yield Label("Slurm Partition")
                        yield Input(value=str(self.cfg["slurm"].get("partition", "")), id="slurm_partition")
                        yield Label("Slurm Constraint")
                        yield Input(value=str(self.cfg["slurm"].get("constraint", "")), id="slurm_constraint")
                        yield Label("Slurm Nodelist")
                        yield Input(value=str(self.cfg["slurm"].get("nodelist", "")), id="slurm_nodelist")
                        yield Label("Smoke")
                        yield Switch(value=False, id="smoke")
                        yield Label("Force")
                        yield Switch(value=False, id="force")
                        yield Label("Execution")
                        yield Button("Execute Local", id="queue_local", variant="primary")
                        yield Button("Execute Slurm", id="queue_slurm", variant="success")
                        yield Button("Interactive Slurm Smoke", id="queue_interactive")
                        yield Static(classes="spacer")
                        yield Button("Refresh Checkpoints", id="refresh_checkpoints")
                        yield Static(classes="spacer")
                        yield Label("Analyze Dataset")
                        yield Input(placeholder="repoeval", id="analysis_dataset")
                        yield Label("Analyze Runs (space-separated NAME=PATH)")
                        yield Input(placeholder="base=/.../raw.jsonl ckpt=/.../raw.jsonl", id="analysis_runs")
                        yield Button("Run Analysis", id="run_analysis", variant="warning")

                    with Vertical(id="status"):
                        yield Static("Checkpoints: loading...", id="checkpoint_summary")
                        yield Static("Preflight: pending...", id="preflight_summary")
                        yield RichLog(id="log", highlight=True, markup=False)

            with TabPane("Slurm", id="slurm_tab"):
                with Vertical(id="slurm_status"):
                    yield Label("Queued sbatch Jobs")
                    yield Button("Refresh Queue", id="refresh_slurm")
                    yield DataTable(id="jobs_table")
                    yield Static("Running tails: pending refresh...", id="jobs_progress")
            with TabPane("GPU Availability", id="gpu_tab"):
                with Vertical(id="slurm_status"):
                    yield Label("Fully Available GPUs (allocated=0)")
                    yield Button("Refresh GPU Availability", id="refresh_gpu_availability")
                    yield DataTable(id="free_gpu_table")
                    yield Label("Partially Available GPUs (allocated>0, available>0)")
                    yield DataTable(id="partial_gpu_table")
            with TabPane("Todo", id="todo_tab"):
                with Vertical(id="slurm_status"):
                    yield Label("Benchmark Todo")
                    yield Button("Refresh Todo", id="refresh_todo")
                    yield Static("Todo summary: pending refresh...", id="todo_summary")
                    yield DataTable(id="todo_table")
                    yield Static("Manifest details: select a completed row to view run manifest metadata.", id="todo_manifest")
            with TabPane("Execute Common Commands", id="common_tab"):
                with Horizontal(id="benchmark_main"):
                    with Vertical(id="controls"):
                        yield Label("Common Command")
                        yield Select(options=common_commands, value="queue_base_slurm", id="common_command")
                        yield Label("Dataset")
                        yield Select(options=dataset_options, value=dataset_options[0][1], id="common_dataset")
                        yield Label("Smoke")
                        yield Switch(value=False, id="common_smoke")
                        yield Label("Force")
                        yield Switch(value=True, id="common_force")
                        yield Label("Task Name (optional)")
                        yield Input(placeholder="e.g. quick_repoeval_base", id="common_task_name")
                        yield Button("Execute Common Command", id="execute_common", variant="success")
                    with Vertical(id="status"):
                        yield Static(
                            "Commands:\n"
                            "- Queue Base On Slurm\n"
                            "- Queue Finetuned On Slurm\n"
                            "- Queue Base Locally\n"
                            "Pick a dataset and run.",
                            id="common_help",
                        )
                        yield RichLog(id="common_log", highlight=True, markup=False)

        yield Footer()

    def on_mount(self) -> None:
        jobs_table = self.query_one("#jobs_table", DataTable)
        jobs_table.add_columns("job_id", "benchmark", "name", "state", "runtime", "nodes", "reason", "progress")
        todo_table = self.query_one("#todo_table", DataTable)
        todo_table.add_columns("benchmark", "dataset", "model", "status", "detail")
        free_gpu_table = self.query_one("#free_gpu_table", DataTable)
        free_gpu_table.add_columns("node", "partition", "gpu_model", "gpus", "state")
        partial_gpu_table = self.query_one("#partial_gpu_table", DataTable)
        partial_gpu_table.add_columns("node", "partition", "gpu_model", "available/total", "state")

        self.refresh_all_async()
        poll_sec = int(self.cfg["slurm"].get("poll_interval_sec", 10))
        self.set_interval(poll_sec, self.refresh_slurm_async)

    def _log(self, message: str) -> None:
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        self.query_one("#log", RichLog).write(f"[{stamp}] {message}")

    def _common_log(self, message: str) -> None:
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        self.query_one("#common_log", RichLog).write(f"[{stamp}] {message}")

    def _common_dataset_options(self) -> list[tuple[str, str]]:
        opts: list[tuple[str, str]] = [("repoeval", "repoeval"), ("swe-bench-lite", "swe-bench-lite")]
        groups = self.cfg["benchmarks"]["coir"]["groups"]
        seen: set[str] = set()
        for group_name, datasets in groups.items():
            for dataset in datasets:
                if dataset in seen:
                    continue
                seen.add(dataset)
                opts.append((f"coir/{dataset} ({group_name})", f"coir::{dataset}"))
        return opts

    def _common_target(self) -> tuple[str, str | None, str]:
        raw = str(self.query_one("#common_dataset", Select).value)
        if raw == "repoeval":
            return "repoeval", None, "repoeval"
        if raw == "swe-bench-lite":
            return "swe-bench-lite", None, "swe-bench-lite"
        if raw.startswith("coir::"):
            dataset = raw.split("::", 1)[1]
            return "coir", "__single__", dataset
        raise ValueError(f"Unknown dataset selection '{raw}'.")

    def _get_checkpoint_step(self) -> int | None:
        raw = self.query_one("#checkpoint_step", Input).value.strip()
        if not raw:
            return None
        return int(raw)

    def _get_dataset_group(self) -> str | None:
        value = self.query_one("#dataset_group", Select).value
        if not value or value == "all":
            return None
        return str(value)

    def _selected_config(self) -> dict[str, object]:
        benchmark = str(self.query_one("#benchmark", Select).value)
        model_profile = str(self.query_one("#model_profile", Select).value)
        if model_profile == "__none__":
            raise ValueError("No enabled model profile in config. Enable one in `benchmark config show` output.")
        checkpoint_step = self._get_checkpoint_step()
        smoke = bool(self.query_one("#smoke", Switch).value)
        force = bool(self.query_one("#force", Switch).value)
        slurm_gpus = int((self.query_one("#slurm_gpus", Input).value or "1").strip())
        slurm_partition = self.query_one("#slurm_partition", Input).value.strip() or None
        slurm_constraint = self.query_one("#slurm_constraint", Input).value.strip() or None
        slurm_nodelist = self.query_one("#slurm_nodelist", Input).value.strip() or None
        task_name = self.query_one("#task_name", Input).value.strip() or None
        dataset_group = self._get_dataset_group()
        return {
            "benchmark": benchmark,
            "model_profile": model_profile,
            "checkpoint_step": checkpoint_step,
            "dataset_group": dataset_group,
            "smoke": smoke,
            "force": force,
            "slurm_gpus": slurm_gpus,
            "slurm_partition": slurm_partition,
            "slurm_constraint": slurm_constraint,
            "slurm_nodelist": slurm_nodelist,
            "task_name": task_name,
        }

    def _dataset_root(self) -> Path:
        return Path(str(self.cfg["paths"]["dataset_root"])).expanduser()

    def _refresh_config_from_env(self) -> None:
        # Keep runtime config in sync with .env/.env.local changes.
        self.cfg = load_config()

    def _auto_repair_dataset_root(self) -> str | None:
        root = self._dataset_root()
        if root.exists():
            return None
        # If dataset root is explicitly provided via env, fail fast and keep source-of-truth in .env.
        if os.environ.get("BENCHMARK_DATASET_ROOT", "").strip():
            return None
        # Common mismatch seen on scratch: .../benchmark_cli/benchmark_cli/datasets
        # while prepared datasets are at .../benchmark_cli/datasets.
        alt = root.parent.parent / "datasets"
        if alt.exists() and alt.is_dir():
            root.parent.mkdir(parents=True, exist_ok=True)
            root.symlink_to(alt, target_is_directory=True)
            return f"Auto-repaired dataset_root -> {alt}"
        return None

    def _preflight_for_params(self, params: dict[str, object], use_slurm: bool) -> tuple[list[str], list[str]]:
        warnings: list[str] = []
        errors: list[str] = []

        benchmark = str(params["benchmark"])
        dataset_root = self._dataset_root()
        if not dataset_root.exists():
            env_root = os.environ.get("BENCHMARK_DATASET_ROOT", "").strip()
            if env_root:
                errors.append(f"dataset_root not found from BENCHMARK_DATASET_ROOT: {dataset_root}")
            else:
                errors.append(
                    f"dataset_root not found: {dataset_root}. Set BENCHMARK_DATASET_ROOT in .env/.env.local."
                )
            return warnings, errors

        if benchmark == "repoeval":
            repoeval_meta = Path(str(self.cfg["paths"]["repoeval_dataset_path"])).expanduser()
            if not repoeval_meta.exists():
                env_meta = os.environ.get("BENCHMARK_REPOEVAL_DATASET_PATH", "").strip()
                if env_meta:
                    errors.append(f"RepoEval metadata missing from BENCHMARK_REPOEVAL_DATASET_PATH: {repoeval_meta}")
                else:
                    errors.append(
                        f"RepoEval metadata missing: {repoeval_meta}. Set BENCHMARK_REPOEVAL_DATASET_PATH in .env/.env.local."
                    )

        if benchmark == "swe-bench-lite":
            swe_dirs = list(dataset_root.glob("swe-bench-lite_*"))
            if not swe_dirs:
                errors.append(f"No swe-bench-lite_* dataset dirs under {dataset_root}")

        if benchmark == "coir":
            coir_sets = set(sum(self.cfg["benchmarks"]["coir"]["groups"].values(), []))
            missing = [d for d in sorted(coir_sets) if not list(dataset_root.glob(f"{d}_*"))]
            if missing:
                errors.append("Missing COIR datasets: " + ", ".join(missing[:6]) + ("..." if len(missing) > 6 else ""))

        if use_slurm:
            tools = slurm_tool_status()
            if not tools["sbatch"]:
                errors.append("`sbatch` is not available in this environment.")
                return warnings, errors

            slurm_partition = params["slurm_partition"]
            slurm_nodelist = params["slurm_nodelist"]

            if slurm_partition in (None, ""):
                warnings.append("No partition set; cluster default partition will be used.")
            if slurm_partition == "scavenger":
                warnings.append("Partition=scavenger may reject submissions depending on account/QoS.")

            if slurm_nodelist and "," in str(slurm_nodelist):
                warnings.append(
                    "Comma-separated nodelist detected; selecting a single node at submit time to avoid multi-node spec issues."
                )

        return warnings, errors

    def _format_preflight(self, warnings: list[str], errors: list[str]) -> str:
        if errors:
            return "Preflight errors:\n- " + "\n- ".join(errors)
        if warnings:
            return "Preflight warnings:\n- " + "\n- ".join(warnings)
        return "Preflight: OK"

    def _normalize_slurm_nodelist(self, slurm_nodelist: str | None, partition: str | None) -> str | None:
        if not slurm_nodelist or "," not in slurm_nodelist:
            return slurm_nodelist
        allowlist = [item.strip() for item in slurm_nodelist.split(",") if item.strip()]
        if not allowlist:
            return None
        by_name = {row.node: row for row in list_gpu_nodes()}
        candidates = []
        for node in allowlist:
            row = by_name.get(node)
            if row is None:
                continue
            if partition and partition not in row.partition.split(","):
                continue
            candidates.append(row)
        candidates.sort(key=lambda r: (r.available_gpus, 0 if "IDLE" in r.state else 1), reverse=True)
        if candidates:
            return candidates[0].node
        return allowlist[0]

    @work(thread=True)
    def refresh_all_async(self) -> None:
        self.refresh_checkpoints_async()
        self.refresh_slurm_async()
        self.refresh_preflight_async()
        self.refresh_todo_async()
        self.refresh_full_free_gpus_async()

    @work(thread=True)
    def refresh_checkpoints_async(self) -> None:
        try:
            state = list_hf_checkpoints(self.cfg)
            recommended = state.recommended_steps if state.available_steps else state.inferred_schedule
            recommended_preview = ", ".join(str(step) for step in recommended[:12]) if recommended else "none"
            text = (
                f"HF: {state.repo_id}\n"
                f"latest: {state.latest_step}\n"
                f"available: {len(state.available_steps)}\n"
                f"recommended: {recommended_preview}"
            )
            if state.notes:
                text += "\nnotes: " + "; ".join(state.notes[:2])
            text = text.replace("\\n", "\n")
            self.call_from_thread(self.query_one("#checkpoint_summary", Static).update, text)
            self.call_from_thread(self._log, "Checkpoint list refreshed.")
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self.query_one("#checkpoint_summary", Static).update, f"Checkpoint refresh failed: {exc}")
            self.call_from_thread(self._log, f"Checkpoint refresh failed: {exc}")

    @work(thread=True)
    def refresh_slurm_async(self) -> None:
        self.refresh_jobs_async()
        self.refresh_todo_async()
        self.refresh_full_free_gpus_async()

    @work(thread=True)
    def refresh_preflight_async(self) -> None:
        try:
            self._refresh_config_from_env()
            params = self._selected_config()
            warnings, errors = self._preflight_for_params(params, use_slurm=True)
            text = self._format_preflight(warnings, errors)
            self.call_from_thread(self.query_one("#preflight_summary", Static).update, text)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self.query_one("#preflight_summary", Static).update, f"Preflight check failed: {exc}")

    @work(thread=True)
    def refresh_jobs_async(self) -> None:
        table = self.query_one("#jobs_table", DataTable)
        progress_panel = self.query_one("#jobs_progress", Static)
        user = os.environ.get("USER", "")
        jobs = list_jobs(user=user)
        jobs = [job for job in jobs if job.state.upper() not in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}]
        tools = slurm_tool_status()
        progress_map: dict[str, str] = {}
        tails: list[str] = []

        for job in jobs:
            if job.state.upper() != "RUNNING":
                continue
            progress, tail = self._job_log_progress(job.job_id)
            if progress:
                progress_map[job.job_id] = progress
            else:
                progress_map[job.job_id] = "-"
            if tail:
                tails.append(f"{job.job_id} {job.name}: {tail}")
            else:
                tails.append(f"{job.job_id} {job.name}: (no log tail yet)")

        def update() -> None:
            table.clear()
            for job in jobs:
                table.add_row(
                    job.job_id,
                    self._benchmark_from_job_name(job.name),
                    job.name,
                    job.state,
                    job.runtime,
                    job.nodes,
                    job.reason,
                    progress_map.get(job.job_id, "-"),
                )
            progress_text = "Running tails:\n"
            if tails:
                progress_text += "\n".join(f"- {line}" for line in tails[:12])
            else:
                progress_text += "- no running jobs"
            progress_panel.update(progress_text)
            pending_reqnode = sum(
                1
                for job in jobs
                if job.state.upper() == "PENDING" and "ReqNodeNotAvail" in (job.reason or "")
            )
            if pending_reqnode:
                self._log(
                    f"{pending_reqnode} job(s) pending due to node pinning (ReqNodeNotAvail). Consider relaxing nodelist/constraint."
                )
            if jobs:
                self._warned_no_squeue = False
            if not jobs and not tools["squeue"] and not self._warned_no_squeue:
                self._log("No Slurm queue info: `squeue` is not available in this environment.")
                self._warned_no_squeue = True

        self.call_from_thread(update)

    @work(thread=True)
    def refresh_full_free_gpus_async(self) -> None:
        table = self.query_one("#free_gpu_table", DataTable)
        partial_table = self.query_one("#partial_gpu_table", DataTable)
        nodes = list_gpu_nodes()
        free_nodes = [
            node
            for node in nodes
            if node.total_gpus > 0 and node.allocated_gpus == 0 and node.available_gpus == node.total_gpus
        ]
        partial_nodes = [
            node
            for node in nodes
            if node.total_gpus > 0 and node.allocated_gpus > 0 and node.available_gpus > 0
        ]
        free_nodes.sort(
            key=lambda n: (
                n.available_gpus,
                n.gpu_ram_gb if n.gpu_ram_gb is not None else -1,
                n.node,
            ),
            reverse=True,
        )
        partial_nodes.sort(
            key=lambda n: (
                n.available_gpus,
                n.gpu_ram_gb if n.gpu_ram_gb is not None else -1,
                n.node,
            ),
            reverse=True,
        )

        def update() -> None:
            table.clear()
            for node in free_nodes:
                table.add_row(
                    node.node,
                    node.partition,
                    node.gpu_model or "-",
                    str(node.available_gpus),
                    node.state,
                )
            partial_table.clear()
            for node in partial_nodes:
                partial_table.add_row(
                    node.node,
                    node.partition,
                    node.gpu_model or "-",
                    f"{node.available_gpus}/{node.total_gpus}",
                    node.state,
                )
            self._log(f"GPU availability refreshed: free={len(free_nodes)}, partial={len(partial_nodes)}")

        self.call_from_thread(update)

    def _dataset_from_job_name(self, name: str) -> str:
        marker = "__bench_"
        core = name
        if marker in name:
            core = name.split(marker, 1)[1]
        elif name.startswith("bench_"):
            core = name[len("bench_"):]
        else:
            return "-"
        cut = core.find("_qwen3-embed")
        if cut == -1:
            return core
        return core[:cut]

    def _target_from_job_name(self, name: str) -> tuple[str, str] | None:
        marker = "__bench_"
        core = name
        if marker in name:
            core = name.split(marker, 1)[1]
        elif name.startswith("bench_"):
            core = name[len("bench_"):]
        else:
            return None

        ft_suffix = "_qwen3-embed-0.6b-finetuned-latest"
        base_suffix = "_qwen3-embed-0.6b"
        if core.endswith(ft_suffix):
            return core[: -len(ft_suffix)], "finetuned-latest"
        if core.endswith(base_suffix):
            return core[: -len(base_suffix)], "base"
        return None

    def _benchmark_from_job_name(self, name: str) -> str:
        dataset = self._dataset_from_job_name(name)
        if dataset == "repoeval":
            return "repoeval"
        if dataset == "swe-bench-lite":
            return "swe-bench-lite"
        coir_sets = set(sum(self.cfg["benchmarks"]["coir"]["groups"].values(), []))
        if dataset in coir_sets:
            return f"coir/{dataset}"
        return dataset

    def _todo_targets(self) -> list[tuple[str, str, str]]:
        datasets: list[tuple[str, str]] = [("repoeval", "repoeval"), ("swe-bench-lite", "swe-bench-lite")]
        for ds in sum(self.cfg["benchmarks"]["coir"]["groups"].values(), []):
            datasets.append((f"coir/{ds}", ds))
        models = [("base", "qwen3-embed-0.6b"), ("finetuned-latest", "qwen3-embed-0.6b-finetuned-latest")]
        targets: list[tuple[str, str, str]] = []
        for bench, ds in datasets:
            for model_tag, _ in models:
                targets.append((bench, ds, model_tag))
        return targets

    def _has_completed_summary(self, dataset: str, model_tag: str) -> bool:
        run_name = "qwen3-embed-0.6b" if model_tag == "base" else "qwen3-embed-0.6b-finetuned-latest"
        for root in self._candidate_results_roots():
            if not root.exists():
                continue
            pattern = f"**/{dataset}/{run_name}/summary.json"
            for summary_path in root.glob(pattern):
                try:
                    if self._summary_is_complete(summary_path):
                        return True
                except Exception:  # noqa: BLE001
                    continue
        return False

    def _find_latest_completed_summary(self, dataset: str, model_tag: str) -> Path | None:
        run_name = "qwen3-embed-0.6b" if model_tag == "base" else "qwen3-embed-0.6b-finetuned-latest"
        candidates: list[Path] = []
        for root in self._candidate_results_roots():
            if not root.exists():
                continue
            pattern = f"**/{dataset}/{run_name}/summary.json"
            for summary_path in root.glob(pattern):
                try:
                    if self._summary_is_complete(summary_path):
                        candidates.append(summary_path)
                except Exception:  # noqa: BLE001
                    continue
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _manifest_details_for_target(self, dataset: str, model_tag: str) -> str:
        summary_path = self._find_latest_completed_summary(dataset, model_tag)
        if summary_path is None:
            return "Manifest details: no valid completed summary found for this target."

        run_root = summary_path.parents[2]
        manifest_path = run_root / "run_manifest.json"
        if not manifest_path.exists():
            return f"Manifest details: completed summary found but run_manifest.json missing at {run_root}."

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as exc:  # noqa: BLE001
            return f"Manifest details: failed to parse {manifest_path}: {exc}"

        result_entry = None
        for item in payload.get("results", []):
            if item.get("dataset") == dataset and item.get("run_name") == ("qwen3-embed-0.6b" if model_tag == "base" else "qwen3-embed-0.6b-finetuned-latest"):
                result_entry = item
                break

        details = {
            "run_id": payload.get("run_id"),
            "benchmark": payload.get("benchmark"),
            "dataset_group": payload.get("dataset_group"),
            "model": payload.get("model"),
            "smoke": payload.get("smoke"),
            "slurm": payload.get("slurm"),
            "dataset_result": result_entry,
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
        }
        return "Manifest details:\n" + json.dumps(details, indent=2)

    def _summary_is_complete(self, summary_path: Path) -> bool:
        if not summary_path.exists() or summary_path.stat().st_size <= 0:
            return False
        payload = json.loads(summary_path.read_text(encoding="utf-8", errors="ignore"))
        if not isinstance(payload, dict):
            return False
        # Require at least one expected metric/result key to consider this complete.
        return (
            "num_queries" in payload
            or "mrr" in payload
            or "metrics" in payload
            or "ndcg" in payload
        )

    def _cleanup_empty_summaries(self) -> int:
        removed = 0
        for root in self._candidate_results_roots():
            if not root.exists():
                continue
            for summary_path in root.glob("**/summary.json"):
                try:
                    if not summary_path.is_file():
                        continue
                    if summary_path.stat().st_size == 0:
                        summary_path.unlink(missing_ok=True)
                        removed += 1
                except Exception:  # noqa: BLE001
                    continue
        return removed

    @work(thread=True)
    def refresh_todo_async(self) -> None:
        table = self.query_one("#todo_table", DataTable)
        summary = self.query_one("#todo_summary", Static)
        manifest_panel = self.query_one("#todo_manifest", Static)
        removed = self._cleanup_empty_summaries()
        user = os.environ.get("USER", "")
        jobs = list_jobs(user=user)

        active: dict[tuple[str, str], str] = {}
        for job in jobs:
            parsed = self._target_from_job_name(job.name)
            if parsed is None:
                continue
            key = (parsed[0], parsed[1])
            state = job.state.upper()
            if key not in active:
                active[key] = state
            elif active[key] != "RUNNING" and state == "RUNNING":
                active[key] = "RUNNING"

        targets = self._todo_targets()
        rows: list[tuple[str, str, str, str, str]] = []
        manifest_by_key: dict[str, str] = {}
        left = running = completed = 0
        for bench, dataset, model_tag in targets:
            key = (dataset, model_tag)
            row_key = f"{bench}|{dataset}|{model_tag}"
            if key in active:
                st = "running" if active[key] == "RUNNING" else "queued"
                detail = active[key]
                running += 1
            elif self._has_completed_summary(dataset, model_tag):
                st = "completed"
                detail = "summary.json present"
                manifest_by_key[row_key] = self._manifest_details_for_target(dataset, model_tag)
                completed += 1
            else:
                st = "left"
                detail = "-"
                left += 1
            rows.append((bench, dataset, model_tag, st, detail, row_key))

        def update() -> None:
            table.clear()
            self._todo_manifest_details = manifest_by_key
            for row in rows:
                bench, dataset, model_tag, st, detail, row_key = row
                table.add_row(bench, dataset, model_tag, st, detail, key=row_key)
            cleanup_note = f", cleaned_empty_summary={removed}" if removed else ""
            summary.update(
                f"Todo summary: left={left}, running/queued={running}, completed={completed}{cleanup_note}"
            )
            manifest_panel.update("Manifest details: select a completed row to view run manifest metadata.")

        self.call_from_thread(update)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "todo_table":
            return
        panel = self.query_one("#todo_manifest", Static)
        row_key = str(event.row_key.value)
        details = self._todo_manifest_details.get(row_key)
        if details:
            panel.update(details)
        else:
            panel.update("Manifest details: selected row is not completed or no manifest is available.")

    def _candidate_results_roots(self) -> list[Path]:
        roots: list[Path] = []
        configured = Path(str(self.cfg["paths"]["results_root"])).expanduser()
        roots.append(configured)

        user = os.environ.get("USER", "").strip()
        if user:
            scratch = Path(f"/fs/cml-scratch/{user}/benchmark_cli/benchmark_cli/results")
            roots.append(scratch)

        dedup: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root.resolve()) if root.exists() else str(root)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(root)
        return dedup

    def _find_job_log(self, job_id: str) -> Path | None:
        for root in self._candidate_results_roots():
            if not root.exists():
                continue
            err_matches = list(root.glob(f"**/*-{job_id}.err"))
            if err_matches:
                err_matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return err_matches[0]
            out_matches = list(root.glob(f"**/*-{job_id}.out"))
            if out_matches:
                out_matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return out_matches[0]
        return None

    def _tail_text(self, path: Path, max_bytes: int = 256_000) -> str:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, os.SEEK_END)
            return f.read().decode("utf-8", errors="ignore")

    def _job_log_progress(self, job_id: str) -> tuple[str, str]:
        log_path = self._find_job_log(job_id)
        if log_path is None:
            return "-", ""
        try:
            text = self._tail_text(log_path)
        except Exception:  # noqa: BLE001
            return "-", ""

        lines = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]
        if not lines:
            return "-", ""

        progress_line = "-"
        for line in reversed(lines):
            if "Batches:" in line or ("%|" in line) or ("it/s" in line):
                progress_line = line
                break
        if len(progress_line) > 140:
            progress_line = progress_line[-140:]

        tail_line = lines[-1]
        if len(tail_line) > 180:
            tail_line = tail_line[-180:]
        return progress_line, tail_line

    @work(thread=True)
    def queue_run_async(self, use_slurm: bool) -> None:
        try:
            self._refresh_config_from_env()
            repair_note = self._auto_repair_dataset_root()
            if repair_note:
                self.call_from_thread(self._log, repair_note)

            params = self._selected_config()
            warnings, errors = self._preflight_for_params(params, use_slurm=use_slurm)
            self.call_from_thread(self.query_one("#preflight_summary", Static).update, self._format_preflight(warnings, errors))
            for warning in warnings:
                self.call_from_thread(self._log, f"Preflight warning: {warning}")
            if errors:
                for error in errors:
                    self.call_from_thread(self._log, f"Preflight error: {error}")
                return

            slurm_nodelist = params["slurm_nodelist"]
            if use_slurm:
                normalized_nodelist = self._normalize_slurm_nodelist(
                    slurm_nodelist=str(slurm_nodelist) if slurm_nodelist else None,
                    partition=str(params["slurm_partition"]) if params["slurm_partition"] else None,
                )
                if normalized_nodelist != slurm_nodelist:
                    self.call_from_thread(
                        self._log,
                        f"Using single-node nodelist '{normalized_nodelist}' from allowlist '{slurm_nodelist}'.",
                    )
                slurm_nodelist = normalized_nodelist

            run_root, results = execute_runs(
                config=self.cfg,
                benchmark=str(params["benchmark"]),
                model_profile_key=str(params["model_profile"]),
                checkpoint_step=params["checkpoint_step"],
                dataset_group=params["dataset_group"],
                smoke=bool(params["smoke"]),
                force=bool(params["force"]),
                run_id=str(params["task_name"]) if params.get("task_name") else None,
                use_slurm=use_slurm,
                slurm_gpus=int(params["slurm_gpus"]),
                slurm_partition=params["slurm_partition"],
                slurm_constraint=params["slurm_constraint"],
                slurm_nodelist=slurm_nodelist,
            )
            self.call_from_thread(
                self._log,
                "Submitted benchmark="
                f"{params['benchmark']} dataset_group={params['dataset_group'] or '-'} "
                f"model={params['model_profile']} task={params.get('task_name') or 'auto'}",
            )
            self.call_from_thread(self._log, f"Run created at {run_root}")
            for item in results:
                job = f" job_id={item.job_id}" if item.job_id else ""
                self.call_from_thread(
                    self._log,
                    f"{item.dataset}/{item.run_name}: {item.status}{job} log={item.log_file}",
                )
            self.refresh_jobs_async()
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._log, f"Run failed: {exc}")

    @work(thread=True)
    def interactive_smoke_async(self) -> None:
        try:
            from .runner import preview_commands

            params = self._selected_config()
            commands = preview_commands(
                config=self.cfg,
                benchmark=str(params["benchmark"]),
                model_profile_key=str(params["model_profile"]),
                checkpoint_step=params["checkpoint_step"],
                dataset_group=params["dataset_group"],
                smoke=True,
            )
            if len(commands) != 1:
                self.call_from_thread(self._log, "Interactive mode requires exactly one dataset selection.")
                return
            dataset, command = commands[0]
            self.call_from_thread(self._log, f"Launching interactive srun smoke for {dataset}")
            code = launch_interactive_srun(
                command=command,
                slurm_cfg=self.cfg["slurm"],
                gpus=int(params["slurm_gpus"]),
                partition=params["slurm_partition"],
                constraint=params["slurm_constraint"],
                nodelist=params["slurm_nodelist"],
            )
            self.call_from_thread(self._log, f"Interactive run exit code: {code}")
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._log, f"Interactive smoke failed: {exc}")

    @work(thread=True)
    def run_analysis_async(self) -> None:
        try:
            dataset = self.query_one("#analysis_dataset", Input).value.strip()
            runs_raw = self.query_one("#analysis_runs", Input).value.strip()
            if not dataset or not runs_raw:
                self.call_from_thread(self._log, "Analysis requires dataset and run specs.")
                return
            run_specs = runs_raw.split()
            output_dir = Path(self.cfg["paths"]["results_root"]) / "analysis" / dataset / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir.mkdir(parents=True, exist_ok=True)
            code, output = run_analysis(
                config=self.cfg,
                dataset=dataset,
                run_specs=run_specs,
                output_dir=output_dir,
                k=10,
                num_cases=8,
            )
            (output_dir / "analysis.log").write_text(output, encoding="utf-8")
            self.call_from_thread(self._log, f"Analysis return code={code} output_dir={output_dir}")
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._log, f"Analysis failed: {exc}")

    @work(thread=True)
    def execute_common_async(self) -> None:
        try:
            self._refresh_config_from_env()
            command = str(self.query_one("#common_command", Select).value)
            benchmark, dataset_group, dataset_name = self._common_target()
            smoke = bool(self.query_one("#common_smoke", Switch).value)
            force = bool(self.query_one("#common_force", Switch).value)
            common_task_name = self.query_one("#common_task_name", Input).value.strip() or None

            cfg = self.cfg
            if benchmark == "coir" and dataset_group == "__single__":
                cfg = deepcopy(self.cfg)
                cfg["benchmarks"]["coir"]["groups"]["__single__"] = [dataset_name]

            model_profile = "qwen3_embed_0_6b"
            use_slurm = False
            checkpoint_step = None
            if command == "queue_base_slurm":
                model_profile = "qwen3_embed_0_6b"
                use_slurm = True
            elif command == "queue_finetuned_slurm":
                if "qwen3_embed_0_6b_finetuned_latest" not in visible_model_profiles(cfg):
                    raise ValueError("Model profile 'qwen3_embed_0_6b_finetuned_latest' is not enabled.")
                model_profile = "qwen3_embed_0_6b_finetuned_latest"
                use_slurm = True
            elif command == "queue_base_local":
                model_profile = "qwen3_embed_0_6b"
                use_slurm = False
            else:
                raise ValueError(f"Unknown common command '{command}'.")

            self.call_from_thread(self._common_log, f"Executing {command} on dataset={dataset_name}")
            run_root, results = execute_runs(
                config=cfg,
                benchmark=benchmark,
                model_profile_key=model_profile,
                checkpoint_step=checkpoint_step,
                dataset_group=dataset_group,
                smoke=smoke,
                force=force,
                run_id=common_task_name,
                use_slurm=use_slurm,
                slurm_gpus=int((self.query_one("#slurm_gpus", Input).value or "1").strip()),
                slurm_partition=self.query_one("#slurm_partition", Input).value.strip() or None,
                slurm_constraint=self.query_one("#slurm_constraint", Input).value.strip() or None,
                slurm_nodelist=self.query_one("#slurm_nodelist", Input).value.strip() or None,
            )
            self.call_from_thread(
                self._common_log,
                f"Submitted benchmark={benchmark} dataset={dataset_name} model={model_profile} task={common_task_name or 'auto'}",
            )
            self.call_from_thread(self._common_log, f"Run created at {run_root}")
            for item in results:
                job = f" job_id={item.job_id}" if item.job_id else ""
                self.call_from_thread(
                    self._common_log,
                    f"{item.dataset}/{item.run_name}: {item.status}{job} log={item.log_file}",
                )
            self.refresh_jobs_async()
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._common_log, f"Common command failed: {exc}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "refresh_checkpoints":
            self.refresh_checkpoints_async()
        elif button_id == "refresh_slurm":
            self.refresh_slurm_async()
        elif button_id == "refresh_todo":
            self.refresh_todo_async()
        elif button_id == "refresh_gpu_availability":
            self.refresh_full_free_gpus_async()
        elif button_id == "queue_local":
            self.queue_run_async(use_slurm=False)
        elif button_id == "queue_slurm":
            self.queue_run_async(use_slurm=True)
        elif button_id == "queue_interactive":
            self.interactive_smoke_async()
        elif button_id == "run_analysis":
            self.run_analysis_async()
        elif button_id == "execute_common":
            self.execute_common_async()

    def action_refresh_all(self) -> None:
        self.refresh_all_async()
