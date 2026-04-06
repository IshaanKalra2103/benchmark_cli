from __future__ import annotations

import datetime as dt
import os
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
from .slurm import GpuNode, launch_interactive_srun, list_gpu_nodes, list_jobs, slurm_tool_status


class BenchmarkTuiApp(App[None]):
    CSS = """
    Screen {
      layout: vertical;
    }

    #benchmark_main, #slurm_main {
      height: 1fr;
    }

    #controls {
      width: 50;
      padding: 1;
      border: solid $primary;
      overflow-y: auto;
    }

    #status {
      width: 1fr;
      padding: 1;
      border: solid $secondary;
    }

    #slurm_filters {
      width: 42;
      padding: 1;
      border: solid $primary;
      overflow-y: auto;
    }

    #slurm_status {
      width: 1fr;
      padding: 1;
      border: solid $secondary;
    }

    .spacer {
      height: 1;
    }

    #gpu_table {
      height: 14;
    }

    #jobs_table {
      height: 10;
    }

    #log {
      height: 1fr;
      border: solid $accent;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_all", "Refresh"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.cfg = load_config()
        self._warned_no_gpu_inventory = False
        self._warned_no_squeue = False
        self._gpu_nodes_by_name: dict[str, GpuNode] = {}

    def compose(self) -> ComposeResult:
        profiles = visible_model_profiles(self.cfg)
        benchmark_options = [("repoeval", "repoeval"), ("swe-bench-lite", "swe-bench-lite"), ("coir", "coir")]
        model_options = [(v["name"], k) for k, v in profiles.items()]
        if not model_options:
            model_options = [("(no enabled model profiles)", "__none__")]
        coir_groups = list(self.cfg["benchmarks"]["coir"]["groups"].keys())
        group_options = [("(all)", "all")] + [(g, g) for g in coir_groups]

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
                        yield RichLog(id="log", highlight=True, markup=False)

            with TabPane("Slurm", id="slurm_tab"):
                with Horizontal(id="slurm_main"):
                    with Vertical(id="slurm_filters"):
                        yield Label("GPU Name Filter (GRES)")
                        yield Input(placeholder="e.g. a100, h100, rtx", id="gpu_model_filter")
                        yield Label("GPU RAM >= (GB)")
                        yield Input(placeholder="e.g. 80", id="gpu_ram_min")
                        yield Label("State")
                        yield Select(
                            options=[
                                ("All", "all"),
                                ("IDLE", "IDLE"),
                                ("MIXED", "MIXED"),
                                ("ALLOCATED", "ALLOCATED"),
                                ("PLANNED", "PLANNED"),
                                ("DOWN", "DOWN"),
                                ("DRAIN", "DRAIN"),
                            ],
                            value="all",
                            id="gpu_state_filter",
                        )
                        yield Label("Available Only")
                        yield Switch(value=False, id="gpu_available_only")
                        yield Static(classes="spacer")
                        yield Label("Quick Filters")
                        yield Button("H100 Only", id="quick_h100")
                        yield Button("A100 Only", id="quick_a100")
                        yield Button("Idle + Available", id="quick_idle_available")
                        yield Static(classes="spacer")
                        yield Button("Apply Filters", id="apply_gpu_filters")
                        yield Button("Refresh GPUs/Jobs", id="refresh_slurm")

                    with Vertical(id="slurm_status"):
                        yield Static("Select a GPU node row to load Slurm run settings.", id="gpu_selection_info")
                        yield Label("GPU Availability")
                        yield DataTable(id="gpu_table")
                        yield Label("Slurm Queue")
                        yield DataTable(id="jobs_table")

        yield Footer()

    def on_mount(self) -> None:
        gpu_table = self.query_one("#gpu_table", DataTable)
        gpu_table.add_columns("node", "partition", "total", "allocated", "available", "state", "gres", "model", "ram_gb")

        jobs_table = self.query_one("#jobs_table", DataTable)
        jobs_table.add_columns("job_id", "name", "state", "runtime", "nodes", "reason")

        self.refresh_all_async()
        poll_sec = int(self.cfg["slurm"].get("poll_interval_sec", 10))
        self.set_interval(poll_sec, self.refresh_jobs_async)

    def _log(self, message: str) -> None:
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        self.query_one("#log", RichLog).write(f"[{stamp}] {message}")

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
        }

    def _update_selected_gpu_info(self, node: GpuNode | None) -> None:
        if node is None:
            self.query_one("#gpu_selection_info", Static).update("Select a GPU node row to load Slurm run settings.")
            return
        gpu_ram = f"{node.gpu_ram_gb} GB" if node.gpu_ram_gb is not None else "unknown"
        node_ram = f"{node.real_memory_mb // 1024} GB" if node.real_memory_mb is not None else "unknown"
        alloc_mem = f"{node.alloc_mem_mb // 1024} GB" if node.alloc_mem_mb is not None else "unknown"
        text = (
            f"Selected node: {node.node} | partition={node.partition} | state={node.state}\n"
            f"GPU model={node.gpu_model or 'unknown'} | GPU RAM={gpu_ram} | "
            f"GPUs {node.available_gpus}/{node.total_gpus} available | "
            f"Node RAM={node_ram} (allocated={alloc_mem})\n"
            "Selecting a row auto-fills Benchmark tab Slurm fields."
        )
        self.query_one("#gpu_selection_info", Static).update(text)

    def _apply_quick_filter(self, preset: str) -> None:
        model_input = self.query_one("#gpu_model_filter", Input)
        ram_input = self.query_one("#gpu_ram_min", Input)
        state_select = self.query_one("#gpu_state_filter", Select)
        available_switch = self.query_one("#gpu_available_only", Switch)

        if preset == "h100":
            model_input.value = "h100"
            ram_input.value = ""
            state_select.value = "all"
            available_switch.value = False
            self._log("Applied quick filter: H100 only.")
        elif preset == "a100":
            model_input.value = "a100"
            ram_input.value = ""
            state_select.value = "all"
            available_switch.value = False
            self._log("Applied quick filter: A100 only.")
        elif preset == "idle_available":
            model_input.value = ""
            ram_input.value = ""
            state_select.value = "IDLE"
            available_switch.value = True
            self._log("Applied quick filter: IDLE + available GPUs.")
        else:
            return

        self.refresh_gpus_async()

    @work(thread=True)
    def refresh_all_async(self) -> None:
        self.refresh_checkpoints_async()
        self.refresh_slurm_async()

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
        self.refresh_gpus_async()
        self.refresh_jobs_async()

    @work(thread=True)
    def refresh_gpus_async(self) -> None:
        table = self.query_one("#gpu_table", DataTable)
        nodes = list_gpu_nodes()
        tools = slurm_tool_status()

        def update() -> None:
            gres_filter = self.query_one("#gpu_model_filter", Input).value.strip().lower()
            ram_filter_raw = self.query_one("#gpu_ram_min", Input).value.strip()
            state_select = self.query_one("#gpu_state_filter", Select)
            current_state = state_select.value if isinstance(state_select.value, str) else "all"
            states = sorted({node.state.upper() for node in nodes if node.state})
            state_options = [("All", "all")] + [(state, state) for state in states]
            state_select.set_options(state_options)
            valid_values = {value for _, value in state_options}
            if current_state not in valid_values:
                current_state = "all"
            state_select.value = current_state
            state_filter = current_state
            available_only = bool(self.query_one("#gpu_available_only", Switch).value)

            min_ram_gb: int | None = None
            if ram_filter_raw:
                try:
                    min_ram_gb = int(ram_filter_raw)
                except ValueError:
                    self._log(f"Ignoring invalid GPU RAM filter '{ram_filter_raw}'. Use an integer GB value.")

            filtered = []
            for node in nodes:
                if gres_filter:
                    haystack = f"{node.gpu_model} {node.gres}".lower()
                    if gres_filter not in haystack:
                        continue
                if min_ram_gb is not None:
                    if node.gpu_ram_gb is None or node.gpu_ram_gb < min_ram_gb:
                        continue
                if state_filter != "all" and node.state.upper() != state_filter:
                    continue
                if available_only and node.available_gpus <= 0:
                    continue
                filtered.append(node)

            self._gpu_nodes_by_name = {node.node: node for node in nodes}
            table.clear()
            for node in filtered:
                table.add_row(
                    node.node,
                    node.partition,
                    str(node.total_gpus),
                    str(node.allocated_gpus),
                    str(node.available_gpus),
                    node.state,
                    node.gres,
                    node.gpu_model or "-",
                    str(node.gpu_ram_gb) if node.gpu_ram_gb is not None else "-",
                )
            selected_nodelist = self.query_one("#slurm_nodelist", Input).value.strip()
            self._update_selected_gpu_info(self._gpu_nodes_by_name.get(selected_nodelist))
            if nodes:
                self._log(f"GPU nodes refreshed: showing {len(filtered)}/{len(nodes)} nodes after filters.")
                self._warned_no_gpu_inventory = False
            elif not self._warned_no_gpu_inventory:
                self._log(
                    "No GPU inventory found. "
                    f"slurm_tools(scontrol={tools['scontrol']}, sinfo={tools['sinfo']})"
                )
                self._warned_no_gpu_inventory = True

        self.call_from_thread(update)

    @work(thread=True)
    def refresh_jobs_async(self) -> None:
        table = self.query_one("#jobs_table", DataTable)
        user = os.environ.get("USER", "")
        jobs = list_jobs(user=user)
        tools = slurm_tool_status()

        def update() -> None:
            table.clear()
            for job in jobs:
                table.add_row(job.job_id, job.name, job.state, job.runtime, job.nodes, job.reason)
            if jobs:
                self._warned_no_squeue = False
            if not jobs and not tools["squeue"] and not self._warned_no_squeue:
                self._log("No Slurm queue info: `squeue` is not available in this environment.")
                self._warned_no_squeue = True

        self.call_from_thread(update)

    @work(thread=True)
    def queue_run_async(self, use_slurm: bool) -> None:
        try:
            params = self._selected_config()
            run_root, results = execute_runs(
                config=self.cfg,
                benchmark=str(params["benchmark"]),
                model_profile_key=str(params["model_profile"]),
                checkpoint_step=params["checkpoint_step"],
                dataset_group=params["dataset_group"],
                smoke=bool(params["smoke"]),
                force=bool(params["force"]),
                run_id=None,
                use_slurm=use_slurm,
                slurm_gpus=int(params["slurm_gpus"]),
                slurm_partition=params["slurm_partition"],
                slurm_constraint=params["slurm_constraint"],
                slurm_nodelist=params["slurm_nodelist"],
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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "refresh_checkpoints":
            self.refresh_checkpoints_async()
        elif button_id == "refresh_slurm":
            self.refresh_slurm_async()
        elif button_id == "quick_h100":
            self._apply_quick_filter("h100")
        elif button_id == "quick_a100":
            self._apply_quick_filter("a100")
        elif button_id == "quick_idle_available":
            self._apply_quick_filter("idle_available")
        elif button_id == "apply_gpu_filters":
            self.refresh_gpus_async()
        elif button_id == "queue_local":
            self.queue_run_async(use_slurm=False)
        elif button_id == "queue_slurm":
            self.queue_run_async(use_slurm=True)
        elif button_id == "queue_interactive":
            self.interactive_smoke_async()
        elif button_id == "run_analysis":
            self.run_analysis_async()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "gpu_table":
            try:
                row = event.data_table.get_row(event.row_key)
                if not row:
                    return
                node_name = str(row[0])
                if node_name.startswith("("):
                    return
                node = self._gpu_nodes_by_name.get(node_name)
                if node is None:
                    return
                available = int(node.available_gpus)
                model = node.gpu_model.strip()
                partition = node.partition.split(",")[0].replace("*", "").strip()

                self.query_one("#slurm_nodelist", Input).value = node_name
                if partition:
                    self.query_one("#slurm_partition", Input).value = partition
                selected_constraint = "(unchanged)"
                if model and model != "-":
                    self.query_one("#slurm_constraint", Input).value = model
                    selected_constraint = model

                try:
                    current_gpus = int((self.query_one("#slurm_gpus", Input).value or "1").strip())
                except ValueError:
                    current_gpus = 1
                if available > 0 and current_gpus > available:
                    self.query_one("#slurm_gpus", Input).value = str(available)

                self._update_selected_gpu_info(node)
                self.query_one(TabbedContent).active = "benchmark_tab"
            except Exception:  # noqa: BLE001
                return
            return

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in {"gpu_model_filter", "gpu_ram_min"}:
            self.refresh_gpus_async()
            return

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "gpu_available_only":
            self.refresh_gpus_async()
            return

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "gpu_state_filter":
            self.refresh_gpus_async()
            return

    def action_refresh_all(self) -> None:
        self.refresh_all_async()
