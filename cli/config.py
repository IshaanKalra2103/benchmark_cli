from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "defaults.json"
USER_CONFIG_PATH = PROJECT_ROOT / "config" / "local.json"
_REPOEVAL_METADATA_REL = Path("repoeval_metadata") / "function_level_completion_2k_context_codex.test.clean.jsonl"

_SCRATCH_ENV_VARS = (
    "SCRATCH",
    "SLURM_TMPDIR",
    "LOCAL_SCRATCH",
    "LSCRATCH",
    "PBS_JOBFS",
    "TMPDIR",
)

_CLUSTER_MARKERS = (
    "SLURM_CLUSTER_NAME",
    "SLURM_JOB_ID",
    "SLURM_SUBMIT_DIR",
    "SLURM_JOB_USER",
    "PBS_JOBID",
    "LSB_JOBID",
)

_ENV_FILE_NAMES = (".env.local", ".env")

_ENV_PATH_OVERRIDES = {
    "BENCHMARK_DATASET_ROOT": ("paths", "dataset_root"),
    "BENCHMARK_RESULTS_ROOT": ("paths", "results_root"),
    "BENCHMARK_CACHE_DIR": ("paths", "cache_dir"),
    "BENCHMARK_REPOEVAL_DATASET_PATH": ("paths", "repoeval_dataset_path"),
}

_ENV_SLURM_OVERRIDES = {
    "BENCHMARK_SLURM_PARTITION": ("slurm", "partition"),
    "BENCHMARK_SLURM_ACCOUNT": ("slurm", "account"),
    "BENCHMARK_SLURM_QOS": ("slurm", "qos"),
    "BENCHMARK_SLURM_TIME": ("slurm", "time"),
    "BENCHMARK_SLURM_MEM": ("slurm", "mem"),
    "BENCHMARK_SLURM_CONSTRAINT": ("slurm", "constraint"),
    "BENCHMARK_SLURM_NODELIST": ("slurm", "nodelist"),
}

_ENV_INT_OVERRIDES = {
    "BENCHMARK_SLURM_GPUS": ("slurm", "gpus"),
    "BENCHMARK_SLURM_CPUS_PER_TASK": ("slurm", "cpus_per_task"),
    "BENCHMARK_SLURM_POLL_INTERVAL_SEC": ("slurm", "poll_interval_sec"),
    "BENCHMARK_BATCH_SIZE": ("models", "defaults", "batch_size"),
}


def _as_bool_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _first_writable_dir(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        try:
            expanded = candidate.expanduser()
            expanded.mkdir(parents=True, exist_ok=True)
            if expanded.is_dir() and os.access(expanded, os.W_OK):
                return expanded
        except OSError:
            continue
    return None


def _detect_scratch_workspace() -> tuple[Path | None, str | None]:
    explicit = os.environ.get("BENCHMARK_SCRATCH_WORKSPACE", "").strip()
    if explicit:
        return Path(explicit).expanduser(), "BENCHMARK_SCRATCH_WORKSPACE"

    if _as_bool_env(os.environ.get("BENCHMARK_DISABLE_SCRATCH_AUTODETECT")):
        return None, None

    in_cluster = any(os.environ.get(marker, "").strip() for marker in _CLUSTER_MARKERS)
    if not in_cluster:
        return None, None

    env_candidates: list[Path] = []
    for key in _SCRATCH_ENV_VARS:
        raw = os.environ.get(key, "").strip()
        if raw:
            env_candidates.append(Path(raw))
    scratch = _first_writable_dir(env_candidates)
    if scratch is not None:
        return scratch, "env_scratch"

    user = os.environ.get("USER", "").strip()
    fallback_candidates = []
    if user:
        fallback_candidates.append(Path("/scratch") / user)
    fallback_candidates.append(Path("/scratch"))
    scratch = _first_writable_dir(fallback_candidates)
    if scratch is not None:
        return scratch, "fallback_scratch"

    return None, None


def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_dotenv() -> None:
    for name in _ENV_FILE_NAMES:
        _load_dotenv_file(PROJECT_ROOT / name)


def _set_nested(config: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    cursor: dict[str, Any] = config
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _apply_env_overrides(config: dict[str, Any]) -> None:
    for env_key, config_path in _ENV_PATH_OVERRIDES.items():
        raw = os.environ.get(env_key, "").strip()
        if raw:
            _set_nested(config, config_path, raw)

    for env_key, config_path in _ENV_SLURM_OVERRIDES.items():
        raw = os.environ.get(env_key, "").strip()
        if raw:
            _set_nested(config, config_path, raw)

    for env_key, config_path in _ENV_INT_OVERRIDES.items():
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            continue
        try:
            _set_nested(config, config_path, int(raw))
        except ValueError:
            continue


def _apply_scratch_workspace_paths(config: dict[str, Any]) -> None:
    scratch_root, source = _detect_scratch_workspace()
    if scratch_root is None:
        return

    workspace_root = (scratch_root / "benchmark_cli").expanduser()
    paths = config.setdefault("paths", {})

    def maybe_relocate(key: str, target: Path) -> None:
        raw = paths.get(key)
        if raw is None:
            paths[key] = str(target)
            return
        current = Path(str(raw)).expanduser()
        if _path_is_under(current, PROJECT_ROOT):
            paths[key] = str(target)

    dataset_root = workspace_root / "datasets"
    results_root = workspace_root / "results"
    cache_dir = workspace_root / "cache" / "retrieval_embeddings"
    repoeval_dataset_path = dataset_root / _REPOEVAL_METADATA_REL

    maybe_relocate("dataset_root", dataset_root)
    maybe_relocate("results_root", results_root)
    maybe_relocate("cache_dir", cache_dir)
    maybe_relocate("repoeval_dataset_path", repoeval_dataset_path)

    for key in ("dataset_root", "results_root", "cache_dir"):
        value = paths.get(key)
        if value:
            Path(str(value)).expanduser().mkdir(parents=True, exist_ok=True)

    paths["workspace_root"] = str(workspace_root)
    paths["workspace_source"] = source


def _default_config() -> dict[str, Any]:
    return {
        "paths": {
            "dataset_root": str(PROJECT_ROOT / "datasets"),
            "results_root": str(PROJECT_ROOT / "results"),
            "cache_dir": str(PROJECT_ROOT / "cache" / "retrieval_embeddings"),
            "repoeval_dataset_path": str(
                PROJECT_ROOT / "datasets" / _REPOEVAL_METADATA_REL
            ),
        },
        "models": {
            "show_8b": False,
            "profiles": {
                "qwen3_embed_0_6b": {
                    "name": "qwen3-embed-0.6b",
                    "model": "Qwen/Qwen3-Embedding-0.6B",
                    "reranker_model": None,
                    "enabled": True,
                    "tag": "baseline",
                },
                "qwen3_embed_0_6b_rerank": {
                    "name": "qwen3-embed-0.6b-rerank",
                    "model": "Qwen/Qwen3-Embedding-0.6B",
                    "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
                    "enabled": True,
                    "tag": "baseline",
                },
                "qwen3_embed_8b": {
                    "name": "qwen3-embed-8b",
                    "model": "Qwen/Qwen3-Embedding-8B",
                    "reranker_model": None,
                    "enabled": False,
                    "tag": "baseline",
                },
                "qwen3_embed_8b_rerank": {
                    "name": "qwen3-embed-8b-rerank",
                    "model": "Qwen/Qwen3-Embedding-8B",
                    "reranker_model": "Qwen/Qwen3-Reranker-8B",
                    "enabled": False,
                    "tag": "baseline",
                },
            },
            "finetune": {
                "hf_repo": "aysinghal/ide-code-retrieval-qwen3-0.6b",
                "checkpoint_pattern": "checkpoint-",
                "embed_only": True,
                "default_reranker_model": "Qwen/Qwen3-Reranker-0.6B",
            },
            "defaults": {
                "batch_size": 32,
                "top_k": 10,
                "candidate_k": 100,
                "normalize_embeddings": True,
                "trust_remote_code": True,
                "query_prefix": "query: ",
                "doc_prefix": "passage: ",
                "reranker_batch_size": 8,
                "reranker_max_length": 512,
            },
        },
        "checkpoint_policy": {
            "include_first": True,
            "anchor_step": 3000,
            "interval": 1000,
        },
        "benchmarks": {
            "repoeval": {
                "dataset": "repoeval",
                "supports_analysis": True,
            },
            "swe-bench-lite": {
                "dataset": "swe-bench-lite",
                "supports_analysis": True,
            },
            "coir": {
                "dataset": "coir",
                "supports_analysis": False,
                "groups": {
                    "text-to-code": [
                        "apps",
                        "cosqa",
                        "synthetic_text2sql",
                        "codesearchnet",
                    ],
                    "code-to-code": [
                        "codesearchnet-ccr",
                        "codetransocean-dl",
                        "codetransocean-contest",
                    ],
                    "hybrid-qa": [
                        "stackoverflow-qa",
                        "codefeedback-st",
                        "codefeedback-mt",
                    ],
                },
            },
        },
        "slurm": {
            "partition": "",
            "account": "",
            "qos": "",
            "time": "01:00:00",
            "mem": "32G",
            "gpus": 1,
            "cpus_per_task": 8,
            "constraint": "",
            "nodelist": "",
            "extra_args": [],
            "poll_interval_sec": 10,
        },
    }


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_default_config() -> None:
    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_CONFIG_PATH.exists():
        DEFAULT_CONFIG_PATH.write_text(json.dumps(_default_config(), indent=2) + "\n", encoding="utf-8")


def load_config() -> dict[str, Any]:
    load_dotenv()
    ensure_default_config()
    base = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if USER_CONFIG_PATH.exists():
        user_cfg = json.loads(USER_CONFIG_PATH.read_text(encoding="utf-8"))
        merged = _deep_merge(base, user_cfg)
    else:
        merged = base
    _apply_env_overrides(merged)
    _apply_scratch_workspace_paths(merged)
    return merged


def save_user_config(config: dict[str, Any]) -> None:
    USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def write_default_config(force: bool = False) -> Path:
    ensure_default_config()
    if force or not USER_CONFIG_PATH.exists():
        USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        USER_CONFIG_PATH.write_text(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return USER_CONFIG_PATH
