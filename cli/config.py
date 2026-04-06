from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "defaults.json"
USER_CONFIG_PATH = PROJECT_ROOT / "config" / "local.json"


def _default_config() -> dict[str, Any]:
    return {
        "paths": {
            "dataset_root": str(PROJECT_ROOT / "datasets"),
            "results_root": str(PROJECT_ROOT / "results"),
            "cache_dir": str(PROJECT_ROOT / "cache" / "retrieval_embeddings"),
            "repoeval_dataset_path": str(
                PROJECT_ROOT / "datasets" / "repoeval_metadata" / "function_level_completion_2k_context_codex.test.clean.jsonl"
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
    ensure_default_config()
    base = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if USER_CONFIG_PATH.exists():
        user_cfg = json.loads(USER_CONFIG_PATH.read_text(encoding="utf-8"))
        merged = _deep_merge(base, user_cfg)
    else:
        merged = base
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
