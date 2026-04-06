from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from beir import util


DEFAULT_COIR_DATASETS = [
    "apps",
    "cosqa",
    "synthetic_text2sql",
    "codesearchnet",
    "codesearchnet-ccr",
    "codetransocean-dl",
    "codetransocean-contest",
    "stackoverflow-qa",
    "codefeedback-st",
    "codefeedback-mt",
]

HF_REPO_MAP = {
    "apps": "CoIR-Retrieval/apps",
    "cosqa": "CoIR-Retrieval/cosqa",
    "synthetic_text2sql": "CoIR-Retrieval/synthetic-text2sql",
    "codesearchnet": "CoIR-Retrieval/CodeSearchNet",
    "codesearchnet-ccr": "CoIR-Retrieval/CodeSearchNet-ccr",
    "codetransocean-dl": "CoIR-Retrieval/codetrans-dl",
    "codetransocean-contest": "CoIR-Retrieval/codetrans-contest",
    "stackoverflow-qa": "CoIR-Retrieval/stackoverflow-qa",
    "codefeedback-st": "CoIR-Retrieval/codefeedback-st",
    "codefeedback-mt": "CoIR-Retrieval/codefeedback-mt",
}


def _apply_shard(items: list[str], shard_id: int, num_shards: int) -> list[str]:
    if num_shards <= 1:
        return items
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard_id {shard_id} for num_shards {num_shards}")
    shard_size = len(items) // num_shards
    start_idx = shard_id * shard_size
    end_idx = start_idx + shard_size
    if shard_id == num_shards - 1:
        end_idx = len(items)
    print(f"Sharding datasets: shard {shard_id}/{num_shards} -> items {start_idx}:{end_idx}")
    return items[start_idx:end_idx]


def _download_beir_dataset(dataset_root: Path, dataset_name: str) -> None:
    # Standard BEIR host pattern. If a dataset doesn't exist at this URL, the run logs the failure
    # so users can provide a custom source later.
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    target_dir = dataset_root / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset_name} from {url}")
    util.download_and_unzip(url, target_dir.as_posix())


def _download_hf_dataset(dataset_root: Path, dataset_name: str, repo_id: str, revision: str | None) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    target_dir = dataset_root / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_repo = HF_REPO_MAP.get(dataset_name, repo_id)

    def _concat_parquet(files: list[Path]) -> pd.DataFrame:
        frames = [pd.read_parquet(p) for p in files]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _write_beir_files(snapshot_root: Path, out_dir: Path) -> None:
        query_files = sorted(snapshot_root.rglob("*queries*.parquet"))
        corpus_files = sorted(snapshot_root.rglob("*corpus*.parquet"))
        qrels_files = sorted(snapshot_root.rglob("*qrels/test*.parquet"))
        if not qrels_files:
            qrels_files = sorted(snapshot_root.rglob("data/test*.parquet"))
        if not query_files or not corpus_files or not qrels_files:
            raise FileNotFoundError("Missing parquet components for queries/corpus/qrels.")

        qdf = _concat_parquet(query_files)
        cdf = _concat_parquet(corpus_files)
        rdf = _concat_parquet(qrels_files)

        query_id_col = "_id" if "_id" in qdf.columns else "query-id"
        query_text_col = "text" if "text" in qdf.columns else "query"
        corpus_id_col = "_id" if "_id" in cdf.columns else "corpus-id"
        corpus_text_col = "text" if "text" in cdf.columns else "corpus"
        corpus_title_col = "title" if "title" in cdf.columns else None

        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "queries.jsonl").open("w", encoding="utf-8") as f:
            for _, row in qdf.iterrows():
                f.write(json.dumps({"_id": str(row[query_id_col]), "text": str(row[query_text_col])}, ensure_ascii=False) + "\n")

        with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
            for _, row in cdf.iterrows():
                payload = {"_id": str(row[corpus_id_col]), "text": str(row[corpus_text_col])}
                if corpus_title_col is not None:
                    payload["title"] = str(row[corpus_title_col]) if pd.notna(row[corpus_title_col]) else ""
                else:
                    payload["title"] = ""
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        qrels_dir = out_dir / "qrels"
        qrels_dir.mkdir(parents=True, exist_ok=True)
        with (qrels_dir / "test.tsv").open("w", encoding="utf-8") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for _, row in rdf.iterrows():
                qid = str(row["query-id"] if "query-id" in rdf.columns else row[query_id_col])
                cid = str(row["corpus-id"] if "corpus-id" in rdf.columns else row[corpus_id_col])
                score = row["score"] if "score" in rdf.columns else 1
                f.write(f"{qid}\t{cid}\t{score}\n")

    # First try a direct zip artifact.
    try:
        zip_path = hf_hub_download(
            repo_id=resolved_repo,
            filename=f"{dataset_name}.zip",
            repo_type="dataset",
            revision=revision,
        )
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        return
    except Exception:
        pass

    # Fallback to parquet snapshot conversion.
    with tempfile.TemporaryDirectory(prefix=f"coir_{dataset_name}_") as tmp:
        tmp_dir = Path(tmp)
        snapshot_download(
            repo_id=resolved_repo,
            repo_type="dataset",
            revision=revision,
            local_dir=tmp_dir.as_posix(),
            allow_patterns=["**/*.parquet", "*.parquet"],
            local_dir_use_symlinks=False,
        )
        _write_beir_files(tmp_dir, target_dir)
        return

    raise FileNotFoundError(f"Could not download '{dataset_name}' from '{resolved_repo}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download COIR/BEIR-style datasets with optional sharding.")
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names. Defaults to configured COIR list.")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--source", choices=["beir", "hf"], default="beir")
    parser.add_argument("--hf_repo_id", type=str, default="CoIR-Retrieval/datasets")
    parser.add_argument("--hf_revision", type=str, default=None)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    datasets = args.datasets if args.datasets else list(DEFAULT_COIR_DATASETS)
    datasets = _apply_shard(datasets, shard_id=args.shard_id, num_shards=args.num_shards)

    for name in datasets:
        expected = dataset_root / name
        if args.skip_existing and expected.exists() and any(expected.iterdir()):
            print(f"Skipping existing dataset: {name}")
            continue
        try:
            if args.source == "hf":
                print(f"Downloading {name} from HF dataset repo {args.hf_repo_id}")
                _download_hf_dataset(
                    dataset_root=dataset_root,
                    dataset_name=name,
                    repo_id=args.hf_repo_id,
                    revision=args.hf_revision,
                )
            else:
                _download_beir_dataset(dataset_root, name)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed dataset {name}: {exc}")


if __name__ == "__main__":
    main()
