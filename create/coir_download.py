from __future__ import annotations

import argparse
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download COIR/BEIR-style datasets with optional sharding.")
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names. Defaults to configured COIR list.")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")
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
            _download_beir_dataset(dataset_root, name)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed dataset {name}: {exc}")


if __name__ == "__main__":
    main()
