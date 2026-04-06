import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from beir.datasets.data_loader import GenericDataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare retrieval runs and inspect positive/failure cases for RepoEval/SWE-Bench."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset prefix, e.g. repoeval, swe-bench-lite.")
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="One or more NAME=/path/to/raw_results.jsonl entries.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num_cases", type=int, default=8)
    parser.add_argument("--snippet_chars", type=int, default=240)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def parse_run_specs(run_specs: list[str]) -> dict[str, str]:
    out = {}
    for spec in run_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid run spec '{spec}'. Expected NAME=PATH.")
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(f"Invalid run spec '{spec}'. Missing NAME.")
        if not path:
            raise ValueError(f"Invalid run spec '{spec}'. Missing PATH.")
        out[name] = path
    return out


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_instance_dirs(dataset_root: str, dataset: str) -> list[str]:
    dirs = []
    for name in os.listdir(dataset_root):
        path = os.path.join(dataset_root, name)
        if os.path.isdir(path) and name.startswith(f"{dataset}_"):
            dirs.append(name)
    dirs.sort()
    return dirs


def load_dataset_index(dataset_root: str, dataset: str) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, set[str]]]:
    query_text: dict[str, str] = {}
    corpus_index: dict[str, dict[str, Any]] = {}
    qrels_index: dict[str, set[str]] = defaultdict(set)

    for instance in get_instance_dirs(dataset_root, dataset):
        corpus, queries, qrels = GenericDataLoader(
            data_folder=os.path.join(dataset_root, instance)
        ).load(split="test")
        for query_id, query in queries.items():
            query_text[query_id] = query
        for doc_id, doc in corpus.items():
            corpus_index[doc_id] = doc
        for query_id, rels in qrels.items():
            for doc_id, rel in rels.items():
                if float(rel) > 0:
                    qrels_index[query_id].add(doc_id)
    return query_text, corpus_index, qrels_index


def summarize_doc(doc_id: str, corpus_index: dict[str, dict[str, Any]], snippet_chars: int) -> dict[str, Any]:
    doc = corpus_index.get(doc_id, {})
    title = doc.get("title", "")
    text = doc.get("text", "")
    snippet = text[:snippet_chars].replace("\n", " ") if text else ""
    return {"doc_id": doc_id, "title": title, "snippet": snippet}


def compute_first_rank(ranked_doc_ids: list[str], relevant: set[str]) -> int | None:
    for i, doc_id in enumerate(ranked_doc_ids):
        if doc_id in relevant:
            return i + 1
    return None


def compute_per_query_metrics(ranked_doc_ids: list[str], relevant: set[str], k: int) -> dict[str, Any]:
    rank = compute_first_rank(ranked_doc_ids, relevant)
    hit = rank is not None and rank <= k
    mrr = (1.0 / rank) if hit and rank is not None else 0.0
    return {"first_relevant_rank": rank, f"hit@{k}": hit, f"mrr@{k}": mrr}


def bucket_query_length(words: int) -> str:
    if words <= 40:
        return "<=40"
    if words <= 120:
        return "41-120"
    return ">120"


def bucket_relevant_count(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    return "4+"


def median_or_zero(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def main() -> None:
    args = parse_args()
    run_specs = parse_run_specs(args.runs)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    query_text, corpus_index, qrels_index = load_dataset_index(
        dataset_root=args.dataset_root, dataset=args.dataset
    )

    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in run_specs.items():
        rows = load_jsonl(path)
        run_map = {row["query_id"]: row for row in rows}
        runs[name] = run_map

    common_queries = sorted(
        set(qrels_index.keys()).intersection(*(set(run.keys()) for run in runs.values()))
    )
    if not common_queries:
        raise ValueError("No overlapping queries across runs and dataset qrels.")

    per_model_stats: dict[str, dict[str, Any]] = {}
    for model_name, run in runs.items():
        hits = 0
        mrr_sum = 0.0
        per_query = {}
        for query_id in common_queries:
            relevant = qrels_index.get(query_id, set())
            ranked_doc_ids = run[query_id].get("ranked_doc_ids", [])
            metrics = compute_per_query_metrics(ranked_doc_ids, relevant, args.k)
            hits += 1 if metrics[f"hit@{args.k}"] else 0
            mrr_sum += metrics[f"mrr@{args.k}"]
            per_query[query_id] = metrics
        per_model_stats[model_name] = {
            "num_queries": len(common_queries),
            f"hit_rate@{args.k}": hits / float(len(common_queries)),
            f"mrr@{args.k}": mrr_sum / float(len(common_queries)),
            "per_query": per_query,
        }

    ranking = sorted(
        per_model_stats.items(),
        key=lambda x: (x[1][f"hit_rate@{args.k}"], x[1][f"mrr@{args.k}"]),
        reverse=True,
    )
    best_model = ranking[0][0]
    other_models = [name for name in runs if name != best_model]

    best_per_query = per_model_stats[best_model]["per_query"]
    positives = []
    failures = []
    rescue_by_model: Counter[str] = Counter()

    for query_id in common_queries:
        best_hit = best_per_query[query_id][f"hit@{args.k}"]
        best_rank = best_per_query[query_id]["first_relevant_rank"]
        query = query_text.get(query_id, "")
        relevant = sorted(qrels_index.get(query_id, set()))
        best_row = runs[best_model][query_id]
        best_ranked = best_row.get("ranked_doc_ids", [])

        other_hit_flags = {
            model: per_model_stats[model]["per_query"][query_id][f"hit@{args.k}"] for model in other_models
        }

        base_case = {
            "query_id": query_id,
            "query": query,
            "query_words": len(query.split()),
            "num_relevant_docs": len(relevant),
            "best_first_relevant_rank": best_rank,
            "best_top_docs": [
                summarize_doc(doc_id, corpus_index, args.snippet_chars)
                for doc_id in best_ranked[: min(args.k, 5)]
            ],
            "relevant_docs": [
                summarize_doc(doc_id, corpus_index, args.snippet_chars)
                for doc_id in relevant[:5]
            ],
            "other_models_hit": other_hit_flags,
        }

        if best_hit:
            gain = len(other_models) - sum(1 for v in other_hit_flags.values() if v)
            positives.append((gain, 1e9 if best_rank is None else best_rank, base_case))
        else:
            if other_models:
                rescuers = [m for m, hit in other_hit_flags.items() if hit]
                if rescuers:
                    best_rescuer = max(
                        rescuers,
                        key=lambda m: (
                            per_model_stats[m]["per_query"][query_id][f"hit@{args.k}"],
                            -1e9
                            if per_model_stats[m]["per_query"][query_id]["first_relevant_rank"] is None
                            else -per_model_stats[m]["per_query"][query_id]["first_relevant_rank"],
                        ),
                    )
                    rescue_by_model[best_rescuer] += 1
            failures.append((sum(1 for v in other_hit_flags.values() if v), base_case))

    positives.sort(key=lambda x: (-x[0], x[1]))
    failures.sort(key=lambda x: (-x[0], x[1]["query_words"]))
    positive_cases = [x[2] for x in positives[: args.num_cases]]
    failure_cases = [x[1] for x in failures[: args.num_cases]]

    success_query_ids = [q for q in common_queries if best_per_query[q][f"hit@{args.k}"]]
    failure_query_ids = [q for q in common_queries if not best_per_query[q][f"hit@{args.k}"]]

    success_lengths = [len(query_text.get(q, "").split()) for q in success_query_ids]
    failure_lengths = [len(query_text.get(q, "").split()) for q in failure_query_ids]
    success_relevant = [len(qrels_index.get(q, set())) for q in success_query_ids]
    failure_relevant = [len(qrels_index.get(q, set())) for q in failure_query_ids]

    fail_rate_by_query_len: dict[str, dict[str, Any]] = {}
    fail_rate_by_relevant_count: dict[str, dict[str, Any]] = {}
    len_buckets = defaultdict(lambda: {"failures": 0, "total": 0})
    rel_buckets = defaultdict(lambda: {"failures": 0, "total": 0})

    for query_id in common_queries:
        q_words = len(query_text.get(query_id, "").split())
        rel_count = len(qrels_index.get(query_id, set()))
        lb = bucket_query_length(q_words)
        rb = bucket_relevant_count(rel_count)
        len_buckets[lb]["total"] += 1
        rel_buckets[rb]["total"] += 1
        if not best_per_query[query_id][f"hit@{args.k}"]:
            len_buckets[lb]["failures"] += 1
            rel_buckets[rb]["failures"] += 1

    for key, counts in sorted(len_buckets.items()):
        fail_rate_by_query_len[key] = {
            "failures": counts["failures"],
            "total": counts["total"],
            f"failure_rate@{args.k}": counts["failures"] / float(counts["total"]) if counts["total"] else 0.0,
        }
    for key, counts in sorted(rel_buckets.items()):
        fail_rate_by_relevant_count[key] = {
            "failures": counts["failures"],
            "total": counts["total"],
            f"failure_rate@{args.k}": counts["failures"] / float(counts["total"]) if counts["total"] else 0.0,
        }

    summary = {
        "dataset": args.dataset,
        "k": args.k,
        "models_ranked": [
            {
                "model": name,
                f"hit_rate@{args.k}": stats[f"hit_rate@{args.k}"],
                f"mrr@{args.k}": stats[f"mrr@{args.k}"],
                "num_queries": stats["num_queries"],
            }
            for name, stats in ranking
        ],
        "best_model": best_model,
        "best_model_stats": {
            f"hit_rate@{args.k}": per_model_stats[best_model][f"hit_rate@{args.k}"],
            f"mrr@{args.k}": per_model_stats[best_model][f"mrr@{args.k}"],
            "num_failures": len(failure_query_ids),
            "num_successes": len(success_query_ids),
        },
        "best_model_failure_trends": {
            "median_query_words_success": median_or_zero(success_lengths),
            "median_query_words_failure": median_or_zero(failure_lengths),
            "median_num_relevant_success": median_or_zero(success_relevant),
            "median_num_relevant_failure": median_or_zero(failure_relevant),
            "failure_rate_by_query_length_bucket": fail_rate_by_query_len,
            "failure_rate_by_relevant_count_bucket": fail_rate_by_relevant_count,
            "rescue_model_counts": dict(rescue_by_model),
        },
        "positive_cases_file": "positive_cases.jsonl",
        "failure_cases_file": "failure_cases.jsonl",
    }

    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(Path(args.output_dir) / "positive_cases.jsonl", "w", encoding="utf-8") as f:
        for row in positive_cases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(Path(args.output_dir) / "failure_cases.jsonl", "w", encoding="utf-8") as f:
        for row in failure_cases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_lines = []
    report_lines.append(f"# Retrieval Case Analysis ({args.dataset})")
    report_lines.append("")
    report_lines.append(f"- Evaluated queries: {len(common_queries)}")
    report_lines.append(f"- Best model @ {args.k}: `{best_model}`")
    report_lines.append("")
    report_lines.append("## Model Ranking")
    report_lines.append("")
    for i, item in enumerate(summary["models_ranked"], start=1):
        report_lines.append(
            f"{i}. `{item['model']}` | hit@{args.k}={item[f'hit_rate@{args.k}']:.4f} | mrr@{args.k}={item[f'mrr@{args.k}']:.4f}"
        )
    report_lines.append("")
    report_lines.append("## Failure Trends")
    report_lines.append("")
    trend = summary["best_model_failure_trends"]
    report_lines.append(
        f"- Median query words (success vs failure): {trend['median_query_words_success']:.1f} vs {trend['median_query_words_failure']:.1f}"
    )
    report_lines.append(
        f"- Median #relevant docs (success vs failure): {trend['median_num_relevant_success']:.1f} vs {trend['median_num_relevant_failure']:.1f}"
    )
    report_lines.append("")
    report_lines.append(f"- Failure rate buckets by query length (@{args.k}):")
    for key, value in trend["failure_rate_by_query_length_bucket"].items():
        report_lines.append(
            f"  - {key}: {value['failures']}/{value['total']} ({value[f'failure_rate@{args.k}']:.3f})"
        )
    report_lines.append("")
    report_lines.append(f"- Failure rate buckets by #relevant docs (@{args.k}):")
    for key, value in trend["failure_rate_by_relevant_count_bucket"].items():
        report_lines.append(
            f"  - {key}: {value['failures']}/{value['total']} ({value[f'failure_rate@{args.k}']:.3f})"
        )

    if trend["rescue_model_counts"]:
        report_lines.append("")
        report_lines.append("- Models that recover failures of the best model:")
        for model, count in sorted(trend["rescue_model_counts"].items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"  - `{model}`: {count}")

    with open(Path(args.output_dir) / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
