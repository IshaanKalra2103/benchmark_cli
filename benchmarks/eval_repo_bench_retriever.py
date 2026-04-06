import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


K_VALUES = [1, 3, 5, 10, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repo-level retrieval for RepoEval/SWE-Bench with optional reranking."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset prefix, e.g. repoeval, swe-bench-lite, apps.")
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--model", type=str, required=True, help="Embedding model id/path.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--query_prefix", type=str, default="")
    parser.add_argument("--doc_prefix", type=str, default="")
    parser.add_argument("--normalize_embeddings", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--reranker_model", type=str, default=None, help="Optional sequence-classification reranker.")
    parser.add_argument("--reranker_batch_size", type=int, default=8)
    parser.add_argument("--reranker_max_length", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=10, help="Final top-k docs saved to results.")
    parser.add_argument("--candidate_k", type=int, default=100, help="Dense top-k candidates for reranking.")
    parser.add_argument("--cache_dir", type=str, default="cache/retrieval_embeddings")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--instance_regex", type=str, default=None, help="Only run instances matching regex.")
    parser.add_argument("--repoeval_dataset_path", type=str, default="output/repoeval/datasets/function_level_completion_2k_context_codex.test.clean.jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="Summary metrics output json.")
    parser.add_argument("--results_file", type=str, required=True, help="Generation-compatible retrieval jsonl.")
    parser.add_argument("--raw_results_file", type=str, required=True, help="Per-query ranked doc IDs/scores jsonl.")
    parser.add_argument("--per_query_metrics_file", type=str, default=None, help="Optional per-query metrics jsonl.")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def text_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def format_doc_text(doc: dict[str, Any], doc_prefix: str) -> str:
    title = doc.get("title", "")
    text = doc.get("text", "")
    combined = f"{title}\n{text}".strip() if title else text
    return f"{doc_prefix}{combined}"


def format_query_text(query: str, query_prefix: str) -> str:
    return f"{query_prefix}{query}"


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    normalize_embeddings: bool,
    desc: str,
) -> np.ndarray:
    if not texts:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def cache_paths(
    cache_dir: str,
    dataset: str,
    instance_dir: str,
    model_name: str,
    normalize_embeddings: bool,
    doc_prefix: str,
) -> tuple[Path, Path]:
    model_tag = safe_name(model_name)
    prefix_tag = text_hash(doc_prefix)
    cache_base = (
        Path(cache_dir)
        / dataset
        / instance_dir
        / f"{model_tag}__norm{int(normalize_embeddings)}__pref{prefix_tag}"
    )
    return cache_base.with_suffix(".ids.json"), cache_base.with_suffix(".emb.npy")


def load_or_encode_doc_embeddings(
    model: SentenceTransformer,
    docs: list[str],
    doc_ids: list[str],
    args: argparse.Namespace,
    instance_dir: str,
) -> np.ndarray:
    ids_path, emb_path = cache_paths(
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        instance_dir=instance_dir,
        model_name=args.model,
        normalize_embeddings=args.normalize_embeddings,
        doc_prefix=args.doc_prefix,
    )

    if not args.no_cache and ids_path.exists() and emb_path.exists():
        with open(ids_path, "r", encoding="utf-8") as f:
            cached_ids = json.load(f)
        if cached_ids == doc_ids:
            return np.load(emb_path)

    embeddings = encode_texts(
        model=model,
        texts=docs,
        batch_size=args.batch_size,
        normalize_embeddings=args.normalize_embeddings,
        desc=f"docs::{instance_dir}",
    )
    if not args.no_cache:
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embeddings)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(doc_ids, f)
    return embeddings


def get_relevant_set(qrels: dict[str, dict[str, int]], query_id: str) -> set[str]:
    return {doc_id for doc_id, rel in qrels.get(query_id, {}).items() if float(rel) > 0}


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0 or k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, scores.size)
    if k == scores.size:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx


def load_reranker(args: argparse.Namespace) -> tuple[Any, Any, torch.device] | tuple[None, None, None]:
    if args.reranker_model is None:
        return None, None, None
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.reranker_model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.reranker_model, trust_remote_code=args.trust_remote_code
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


def rerank_scores(
    query: str,
    docs: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    if not docs:
        return np.array([], dtype=np.float32)
    all_scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            pairs = [[query, doc] for doc in batch_docs]
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs, return_dict=True).logits
            if logits.ndim == 1:
                batch_scores = logits
            elif logits.shape[-1] == 1:
                batch_scores = logits[:, 0]
            else:
                batch_scores = logits[:, -1]
            all_scores.extend(batch_scores.float().cpu().tolist())
    return np.array(all_scores, dtype=np.float32)


def init_metric_buckets() -> dict[str, dict[int, float]]:
    return {
        "ndcg": {k: 0.0 for k in K_VALUES},
        "mrr": {k: 0.0 for k in K_VALUES},
        "recall": {k: 0.0 for k in K_VALUES},
        "precision": {k: 0.0 for k in K_VALUES},
    }


def dcg_at_k(relevance: list[int], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevance[:k]):
        if rel:
            score += 1.0 / np.log2(i + 2.0)
    return score


def per_query_metrics(ranked_doc_ids: list[str], relevant: set[str]) -> dict[str, Any]:
    relevance = [1 if doc_id in relevant else 0 for doc_id in ranked_doc_ids]
    first_rel_rank = None
    for i, rel in enumerate(relevance):
        if rel:
            first_rel_rank = i + 1
            break

    out = {"first_relevant_rank": first_rel_rank, "num_relevant": len(relevant), "metrics": {}}
    for k in K_VALUES:
        top_relevance = relevance[:k]
        hits = int(sum(top_relevance))
        precision = hits / float(k)
        recall = hits / float(len(relevant)) if relevant else 0.0

        dcg = dcg_at_k(relevance, k)
        ideal_relevance = [1] * min(k, len(relevant))
        idcg = dcg_at_k(ideal_relevance, k)
        ndcg = dcg / idcg if idcg > 0 else 0.0

        mrr = 0.0
        if first_rel_rank is not None and first_rel_rank <= k:
            mrr = 1.0 / float(first_rel_rank)

        out["metrics"][k] = {
            "ndcg": ndcg,
            "mrr": mrr,
            "recall": recall,
            "precision": precision,
            "hit": 1.0 if hits > 0 else 0.0,
        }
    return out


def finalize_metric_buckets(buckets: dict[str, dict[int, float]], num_queries: int) -> dict[str, dict[str, float]]:
    if num_queries == 0:
        return {
            "ndcg": {f"NDCG@{k}": 0.0 for k in K_VALUES},
            "mrr": {f"MRR@{k}": 0.0 for k in K_VALUES},
            "recall": {f"Recall@{k}": 0.0 for k in K_VALUES},
            "precision": {f"P@{k}": 0.0 for k in K_VALUES},
        }
    return {
        "ndcg": {f"NDCG@{k}": buckets["ndcg"][k] / num_queries for k in K_VALUES},
        "mrr": {f"MRR@{k}": buckets["mrr"][k] / num_queries for k in K_VALUES},
        "recall": {f"Recall@{k}": buckets["recall"][k] / num_queries for k in K_VALUES},
        "precision": {f"P@{k}": buckets["precision"][k] / num_queries for k in K_VALUES},
    }


def dump_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_instance_dirs(dataset_root: str, dataset: str, max_instances: int | None, instance_regex: str | None) -> list[str]:
    dirs = []
    for name in os.listdir(dataset_root):
        path = os.path.join(dataset_root, name)
        if not os.path.isdir(path):
            continue
        if not name.startswith(f"{dataset}_"):
            continue
        if instance_regex is not None and re.search(instance_regex, name) is None:
            continue
        dirs.append(name)
    dirs.sort()
    if max_instances is not None:
        dirs = dirs[:max_instances]
    return dirs


def build_final_results(
    dataset: str,
    query_to_docs: dict[str, list[dict[str, Any]]],
    query_to_text: dict[str, str],
    repoeval_dataset_path: str,
) -> list[dict[str, Any]]:
    if dataset == "swe-bench-lite":
        out = []
        for query_id in sorted(query_to_docs):
            out.append(
                {
                    "instance_id": query_id,
                    "problem_statement": query_to_text.get(query_id, ""),
                    "docs": query_to_docs[query_id],
                }
            )
        return out

    repoeval_map: dict[str, dict[str, str]] = {}
    if os.path.exists(repoeval_dataset_path):
        with open(repoeval_dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                query_id = item["metadata"]["task_id"]
                repoeval_map[query_id] = {
                    "prompt": item.get("prompt", ""),
                    "reference": item.get("metadata", {}).get("ground_truth", ""),
                }

    out = []
    for query_id in sorted(query_to_docs):
        if dataset == "repoeval":
            row = repoeval_map.get(query_id, {"prompt": query_to_text.get(query_id, ""), "reference": ""})
            out.append({"prompt": row["prompt"], "reference": row["reference"], "docs": query_to_docs[query_id]})
        else:
            out.append(
                {
                    "query_id": query_id,
                    "query": query_to_text.get(query_id, ""),
                    "docs": query_to_docs[query_id],
                }
            )
    return out


def main() -> None:
    args = parse_args()
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.raw_results_file).parent.mkdir(parents=True, exist_ok=True)
    if args.per_query_metrics_file is not None:
        Path(args.per_query_metrics_file).parent.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(
        args.model,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
    )
    rerank_tokenizer, rerank_model, rerank_device = load_reranker(args)

    instance_dirs = get_instance_dirs(
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        max_instances=args.max_instances,
        instance_regex=args.instance_regex,
    )
    if not instance_dirs:
        raise ValueError(f"No dataset directories matched '{args.dataset}_*' under '{args.dataset_root}'.")

    all_metrics = init_metric_buckets()
    per_instance_summary: dict[str, Any] = {}
    query_to_docs: dict[str, list[dict[str, Any]]] = {}
    query_to_text: dict[str, str] = {}
    raw_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    total_queries = 0

    t0 = time.time()
    for instance_dir in instance_dirs:
        data_folder = os.path.join(args.dataset_root, instance_dir)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_folder).load(split="test")
        if len(queries) == 0:
            continue

        doc_ids = list(corpus.keys())
        doc_texts = [format_doc_text(corpus[doc_id], args.doc_prefix) for doc_id in doc_ids]
        doc_embeddings = load_or_encode_doc_embeddings(
            model=model,
            docs=doc_texts,
            doc_ids=doc_ids,
            args=args,
            instance_dir=instance_dir,
        )

        query_ids = list(queries.keys())
        query_texts = [format_query_text(queries[query_id], args.query_prefix) for query_id in query_ids]
        query_embeddings = encode_texts(
            model=model,
            texts=query_texts,
            batch_size=args.batch_size,
            normalize_embeddings=args.normalize_embeddings,
            desc=f"queries::{instance_dir}",
        )

        candidate_k = min(args.candidate_k, len(doc_ids))
        final_top_k = min(args.top_k, candidate_k)

        instance_metrics = init_metric_buckets()
        instance_q_count = 0
        for q_idx, query_id in enumerate(query_ids):
            dense_scores = np.dot(doc_embeddings, query_embeddings[q_idx])
            dense_idx = topk_indices(dense_scores, candidate_k)
            ranked_idx = dense_idx
            ranked_scores = dense_scores[dense_idx]

            if rerank_model is not None and dense_idx.size > 0:
                candidate_docs = [doc_texts[idx] for idx in dense_idx]
                rerank = rerank_scores(
                    query=queries[query_id],
                    docs=candidate_docs,
                    tokenizer=rerank_tokenizer,
                    model=rerank_model,
                    device=rerank_device,
                    batch_size=args.reranker_batch_size,
                    max_length=args.reranker_max_length,
                )
                rerank_order = np.argsort(-rerank)
                ranked_idx = dense_idx[rerank_order]
                ranked_scores = rerank[rerank_order]

            ranked_doc_ids = [doc_ids[idx] for idx in ranked_idx.tolist()]
            ranked_scores_list = ranked_scores.tolist()
            top_doc_ids = ranked_doc_ids[:final_top_k]
            top_docs = [corpus[doc_id] for doc_id in top_doc_ids]

            query_to_docs[query_id] = top_docs
            query_to_text[query_id] = queries[query_id]
            relevant = get_relevant_set(qrels, query_id)
            q_metrics = per_query_metrics(ranked_doc_ids, relevant)

            total_queries += 1
            instance_q_count += 1
            for k in K_VALUES:
                all_metrics["ndcg"][k] += q_metrics["metrics"][k]["ndcg"]
                all_metrics["mrr"][k] += q_metrics["metrics"][k]["mrr"]
                all_metrics["recall"][k] += q_metrics["metrics"][k]["recall"]
                all_metrics["precision"][k] += q_metrics["metrics"][k]["precision"]
                instance_metrics["ndcg"][k] += q_metrics["metrics"][k]["ndcg"]
                instance_metrics["mrr"][k] += q_metrics["metrics"][k]["mrr"]
                instance_metrics["recall"][k] += q_metrics["metrics"][k]["recall"]
                instance_metrics["precision"][k] += q_metrics["metrics"][k]["precision"]

            raw_rows.append(
                {
                    "instance": instance_dir,
                    "query_id": query_id,
                    "query": queries[query_id],
                    "ranked_doc_ids": ranked_doc_ids,
                    "ranked_scores": ranked_scores_list,
                    "relevant_doc_ids": sorted(relevant),
                }
            )
            if args.per_query_metrics_file is not None:
                per_query_rows.append(
                    {
                        "instance": instance_dir,
                        "query_id": query_id,
                        "first_relevant_rank": q_metrics["first_relevant_rank"],
                        "num_relevant": q_metrics["num_relevant"],
                        "metrics": {
                            f"@{k}": q_metrics["metrics"][k] for k in K_VALUES
                        },
                    }
                )

        per_instance_summary[instance_dir] = finalize_metric_buckets(instance_metrics, instance_q_count)

    total_time = time.time() - t0
    summary = finalize_metric_buckets(all_metrics, total_queries)
    summary["time"] = total_time
    summary["num_queries"] = total_queries
    summary["num_instances"] = len(per_instance_summary)
    summary["model"] = args.model
    summary["reranker_model"] = args.reranker_model
    summary["per_instance"] = per_instance_summary

    final_results = build_final_results(
        dataset=args.dataset,
        query_to_docs=query_to_docs,
        query_to_text=query_to_text,
        repoeval_dataset_path=args.repoeval_dataset_path,
    )

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    dump_jsonl(args.results_file, final_results)
    dump_jsonl(args.raw_results_file, raw_rows)
    if args.per_query_metrics_file is not None:
        dump_jsonl(args.per_query_metrics_file, per_query_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
