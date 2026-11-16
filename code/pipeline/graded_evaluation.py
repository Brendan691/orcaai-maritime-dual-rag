#!/usr/bin/env python3
"""Evaluate retrieval stage F1/F2 document rankings against graded development qrels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gate5_score_cache as score_cache  # noqa: E402


REPORT_SCHEMA = "gate5-graded-evaluation-report-v1"
OUTPUT_MANIFEST_SCHEMA = "gate5-graded-evaluation-output-manifest-v1"
EXPECTED_QUERY_IDS = tuple(f"D{index:02d}" for index in range(1, 9))
CONFIRMATION_QUERY_IDS = frozenset(f"H{index:02d}" for index in range(1, 33))
EXPECTED_DOCUMENTS_PER_QUERY = 48
EXPECTED_QREL_PAIRS = 384
DEFAULT_CELL = "E1-R1"
DEFAULT_CUTOFF = 5
F2_AEF_WEIGHT = 0.5
F2_RERANKER_WEIGHT = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as block_reader:
        for block in iter(lambda: block_reader.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_qrels(path: Path) -> list[dict]:
    """Parse TREC-style qrels and reject incomplete or duplicate judgments."""
    rows = []
    seen: set[tuple[str, str]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"qrels row {line_number} must have four tab-separated fields")
        query_id, iteration, doc_id, relevance_text = fields
        if query_id in CONFIRMATION_QUERY_IDS:
            raise ValueError(f"confirmation query {query_id} leaked into evaluation qrels")
        if not query_id or not doc_id:
            raise ValueError(f"qrels row {line_number} has an empty query or document ID")
        if iteration != "0":
            raise ValueError(f"qrels row {line_number} must use iteration 0")
        try:
            relevance = int(relevance_text)
        except ValueError as exc:
            raise ValueError(f"qrels row {line_number} has a non-integer relevance") from exc
        if relevance not in {0, 1, 2, 3}:
            raise ValueError(f"qrels row {line_number} relevance must be in 0..3")
        pair = (query_id, doc_id)
        if pair in seen:
            raise ValueError(f"duplicate qrels pair {query_id}/{doc_id}")
        seen.add(pair)
        rows.append(
            {
                "query_id": query_id,
                "doc_id": doc_id,
                "relevance": relevance,
            }
        )
    if not rows:
        raise ValueError("qrels file is empty")
    return rows


def group_qrels(rows: Sequence[dict]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for row in rows:
        query_id = str(row["query_id"])
        doc_id = str(row["doc_id"])
        relevance = int(row["relevance"])
        if relevance not in {0, 1, 2, 3}:
            raise ValueError("relevance must be in 0..3")
        by_doc = grouped.setdefault(query_id, {})
        if doc_id in by_doc:
            raise ValueError(f"duplicate qrels pair {query_id}/{doc_id}")
        by_doc[doc_id] = relevance
    return grouped


def load_frozen_qrels(qrels_path: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    rows = parse_qrels(qrels_path)
    manifest = read_json(manifest_path)
    if manifest.get("status") != "passed" or manifest.get("final_qrels_frozen") is not True:
        raise ValueError("development qrels manifest is not passed and frozen")
    declared = manifest.get("final_qrels") or {}
    if declared.get("sha256") != sha256_file(qrels_path):
        raise ValueError("development qrels hash does not match its final manifest")

    query_ids = tuple(dict.fromkeys(row["query_id"] for row in rows))
    if query_ids != EXPECTED_QUERY_IDS:
        raise ValueError("development qrels must contain D01-D08 in order")
    counts = Counter(row["query_id"] for row in rows)
    if any(counts[query_id] != EXPECTED_DOCUMENTS_PER_QUERY for query_id in EXPECTED_QUERY_IDS):
        raise ValueError("each development query must have exactly 48 judgments")
    if len(rows) != EXPECTED_QREL_PAIRS:
        raise ValueError("development qrels must contain exactly 384 pairs")
    return rows, manifest


def load_score_cache(cache_dir: Path) -> tuple[list[dict], dict]:
    manifest_path = cache_dir / "manifest.json"
    declared = read_json(manifest_path)
    records, manifest = score_cache.validate_cache(
        cache_dir,
        expected_hashes=declared.get("input_hashes", {}),
    )
    if manifest.get("cell") != DEFAULT_CELL:
        raise ValueError("score cache is not the E1-R1 tracer")
    if manifest.get("status") != "passed":
        raise ValueError("E1-R1 score cache did not pass")
    if any(record.get("status") != "passed" for record in records):
        raise ValueError("E1-R1 score cache contains failed records")
    return records, manifest


def max_pool_candidates(candidates: Sequence[dict], score_field: str) -> dict[str, dict]:
    """Max-pool chunk scores by document with deterministic equal-score ties."""
    pooled: dict[str, dict] = {}
    seen_chunks: set[str] = set()
    for candidate in candidates:
        doc_id = str(candidate.get("doc_id") or "")
        chunk_id = str(candidate.get("chunk_id") or "")
        if not doc_id or not chunk_id:
            raise ValueError("candidate is missing doc_id or chunk_id")
        if chunk_id in seen_chunks:
            raise ValueError(f"duplicate candidate chunk ID: {chunk_id}")
        seen_chunks.add(chunk_id)
        score = float(candidate.get(score_field))
        if not math.isfinite(score):
            raise ValueError(f"candidate {chunk_id} has a non-finite {score_field}")
        previous = pooled.get(doc_id)
        if (
            previous is None
            or score > previous["score"]
            or (score == previous["score"] and chunk_id < previous["best_chunk_id"])
        ):
            pooled[doc_id] = {
                "doc_id": doc_id,
                "score": score,
                "best_chunk_id": chunk_id,
            }
    return {doc_id: pooled[doc_id] for doc_id in sorted(pooled)}


def rank_document_scores(scores: Mapping[str, float]) -> list[dict]:
    """Rank documents by descending score, then ascending document ID."""
    rows = []
    for doc_id, raw_score in scores.items():
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"document {doc_id} has a non-finite score")
        rows.append({"doc_id": str(doc_id), "score": score})
    rows.sort(key=lambda row: (-row["score"], row["doc_id"]))
    return [
        {"rank": rank, "doc_id": row["doc_id"], "score": row["score"]}
        for rank, row in enumerate(rows, start=1)
    ]


def graded_gain(relevance: int) -> float:
    if relevance not in {0, 1, 2, 3}:
        raise ValueError("relevance must be in 0..3")
    return float((1 << relevance) - 1)


def dcg_at_k(relevances: Sequence[int], k: int = DEFAULT_CUTOFF) -> float:
    if k <= 0:
        raise ValueError("NDCG cutoff must be positive")
    return sum(
        graded_gain(int(relevance)) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances[:k], start=1)
    )


def evaluate_ranking(
    ranked_doc_ids: Sequence[str],
    judgments: Mapping[str, int],
    *,
    k: int = DEFAULT_CUTOFF,
) -> dict:
    """Evaluate one query; unranked relevant judgments remain in the ideal DCG."""
    if k <= 0:
        raise ValueError("NDCG cutoff must be positive")
    if len(ranked_doc_ids) != len(set(ranked_doc_ids)):
        raise ValueError("ranking contains duplicate documents")
    unknown = sorted(set(ranked_doc_ids) - set(judgments))
    if unknown:
        raise ValueError(f"ranking contains unjudged documents: {unknown}")

    ranked = [str(doc_id) for doc_id in ranked_doc_ids]
    top_k = ranked[:k]
    top_k_relevances = [int(judgments[doc_id]) for doc_id in top_k]
    dcg = dcg_at_k(top_k_relevances, k)
    ideal_relevances = sorted((int(value) for value in judgments.values()), reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    ndcg = dcg / idcg if idcg > 0.0 else 0.0

    relevant_docs = {doc_id for doc_id, relevance in judgments.items() if relevance > 0}
    ranked_set = set(ranked)
    unranked_relevant = sorted(relevant_docs - ranked_set)
    beyond_cutoff = [doc_id for doc_id in ranked[k:] if judgments[doc_id] > 0]
    return {
        "dcg_at_5": dcg,
        "idcg_at_5": idcg,
        "ndcg_at_5": ndcg,
        "judged_relevant_documents": len(relevant_docs),
        "ranked_relevant_documents": [
            doc_id for doc_id in ranked if judgments[doc_id] > 0
        ],
        "top_5_relevances": top_k_relevances,
        "unranked_relevant_doc_ids": unranked_relevant,
        "beyond_cutoff_relevant_doc_ids": beyond_cutoff,
    }


def fusion_specs() -> dict[str, dict]:
    return {
        "F1": {
            "condition_id": f"{DEFAULT_CELL}-F1",
            "name": "reranker-raw",
            "aef_weight": 0.0,
            "reranker_weight": 1.0,
            "score_transform": "identity",
            "pooling": "max_chunk_score_per_document",
            "tie_break": "score_desc_then_doc_id_asc",
        },
        "F2": {
            "condition_id": f"{DEFAULT_CELL}-F2",
            "name": "equal-fusion-0.50",
            "aef_weight": F2_AEF_WEIGHT,
            "reranker_weight": F2_RERANKER_WEIGHT,
            "score_transform": "identity",
            "pooling": "max_chunk_score_per_document_per_branch_then_weighted_sum",
            "tie_break": "score_desc_then_doc_id_asc",
        },
    }


def _augment_ranking(
    ranking: Sequence[dict],
    judgments: Mapping[str, int],
    aef_pool: Mapping[str, dict] | None = None,
    rerank_pool: Mapping[str, dict] | None = None,
) -> list[dict]:
    rows = []
    for item in ranking:
        doc_id = item["doc_id"]
        row = {
            "rank": item["rank"],
            "doc_id": doc_id,
            "score": item["score"],
            "relevance": int(judgments[doc_id]),
        }
        if rerank_pool is not None:
            row["reranker_best_chunk_id"] = rerank_pool[doc_id]["best_chunk_id"]
        if aef_pool is not None:
            row["aef_best_chunk_id"] = aef_pool[doc_id]["best_chunk_id"]
            row["aef_score"] = aef_pool[doc_id]["score"]
            row["reranker_score"] = rerank_pool[doc_id]["score"]
        rows.append(row)
    return rows


def _macro(queries: Sequence[dict]) -> dict:
    values = [float(query["ndcg_at_5"]) for query in queries]
    if not values:
        raise ValueError("query macro requires at least one query")
    return {
        "query_count": len(values),
        "ndcg_at_5": sum(values) / len(values),
        "minimum_ndcg_at_5": min(values),
        "maximum_ndcg_at_5": max(values),
    }


def evaluate_f1_f2(
    records: Sequence[dict],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    k: int = DEFAULT_CUTOFF,
) -> dict[str, dict]:
    query_ids = [record.get("query", {}).get("query_id") for record in records]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("score cache contains duplicate query IDs")
    if set(query_ids) != set(qrels):
        raise ValueError("score-cache and qrels query sets differ")

    specs = fusion_specs()
    conditions = {
        fusion_id: {**spec, "queries": []}
        for fusion_id, spec in specs.items()
    }
    for record in records:
        query_id = record["query"]["query_id"]
        judgments = qrels[query_id]
        candidates = record.get("candidates") or []
        if not candidates:
            raise ValueError(f"{query_id} has no candidate chunks")
        aef_pool = max_pool_candidates(candidates, "aef_score")
        rerank_pool = max_pool_candidates(candidates, "rerank_score")
        if set(aef_pool) != set(rerank_pool):
            raise ValueError(f"{query_id} AEF and reranker document pools differ")

        f1_scores = {doc_id: item["score"] for doc_id, item in rerank_pool.items()}
        f2_scores = {
            doc_id: (
                F2_AEF_WEIGHT * aef_pool[doc_id]["score"]
                + F2_RERANKER_WEIGHT * rerank_pool[doc_id]["score"]
            )
            for doc_id in sorted(aef_pool)
        }
        for fusion_id, scores, with_aef in (
            ("F1", f1_scores, False),
            ("F2", f2_scores, True),
        ):
            ranking = rank_document_scores(scores)
            metrics = evaluate_ranking(
                [row["doc_id"] for row in ranking],
                judgments,
                k=k,
            )
            conditions[fusion_id]["queries"].append(
                {
                    "query_id": query_id,
                    **metrics,
                    "ranked_documents": _augment_ranking(
                        ranking,
                        judgments,
                        aef_pool=aef_pool if with_aef else None,
                        rerank_pool=rerank_pool,
                    ),
                }
            )

    for condition in conditions.values():
        condition["queries"].sort(key=lambda row: row["query_id"])
        condition["macro"] = _macro(condition["queries"])
    return conditions


def assemble_report(
    *,
    conditions: dict[str, dict],
    qrels_rows: Sequence[dict],
    qrels_path: Path,
    qrels_manifest_path: Path,
    score_cache_dir: Path,
    score_manifest: dict,
    script_path: Path,
    project_root: Path,
    run_date: str,
) -> dict:
    distribution = Counter(str(row["relevance"]) for row in qrels_rows)

    def display(path: Path) -> str:
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(project_root.resolve()))
        except ValueError:
            return str(resolved)

    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed",
        "run_date": run_date,
        "scope": {
            "gate": "retrieval stage development evaluation",
            "development_only": True,
            "condition_ids": [conditions["F1"]["condition_id"], conditions["F2"]["condition_id"]],
            "forty_five_condition_matrix_executed": False,
            "f5_weights_generated": False,
            "confirmation_set_accessed": False,
        },
        "inputs": {
            "qrels": {
                "path": display(qrels_path),
                "sha256": sha256_file(qrels_path),
            },
            "qrels_manifest": {
                "path": display(qrels_manifest_path),
                "sha256": sha256_file(qrels_manifest_path),
            },
            "score_cache_records": {
                "path": display(score_cache_dir / "records.jsonl"),
                "sha256": sha256_file(score_cache_dir / "records.jsonl"),
            },
            "score_cache_manifest": {
                "path": display(score_cache_dir / "manifest.json"),
                "sha256": sha256_file(score_cache_dir / "manifest.json"),
                "records_sha256": score_manifest.get("records_sha256"),
            },
            "evaluator_script": {
                "path": display(script_path),
                "sha256": sha256_file(script_path),
            },
        },
        "qrels_summary": {
            "queries": len({row["query_id"] for row in qrels_rows}),
            "documents_per_query": EXPECTED_DOCUMENTS_PER_QUERY,
            "pairs": len(qrels_rows),
            "relevance_distribution": {
                relevance: distribution.get(relevance, 0) for relevance in ("0", "1", "2", "3")
            },
        },
        "metric_definition": {
            "metric": "query_macro_ndcg@5",
            "cutoff": DEFAULT_CUTOFF,
            "gain": "2^relevance - 1",
            "discount": "log2(rank + 1)",
            "ideal_ranking": "all judged documents sorted by relevance, including unranked relevant documents",
            "no_relevant_documents": "NDCG is 0.0 when ideal DCG is zero",
            "query_macro": "unweighted arithmetic mean over D01-D08",
            "document_tie_break": "score descending, then doc_id ascending",
            "chunk_tie_break": "max score; equal scores use lexicographically smaller chunk_id",
        },
        "conditions": conditions,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# retrieval stage E1-R1 F1/F2 Evaluation",
        "",
        f"- Status: `{report['status']}`",
        f"- Qrels: {report['qrels_summary']['pairs']} pairs across "
        f"{report['qrels_summary']['queries']} development queries",
        "- Scope: development only; no 45-condition matrix, F5 weights, or confirmation-set data",
        "",
        "## Macro NDCG@5",
        "",
        "| Condition | Fusion | Query-macro NDCG@5 |",
        "|---|---|---:|",
    ]
    for fusion_id in ("F1", "F2"):
        condition = report["conditions"][fusion_id]
        lines.append(
            f"| {condition['condition_id']} | {condition['name']} | "
            f"{condition['macro']['ndcg_at_5']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Per-query Metrics",
            "",
            "| Query | F1 NDCG@5 | F1 unranked relevant | F2 NDCG@5 | F2 unranked relevant |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    f1_by_query = {
        row["query_id"]: row for row in report["conditions"]["F1"]["queries"]
    }
    f2_by_query = {
        row["query_id"]: row for row in report["conditions"]["F2"]["queries"]
    }
    for query_id in EXPECTED_QUERY_IDS:
        f1 = f1_by_query[query_id]
        f2 = f2_by_query[query_id]
        lines.append(
            f"| {query_id} | {f1['ndcg_at_5']:.6f} | "
            f"{len(f1['unranked_relevant_doc_ids'])} | {f2['ndcg_at_5']:.6f} | "
            f"{len(f2['unranked_relevant_doc_ids'])} |"
        )

    definition = report["metric_definition"]
    lines.extend(
        [
            "",
            "## Definitions",
            "",
            f"- Gain: `{definition['gain']}`; discount: `{definition['discount']}`.",
            f"- Ideal ranking: {definition['ideal_ranking']}.",
            f"- Document ties: {definition['document_tie_break']}.",
            f"- Chunk ties: {definition['chunk_tie_break']}.",
            "- F1 uses raw document-max reranker scores.",
            "- F2 uses `0.5 * document-max AEF + 0.5 * document-max reranker`, "
            "with identity score transforms.",
            "",
        ]
    )
    return "\n".join(lines)


def render_tsv(report: dict) -> str:
    header = [
        "condition_id",
        "query_id",
        "ndcg_at_5",
        "dcg_at_5",
        "idcg_at_5",
        "judged_relevant_documents",
        "unranked_relevant_documents",
        "top_1_doc_id",
        "top_1_relevance",
        "top_1_score",
    ]
    rows = ["\t".join(header)]
    for fusion_id in ("F1", "F2"):
        condition = report["conditions"][fusion_id]
        for query in condition["queries"]:
            top = query["ranked_documents"][0]
            rows.append(
                "\t".join(
                    [
                        condition["condition_id"],
                        query["query_id"],
                        f"{query['ndcg_at_5']:.12f}",
                        f"{query['dcg_at_5']:.12f}",
                        f"{query['idcg_at_5']:.12f}",
                        str(query["judged_relevant_documents"]),
                        str(len(query["unranked_relevant_doc_ids"])),
                        top["doc_id"],
                        str(top["relevance"]),
                        f"{top['score']:.12f}",
                    ]
                )
            )
    return "\n".join(rows) + "\n"


def write_report_bundle(report: dict, output_dir: Path) -> dict:
    run_date = report["run_date"]
    stem = f"E1-R1-F1-F2-{run_date}"
    json_path = output_dir / f"{stem}-REPORT.json"
    markdown_path = output_dir / f"{stem}-REPORT.md"
    tsv_path = output_dir / f"{stem}-PER-QUERY.tsv"
    manifest_path = output_dir / f"{stem}-MANIFEST.json"

    write_json(json_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    tsv_path.write_text(render_tsv(report), encoding="utf-8")
    output_files = {}
    for path in (json_path, markdown_path, tsv_path):
        output_files[path.name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": OUTPUT_MANIFEST_SCHEMA,
        "status": "passed",
        "run_date": run_date,
        "scope": "development E1-R1 F1/F2 evaluation only",
        "report_sha256": output_files[json_path.name]["sha256"],
        "outputs": output_files,
        "input_hashes": {
            "qrels_sha256": report["inputs"]["qrels"]["sha256"],
            "qrels_manifest_sha256": report["inputs"]["qrels_manifest"]["sha256"],
            "score_cache_records_sha256": report["inputs"]["score_cache_records"]["sha256"],
            "score_cache_manifest_sha256": report["inputs"]["score_cache_manifest"]["sha256"],
            "evaluator_script_sha256": report["inputs"]["evaluator_script"]["sha256"],
        },
    }
    write_json(manifest_path, manifest)
    return {
        "report": json_path,
        "markdown": markdown_path,
        "per_query": tsv_path,
        "manifest": manifest_path,
        "outputs": output_files,
        "manifest_sha256": sha256_file(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--qrels", type=Path)
    parser.add_argument("--qrels-manifest", type=Path)
    parser.add_argument("--score-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    qrels_path = (
        args.qrels or root / "experiments/results/gate5-dev-qrels-20261102/final-20261103/dev-qrels-final.tsv"
    ).resolve()
    qrels_manifest_path = (
        args.qrels_manifest
        or root
        / "experiments/results/gate5-dev-qrels-20261102/final-20261103/GATE5-DEV-QRELS-FINAL-MANIFEST-20261103.json"
    ).resolve()
    score_cache_dir = (
        args.score_cache or root / "experiments/results/gate5-execution-v2/score-cache/E1-R1"
    ).resolve()
    output_dir = (
        args.output_dir or root / "experiments/results/gate5-execution-v2/evaluation"
    ).resolve()

    qrels_rows, _ = load_frozen_qrels(qrels_path, qrels_manifest_path)
    qrels = group_qrels(qrels_rows)
    records, score_manifest = load_score_cache(score_cache_dir)
    conditions = evaluate_f1_f2(records, qrels)
    run_date = str(score_manifest["run_date"])
    report = assemble_report(
        conditions=conditions,
        qrels_rows=qrels_rows,
        qrels_path=qrels_path,
        qrels_manifest_path=qrels_manifest_path,
        score_cache_dir=score_cache_dir,
        score_manifest=score_manifest,
        script_path=Path(__file__).resolve(),
        project_root=root,
        run_date=run_date,
    )
    bundle = write_report_bundle(report, output_dir)
    print(
        json.dumps(
            {
                "status": report["status"],
                "macro_ndcg_at_5": {
                    fusion_id: conditions[fusion_id]["macro"]["ndcg_at_5"]
                    for fusion_id in ("F1", "F2")
                },
                "report": str(bundle["report"]),
                "report_sha256": bundle["outputs"][bundle["report"].name]["sha256"],
                "manifest": str(bundle["manifest"]),
                "manifest_sha256": bundle["manifest_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
