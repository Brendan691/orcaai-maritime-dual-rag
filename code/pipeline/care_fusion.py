#!/usr/bin/env python3
"""Deterministic smooth-CARE and routed-CARE fusion from retrieval stage caches."""

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
PROJECT_ROOT = SCRIPTS_DIR.parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import embedding_caches as embedding_caches  # noqa: E402
import graded_evaluation as evaluation  # noqa: E402
import rerank_caches as rerank_caches  # noqa: E402


RANKINGS_SCHEMA = "gate5-care-f3-f4-rankings-v1"
REPORT_SCHEMA = "gate5-care-f3-f4-report-v1"
OUTPUT_MANIFEST_SCHEMA = "gate5-care-f3-f4-output-manifest-v1"
RUN_DATE = "2026-11-04"
FUSION_IDS = ("F3", "F4")
FUSION_NAMES = {"F3": "smooth-CARE", "F4": "routed-CARE"}
GATE_FEATURES = ("baseline_margin", "gte_margin", "top1_agreement")
EXPECTED_GATE_SHA256 = "c6f1c1efa52a894e1d9696de7ffc8b1c0a4236db523f52b24ab0e0698de579ce"
EXPECTED_ROUTE_THRESHOLD = 0.025
ROUTE_RULE = "top1_disagreement_and_baseline_advantage"
RERANK_RECIPROCAL_WEIGHT = 0.4
RERANK_SCORE_WEIGHT = 0.6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def artifact_entry(path: Path, project_root: Path, **extra: Any) -> dict:
    if not path.is_file():
        raise ValueError(f"missing artifact: {path}")
    return {
        "path": display_path(path, project_root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        **extra,
    }


def load_frozen_gate(
    path: Path,
    *,
    expected_sha256: str | None = EXPECTED_GATE_SHA256,
) -> dict:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("frozen CARE gate hash mismatch")
    gate = read_json(path)
    if gate.get("form") != "logistic":
        raise ValueError("frozen CARE gate must use the logistic form")
    if tuple(gate.get("features", ())) != GATE_FEATURES:
        raise ValueError("frozen CARE gate feature order drift")
    standardization = gate.get("standardization", {})
    coefficients = gate.get("coefficients", {})
    if tuple(standardization) != GATE_FEATURES:
        raise ValueError("frozen CARE standardization drift")
    for feature in GATE_FEATURES:
        params = standardization[feature]
        for field in ("mean", "scale"):
            value = float(params[field])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {feature} {field}")
        if float(params["scale"]) <= 0.0:
            raise ValueError(f"{feature} scale must be positive")
        if not math.isfinite(float(coefficients[feature])):
            raise ValueError(f"non-finite {feature} coefficient")
    if not math.isfinite(float(coefficients["intercept"])):
        raise ValueError("non-finite gate intercept")
    return gate


def load_route_rule(results_path: Path, implementation_path: Path) -> dict:
    results = read_json(results_path)
    if results.get("status") != "completed":
        raise ValueError("CARE route selection did not complete")
    if results.get("selected_feature_set") != "margin_agreement":
        raise ValueError("CARE route feature set drift")
    if tuple(results.get("feature_names", ())) != GATE_FEATURES:
        raise ValueError("CARE route feature order drift")
    threshold = float(
        results.get("development_only_selection", {}).get(
            "disagreement_threshold",
            float("nan"),
        )
    )
    if not math.isfinite(threshold) or threshold != EXPECTED_ROUTE_THRESHOLD:
        raise ValueError("CARE disagreement route threshold drift")
    if not implementation_path.is_file():
        raise ValueError("CARE route implementation is missing")
    return {
        "rule": ROUTE_RULE,
        "threshold": threshold,
        "condition": "top1_agreement < 0.5 and baseline_margin - rerank_margin >= threshold",
        "fallback": "aef_only",
        "selection_scope": "development_only",
        "source_results": {
            "path": str(results_path),
            "sha256": sha256_file(results_path),
        },
        "source_implementation": {
            "path": str(implementation_path),
            "sha256": sha256_file(implementation_path),
        },
    }


def evaluate_gate(gate: Mapping[str, Any], features: Mapping[str, float]) -> dict:
    if tuple(features) != GATE_FEATURES:
        raise ValueError("CARE gate feature order drift")
    standardized = {}
    logit = float(gate["coefficients"]["intercept"])
    for feature in GATE_FEATURES:
        value = float(features[feature])
        if not math.isfinite(value):
            raise ValueError(f"non-finite CARE feature: {feature}")
        params = gate["standardization"][feature]
        z_value = (value - float(params["mean"])) / float(params["scale"])
        standardized[feature] = z_value
        logit += float(gate["coefficients"][feature]) * z_value
    clipped_logit = max(-30.0, min(30.0, logit))
    weight = 1.0 / (1.0 + math.exp(-clipped_logit))
    return {
        "features": {feature: float(features[feature]) for feature in GATE_FEATURES},
        "standardized_features": standardized,
        "logit": logit,
        "clipped_logit": clipped_logit,
        "weight": weight,
    }


def route_gate_weight(gate_weight: float, features: Mapping[str, float], threshold: float) -> dict:
    if not 0.0 <= gate_weight <= 1.0:
        raise ValueError("smooth CARE weight must be in [0, 1]")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("route threshold must be finite and non-negative")
    agreement = float(features["top1_agreement"])
    baseline_advantage = float(features["baseline_margin"]) - float(features["gte_margin"])
    triggered = agreement < 0.5 and baseline_advantage >= threshold
    return {
        "rule": ROUTE_RULE,
        "threshold": threshold,
        "top1_agreement": agreement,
        "baseline_advantage": baseline_advantage,
        "triggered": triggered,
        "branch": "aef_fallback" if triggered else "smooth_care",
        "effective_weight": 0.0 if triggered else gate_weight,
    }


def branch_margins(ranked: Sequence[dict]) -> tuple[float, float]:
    if not ranked:
        raise ValueError("CARE diagnostics require a non-empty document pool")
    top = float(ranked[0]["score"])
    second = float(ranked[1]["score"]) if len(ranked) > 1 else top
    return top, max(0.0, top - second)


def care_query_result(
    candidates: Sequence[dict],
    gate: Mapping[str, Any],
    route_threshold: float,
) -> tuple[dict[str, float], dict]:
    if not candidates:
        raise ValueError("CARE fusion requires candidate chunks")
    aef_pool = evaluation.max_pool_candidates(candidates, "aef_score")
    rerank_pool = evaluation.max_pool_candidates(candidates, "rerank_score")
    if set(aef_pool) != set(rerank_pool):
        raise ValueError("AEF and reranker document pools differ")

    aef_scores = {doc_id: item["score"] for doc_id, item in aef_pool.items()}
    rerank_scores = {doc_id: item["score"] for doc_id, item in rerank_pool.items()}
    aef_ranking = evaluation.rank_document_scores(aef_scores)
    rerank_ranking = evaluation.rank_document_scores(rerank_scores)
    aef_top, baseline_margin = branch_margins(aef_ranking)
    rerank_top, rerank_margin = branch_margins(rerank_ranking)
    features = {
        "baseline_margin": baseline_margin,
        "gte_margin": rerank_margin,
        "top1_agreement": float(aef_ranking[0]["doc_id"] == rerank_ranking[0]["doc_id"]),
    }
    gate_diagnostics = evaluate_gate(gate, features)
    route_diagnostics = route_gate_weight(
        gate_diagnostics["weight"],
        features,
        route_threshold,
    )
    aef_rank = {row["doc_id"]: row["rank"] for row in aef_ranking}
    rerank_rank = {row["doc_id"]: row["rank"] for row in rerank_ranking}

    def fusion_scores(weight: float) -> dict[str, float]:
        scores = {}
        for doc_id in sorted(aef_pool):
            aef_rr = 1.0 / aef_rank[doc_id]
            rerank_rr = 1.0 / rerank_rank[doc_id]
            scores[doc_id] = (
                (1.0 - weight) * aef_rr
                + weight
                * (
                    RERANK_RECIPROCAL_WEIGHT * rerank_rr
                    + RERANK_SCORE_WEIGHT * rerank_pool[doc_id]["score"]
                )
            )
        return scores

    f3_scores = fusion_scores(float(gate_diagnostics["weight"]))
    f4_scores = fusion_scores(float(route_diagnostics["effective_weight"]))
    diagnostics = {
        "gate": gate_diagnostics,
        "route": route_diagnostics,
        "document_pools": {
            "aef": aef_pool,
            "rerank": rerank_pool,
        },
        "branch_ranks": {
            "aef": aef_rank,
            "rerank": rerank_rank,
        },
        "branch_top_scores": {
            "aef": aef_top,
            "rerank": rerank_top,
        },
    }
    return {"F3": f3_scores, "F4": f4_scores}, diagnostics


def augment_ranking(
    ranking: Sequence[dict],
    judgments: Mapping[str, int],
    diagnostics: Mapping[str, Any],
) -> list[dict]:
    aef_pool = diagnostics["document_pools"]["aef"]
    rerank_pool = diagnostics["document_pools"]["rerank"]
    aef_rank = diagnostics["branch_ranks"]["aef"]
    rerank_rank = diagnostics["branch_ranks"]["rerank"]
    return [
        {
            "rank": row["rank"],
            "doc_id": row["doc_id"],
            "score": row["score"],
            "relevance": int(judgments[row["doc_id"]]),
            "aef_score": aef_pool[row["doc_id"]]["score"],
            "aef_best_chunk_id": aef_pool[row["doc_id"]]["best_chunk_id"],
            "aef_rank": aef_rank[row["doc_id"]],
            "reranker_score": rerank_pool[row["doc_id"]]["score"],
            "reranker_best_chunk_id": rerank_pool[row["doc_id"]]["best_chunk_id"],
            "reranker_rank": rerank_rank[row["doc_id"]],
        }
        for row in ranking
    ]


def build_condition_query(
    *,
    cell_id: str,
    fusion_id: str,
    query: Mapping[str, Any],
    candidates: Sequence[dict],
    judgments: Mapping[str, int],
    gate: Mapping[str, Any],
    route_threshold: float,
    cache_identity: Mapping[str, Any],
    k: int = evaluation.DEFAULT_CUTOFF,
) -> dict:
    scores_by_fusion, diagnostics = care_query_result(candidates, gate, route_threshold)
    ranking = evaluation.rank_document_scores(scores_by_fusion[fusion_id])
    doc_ids = [row["doc_id"] for row in ranking]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError(f"{cell_id}/{query['query_id']} produced duplicate documents")
    metrics = evaluation.evaluate_ranking(doc_ids, judgments, k=k)
    weighted_ranking = augment_ranking(ranking, judgments, diagnostics)
    effective_weight = (
        diagnostics["route"]["effective_weight"]
        if fusion_id == "F4"
        else diagnostics["gate"]["weight"]
    )
    return {
        "condition_id": f"{cell_id}-{fusion_id}",
        "cell_id": cell_id,
        "fusion_id": fusion_id,
        "query_id": str(query["query_id"]),
        "query_sha256": query.get("query_sha256"),
        "gate_weight": diagnostics["gate"]["weight"],
        "effective_weight": effective_weight,
        "gate": diagnostics["gate"],
        "route": diagnostics["route"],
        "cache_identity": dict(cache_identity),
        **metrics,
        "ranked_documents": weighted_ranking,
    }


def macro_metrics(queries: Sequence[dict]) -> dict:
    return evaluation._macro(queries)


def merge_cached_record(candidate_record: dict, rerank_record: dict) -> tuple[dict, dict]:
    scores_by_chunk = {}
    for score in rerank_record.get("scores") or []:
        chunk_id = str(score.get("chunk_id") or "")
        if not chunk_id or chunk_id in scores_by_chunk:
            raise ValueError("rerank scores contain an empty or duplicate chunk ID")
        scores_by_chunk[chunk_id] = float(score["score"])
    merged_candidates = []
    for candidate in candidate_record["candidates"]:
        chunk_id = candidate["chunk_id"]
        if chunk_id not in scores_by_chunk:
            raise ValueError(f"missing rerank score for candidate chunk: {chunk_id}")
        merged_candidates.append({**candidate, "rerank_score": scores_by_chunk[chunk_id]})
    if set(scores_by_chunk) != {candidate["chunk_id"] for candidate in candidate_record["candidates"]}:
        raise ValueError("rerank score chunks do not match the candidate cache")
    cache_identity = {
        "candidate_record_sha256": candidate_record["record_sha256"],
        "candidate_set_sha256": rerank_record["cache_identity"]["candidate_set_sha256"],
        "rerank_record_sha256": rerank_record["record_sha256"],
        "rerank_request_sha256": rerank_record["cache_identity"]["request_sha256"],
        "rerank_cache_key": rerank_record["cache_identity"]["cache_key"],
        "reranker_identity_sha256": rerank_record["cache_identity"][
            "reranker_identity_sha256"
        ],
    }
    return {"query": candidate_record["query"], "candidates": merged_candidates}, cache_identity


def evaluate_cache_bundle(
    *,
    candidate_records: Sequence[dict],
    rerank_groups: Sequence[dict],
    qrels: Mapping[str, Mapping[str, int]],
    gate: Mapping[str, Any],
    route_threshold: float,
    group_hashes: Mapping[str, str],
    k: int = evaluation.DEFAULT_CUTOFF,
) -> dict[str, dict]:
    candidates_by_key = {
        (record["embedding_id"], record["query"]["query_id"]): record
        for record in candidate_records
    }
    records_by_cell: dict[str, list[dict]] = {
        cell_id: [] for cell_id in embedding_caches.EXPECTED_CELL_IDS
    }
    for group in rerank_groups:
        cell_id = group["cell_id"]
        group_hash = group_hashes[group["group_id"]]
        for rerank_record in group["records"]:
            query_id = rerank_record["query"]["query_id"]
            key = (rerank_record["embedding_id"], query_id)
            if key not in candidates_by_key:
                raise ValueError(f"candidate cache is missing fused input: {key}")
            merged, cache_identity = merge_cached_record(
                candidates_by_key[key],
                rerank_record,
            )
            records_by_cell[cell_id].append(
                {
                    "query": merged["query"],
                    "candidates": merged["candidates"],
                    "cache_identity": {
                        **cache_identity,
                        "rerank_group_id": group["group_id"],
                        "rerank_group_sha256": group_hash,
                    },
                }
            )

    conditions = {}
    for cell_id in embedding_caches.EXPECTED_CELL_IDS:
        records = sorted(records_by_cell[cell_id], key=lambda row: row["query"]["query_id"])
        query_ids = [row["query"]["query_id"] for row in records]
        if query_ids != list(embedding_caches.EXPECTED_QUERY_IDS):
            raise ValueError(f"{cell_id} does not cover D01-D08 exactly once")
        for fusion_id in FUSION_IDS:
            condition_id = f"{cell_id}-{fusion_id}"
            query_results = []
            for record in records:
                query_id = record["query"]["query_id"]
                query_results.append(
                    build_condition_query(
                        cell_id=cell_id,
                        fusion_id=fusion_id,
                        query=record["query"],
                        candidates=record["candidates"],
                        judgments=qrels[query_id],
                        gate=gate,
                        route_threshold=route_threshold,
                        cache_identity=record["cache_identity"],
                        k=k,
                    )
                )
            conditions[condition_id] = {
                "condition_id": condition_id,
                "cell_id": cell_id,
                "fusion_id": fusion_id,
                "name": FUSION_NAMES[fusion_id],
                "queries": query_results,
                "macro": macro_metrics(query_results),
            }
    return conditions


def render_rankings(conditions: Mapping[str, dict]) -> list[dict]:
    rows = []
    for condition_id in sorted(conditions):
        condition = conditions[condition_id]
        for query in condition["queries"]:
            rows.append(
                {
                    "schema_version": RANKINGS_SCHEMA,
                    "run_date": RUN_DATE,
                    "condition_id": condition_id,
                    "cell_id": condition["cell_id"],
                    "fusion_id": condition["fusion_id"],
                    "query_id": query["query_id"],
                    "query_sha256": query["query_sha256"],
                    "gate_weight": query["gate_weight"],
                    "effective_weight": query["effective_weight"],
                    "gate": query["gate"],
                    "route": query["route"],
                    "cache_identity": query["cache_identity"],
                    "metrics": {
                        key: query[key]
                        for key in (
                            "dcg_at_5",
                            "idcg_at_5",
                            "ndcg_at_5",
                            "judged_relevant_documents",
                            "ranked_relevant_documents",
                            "top_5_relevances",
                            "unranked_relevant_doc_ids",
                            "beyond_cutoff_relevant_doc_ids",
                        )
                    },
                    "ranked_documents": query["ranked_documents"],
                }
            )
    return rows


def render_tsv(conditions: Mapping[str, dict]) -> str:
    header = [
        "condition_id",
        "cell_id",
        "fusion_id",
        "query_id",
        "ndcg_at_5",
        "gate_weight",
        "effective_weight",
        "route_triggered",
        "top1_agreement",
        "baseline_margin",
        "rerank_margin",
        "baseline_advantage",
        "top_1_doc_id",
    ]
    rows = ["\t".join(header)]
    for condition_id in sorted(conditions):
        condition = conditions[condition_id]
        for query in condition["queries"]:
            route = query["route"]
            features = query["gate"]["features"]
            rows.append(
                "\t".join(
                    [
                        condition_id,
                        condition["cell_id"],
                        condition["fusion_id"],
                        query["query_id"],
                        f"{query['ndcg_at_5']:.12f}",
                        f"{query['gate_weight']:.12f}",
                        f"{query['effective_weight']:.12f}",
                        str(route["triggered"]).lower(),
                        f"{route['top1_agreement']:.12f}",
                        f"{features['baseline_margin']:.12f}",
                        f"{features['gte_margin']:.12f}",
                        f"{route['baseline_advantage']:.12f}",
                        query["ranked_documents"][0]["doc_id"],
                    ]
                )
            )
    return "\n".join(rows) + "\n"


def report_query_summary(query: dict) -> dict:
    return {
        key: value
        for key, value in query.items()
        if key != "ranked_documents"
    } | {"ranked_doc_ids": [row["doc_id"] for row in query["ranked_documents"]]}


def build_report(
    *,
    conditions: Mapping[str, dict],
    qrels_rows: Sequence[dict],
    qrels_path: Path,
    qrels_manifest_path: Path,
    input_artifacts: Mapping[str, dict],
    gate_path: Path,
    gate: Mapping[str, Any],
    route_rule: Mapping[str, Any],
    rerank_manifest: Mapping[str, Any],
    project_root: Path,
    script_path: Path,
) -> dict:
    relevance_distribution = Counter(str(row["relevance"]) for row in qrels_rows)
    routed = {
        condition_id: sum(query["route"]["triggered"] for query in condition["queries"])
        for condition_id, condition in conditions.items()
        if condition["fusion_id"] == "F4"
    }
    weights = [
        query["gate_weight"]
        for condition_id, condition in conditions.items()
        if condition["fusion_id"] == "F3"
        for query in condition["queries"]
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed",
        "run_date": RUN_DATE,
        "scope": {
            "gate": "retrieval stage development F3/F4 fusion",
            "development_only": True,
            "condition_ids": sorted(conditions),
            "condition_count": len(conditions),
            "cells": list(embedding_caches.EXPECTED_CELL_IDS),
            "fusion_ids": list(FUSION_IDS),
            "forty_five_condition_matrix_executed": False,
            "f5_weights_generated": False,
            "confirmation_set_accessed": False,
            "model_calls": 0,
        },
        "fusion_definition": {
            "ranking_unit": "document",
            "pooling": "max_chunk_score_per_document_per_branch",
            "document_tie_break": "score_desc_then_doc_id_asc",
            "chunk_tie_break": "score_desc_then_chunk_id_asc",
            "smooth_care_formula": "(1-w)/rank_aef + w*(0.4/rank_rerank + 0.6*rerank_score)",
            "gate_formula": "sigmoid(clip(intercept + sum(coef_i*z_i), -30, 30))",
            "route_rule": dict(route_rule),
        },
        "frozen_gate": {
            "path": display_path(gate_path, project_root),
            "sha256": sha256_file(gate_path),
            "form": gate["form"],
            "features": list(gate["features"]),
            "standardization": gate["standardization"],
            "coefficients": gate["coefficients"],
        },
        "inputs": {
            **input_artifacts,
            "qrels": artifact_entry(qrels_path, project_root),
            "qrels_manifest": artifact_entry(qrels_manifest_path, project_root),
            "fusion_script": artifact_entry(script_path, project_root),
        },
        "qrels_summary": {
            "queries": len({row["query_id"] for row in qrels_rows}),
            "pairs": len(qrels_rows),
            "relevance_distribution": dict(sorted(relevance_distribution.items())),
        },
        "coverage": {
            "cells": len(embedding_caches.EXPECTED_CELL_IDS),
            "conditions": len(conditions),
            "queries_per_condition": len(embedding_caches.EXPECTED_QUERY_IDS),
            "condition_query_records": sum(len(condition["queries"]) for condition in conditions.values()),
            "f3_gate_weight_min": min(weights),
            "f3_gate_weight_max": max(weights),
            "f4_routed_queries_by_cell": routed,
            "rerank_records": rerank_manifest["counts"]["query_records"],
        },
        "conditions": {
            condition_id: {
                **{key: value for key, value in condition.items() if key != "queries"},
                "queries": [report_query_summary(query) for query in condition["queries"]],
            }
            for condition_id, condition in sorted(conditions.items())
        },
    }


def write_report_bundle(
    *,
    report: Mapping[str, Any],
    conditions: Mapping[str, dict],
    output_dir: Path,
) -> dict:
    stem = f"GATE5-F3-F4-{RUN_DATE}"
    rankings_path = output_dir / f"{stem}-RANKINGS.jsonl"
    report_path = output_dir / f"{stem}-REPORT.json"
    tsv_path = output_dir / f"{stem}-PER-QUERY.tsv"
    manifest_path = output_dir / f"{stem}-MANIFEST.json"

    write_jsonl(rankings_path, render_rankings(conditions))
    write_json(report_path, dict(report))
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    tsv_path.write_text(render_tsv(conditions), encoding="utf-8")

    outputs = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (rankings_path, report_path, tsv_path)
    }
    input_hashes = {}
    for name, artifact in report["inputs"].items():
        if "sha256" in artifact:
            input_hashes[name] = artifact["sha256"]
            continue
        for child_name, child in artifact.items():
            if not isinstance(child, dict) or "sha256" not in child:
                raise ValueError(f"input artifact lacks a hash: {name}.{child_name}")
            input_hashes[f"{name}.{child_name}"] = child["sha256"]
    manifest = {
        "schema_version": OUTPUT_MANIFEST_SCHEMA,
        "status": "passed",
        "run_date": RUN_DATE,
        "scope": "development F3/F4 fusion only",
        "report_sha256": outputs[report_path.name]["sha256"],
        "rankings_sha256": outputs[rankings_path.name]["sha256"],
        "outputs": outputs,
        "input_hashes": input_hashes,
        "checks": {
            "condition_count": len(conditions),
            "conditions_are_f3_f4_only": all(
                condition["fusion_id"] in FUSION_IDS for condition in conditions.values()
            ),
            "unique_document_rankings": True,
            "forty_five_condition_matrix_executed": False,
            "f5_weights_generated": False,
            "confirmation_set_accessed": False,
            "model_calls": 0,
        },
    }
    write_json(manifest_path, manifest)
    return {
        "rankings": rankings_path,
        "report": report_path,
        "per_query": tsv_path,
        "manifest": manifest_path,
        "outputs": outputs,
        "manifest_sha256": sha256_file(manifest_path),
    }


def collect_input_artifacts(
    *,
    project_root: Path,
    execution_root: Path,
    embedding_root: Path,
    rerank_root: Path,
    gate_path: Path,
    route_results_path: Path,
    route_implementation_path: Path,
    rerank_manifest: Mapping[str, Any],
) -> dict:
    artifacts = {
        "execution_manifest": artifact_entry(
            execution_root / "GATE5-EXECUTION-MANIFEST-20261103.json",
            project_root,
        ),
        "execution_guard_report": artifact_entry(
            execution_root / "GATE5-EXECUTION-GUARD-REPORT-20261103.json",
            project_root,
        ),
        "embedding_cache_manifest": artifact_entry(
            embedding_root / "manifest.json",
            project_root,
        ),
        "candidate_manifest": artifact_entry(
            embedding_root / "candidates" / "manifest.json",
            project_root,
        ),
        "candidate_records": artifact_entry(
            embedding_root / "candidates" / "records.jsonl",
            project_root,
        ),
        "rerank_cache_manifest": artifact_entry(
            rerank_root / "manifest.json",
            project_root,
        ),
        "rerank_validation_report": artifact_entry(
            rerank_root / "validation-report.json",
            project_root,
        ),
        "rerank_replay_report": artifact_entry(
            rerank_root / "replay-report.json",
            project_root,
        ),
        "frozen_gate": artifact_entry(gate_path, project_root),
        "route_selection_results": artifact_entry(route_results_path, project_root),
        "route_implementation": artifact_entry(route_implementation_path, project_root),
    }
    for embedding_id in embedding_caches.EMBEDDING_IDS:
        path = embedding_root / embedding_id / "manifest.json"
        artifacts[f"embedding_manifest_{embedding_id}"] = artifact_entry(path, project_root)
    artifacts["rerank_groups"] = {
        entry["group_id"]: {
            "path": entry["path"],
            "sha256": entry["sha256"],
            "bytes": entry["bytes"],
            "cell_id": entry["cell_id"],
            "query_ids": entry["query_ids"],
        }
        for entry in rerank_manifest["groups"]
    }
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--embedding-cache-root", type=Path)
    parser.add_argument("--rerank-cache-root", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--route-results", type=Path)
    parser.add_argument("--route-implementation", type=Path)
    parser.add_argument("--qrels", type=Path)
    parser.add_argument("--qrels-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    execution_root = root / "experiments/results/gate5-execution-v2"
    embedding_root = (args.embedding_cache_root or execution_root / "embedding-caches").resolve()
    rerank_root = (args.rerank_cache_root or execution_root / "rerank-caches").resolve()
    gate_path = (
        args.gate
        or root / "experiments/results/gate-p-readiness-20261017/gate-weights-v2-20261023.json"
    ).resolve()
    route_results_path = (
        args.route_results
        or root / "experiments/results/care-router-20261009/main-cv-stable/results.json"
    ).resolve()
    route_implementation_path = (
        args.route_implementation
        or root / "experiments/scripts/develop_care_router.py"
    ).resolve()
    qrels_path = (
        args.qrels
        or root / "experiments/results/gate5-dev-qrels-20261102/final-20261103/dev-qrels-final.tsv"
    ).resolve()
    qrels_manifest_path = (
        args.qrels_manifest
        or root
        / "experiments/results/gate5-dev-qrels-20261102/final-20261103/GATE5-DEV-QRELS-FINAL-MANIFEST-20261103.json"
    ).resolve()
    output_dir = (args.output_dir or execution_root / "evaluation").resolve()

    gate = load_frozen_gate(gate_path)
    route_rule = load_route_rule(route_results_path, route_implementation_path)
    qrels_rows, _ = evaluation.load_frozen_qrels(qrels_path, qrels_manifest_path)
    qrels = evaluation.group_qrels(qrels_rows)
    _, _, candidate_records = rerank_caches.load_candidate_cache(embedding_root)
    groups, rerank_manifest = rerank_caches.validate_bundle(
        output_dir=rerank_root,
        project_root=root,
        candidate_records=candidate_records,
    )
    if rerank_manifest.get("status") != "passed":
        raise ValueError("rerank cache bundle did not pass")
    group_hashes = {group["group_id"]: group["group_sha256"] for group in groups}
    conditions = evaluate_cache_bundle(
        candidate_records=candidate_records,
        rerank_groups=groups,
        qrels=qrels,
        gate=gate,
        route_threshold=float(route_rule["threshold"]),
        group_hashes=group_hashes,
    )
    input_artifacts = collect_input_artifacts(
        project_root=root,
        execution_root=execution_root,
        embedding_root=embedding_root,
        rerank_root=rerank_root,
        gate_path=gate_path,
        route_results_path=route_results_path,
        route_implementation_path=route_implementation_path,
        rerank_manifest=rerank_manifest,
    )
    report = build_report(
        conditions=conditions,
        qrels_rows=qrels_rows,
        qrels_path=qrels_path,
        qrels_manifest_path=qrels_manifest_path,
        input_artifacts=input_artifacts,
        gate_path=gate_path,
        gate=gate,
        route_rule=route_rule,
        rerank_manifest=rerank_manifest,
        project_root=root,
        script_path=Path(__file__).resolve(),
    )
    bundle = write_report_bundle(report=report, conditions=conditions, output_dir=output_dir)
    print(
        json.dumps(
            {
                "status": report["status"],
                "conditions": len(conditions),
                "rankings": str(bundle["rankings"]),
                "rankings_sha256": bundle["outputs"][bundle["rankings"].name]["sha256"],
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
