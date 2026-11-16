#!/usr/bin/env python3
"""Schema, validation, and replay helpers for retrieval stage rerank score caches."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import embedding_caches as embedding_caches
from provider_cache import normalize_text


RERANK_CACHE_SCHEMA = "gate5-rerank-score-cache-v1"
RERANK_GROUP_SCHEMA = "gate5-rerank-cache-group-v1"
RERANK_MANIFEST_SCHEMA = "gate5-rerank-caches-manifest-v1"
RERANK_VALIDATION_SCHEMA = "gate5-rerank-cache-validation-v1"
RERANK_REPLAY_SCHEMA = "gate5-rerank-cache-replay-v1"

EXPECTED_QUERY_IDS = embedding_caches.EXPECTED_QUERY_IDS
EMBEDDING_IDS = embedding_caches.EMBEDDING_IDS
RERANKER_IDS = embedding_caches.RERANKER_IDS
EXPECTED_CELL_IDS = embedding_caches.EXPECTED_CELL_IDS
CONFIRMATION_QUERY_IDS = embedding_caches.CONFIRMATION_QUERY_IDS
CANDIDATE_COUNT = 20

# Ticket 04 names 27 cache groups. Each E/R cell therefore has three fixed
# query shards, while every one of the 72 cell-query combinations remains a
# separate record with its own identity and hashes.
QUERY_SHARDS = (
    ("G1", ("D01", "D02", "D03")),
    ("G2", ("D04", "D05", "D06")),
    ("G3", ("D07", "D08")),
)
EXPECTED_GROUP_IDS = tuple(
    f"{cell_id}-{group_id}"
    for cell_id in EXPECTED_CELL_IDS
    for group_id, _ in QUERY_SHARDS
)

RERANKER_CONTRACT = {
    "R1": {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "runtime": "local",
        "backend": "sequence-classification-cross-encoder",
        "max_length": 512,
        "model_weight_sha256": "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286",
    },
    "R2": {
        "model_id": "mixedbread-ai/mxbai-rerank-large-v2",
        "runtime": "local",
        "backend": "sentence-transformers-generative-cross-encoder",
        "max_length": 512,
        "model_weight_sha256": "430e47b1754d2306fce32e560d3386a6a79850a91e5adf78d51fba9647022730",
    },
    "R3": {
        "model_id": "gte-rerank-v2",
        "runtime": "dashscope",
        "backend": "dashscope-text-rerank-api",
        "max_length": None,
        "model_weight_sha256": None,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
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


def records_sha256(records: Iterable[dict]) -> str:
    return sha256_value(list(records))


def group_path(output_dir: Path, cell_id: str, shard_id: str) -> Path:
    return output_dir / "cache-groups" / cell_id / f"{shard_id}.json"


def cell_id_for(embedding_id: str, reranker_id: str) -> str:
    return f"{embedding_id}-{reranker_id}"


def query_shard(shard_id: str) -> tuple[str, ...]:
    for expected_id, query_ids in QUERY_SHARDS:
        if expected_id == shard_id:
            return query_ids
    raise ValueError(f"unknown query shard: {shard_id}")


def load_candidate_cache(embedding_root: Path) -> tuple[dict, dict, list[dict]]:
    bundle_path = embedding_root / "manifest.json"
    candidate_manifest_path = embedding_root / "candidates" / "manifest.json"
    records_path = embedding_root / "candidates" / "records.jsonl"
    for path in (bundle_path, candidate_manifest_path, records_path):
        if not path.is_file():
            raise ValueError(f"embedding/candidate cache is incomplete: {path}")

    bundle = read_json(bundle_path)
    candidate_manifest = read_json(candidate_manifest_path)
    records = read_jsonl(records_path)

    if bundle.get("status") != "passed" or bundle.get("run_scope") != "development_only":
        raise ValueError("embedding cache bundle status or scope drift")
    if candidate_manifest.get("status") != "passed":
        raise ValueError("candidate cache manifest did not pass")
    if candidate_manifest.get("outputs", {}).get("records", {}).get("sha256") != sha256_file(
        records_path
    ):
        raise ValueError("candidate records file hash drift")
    if len(records) != len(EMBEDDING_IDS) * len(EXPECTED_QUERY_IDS):
        raise ValueError("candidate cache must contain 24 embedding-query records")

    expected_pairs = {
        (embedding_id, query_id)
        for embedding_id in EMBEDDING_IDS
        for query_id in EXPECTED_QUERY_IDS
    }
    actual_pairs: set[tuple[str, str]] = set()
    for record in records:
        embedding_id = str(record.get("embedding_id") or "")
        query = record.get("query", {})
        query_id = str(query.get("query_id") or "")
        pair = (embedding_id, query_id)
        if pair in actual_pairs:
            raise ValueError(f"duplicate candidate cache key: {pair}")
        actual_pairs.add(pair)
        if query_id in CONFIRMATION_QUERY_IDS:
            raise ValueError("confirmation query leaked into candidate cache")
        if sha256_text(str(query.get("text") or "")) != query.get("query_sha256"):
            raise ValueError(f"candidate query hash drift: {pair}")
        candidates = record.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != CANDIDATE_COUNT:
            raise ValueError(f"candidate count drift: {pair}")
        if len({candidate.get("chunk_index") for candidate in candidates}) != CANDIDATE_COUNT:
            raise ValueError(f"duplicate candidate index: {pair}")
        if len({candidate.get("chunk_id") for candidate in candidates}) != CANDIDATE_COUNT:
            raise ValueError(f"duplicate candidate chunk: {pair}")
        record_without_hash = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        if record.get("record_sha256") != sha256_value(record_without_hash):
            raise ValueError(f"candidate record hash drift: {pair}")
    if actual_pairs != expected_pairs:
        raise ValueError("candidate cache does not cover E1/E2/E3 x D01-D08")
    return bundle, candidate_manifest, records


def make_reranker_identity(
    reranker_id: str,
    *,
    model_path: Path | None,
    provider_endpoint: str | None = None,
) -> dict:
    if reranker_id not in RERANKER_CONTRACT:
        raise ValueError(f"unknown reranker: {reranker_id}")
    contract = RERANKER_CONTRACT[reranker_id]
    identity = {
        "id": reranker_id,
        **copy.deepcopy(contract),
        "model_path": str(model_path) if model_path is not None else None,
        "provider": "dashscope" if reranker_id == "R3" else None,
        "provider_endpoint": provider_endpoint,
        "backend_verified": True,
    }
    identity["identity_sha256"] = sha256_value(identity)
    return identity


def normalized_request(query: str, documents: list[str]) -> dict:
    return {
        "query": normalize_text(query),
        "documents": [normalize_text(document) for document in documents],
        "parameters": {
            "top_n": len(documents),
            "return_documents": False,
        },
    }


def make_cache_identity(
    *,
    cell_id: str,
    candidate_record: dict,
    reranker_identity: dict,
    raw_request: dict,
    provider_cache_key: str | None = None,
) -> dict:
    query = candidate_record["query"]
    query_id = query["query_id"]
    candidates = candidate_record["candidates"]
    candidate_set_sha256 = sha256_value(candidates)
    request_sha256 = sha256_value(raw_request)
    payload = {
        "schema_version": RERANK_CACHE_SCHEMA,
        "cell_id": cell_id,
        "embedding_id": candidate_record["embedding_id"],
        "reranker_id": reranker_identity["id"],
        "query_id": query_id,
        "query_sha256": query["query_sha256"],
        "candidate_record_sha256": candidate_record["record_sha256"],
        "candidate_set_sha256": candidate_set_sha256,
        "reranker_identity_sha256": reranker_identity["identity_sha256"],
        "request_sha256": request_sha256,
        "provider_cache_key": provider_cache_key,
    }
    return {
        "schema_version": RERANK_CACHE_SCHEMA,
        "cell_id": cell_id,
        "embedding_id": candidate_record["embedding_id"],
        "reranker_id": reranker_identity["id"],
        "query_id": query_id,
        "query_sha256": query["query_sha256"],
        "candidate_record_sha256": candidate_record["record_sha256"],
        "candidate_set_sha256": candidate_set_sha256,
        "reranker_identity_sha256": reranker_identity["identity_sha256"],
        "request_sha256": request_sha256,
        "provider_cache_key": provider_cache_key,
        "cache_key": sha256_value(payload),
        "identity_payload_sha256": sha256_value(payload),
    }


def parse_raw_response(raw_response: Any, expected_count: int) -> list[dict]:
    if not isinstance(raw_response, dict):
        raise ValueError("rerank raw response is not an object")
    results = raw_response.get("results")
    if results is None:
        results = (raw_response.get("output") or {}).get("results")
    if not isinstance(results, list):
        raise ValueError("rerank raw response has no results list")
    if len(results) != expected_count:
        raise ValueError(
            f"rerank raw response has {len(results)} results, expected {expected_count}"
        )
    parsed: list[dict] = []
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("rerank raw response item is not an object")
        index = item.get("index")
        score = item.get("relevance_score")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("rerank raw response index is not an integer")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("rerank raw response score is not numeric")
        parsed.append({"request_index": index, "rerank_score": float(score)})
    indexes = sorted(item["request_index"] for item in parsed)
    if indexes != list(range(expected_count)):
        raise ValueError("rerank raw response indexes are incomplete or duplicated")
    if not all(math.isfinite(item["rerank_score"]) for item in parsed):
        raise ValueError("rerank raw response contains a non-finite score")
    return sorted(parsed, key=lambda item: item["request_index"])


def parsed_scores(parsed_response: list[dict], candidates: list[dict]) -> list[dict]:
    return [
        {
            "request_index": row["request_index"],
            "chunk_index": candidates[row["request_index"]]["chunk_index"],
            "chunk_id": candidates[row["request_index"]]["chunk_id"],
            "score": row["rerank_score"],
        }
        for row in parsed_response
    ]


def make_score_record(
    *,
    group_id: str,
    cell_id: str,
    candidate_record: dict,
    reranker_identity: dict,
    raw_request: dict,
    raw_response: dict | None,
    latency_ms: dict[str, float],
    cache: dict,
    failure: dict | None = None,
) -> dict:
    query = candidate_record["query"]
    candidates = copy.deepcopy(candidate_record["candidates"])
    cache_identity = make_cache_identity(
        cell_id=cell_id,
        candidate_record=candidate_record,
        reranker_identity=reranker_identity,
        raw_request=raw_request,
        provider_cache_key=cache.get("provider_cache_key"),
    )
    record = {
        "schema_version": RERANK_CACHE_SCHEMA,
        "run_scope": "development_only",
        "group_id": group_id,
        "cell_id": cell_id,
        "embedding_id": candidate_record["embedding_id"],
        "reranker_id": reranker_identity["id"],
        "query": copy.deepcopy(query),
        "reranker": copy.deepcopy(reranker_identity),
        "candidates": candidates,
        "raw_request": copy.deepcopy(raw_request),
        "cache": copy.deepcopy(cache),
        "latency_ms": {key: float(value) for key, value in latency_ms.items()},
        "cache_identity": cache_identity,
    }
    if failure is None:
        parsed_response = parse_raw_response(raw_response, len(candidates))
        scores = parsed_scores(parsed_response, candidates)
        record.update(
            {
                "status": "passed",
                "raw_response": copy.deepcopy(raw_response),
                "parsed_response": parsed_response,
                "scores": scores,
                "failure": {
                    "failed": False,
                    "category": None,
                    "type": None,
                    "message": None,
                    "status_code": None,
                    "response_body": None,
                },
            }
        )
    else:
        record.update(
            {
                "status": "failed",
                "raw_response": copy.deepcopy(raw_response),
                "parsed_response": None,
                "scores": None,
                "failure": {
                    "failed": True,
                    "category": str(failure.get("category") or "rerank_failure"),
                    "type": failure.get("type"),
                    "message": str(failure.get("message") or "rerank failed"),
                    "status_code": failure.get("status_code"),
                    "response_body": failure.get("response_body"),
                },
            }
        )
    record["raw_request_sha256"] = sha256_value(record["raw_request"])
    record["raw_response_sha256"] = (
        sha256_value(record["raw_response"]) if record["raw_response"] is not None else None
    )
    record["parsed_response_sha256"] = (
        sha256_value(record["parsed_response"])
        if record["parsed_response"] is not None
        else None
    )
    record["record_sha256"] = sha256_value(record)
    return record


def make_group(
    *,
    group_id: str,
    cell_id: str,
    records: list[dict],
) -> dict:
    embedding_id, reranker_id = cell_id.split("-", 1)
    query_ids = [record["query"]["query_id"] for record in records]
    expected_shard_id = group_id.rsplit("-", 1)[-1]
    if query_ids != list(query_shard(expected_shard_id)):
        raise ValueError(f"query shard coverage drift for {group_id}")
    status = "passed" if all(record.get("status") == "passed" for record in records) else "failed"
    group = {
        "schema_version": RERANK_GROUP_SCHEMA,
        "status": status,
        "run_scope": "development_only",
        "group_id": group_id,
        "cell_id": cell_id,
        "embedding_id": embedding_id,
        "reranker_id": reranker_id,
        "query_ids": query_ids,
        "record_count": len(records),
        "records_sha256": records_sha256(records),
        "records": records,
    }
    group["group_sha256"] = sha256_value(group)
    return group


def validate_score_record(
    record: dict,
    *,
    expected_group_id: str,
    expected_cell_id: str,
    expected_query_id: str,
    candidate_record: dict,
    reranker_identity: dict,
) -> None:
    if record.get("schema_version") != RERANK_CACHE_SCHEMA:
        raise ValueError("rerank cache schema drift")
    if record.get("run_scope") != "development_only":
        raise ValueError("rerank cache scope drift")
    if expected_query_id in CONFIRMATION_QUERY_IDS:
        raise ValueError("confirmation query leaked into rerank cache")
    for field, expected in (
        ("group_id", expected_group_id),
        ("cell_id", expected_cell_id),
        ("embedding_id", candidate_record["embedding_id"]),
        ("reranker_id", reranker_identity["id"]),
    ):
        if record.get(field) != expected:
            raise ValueError(f"rerank cache {field} drift: {expected_query_id}")
    if record.get("query", {}).get("query_id") != expected_query_id:
        raise ValueError(f"rerank cache query drift: {expected_query_id}")
    if record.get("reranker") != reranker_identity:
        raise ValueError(f"reranker identity drift: {expected_query_id}")

    candidates = record.get("candidates")
    if candidates != candidate_record["candidates"]:
        raise ValueError(f"rerank candidate payload drift: {expected_query_id}")
    if len(candidates) != CANDIDATE_COUNT:
        raise ValueError(f"rerank candidate count drift: {expected_query_id}")
    request_indexes = [candidate.get("rank") for candidate in candidates]
    if request_indexes != list(range(1, CANDIDATE_COUNT + 1)):
        raise ValueError(f"rerank candidate ranks drift: {expected_query_id}")
    chunk_indexes = [candidate.get("chunk_index") for candidate in candidates]
    if len(set(chunk_indexes)) != CANDIDATE_COUNT:
        raise ValueError(f"duplicate rerank candidate indexes: {expected_query_id}")

    expected_identity = make_cache_identity(
        cell_id=expected_cell_id,
        candidate_record=candidate_record,
        reranker_identity=reranker_identity,
        raw_request=record.get("raw_request", {}),
        provider_cache_key=record.get("cache", {}).get("provider_cache_key"),
    )
    if record.get("cache_identity") != expected_identity:
        raise ValueError(f"rerank cache identity drift: {expected_query_id}")
    if record.get("raw_request_sha256") != sha256_value(record.get("raw_request")):
        raise ValueError(f"rerank raw request hash drift: {expected_query_id}")

    failure = record.get("failure", {})
    status = record.get("status")
    if status == "passed":
        if failure.get("failed") is not False:
            raise ValueError(f"passed rerank record has failure flag: {expected_query_id}")
        parsed_response = parse_raw_response(record.get("raw_response"), CANDIDATE_COUNT)
        if record.get("parsed_response") != parsed_response:
            raise ValueError(f"parsed rerank response drift: {expected_query_id}")
        expected_scores = parsed_scores(parsed_response, candidates)
        if record.get("scores") != expected_scores:
            raise ValueError(f"rerank score mapping drift: {expected_query_id}")
        if len(record["scores"]) != CANDIDATE_COUNT:
            raise ValueError(f"rerank score count drift: {expected_query_id}")
        if len({row["request_index"] for row in record["scores"]}) != CANDIDATE_COUNT:
            raise ValueError(f"duplicate rerank score indexes: {expected_query_id}")
        if record.get("raw_response_sha256") != sha256_value(record["raw_response"]):
            raise ValueError(f"raw rerank response hash drift: {expected_query_id}")
        if record.get("parsed_response_sha256") != sha256_value(parsed_response):
            raise ValueError(f"parsed rerank response hash drift: {expected_query_id}")
    elif status == "failed":
        if failure.get("failed") is not True:
            raise ValueError(f"failed rerank record has no failure flag: {expected_query_id}")
        if not failure.get("category") or not failure.get("message"):
            raise ValueError(f"failed rerank record lacks details: {expected_query_id}")
        if record.get("parsed_response") is not None or record.get("scores") is not None:
            raise ValueError(f"failed rerank record contains default scores: {expected_query_id}")
    else:
        raise ValueError(f"invalid rerank record status: {expected_query_id}")

    latency = record.get("latency_ms", {})
    for field in ("rerank", "provider", "cache_lookup"):
        value = latency.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"invalid rerank latency {field}: {expected_query_id}")
    cache = record.get("cache", {})
    if cache.get("mode") not in {"live", "offline"}:
        raise ValueError(f"invalid rerank cache mode: {expected_query_id}")
    if not cache.get("source"):
        raise ValueError(f"rerank cache source is missing: {expected_query_id}")
    if reranker_identity["runtime"] == "dashscope":
        if not cache.get("provider_cache_key") or not cache.get("provider_cache_path"):
            raise ValueError(f"R3 provider cache identity is missing: {expected_query_id}")

    record_without_hash = {
        key: value for key, value in record.items() if key != "record_sha256"
    }
    if record.get("record_sha256") != sha256_value(record_without_hash):
        raise ValueError(f"rerank record hash drift: {expected_query_id}")


def validate_group(
    group: dict,
    *,
    candidate_records: dict[tuple[str, str], dict],
    reranker_identities: dict[str, dict],
) -> None:
    if group.get("schema_version") != RERANK_GROUP_SCHEMA:
        raise ValueError("rerank group schema drift")
    group_id = str(group.get("group_id") or "")
    cell_id = str(group.get("cell_id") or "")
    if group_id not in EXPECTED_GROUP_IDS or cell_id not in EXPECTED_CELL_IDS:
        raise ValueError(f"unexpected rerank group: {group_id}")
    embedding_id, reranker_id = cell_id.split("-", 1)
    shard_id = group_id.rsplit("-", 1)[-1]
    expected_queries = list(query_shard(shard_id))
    if group.get("query_ids") != expected_queries:
        raise ValueError(f"rerank group query coverage drift: {group_id}")
    records = group.get("records")
    if not isinstance(records, list) or len(records) != len(expected_queries):
        raise ValueError(f"rerank group record count drift: {group_id}")
    if [record.get("query", {}).get("query_id") for record in records] != expected_queries:
        raise ValueError(f"rerank group record order drift: {group_id}")
    if group.get("record_count") != len(records):
        raise ValueError(f"rerank group record metadata drift: {group_id}")
    if group.get("records_sha256") != records_sha256(records):
        raise ValueError(f"rerank group records hash drift: {group_id}")
    if group.get("status") != (
        "passed" if all(record.get("status") == "passed" for record in records) else "failed"
    ):
        raise ValueError(f"rerank group status drift: {group_id}")
    for record, query_id in zip(records, expected_queries, strict=True):
        validate_score_record(
            record,
            expected_group_id=group_id,
            expected_cell_id=cell_id,
            expected_query_id=query_id,
            candidate_record=candidate_records[(embedding_id, query_id)],
            reranker_identity=reranker_identities[reranker_id],
        )
    group_without_hash = {key: value for key, value in group.items() if key != "group_sha256"}
    if group.get("group_sha256") != sha256_value(group_without_hash):
        raise ValueError(f"rerank group hash drift: {group_id}")


def write_group(output_dir: Path, group: dict) -> Path:
    path = group_path(output_dir, group["cell_id"], group["group_id"].rsplit("-", 1)[-1])
    write_json(path, group)
    return path


def build_manifest(
    *,
    output_dir: Path,
    project_root: Path,
    groups: list[dict],
    execution_manifest: Path,
    embedding_manifest: Path,
    candidate_manifest: Path,
    candidate_records_path: Path,
    reranker_identities: dict[str, dict],
    provider_cache_manifest: dict | None,
    generated_at: str,
) -> dict:
    groups_by_cell: dict[str, list[dict]] = {cell_id: [] for cell_id in EXPECTED_CELL_IDS}
    group_entries = []
    for group in groups:
        groups_by_cell[group["cell_id"]].append(group)
        path = group_path(
            output_dir,
            group["cell_id"],
            group["group_id"].rsplit("-", 1)[-1],
        )
        group_entries.append(
            artifact_entry(
                path,
                project_root,
                group_id=group["group_id"],
                cell_id=group["cell_id"],
                query_ids=group["query_ids"],
                record_count=group["record_count"],
                records_sha256=group["records_sha256"],
            )
        )

    cells: dict[str, dict] = {}
    all_records = [record for group in groups for record in group["records"]]
    for cell_id in EXPECTED_CELL_IDS:
        cell_groups = sorted(groups_by_cell[cell_id], key=lambda item: item["group_id"])
        records = [record for group in cell_groups for record in group["records"]]
        cells[cell_id] = {
            "embedding_id": cell_id.split("-", 1)[0],
            "reranker_id": cell_id.split("-", 1)[1],
            "group_ids": [group["group_id"] for group in cell_groups],
            "query_ids": [record["query"]["query_id"] for record in records],
            "group_count": len(cell_groups),
            "record_count": len(records),
            "records_sha256": records_sha256(records),
            "status": "passed" if all(record["status"] == "passed" for record in records) else "failed",
        }

    failed_records = sum(record["status"] == "failed" for record in all_records)
    passed_records = sum(record["status"] == "passed" for record in all_records)
    manifest = {
        "schema_version": RERANK_MANIFEST_SCHEMA,
        "status": "passed" if failed_records == 0 and len(groups) == 27 else "failed",
        "run_scope": "development_only",
        "generated_at": generated_at,
        "execution_manifest": artifact_entry(execution_manifest, project_root),
        "embedding_cache_manifest": artifact_entry(embedding_manifest, project_root),
        "candidate_manifest": artifact_entry(candidate_manifest, project_root),
        "candidate_records": artifact_entry(candidate_records_path, project_root),
        "rerankers": reranker_identities,
        "provider_cache_manifest": provider_cache_manifest,
        "groups": group_entries,
        "cells": cells,
        "counts": {
            "embeddings": len(EMBEDDING_IDS),
            "rerankers": len(RERANKER_IDS),
            "cells": len(EXPECTED_CELL_IDS),
            "groups_per_cell": len(QUERY_SHARDS),
            "cache_groups": len(groups),
            "query_records": len(all_records),
            "candidate_scores": passed_records * CANDIDATE_COUNT,
            "passed_records": passed_records,
            "failed_records": failed_records,
        },
        "checks": {
            "matrix_conditions_executed": 0,
            "f5_weights_generated": False,
            "confirmation_inputs_used": False,
        },
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    return manifest


def validate_bundle(
    *,
    output_dir: Path,
    project_root: Path,
    candidate_records: list[dict],
) -> tuple[list[dict], dict]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("rerank cache bundle manifest is missing")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != RERANK_MANIFEST_SCHEMA:
        raise ValueError("rerank cache manifest schema drift")
    if manifest.get("run_scope") != "development_only":
        raise ValueError("rerank cache manifest scope drift")
    if tuple(manifest.get("cells", {})) != EXPECTED_CELL_IDS:
        raise ValueError("rerank cache manifest cell order drift")
    expected_counts = {
        "embeddings": 3,
        "rerankers": 3,
        "cells": 9,
        "groups_per_cell": 3,
        "cache_groups": 27,
        "query_records": 72,
    }
    counts = manifest.get("counts", {})
    for field, expected in expected_counts.items():
        if counts.get(field) != expected:
            raise ValueError(f"rerank cache count drift: {field}")
    if counts.get("failed_records", 0) + counts.get("passed_records", 0) != 72:
        raise ValueError("rerank cache pass/fail counts do not cover 72 records")
    if counts.get("candidate_scores") != counts.get("passed_records") * CANDIDATE_COUNT:
        raise ValueError("rerank candidate score count drift")
    expected_status = "passed" if counts.get("failed_records") == 0 else "failed"
    if manifest.get("status") != expected_status:
        raise ValueError("rerank cache manifest status drift")

    for artifact_field in (
        "execution_manifest",
        "embedding_cache_manifest",
        "candidate_manifest",
        "candidate_records",
    ):
        artifact = manifest.get(artifact_field, {})
        path = project_root / artifact.get("path", "")
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise ValueError(f"rerank cache input artifact drift: {artifact_field}")
    provider_manifest = manifest.get("provider_cache_manifest")
    if provider_manifest is not None:
        path = project_root / provider_manifest.get("path", "")
        if not path.is_file() or sha256_file(path) != provider_manifest.get("sha256"):
            raise ValueError("rerank provider cache manifest drift")

    reranker_identities = manifest.get("rerankers", {})
    if tuple(reranker_identities) != RERANKER_IDS:
        raise ValueError("reranker identity order drift")
    for reranker_id, contract in RERANKER_CONTRACT.items():
        identity = reranker_identities[reranker_id]
        for field, expected in contract.items():
            if identity.get(field) != expected:
                raise ValueError(f"{reranker_id} contract drift: {field}")
        if identity.get("backend_verified") is not True:
            raise ValueError(f"{reranker_id} backend was not verified")
        identity_without_hash = {
            key: value for key, value in identity.items() if key != "identity_sha256"
        }
        if identity.get("identity_sha256") != sha256_value(identity_without_hash):
            raise ValueError(f"{reranker_id} identity hash drift")

    candidate_records_by_pair = {
        (record["embedding_id"], record["query"]["query_id"]): record
        for record in candidate_records
    }
    group_entries = manifest.get("groups")
    if not isinstance(group_entries, list) or len(group_entries) != 27:
        raise ValueError("rerank cache manifest must list exactly 27 groups")
    if [entry.get("path") for entry in group_entries] != [
        display_path(
            group_path(output_dir, cell_id, shard_id),
            project_root,
        )
        for cell_id in EXPECTED_CELL_IDS
        for shard_id, _ in QUERY_SHARDS
    ]:
        raise ValueError("rerank cache group file order drift")
    if [entry.get("group_id") for entry in group_entries] != list(EXPECTED_GROUP_IDS):
        raise ValueError("rerank cache group IDs drift")

    groups = []
    for entry in group_entries:
        path = project_root / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"rerank group file hash drift: {entry['group_id']}")
        group = read_json(path)
        validate_group(
            group,
            candidate_records=candidate_records_by_pair,
            reranker_identities=reranker_identities,
        )
        if group["records_sha256"] != entry.get("records_sha256"):
            raise ValueError(f"rerank group manifest hash drift: {entry['group_id']}")
        if group["query_ids"] != entry.get("query_ids"):
            raise ValueError(f"rerank group query manifest drift: {entry['group_id']}")
        groups.append(group)

    all_records = [record for group in groups for record in group["records"]]
    if len(all_records) != 72:
        raise ValueError("rerank bundle does not contain 72 query records")
    for cell_id, cell in manifest["cells"].items():
        records = [record for record in all_records if record["cell_id"] == cell_id]
        if len(records) != 8:
            raise ValueError(f"rerank cell does not contain eight queries: {cell_id}")
        if cell.get("records_sha256") != records_sha256(records):
            raise ValueError(f"rerank cell records hash drift: {cell_id}")
    if manifest.get("checks") != {
        "matrix_conditions_executed": 0,
        "f5_weights_generated": False,
        "confirmation_inputs_used": False,
    }:
        raise ValueError("rerank cache execution guards drifted")
    manifest_without_hash = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if manifest.get("manifest_sha256") != sha256_value(manifest_without_hash):
        raise ValueError("rerank cache manifest hash drift")
    return groups, manifest


def replay_local_record(record: dict) -> list[dict]:
    if record.get("status") != "passed":
        raise ValueError("cannot replay a failed rerank record")
    if record.get("reranker_id") == "R3":
        raise ValueError("R3 must replay through the provider cache adapter")
    parsed_response = parse_raw_response(record.get("raw_response"), len(record["candidates"]))
    return parsed_scores(parsed_response, record["candidates"])

