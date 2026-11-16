#!/usr/bin/env python3
"""retrieval stage score-cache schema, validation, and offline replay."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPTS_DIR))

from retrieval_utils import chinese_tokens  # noqa: E402
from shared_policy import query_intent_analyzer  # noqa: E402


SCORE_CACHE_SCHEMA = "gate5-score-cache-v1"
SCORE_CACHE_MANIFEST_SCHEMA = "gate5-score-cache-manifest-v1"
EXPECTED_QUERY_IDS = tuple(f"D{index:02d}" for index in range(1, 9))
CANDIDATE_COUNT = 20
AEF_WEIGHTS = {"dense": 0.5, "bm25": 0.3, "tag": 0.2}
E1_MODEL_ID = "BAAI/bge-m3"
E1_WEIGHT_SHA256 = "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"
R1_MODEL_ID = "BAAI/bge-reranker-v2-m3"
R1_WEIGHT_SHA256 = "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286"
R1_BACKEND = "sequence-classification-cross-encoder"
R1_MAX_LENGTH = 512
CONFIRMATION_QUERY_IDS = frozenset(f"H{index:02d}" for index in range(1, 33))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as block_reader:
        for block in iter(lambda: block_reader.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def tag_score(query: dict, chunks: list[dict], topics: dict[str, str]) -> np.ndarray:
    label = topics.get(query.get("topic", ""), "")
    if not label:
        return np.zeros(len(chunks), dtype=np.float64)
    variants = query_intent_analyzer.synonym_variants(label)
    return np.asarray(
        [
            1.0
            if any(variant in (chunk.get("doc_title") or "") for variant in variants)
            else 0.0
            for chunk in chunks
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class CacheInputs:
    queries: Path
    chunks: Path
    vectors: Path
    e1_model_weight: Path
    r1_model_weight: Path
    topics: Path
    manifest: Path
    guard_report: Path


def input_hashes(paths: CacheInputs) -> dict[str, str]:
    return {
        "queries_sha256": sha256_file(paths.queries),
        "chunks_sha256": sha256_file(paths.chunks),
        "e1_vectors_sha256": sha256_file(paths.vectors),
        "e1_model_weight_sha256": sha256_file(paths.e1_model_weight),
        "r1_model_weight_sha256": sha256_file(paths.r1_model_weight),
        "topics_sha256": sha256_file(paths.topics),
        "execution_manifest_sha256": sha256_file(paths.manifest),
        "execution_guard_report_sha256": sha256_file(paths.guard_report),
    }


def algorithm_identity() -> dict:
    return {
        "candidate_count": CANDIDATE_COUNT,
        "candidate_selection": "aef_chunk_top_k_stable",
        "aef_weights": dict(AEF_WEIGHTS),
        "aef_normalization": "query_minmax",
        "embedding": {
            "id": "E1",
            "model_id": E1_MODEL_ID,
            "pooling": "cls",
            "normalize": "l2",
            "max_tokens": 1024,
            "shared_query_encoding": True,
        },
        "reranker": {
            "id": "R1",
            "model_id": R1_MODEL_ID,
            "backend": R1_BACKEND,
            "max_length": R1_MAX_LENGTH,
            "score_transform": "sigmoid_round_6",
        },
    }


def cache_identity(query_id: str, query_text: str, hashes: dict[str, str]) -> dict:
    payload = {
        "schema_version": SCORE_CACHE_SCHEMA,
        "cell": {"embedding_id": "E1", "reranker_id": "R1"},
        "query_id": query_id,
        "query_sha256": sha256_text(query_text),
        "inputs": hashes,
        "algorithm": algorithm_identity(),
    }
    return {
        "cell": "E1-R1",
        "query_id": query_id,
        "query_sha256": sha256_text(query_text),
        "input_hashes": hashes,
        "algorithm": algorithm_identity(),
        "identity_payload_sha256": sha256_value(payload),
    }


def select_candidates(
    query: dict,
    query_vector: np.ndarray,
    chunks: list[dict],
    vectors: np.ndarray,
    bm25: Any,
    topics: dict[str, str],
) -> tuple[list[dict], dict[str, float]]:
    text = query["text"]
    dense = np.asarray(vectors @ query_vector, dtype=np.float64)
    lexical = np.asarray(bm25.get_scores(chinese_tokens(text)), dtype=np.float64)
    tag = tag_score(query, chunks, topics)
    aef = (
        AEF_WEIGHTS["dense"] * minmax(dense)
        + AEF_WEIGHTS["bm25"] * minmax(lexical)
        + AEF_WEIGHTS["tag"] * minmax(tag)
    )
    order = np.argsort(-aef, kind="stable")[:CANDIDATE_COUNT]
    candidates = []
    for rank, chunk_index in enumerate(order, start=1):
        index = int(chunk_index)
        chunk = chunks[index]
        candidates.append(
            {
                "rank": rank,
                "chunk_index": index,
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "dense_score": round(float(dense[index]), 8),
                "bm25_score": round(float(lexical[index]), 8),
                "tag_score": round(float(tag[index]), 8),
                "aef_score": round(float(aef[index]), 8),
            }
        )
    if len(candidates) != CANDIDATE_COUNT:
        raise RuntimeError(f"selected {len(candidates)} candidates, expected {CANDIDATE_COUNT}")
    return candidates, {"embedding": sha256_value(query_vector.tolist())}


def parse_raw_response(raw_response: dict, expected_count: int) -> list[dict]:
    results = raw_response.get("results")
    if not isinstance(results, list):
        raise ValueError("raw response has no results list")
    if len(results) != expected_count:
        raise ValueError(f"raw response has {len(results)} results, expected {expected_count}")
    parsed = []
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("raw response result is not an object")
        index = item.get("index")
        score = item.get("relevance_score")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("raw response result has a non-integer index")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("raw response result has a non-finite score")
        parsed.append({"request_index": index, "rerank_score": score})
    if sorted(row["request_index"] for row in parsed) != list(range(expected_count)):
        raise ValueError("raw response indexes are not a complete permutation")
    return parsed


def make_success_record(
    *,
    query: dict,
    query_vector: np.ndarray,
    candidates: list[dict],
    raw_request: dict,
    raw_response: dict,
    hashes: dict[str, str],
    latency: dict[str, float],
    embedding_fingerprint: dict,
) -> dict:
    parsed_raw = parse_raw_response(raw_response, len(candidates))
    parsed = [
        {
            "request_index": row["request_index"],
            "chunk_index": candidates[row["request_index"]]["chunk_index"],
            "rerank_score": row["rerank_score"],
        }
        for row in parsed_raw
    ]
    parsed_by_request = {row["request_index"]: row for row in parsed}
    for request_index, candidate in enumerate(candidates):
        candidate["rerank_score"] = parsed_by_request[request_index]["rerank_score"]
    identity = cache_identity(query["slot"], query["text"], hashes)
    record = {
        "schema_version": SCORE_CACHE_SCHEMA,
        "status": "passed",
        "cache_identity": identity,
        "query": {
            "query_id": query["slot"],
            "text": query["text"],
            "topic": query.get("topic"),
            "topic_label": query.get("topic_label_cn"),
            "query_sha256": sha256_text(query["text"]),
        },
        "embedding": {
            "id": "E1",
            "model_id": E1_MODEL_ID,
            "shared_encoder": True,
            "fingerprint": embedding_fingerprint,
            "query_vector_sha256": sha256_value(query_vector.tolist()),
            "vector_shape": [1, int(query_vector.shape[0])],
        },
        "candidate_selection": {
            "count": len(candidates),
            "source": "AEF top 20 chunks",
            "ordering": "aef_score descending, chunk_index ascending",
        },
        "candidates": candidates,
        "raw_request": raw_request,
        "raw_response": raw_response,
        "parsed_response": parsed,
        "latency_ms": latency,
        "failure": {
            "failed": False,
            "category": None,
            "message": None,
        },
    }
    record["record_sha256"] = sha256_value(record)
    return record


def make_failure_record(
    *,
    query: dict,
    query_vector: np.ndarray | None,
    candidates: list[dict],
    hashes: dict[str, str],
    category: str,
    message: str,
    latency: dict[str, float],
    embedding_fingerprint: dict,
) -> dict:
    identity = cache_identity(query["slot"], query["text"], hashes)
    embedding = {
        "id": "E1",
        "model_id": E1_MODEL_ID,
        "shared_encoder": True,
        "fingerprint": embedding_fingerprint,
    }
    if query_vector is not None:
        embedding["query_vector_sha256"] = sha256_value(query_vector.tolist())
        embedding["vector_shape"] = [1, int(query_vector.shape[0])]
    record = {
        "schema_version": SCORE_CACHE_SCHEMA,
        "status": "failed",
        "cache_identity": identity,
        "query": {
            "query_id": query["slot"],
            "text": query["text"],
            "topic": query.get("topic"),
            "topic_label": query.get("topic_label_cn"),
            "query_sha256": sha256_text(query["text"]),
        },
        "embedding": embedding,
        "candidate_selection": {
            "count": len(candidates),
            "source": "AEF top 20 chunks",
            "ordering": "aef_score descending, chunk_index ascending",
        },
        "candidates": candidates,
        "raw_request": None,
        "raw_response": None,
        "parsed_response": None,
        "latency_ms": latency,
        "failure": {
            "failed": True,
            "category": category,
            "message": message,
        },
    }
    record["record_sha256"] = sha256_value(record)
    return record


def validate_record(
    record: dict,
    *,
    expected_query_id: str,
    expected_hashes: dict[str, str],
) -> None:
    if record.get("schema_version") != SCORE_CACHE_SCHEMA:
        raise ValueError("score cache schema version drift")
    query = record.get("query", {})
    if query.get("query_id") != expected_query_id:
        raise ValueError(f"query ID drift: expected {expected_query_id}")
    if expected_query_id in CONFIRMATION_QUERY_IDS:
        raise ValueError("confirmation query leaked into score cache")
    if sha256_text(query.get("text", "")) != query.get("query_sha256"):
        raise ValueError("query text hash mismatch")
    identity = record.get("cache_identity", {})
    expected_identity = cache_identity(expected_query_id, query["text"], expected_hashes)
    if identity != expected_identity:
        raise ValueError("cache identity mismatch")
    if identity.get("query_id") != expected_query_id:
        raise ValueError("cache identity query drift")

    candidates = record.get("candidates")
    required_candidates = CANDIDATE_COUNT if record.get("status") == "passed" else 0
    if not isinstance(candidates, list) or len(candidates) < required_candidates:
        raise ValueError(f"{expected_query_id} candidate count is not {CANDIDATE_COUNT}")
    if record.get("candidate_selection", {}).get("count") != len(candidates):
        raise ValueError(f"{expected_query_id} candidate count metadata mismatch")
    chunk_ids = [candidate.get("chunk_id") for candidate in candidates]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError(f"{expected_query_id} has duplicate candidate chunk IDs")
    chunk_indexes = [candidate.get("chunk_index") for candidate in candidates]
    if len(set(chunk_indexes)) != len(chunk_indexes):
        raise ValueError(f"{expected_query_id} has duplicate candidate indexes")
    for candidate in candidates:
        for field in ("aef_score", "dense_score", "bm25_score", "tag_score"):
            if not isinstance(candidate.get(field), (int, float)):
                raise ValueError(f"{expected_query_id} candidate is missing {field}")
        if candidate.get("rank") is None or candidate.get("doc_id") is None:
            raise ValueError(f"{expected_query_id} candidate is missing rank or doc_id")

    failure = record.get("failure", {})
    if record.get("status") == "failed":
        if failure.get("failed") is not True:
            raise ValueError(f"{expected_query_id} failed record has inconsistent failure flag")
        if not failure.get("category") or not failure.get("message"):
            raise ValueError(f"{expected_query_id} failed record has no failure category or message")
        if record.get("parsed_response") is not None:
            raise ValueError(f"{expected_query_id} failed record must not contain parsed scores")
        for candidate in candidates:
            if candidate.get("rerank_score") is not None:
                raise ValueError(f"{expected_query_id} failed record must not contain rerank scores")
    elif record.get("status") == "passed":
        if failure.get("failed") is not False:
            raise ValueError(f"{expected_query_id} passed record has failure flag")
        parsed_raw = parse_raw_response(record.get("raw_response", {}), len(candidates))
        expected_parsed = [
            {
                "request_index": row["request_index"],
                "chunk_index": candidates[row["request_index"]]["chunk_index"],
                "rerank_score": row["rerank_score"],
            }
            for row in parsed_raw
        ]
        if record.get("parsed_response") != expected_parsed:
            raise ValueError(f"{expected_query_id} parsed response is not exact")
        parsed_by_request = {
            row["request_index"]: row for row in record.get("parsed_response", [])
        }
        for request_index, candidate in enumerate(candidates):
            if candidate.get("rerank_score") != parsed_by_request[request_index]["rerank_score"]:
                raise ValueError(f"{expected_query_id} rerank score does not match raw response")
    else:
        raise ValueError(f"{expected_query_id} has invalid status")

    latency = record.get("latency_ms", {})
    for field in ("embedding", "first_stage", "rerank", "total"):
        value = latency.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{expected_query_id} has invalid latency for {field}")
    record_without_hash = {key: value for key, value in record.items() if key != "record_sha256"}
    if record.get("record_sha256") != sha256_value(record_without_hash):
        raise ValueError(f"{expected_query_id} record hash mismatch")


def replay_record(record: dict) -> list[dict]:
    if record.get("status") != "passed":
        raise ValueError("cannot replay a failed score-cache record")
    parsed = parse_raw_response(record["raw_response"], len(record["candidates"]))
    parsed_by_request = {row["request_index"]: row for row in parsed}
    return [
        {
            "query_id": record["query"]["query_id"],
            "chunk_id": candidate["chunk_id"],
            "aef_score": candidate["aef_score"],
            "rerank_score": parsed_by_request[position]["rerank_score"],
        }
        for position, candidate in enumerate(record["candidates"])
    ]


def validate_cache(
    output_dir: Path,
    *,
    expected_hashes: dict[str, str],
) -> tuple[list[dict], dict]:
    manifest_path = output_dir / "manifest.json"
    records_path = output_dir / "records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        raise ValueError("score cache is incomplete")
    manifest = read_json(manifest_path)
    records = read_jsonl(records_path)
    if manifest.get("schema_version") != SCORE_CACHE_MANIFEST_SCHEMA:
        raise ValueError("score-cache manifest schema drift")
    if manifest.get("cell") != "E1-R1":
        raise ValueError("score cache is not for E1-R1")
    if manifest.get("record_count") != len(records):
        raise ValueError("manifest record count mismatch")
    query_ids = [record.get("query", {}).get("query_id") for record in records]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("duplicate query-condition cache keys")
    if query_ids != list(EXPECTED_QUERY_IDS):
        raise ValueError("score cache must contain D01-D08 exactly once in order")
    cache_keys = []
    for expected_query_id, record in zip(EXPECTED_QUERY_IDS, records, strict=True):
        validate_record(record, expected_query_id=expected_query_id, expected_hashes=expected_hashes)
        cache_keys.append(record["cache_identity"]["identity_payload_sha256"])
    if len(set(cache_keys)) != len(cache_keys):
        raise ValueError("duplicate query-condition cache keys")
    records_file_hash = sha256_file(records_path)
    if manifest.get("records_sha256") != records_file_hash:
        raise ValueError("records hash mismatch")
    if manifest.get("record_hashes") != [record["record_sha256"] for record in records]:
        raise ValueError("manifest record hash list mismatch")
    expected_status = "passed" if all(r["status"] == "passed" for r in records) else "failed"
    if manifest.get("status") != expected_status:
        raise ValueError("manifest status does not match records")
    return records, manifest


def replay_cache(output_dir: Path, output: Path | None = None) -> list[dict]:
    hashes = read_json(output_dir / "manifest.json").get("input_hashes", {})
    records, _ = validate_cache(output_dir, expected_hashes=hashes)
    replayed = [row for record in records for row in replay_record(record)]
    if output is not None:
        write_jsonl(output, replayed)
    return replayed


def build_manifest(
    *,
    records: list[dict],
    hashes: dict[str, str],
    run_date: str,
    embedding_fingerprint: dict,
) -> dict:
    record_hashes = [record["record_sha256"] for record in records]
    records_text = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    return {
        "schema_version": SCORE_CACHE_MANIFEST_SCHEMA,
        "cell": "E1-R1",
        "embedding_id": "E1",
        "reranker_id": "R1",
        "status": "passed" if all(record["status"] == "passed" for record in records) else "failed",
        "run_date": run_date,
        "record_count": len(records),
        "query_ids": list(EXPECTED_QUERY_IDS),
        "input_hashes": hashes,
        "algorithm": algorithm_identity(),
        "embedding_fingerprint": embedding_fingerprint,
        "record_hashes": record_hashes,
        "records_sha256": sha256_text(records_text),
    }


def generate_records(
    paths: CacheInputs,
    *,
    encode_queries: Any,
    reranker: Any,
    run_date: str = "2026-11-03",
    embedding_fingerprint: dict | None = None,
) -> tuple[list[dict], dict]:
    queries = read_jsonl(paths.queries)
    if [query.get("slot") for query in queries] != list(EXPECTED_QUERY_IDS):
        raise ValueError("input queries are not D01-D08 in frozen order")
    chunks = read_jsonl(paths.chunks)
    vectors = np.load(paths.vectors)
    if vectors.shape != (len(chunks), 1024):
        raise ValueError("E1 vector cache shape mismatch")
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("frozen chunk cache has duplicate IDs")
    e1_ids = read_json(paths.vectors.with_suffix(".ids.json"))
    if e1_ids.get("chunk_ids") != chunk_ids:
        raise ValueError("E1 chunk order drift")
    if e1_ids.get("chunks_sha256") != sha256_file(paths.chunks):
        raise ValueError("E1 vector manifest chunk hash mismatch")
    hashes = input_hashes(paths)
    if hashes["e1_model_weight_sha256"] != E1_WEIGHT_SHA256:
        raise ValueError("E1 model weight drift")
    if hashes["r1_model_weight_sha256"] != R1_WEIGHT_SHA256:
        raise ValueError("R1 model weight drift")
    topics = read_json(paths.topics).get("topics", {})

    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([chinese_tokens(chunk["content"]) for chunk in chunks])
    embedding_started = time.perf_counter()
    query_vectors = np.asarray(
        encode_queries([query["text"] for query in queries]),
        dtype=np.float32,
    )
    embedding_elapsed = (time.perf_counter() - embedding_started) * 1000.0
    if query_vectors.shape != (len(queries), 1024):
        raise ValueError("shared query encoding shape mismatch")
    fingerprint = embedding_fingerprint
    if fingerprint is None:
        fingerprint = getattr(getattr(encode_queries, "module", None), "fingerprint", None)
        fingerprint = fingerprint() if callable(fingerprint) else {
            "implementation": "shared_bge_encoder"
        }

    records = []
    for query, query_vector in zip(queries, query_vectors, strict=True):
        first_started = time.perf_counter()
        try:
            candidates, _ = select_candidates(
                query,
                query_vector,
                chunks,
                vectors,
                bm25,
                topics,
            )
            first_stage_elapsed = (time.perf_counter() - first_started) * 1000.0
            documents = [chunks[candidate["chunk_index"]]["content"] for candidate in candidates]
            raw_request = {"query": query["text"], "documents": documents}
            rerank_started = time.perf_counter()
            scores = reranker.score(query["text"], documents)
            if len(scores) != CANDIDATE_COUNT:
                raise RuntimeError(
                    f"reranker returned {len(scores)} scores for {CANDIDATE_COUNT} documents"
                )
            from local_reranker_server import rank_scores

            ranked = rank_scores(scores)
            raw_response = {
                "output": {
                    "results": ranked,
                    "truncated_pairs": reranker.truncation_count(query["text"], documents),
                    "max_length": reranker.max_length,
                    "backend": reranker.name,
                },
                "results": ranked,
            }
            rerank_elapsed = (time.perf_counter() - rerank_started) * 1000.0
            total_elapsed = embedding_elapsed + first_stage_elapsed + rerank_elapsed
            records.append(
                make_success_record(
                    query=query,
                    query_vector=query_vector,
                    candidates=candidates,
                    raw_request=raw_request,
                    raw_response=raw_response,
                    hashes=hashes,
                    latency={
                        "embedding": round(embedding_elapsed, 6),
                        "first_stage": round(first_stage_elapsed, 6),
                        "rerank": round(rerank_elapsed, 6),
                        "total": round(total_elapsed, 6),
                    },
                    embedding_fingerprint=fingerprint,
                )
            )
        except Exception as exc:  # noqa: BLE001
            first_stage_elapsed = (time.perf_counter() - first_started) * 1000.0
            records.append(
                make_failure_record(
                    query=query,
                    query_vector=query_vector,
                    candidates=[],
                    hashes=hashes,
                    category="tracer_execution_failure",
                    message=f"{type(exc).__name__}: {exc}",
                    latency={
                        "embedding": round(embedding_elapsed, 6),
                        "first_stage": round(first_stage_elapsed, 6),
                        "rerank": 0.0,
                        "total": round(embedding_elapsed + first_stage_elapsed, 6),
                    },
                    embedding_fingerprint=fingerprint,
                )
            )
    manifest = build_manifest(
        records=records,
        hashes=hashes,
        run_date=run_date,
        embedding_fingerprint=fingerprint,
    )
    return records, manifest


def write_cache(output_dir: Path, records: list[dict], manifest: dict) -> None:
    write_jsonl(output_dir / "records.jsonl", records)
    write_json(output_dir / "manifest.json", manifest)
