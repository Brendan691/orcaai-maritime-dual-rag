#!/usr/bin/env python3
"""Schemas, validation, and replay for retrieval stage embedding/candidate caches."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gate5_score_cache as score_cache  # noqa: E402
from chinese_retrieval_utils import chinese_tokens  # noqa: E402
from gate5_provider_cache import (  # noqa: E402
    Gate5Provider,
    Gate5ProviderCache,
    ProviderCacheMiss,
    normalize_text,
)


EMBEDDING_CACHE_SCHEMA = "gate5-embedding-cache-v1"
EMBEDDING_MANIFEST_SCHEMA = "gate5-embedding-manifest-v1"
CANDIDATE_CACHE_SCHEMA = "gate5-candidate-cache-v1"
CANDIDATE_MANIFEST_SCHEMA = "gate5-candidate-manifest-v1"
BUNDLE_MANIFEST_SCHEMA = "gate5-embedding-caches-manifest-v1"
VALIDATION_REPORT_SCHEMA = "gate5-embedding-cache-validation-v1"
REPLAY_REPORT_SCHEMA = "gate5-embedding-cache-replay-v1"

EXPECTED_QUERY_IDS = tuple(f"D{index:02d}" for index in range(1, 9))
EMBEDDING_IDS = ("E1", "E2", "E3")
RERANKER_IDS = ("R1", "R2", "R3")
EXPECTED_CELL_IDS = tuple(
    f"{embedding_id}-{reranker_id}"
    for embedding_id in EMBEDDING_IDS
    for reranker_id in RERANKER_IDS
)
CONFIRMATION_QUERY_IDS = frozenset(f"H{index:02d}" for index in range(1, 33))
E3_PROVIDER_NAME = "dashscope"
E3_MODEL_ID = "text-embedding-v3"
VECTOR_DIMENSION = 1024


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


def jsonl_text(rows: Iterable[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(jsonl_text(rows), encoding="utf-8")
    temporary.replace(path)


def display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    root = project_root.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def artifact_entry(path: Path, project_root: Path, **extra: Any) -> dict:
    if not path.is_file():
        raise ValueError(f"missing artifact: {path}")
    return {
        "path": display_path(path, project_root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        **extra,
    }


def load_dev_queries(path: Path) -> list[dict]:
    queries = read_jsonl(path)
    if len(queries) != len(EXPECTED_QUERY_IDS):
        raise ValueError(
            f"development query cache requires {len(EXPECTED_QUERY_IDS)} rows, "
            f"found {len(queries)}"
        )
    slots = [str(query.get("slot") or "") for query in queries]
    if len(slots) != len(set(slots)):
        raise ValueError("duplicate development query slot")
    if set(slots) != set(EXPECTED_QUERY_IDS):
        raise ValueError(f"development query slots drift: {slots}")
    for query in queries:
        slot = str(query["slot"])
        query_id = str(query.get("query_id") or "")
        if slot in CONFIRMATION_QUERY_IDS or query_id in CONFIRMATION_QUERY_IDS:
            raise ValueError(f"confirmation query leaked into development cache: {slot}")
        if query.get("kind") not in (None, "dev"):
            raise ValueError(f"non-development query kind for {slot}: {query.get('kind')}")
        if not str(query.get("text") or "").strip():
            raise ValueError(f"empty development query text for {slot}")
    order = {query_id: index for index, query_id in enumerate(EXPECTED_QUERY_IDS)}
    return sorted(queries, key=lambda query: order[query["slot"]])


def query_cache_rows(queries: list[dict], *, embedding_id: str, encoded_inputs: list[str]) -> list[dict]:
    if len(queries) != len(encoded_inputs):
        raise ValueError("query/input count mismatch")
    return [
        {
            "embedding_id": embedding_id,
            "query_id": query["slot"],
            "slot": query["slot"],
            "text": query["text"],
            "topic": query.get("topic"),
            "topic_label_cn": query.get("topic_label_cn"),
            "query_sha256": sha256_text(query["text"]),
            "encoded_input_sha256": sha256_text(encoded_input),
            "vector_index": index,
        }
        for index, (query, encoded_input) in enumerate(
            zip(queries, encoded_inputs, strict=True)
        )
    ]


def vector_value_sha256(vector: np.ndarray) -> str:
    return score_cache.sha256_value(np.asarray(vector, dtype=np.float32).tolist())


def add_vector_hashes(rows: list[dict], vectors: np.ndarray) -> list[dict]:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(rows):
        raise ValueError("vector row count mismatch")
    enriched = []
    for row, vector in zip(rows, vectors, strict=True):
        enriched.append({**row, "vector_sha256": vector_value_sha256(vector)})
    return enriched


def document_cache_rows(
    chunks: list[dict],
    *,
    embedding_id: str,
    encoded_inputs: list[str] | None = None,
) -> list[dict]:
    if encoded_inputs is None:
        encoded_inputs = [chunk["content"] for chunk in chunks]
    if len(chunks) != len(encoded_inputs):
        raise ValueError("chunk/input count mismatch")
    return [
        {
            "embedding_id": embedding_id,
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "chunk_index": index,
            "content_sha256": sha256_text(chunk["content"]),
            "encoded_input_sha256": sha256_text(encoded_input),
            "vector_index": index,
        }
        for index, (chunk, encoded_input) in enumerate(
            zip(chunks, encoded_inputs, strict=True)
        )
    ]


def save_vectors(path: Path, vectors: np.ndarray) -> dict:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != VECTOR_DIMENSION:
        raise ValueError(f"unexpected embedding shape: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("embedding cache contains non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)
    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "vector_sha256": sha256_value(array.tolist()),
    }


def load_vectors(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    vectors = np.load(path, allow_pickle=False)
    if vectors.shape != expected_shape:
        raise ValueError(f"vector shape drift at {path}: {vectors.shape}")
    if vectors.dtype != np.float32:
        raise ValueError(f"vector dtype drift at {path}: {vectors.dtype}")
    if not np.isfinite(vectors).all():
        raise ValueError(f"non-finite vector value at {path}")
    return vectors


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)


def BM25Okapi(*args, **kwargs):  # noqa: N802 - mirrors the pinned dependency name
    try:
        from rank_bm25 import BM25Okapi as implementation
    except ModuleNotFoundError:
        project_root = SCRIPTS_DIR.parents[1]
        fallback = (
            project_root
            / ".venv313"
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        if not fallback.is_dir():
            raise
        sys.path.append(str(fallback))
        from rank_bm25 import BM25Okapi as implementation
    return implementation(*args, **kwargs)


def cells_for_embedding(embedding_id: str) -> list[str]:
    return [f"{embedding_id}-{reranker_id}" for reranker_id in RERANKER_IDS]


def build_candidate_records(
    *,
    queries: list[dict],
    chunks: list[dict],
    topics: dict[str, str],
    query_vectors: dict[str, np.ndarray],
    document_vectors: dict[str, np.ndarray],
) -> list[dict]:
    if tuple(query_vectors) != EMBEDDING_IDS or tuple(document_vectors) != EMBEDDING_IDS:
        raise ValueError("candidate cache requires E1/E2/E3 in canonical order")
    bm25 = BM25Okapi([chinese_tokens(chunk["content"]) for chunk in chunks])
    records: list[dict] = []
    for embedding_id in EMBEDDING_IDS:
        vectors = np.asarray(document_vectors[embedding_id], dtype=np.float32)
        encoded_queries = np.asarray(query_vectors[embedding_id], dtype=np.float32)
        if vectors.shape != (len(chunks), VECTOR_DIMENSION):
            raise ValueError(f"{embedding_id} document vector shape drift")
        if encoded_queries.shape != (len(queries), VECTOR_DIMENSION):
            raise ValueError(f"{embedding_id} query vector shape drift")
        for query, query_vector in zip(queries, encoded_queries, strict=True):
            candidates, embedding_hashes = score_cache.select_candidates(
                query,
                query_vector,
                chunks,
                vectors,
                bm25,
                topics,
            )
            record = {
                "schema_version": CANDIDATE_CACHE_SCHEMA,
                "status": "passed",
                "run_scope": "development_only",
                "embedding_id": embedding_id,
                "cell_ids": cells_for_embedding(embedding_id),
                "query": {
                    "query_id": query["slot"],
                    "text": query["text"],
                    "topic": query.get("topic"),
                    "topic_label_cn": query.get("topic_label_cn"),
                    "query_sha256": sha256_text(query["text"]),
                },
                "embedding": {
                    "id": embedding_id,
                    "query_vector_sha256": embedding_hashes["embedding"],
                },
                "candidate_selection": {
                    "count": score_cache.CANDIDATE_COUNT,
                    "source": "AEF top 20 chunks",
                    "ordering": "aef_score descending, chunk_index ascending",
                    "selection": "aef_chunk_top_k_stable",
                    "normalization": "query_minmax",
                    "weights": dict(score_cache.AEF_WEIGHTS),
                },
                "candidates": candidates,
            }
            record["record_sha256"] = sha256_value(record)
            records.append(record)
    validate_candidate_records(records, queries=queries, chunks=chunks)
    return records


def validate_candidate_records(
    records: list[dict],
    *,
    queries: list[dict],
    chunks: list[dict],
) -> None:
    expected_pairs = {(embedding_id, query["slot"]) for embedding_id in EMBEDDING_IDS for query in queries}
    actual_pairs: set[tuple[str, str]] = set()
    chunks_by_index = {index: chunk for index, chunk in enumerate(chunks)}
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    for record in records:
        if record.get("schema_version") != CANDIDATE_CACHE_SCHEMA:
            raise ValueError("candidate cache schema drift")
        if record.get("status") != "passed" or record.get("run_scope") != "development_only":
            raise ValueError("candidate cache status or scope drift")
        embedding_id = str(record.get("embedding_id"))
        query = record.get("query", {})
        query_id = str(query.get("query_id") or "")
        pair = (embedding_id, query_id)
        if pair in actual_pairs:
            raise ValueError(f"duplicate candidate cache key: {pair}")
        actual_pairs.add(pair)
        if query_id in CONFIRMATION_QUERY_IDS:
            raise ValueError(f"confirmation query leaked into candidates: {query_id}")
        if sha256_text(str(query.get("text") or "")) != query.get("query_sha256"):
            raise ValueError(f"candidate query hash mismatch for {embedding_id}/{query_id}")
        if record.get("cell_ids") != cells_for_embedding(embedding_id):
            raise ValueError(f"candidate cell mapping drift for {embedding_id}/{query_id}")
        candidates = record.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != score_cache.CANDIDATE_COUNT:
            raise ValueError(f"candidate count drift for {embedding_id}/{query_id}")
        if [candidate.get("rank") for candidate in candidates] != list(
            range(1, score_cache.CANDIDATE_COUNT + 1)
        ):
            raise ValueError(f"candidate rank drift for {embedding_id}/{query_id}")
        if len({candidate.get("chunk_id") for candidate in candidates}) != len(candidates):
            raise ValueError(f"duplicate candidate chunk for {embedding_id}/{query_id}")
        if len({candidate.get("chunk_index") for candidate in candidates}) != len(candidates):
            raise ValueError(f"duplicate candidate index for {embedding_id}/{query_id}")
        for candidate in candidates:
            chunk_index = candidate.get("chunk_index")
            chunk = chunks_by_index.get(chunk_index)
            if chunk is None or chunks_by_id.get(candidate.get("chunk_id")) != chunk:
                raise ValueError(f"candidate chunk identity drift for {embedding_id}/{query_id}")
            if candidate.get("doc_id") != chunk.get("doc_id"):
                raise ValueError(f"candidate doc identity drift for {embedding_id}/{query_id}")
            if (candidate.get("char_start"), candidate.get("char_end")) != (
                chunk.get("char_start"),
                chunk.get("char_end"),
            ):
                raise ValueError(f"candidate span drift for {embedding_id}/{query_id}")
            for field in ("aef_score", "dense_score", "bm25_score", "tag_score"):
                value = candidate.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"invalid {field} for {embedding_id}/{query_id}")
        record_without_hash = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        if record.get("record_sha256") != sha256_value(record_without_hash):
            raise ValueError(f"candidate record hash mismatch for {embedding_id}/{query_id}")
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        raise ValueError(f"candidate cache coverage drift; missing={missing}, extra={extra}")


def provider_cache_records(
    cache_root: Path,
    *,
    provider_name: str,
    model: str,
) -> list[dict]:
    rows = []
    for path in sorted(cache_root.rglob("*.json")):
        if path.name.startswith("._"):
            continue
        record = read_json(path)
        if record.get("schema_version") != "gate5-provider-cache-v1":
            raise ValueError(f"provider cache schema drift at {path}")
        if record.get("status") != "passed":
            raise ValueError(f"provider cache failure record present at {path}")
        if record.get("operation") != "embedding":
            raise ValueError(f"unexpected provider operation at {path}")
        if record.get("provider") != provider_name or record.get("model") != model:
            raise ValueError(f"provider cache model drift at {path}")
        rows.append(
            {
                "path": str(path.relative_to(cache_root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "cache_key": record["cache_key"],
                "input_sha256": record["input_sha256"],
                "result_sha256": record["result_sha256"],
            }
        )
    if not rows:
        raise ValueError(f"provider cache is empty: {cache_root}")
    return rows


def write_provider_cache_manifest(
    cache_root: Path,
    manifest_path: Path,
    *,
    project_root: Path,
    provider_name: str,
    model: str,
    batch_size: int,
) -> dict:
    files = provider_cache_records(cache_root, provider_name=provider_name, model=model)
    tree_sha256 = sha256_value(
        [{"path": row["path"], "sha256": row["sha256"]} for row in files]
    )
    manifest = {
        "schema_version": "gate5-provider-cache-manifest-v1",
        "status": "passed",
        "run_scope": "development_only",
        "provider": provider_name,
        "model": model,
        "operation": "embedding",
        "batch_size": batch_size,
        "file_count": len(files),
        "tree_sha256": tree_sha256,
        "files": files,
    }
    write_json(manifest_path, manifest)
    return manifest


def provider_embed_in_batches(
    cache: Gate5ProviderCache,
    provider: Gate5Provider,
    texts: list[str],
    *,
    batch_size: int,
    retries: int = 0,
    progress_label: str | None = None,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    vectors: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(texts), batch_size), 1):
        batch = texts[start : start + batch_size]
        attempt = 0
        while True:
            try:
                vectors.extend(
                    cache.embed(
                        provider,
                        E3_MODEL_ID,
                        batch,
                        batch_size=len(batch),
                    )
                )
                break
            except ProviderCacheMiss:
                raise
            except Exception:
                if attempt >= retries:
                    raise
                attempt += 1
        if progress_label:
            print(
                f"{progress_label}: batch {batch_number}/{total_batches}",
                flush=True,
            )
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (len(texts), VECTOR_DIMENSION):
        raise ValueError(f"provider embedding shape drift: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("provider returned non-finite embedding values")
    return array


def offline_provider_vectors(
    *,
    cache_root: Path,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    provider = Gate5Provider(
        name=E3_PROVIDER_NAME,
        key="offline-replay-does-not-use-network",
        base_url="",
        rerank_endpoint="",
    )
    cache = Gate5ProviderCache(cache_root, mode="offline")
    return provider_embed_in_batches(cache, provider, texts, batch_size=batch_size)


def compute_embedding_replay(
    manifest: dict,
    *,
    output_dir: Path,
    project_root: Path,
    queries: list[dict],
    chunks: list[dict],
    batch_size: int,
) -> dict:
    embedding_id = manifest["embedding_id"]
    query_vectors_path = project_root / manifest["outputs"]["query_vectors"]["path"]
    document_vectors_path = project_root / manifest["outputs"]["document_vectors"]["path"]
    query_vectors = load_vectors(
        query_vectors_path,
        (len(queries), VECTOR_DIMENSION),
    )
    document_vectors = load_vectors(
        document_vectors_path,
        (len(chunks), VECTOR_DIMENSION),
    )
    outputs = manifest["outputs"]
    if sha256_file(query_vectors_path) != outputs["query_vectors"]["sha256"]:
        raise ValueError(f"{embedding_id} query vector file hash mismatch during replay")
    if sha256_file(document_vectors_path) != outputs["document_vectors"]["sha256"]:
        raise ValueError(f"{embedding_id} document vector file hash mismatch during replay")
    if sha256_value(query_vectors.tolist()) != outputs["query_vectors"]["vector_sha256"]:
        raise ValueError(f"{embedding_id} query vector value hash mismatch during replay")
    if sha256_value(document_vectors.tolist()) != outputs["document_vectors"]["vector_sha256"]:
        raise ValueError(f"{embedding_id} document vector value hash mismatch during replay")

    details = {
        "mode": "cache_readback",
        "query_vectors_sha256": outputs["query_vectors"]["sha256"],
        "document_vectors_sha256": outputs["document_vectors"]["sha256"],
        "query_replay_equal": True,
        "document_replay_equal": True,
    }
    if embedding_id == "E3":
        encoded_queries = [normalize_text(query["text"]) for query in queries]
        encoded_documents = [normalize_text(chunk["content"]) for chunk in chunks]
        replayed_queries = offline_provider_vectors(
            cache_root=output_dir / "E3" / "provider-cache",
            texts=encoded_queries,
            batch_size=batch_size,
        )
        replayed_documents = offline_provider_vectors(
            cache_root=output_dir / "E3" / "provider-cache",
            texts=encoded_documents,
            batch_size=batch_size,
        )
        query_equal = np.array_equal(query_vectors, replayed_queries)
        document_equal = np.array_equal(document_vectors, replayed_documents)
        if not query_equal or not document_equal:
            raise ValueError("E3 offline provider replay is not bit-identical")
        provider_manifest = read_json(output_dir / "E3" / "provider-cache-manifest.json")
        details.update(
            {
                "mode": "offline_provider_cache",
                "provider_cache_tree_sha256": provider_manifest["tree_sha256"],
                "provider_cache_file_count": provider_manifest["file_count"],
                "offline_query_result_sha256": sha256_value(replayed_queries.tolist()),
                "offline_document_result_sha256": sha256_value(
                    replayed_documents.tolist()
                ),
                "query_replay_equal": query_equal,
                "document_replay_equal": document_equal,
            }
        )
    details["replay_sha256"] = sha256_value(details)
    return details


def replay_candidate_records(
    *,
    output_dir: Path,
    project_root: Path,
    records: list[dict],
    queries: list[dict],
    chunks: list[dict],
    topics: dict[str, str],
) -> dict:
    query_vectors = {}
    document_vectors = {}
    for embedding_id in EMBEDDING_IDS:
        manifest = read_json(output_dir / embedding_id / "manifest.json")
        query_vectors[embedding_id] = load_vectors(
            project_root / manifest["outputs"]["query_vectors"]["path"],
            (len(queries), VECTOR_DIMENSION),
        )
        document_vectors[embedding_id] = load_vectors(
            project_root / manifest["outputs"]["document_vectors"]["path"],
            (len(chunks), VECTOR_DIMENSION),
        )
    replayed = build_candidate_records(
        queries=queries,
        chunks=chunks,
        topics=topics,
        query_vectors=query_vectors,
        document_vectors=document_vectors,
    )
    if replayed != records:
        raise ValueError("candidate cache offline replay mismatch")
    canonical_sha256 = hashlib.sha256(jsonl_text(records).encode("utf-8")).hexdigest()
    replay_sha256 = hashlib.sha256(jsonl_text(replayed).encode("utf-8")).hexdigest()
    if canonical_sha256 != replay_sha256:
        raise ValueError("candidate cache replay hash mismatch")
    return {
        "mode": "recomputed_from_cached_vectors",
        "row_count": len(replayed),
        "records_sha256": canonical_sha256,
        "replayed_records_sha256": replay_sha256,
        "replay_equal": True,
    }


def validate_embedding_manifest(
    manifest: dict,
    *,
    project_root: Path,
    queries: list[dict],
    chunks: list[dict],
    expected_chunk_ids: list[str],
) -> np.ndarray:
    embedding_id = manifest.get("embedding_id")
    if embedding_id not in EMBEDDING_IDS:
        raise ValueError(f"unknown embedding cache: {embedding_id}")
    if manifest.get("schema_version") != EMBEDDING_MANIFEST_SCHEMA:
        raise ValueError(f"{embedding_id} embedding manifest schema drift")
    if manifest.get("status") != "passed" or manifest.get("run_scope") != "development_only":
        raise ValueError(f"{embedding_id} embedding manifest status or scope drift")
    if manifest.get("query_ids") != list(EXPECTED_QUERY_IDS):
        raise ValueError(f"{embedding_id} query coverage drift")
    if manifest.get("cells") != cells_for_embedding(embedding_id):
        raise ValueError(f"{embedding_id} cell mapping drift")
    for key in ("model", "inputs", "outputs", "replay"):
        if not isinstance(manifest.get(key), dict):
            raise ValueError(f"{embedding_id} manifest is missing {key}")

    query_vectors = load_vectors(
        project_root / manifest["outputs"]["query_vectors"]["path"],
        (len(queries), VECTOR_DIMENSION),
    )
    document_vectors = load_vectors(
        project_root / manifest["outputs"]["document_vectors"]["path"],
        (len(chunks), VECTOR_DIMENSION),
    )
    for output_name, array in (
        ("query_vectors", query_vectors),
        ("document_vectors", document_vectors),
    ):
        entry = manifest["outputs"][output_name]
        path = project_root / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{embedding_id} {output_name} file hash mismatch")
        if sha256_value(array.tolist()) != entry["vector_sha256"]:
            raise ValueError(f"{embedding_id} {output_name} value hash mismatch")

    query_rows = read_jsonl(project_root / manifest["outputs"]["queries"]["path"])
    if [row.get("query_id") for row in query_rows] != list(EXPECTED_QUERY_IDS):
        raise ValueError(f"{embedding_id} query metadata coverage drift")
    for row, query, vector in zip(query_rows, queries, query_vectors, strict=True):
        if row.get("query_sha256") != sha256_text(query["text"]):
            raise ValueError(f"{embedding_id} query metadata hash mismatch")
        if row.get("vector_sha256") != vector_value_sha256(vector):
            raise ValueError(f"{embedding_id} query metadata vector mismatch")

    document_rows = read_jsonl(project_root / manifest["outputs"]["documents"]["path"])
    if [row.get("chunk_id") for row in document_rows] != expected_chunk_ids:
        raise ValueError(f"{embedding_id} document metadata chunk order drift")
    for row, chunk, vector in zip(document_rows, chunks, document_vectors, strict=True):
        if row.get("content_sha256") != sha256_text(chunk["content"]):
            raise ValueError(f"{embedding_id} document content hash mismatch")
        if row.get("vector_sha256") != vector_value_sha256(vector):
            raise ValueError(f"{embedding_id} document metadata vector mismatch")

    replay = manifest["replay"]
    replay_without_hash = {
        key: value for key, value in replay.items() if key != "replay_sha256"
    }
    if replay.get("replay_sha256") != sha256_value(replay_without_hash):
        raise ValueError(f"{embedding_id} replay hash mismatch")
    if replay.get("query_replay_equal") is not True or replay.get("document_replay_equal") is not True:
        raise ValueError(f"{embedding_id} replay did not pass")
    return document_vectors


def validate_candidate_manifest(
    manifest: dict,
    *,
    output_dir: Path,
    project_root: Path,
    records: list[dict],
    queries: list[dict],
    chunks: list[dict],
) -> None:
    if manifest.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA:
        raise ValueError("candidate manifest schema drift")
    if manifest.get("status") != "passed" or manifest.get("run_scope") != "development_only":
        raise ValueError("candidate manifest status or scope drift")
    if manifest.get("query_ids") != list(EXPECTED_QUERY_IDS):
        raise ValueError("candidate manifest query coverage drift")
    if tuple(manifest.get("cell_ids", ())) != EXPECTED_CELL_IDS:
        raise ValueError("candidate manifest does not cover exactly nine cells")
    if manifest.get("candidate_count_per_query") != score_cache.CANDIDATE_COUNT:
        raise ValueError("candidate manifest count drift")
    validate_candidate_records(records, queries=queries, chunks=chunks)
    records_path = project_root / manifest["outputs"]["records"]["path"]
    if hashlib.sha256(records_path.read_bytes()).hexdigest() != manifest["outputs"]["records"]["sha256"]:
        raise ValueError("candidate records file hash mismatch")
    for cell_id, cell in manifest.get("cells", {}).items():
        if cell_id not in EXPECTED_CELL_IDS:
            raise ValueError(f"unexpected candidate cell: {cell_id}")
        if cell.get("query_ids") != list(EXPECTED_QUERY_IDS):
            raise ValueError(f"candidate cell query coverage drift: {cell_id}")
        if cell.get("record_count") != len(EXPECTED_QUERY_IDS):
            raise ValueError(f"candidate cell record count drift: {cell_id}")
    if tuple(manifest.get("cells", ())) != EXPECTED_CELL_IDS:
        raise ValueError("candidate cell order drift")
    replay = manifest.get("replay", {})
    if replay.get("replay_equal") is not True:
        raise ValueError("candidate replay did not pass")
    if replay.get("records_sha256") != manifest["outputs"]["records"]["sha256"]:
        raise ValueError("candidate replay/output hash mismatch")


def validate_bundle(
    *,
    output_dir: Path,
    project_root: Path,
    execution_manifest: dict,
    execution_manifest_path: Path,
    guard_report_path: Path,
) -> dict:
    bundle = read_json(output_dir / "manifest.json")
    if bundle.get("schema_version") != BUNDLE_MANIFEST_SCHEMA:
        raise ValueError("embedding bundle manifest schema drift")
    if bundle.get("status") != "passed" or bundle.get("run_scope") != "development_only":
        raise ValueError("embedding bundle status or scope drift")
    if bundle.get("query_ids") != list(EXPECTED_QUERY_IDS):
        raise ValueError("embedding bundle query coverage drift")
    if tuple(bundle.get("cell_ids", ())) != EXPECTED_CELL_IDS:
        raise ValueError("embedding bundle does not cover exactly nine cells")
    expected_cells = [cell["cell_id"] for cell in execution_manifest["f5_cells"]]
    if list(bundle["cell_ids"]) != expected_cells:
        raise ValueError("embedding bundle cell IDs do not match execution manifest")
    if bundle.get("execution_manifest", {}).get("sha256") != sha256_file(execution_manifest_path):
        raise ValueError("embedding bundle execution manifest binding drift")
    if bundle.get("execution_guard_report", {}).get("sha256") != sha256_file(guard_report_path):
        raise ValueError("embedding bundle guard report binding drift")
    guards = bundle.get("guards", {})
    if guards.get("matrix_conditions_executed") != 0:
        raise ValueError("embedding cache run executed matrix conditions")
    if guards.get("f5_weights_generated") is not False:
        raise ValueError("embedding cache run generated F5 weights")
    if guards.get("confirmation_inputs_used") is not False:
        raise ValueError("embedding cache run touched confirmation inputs")

    queries_path = project_root / bundle["inputs"]["queries"]["path"]
    chunks_path = project_root / bundle["inputs"]["chunks"]["path"]
    topics_path = project_root / bundle["inputs"]["topics"]["path"]
    queries = load_dev_queries(queries_path)
    chunks = read_jsonl(chunks_path)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("frozen chunk cache contains duplicate IDs")
    topics = read_json(topics_path).get("topics", {})

    expected_bundle_hashes = {
        "queries_sha256": sha256_file(queries_path),
        "chunks_sha256": sha256_file(chunks_path),
        "topics_sha256": sha256_file(topics_path),
    }
    for key, actual in expected_bundle_hashes.items():
        if bundle["inputs"][key] != actual:
            raise ValueError(f"embedding bundle input hash drift: {key}")

    document_vectors: dict[str, np.ndarray] = {}
    for embedding_id in EMBEDDING_IDS:
        entry = bundle["embeddings"][embedding_id]
        manifest_path = project_root / entry["manifest"]["path"]
        if sha256_file(manifest_path) != entry["manifest"]["sha256"]:
            raise ValueError(f"{embedding_id} embedding manifest hash drift")
        manifest = read_json(manifest_path)
        document_vectors[embedding_id] = validate_embedding_manifest(
            manifest,
            project_root=project_root,
            queries=queries,
            chunks=chunks,
            expected_chunk_ids=chunk_ids,
        )
        if entry["replay_sha256"] != manifest["replay"]["replay_sha256"]:
            raise ValueError(f"{embedding_id} bundle replay binding drift")

    e1_entry = execution_manifest["inputs"]["E1_vectors"]
    e2_entry = execution_manifest["inputs"]["E2_vectors"]
    bundle_e1 = bundle["embeddings"]["E1"]["outputs"]["document_vectors"]
    bundle_e2 = bundle["embeddings"]["E2"]["outputs"]["document_vectors"]
    if bundle_e1["sha256"] != e1_entry["sha256"]:
        raise ValueError("E1 document vectors do not match frozen V9 vectors")
    if bundle_e2["sha256"] != e2_entry["sha256"]:
        raise ValueError("E2 document vectors do not match frozen V9 vectors")

    candidate_manifest_path = project_root / bundle["candidates"]["manifest"]["path"]
    if sha256_file(candidate_manifest_path) != bundle["candidates"]["manifest"]["sha256"]:
        raise ValueError("candidate manifest hash drift")
    candidate_manifest = read_json(candidate_manifest_path)
    candidate_records_path = project_root / candidate_manifest["outputs"]["records"]["path"]
    records = read_jsonl(candidate_records_path)
    validate_candidate_manifest(
        candidate_manifest,
        output_dir=output_dir,
        project_root=project_root,
        records=records,
        queries=queries,
        chunks=chunks,
    )

    replay_report_path = project_root / bundle["reports"]["replay"]["path"]
    if sha256_file(replay_report_path) != bundle["reports"]["replay"]["sha256"]:
        raise ValueError("embedding replay report hash drift")
    replay_report = read_json(replay_report_path)
    if replay_report.get("status") != "passed" or replay_report.get("run_scope") != "development_only":
        raise ValueError("embedding replay report status or scope drift")
    for embedding_id in EMBEDDING_IDS:
        if replay_report["embeddings"][embedding_id]["replay_sha256"] != bundle["embeddings"][embedding_id]["replay_sha256"]:
            raise ValueError(f"{embedding_id} replay report binding drift")
    if not replay_report.get("candidates", {}).get("replay_equal"):
        raise ValueError("candidate replay report did not pass")

    score_manifest_path = project_root / bundle["evidence"]["score_cache_manifest"]["path"]
    if sha256_file(score_manifest_path) != bundle["evidence"]["score_cache_manifest"]["sha256"]:
        raise ValueError("E1-R1 score-cache evidence hash drift")
    score_records_path = score_manifest_path.parent / "records.jsonl"
    score_records = read_jsonl(score_records_path)
    score_by_query = {record["query"]["query_id"]: record for record in score_records}
    candidate_by_query = {
        record["query"]["query_id"]: record
        for record in records
        if record["embedding_id"] == "E1"
    }
    for query_id in EXPECTED_QUERY_IDS:
        score_record = score_by_query[query_id]
        candidate_record = candidate_by_query[query_id]
        if score_record["embedding"]["query_vector_sha256"] != candidate_record["embedding"]["query_vector_sha256"]:
            raise ValueError(f"E1 query encoding drift against score cache: {query_id}")
        score_candidates = [
            {key: value for key, value in candidate.items() if key != "rerank_score"}
            for candidate in score_record["candidates"]
        ]
        if score_candidates != candidate_record["candidates"]:
            raise ValueError(f"E1 candidate drift against score cache: {query_id}")

    return {
        "queries": len(queries),
        "documents": len(chunks),
        "embeddings": len(EMBEDDING_IDS),
        "candidate_records": len(records),
        "candidate_chunks": sum(len(record["candidates"]) for record in records),
        "cells": len(EXPECTED_CELL_IDS),
    }


def write_validation_report(
    *,
    output_dir: Path,
    project_root: Path,
    counts: dict,
) -> dict:
    bundle_path = output_dir / "manifest.json"
    bundle = read_json(bundle_path)
    replay_path = project_root / bundle["reports"]["replay"]["path"]
    report = {
        "schema_version": VALIDATION_REPORT_SCHEMA,
        "status": "passed",
        "run_scope": "development_only",
        "bundle_manifest": artifact_entry(bundle_path, project_root),
        "replay_report": artifact_entry(replay_path, project_root),
        "checks": {
            "development_query_coverage_D01_D08": True,
            "E1_E2_frozen_V9_document_identity": True,
            "E3_live_cache_and_offline_replay": True,
            "nine_embedding_reranker_cells": True,
            "deterministic_candidate_replay": True,
            "manifest_model_input_output_replay_hashes": True,
            "confirmation_input_isolation": True,
            "matrix_conditions_executed": 0,
            "f5_weights_generated": False,
        },
        "counts": counts,
    }
    report["report_sha256"] = sha256_value(report)
    write_json(output_dir / "validation-report.json", report)
    return report


def write_replay_report(
    *,
    output_dir: Path,
    project_root: Path,
    queries: list[dict],
    chunks: list[dict],
    topics: dict[str, str],
    batch_size: int,
) -> dict:
    embedding_replays = {}
    for embedding_id in EMBEDDING_IDS:
        manifest = read_json(output_dir / embedding_id / "manifest.json")
        replay = compute_embedding_replay(
            manifest,
            output_dir=output_dir,
            project_root=project_root,
            queries=queries,
            chunks=chunks,
            batch_size=batch_size,
        )
        if replay != manifest["replay"]:
            raise ValueError(f"{embedding_id} replay differs from manifest")
        embedding_replays[embedding_id] = replay

    candidate_manifest_path = output_dir / "candidates" / "manifest.json"
    candidate_manifest = read_json(candidate_manifest_path)
    records_path = project_root / candidate_manifest["outputs"]["records"]["path"]
    records = read_jsonl(records_path)
    candidate_replay = replay_candidate_records(
        output_dir=output_dir,
        project_root=project_root,
        records=records,
        queries=queries,
        chunks=chunks,
        topics=topics,
    )
    if candidate_replay != candidate_manifest["replay"]:
        raise ValueError("candidate replay differs from manifest")
    report = {
        "schema_version": REPLAY_REPORT_SCHEMA,
        "status": "passed",
        "run_scope": "development_only",
        "embeddings": embedding_replays,
        "candidates": candidate_replay,
        "candidate_manifest": artifact_entry(candidate_manifest_path, project_root),
        "replay_equal": True,
    }
    report["report_sha256"] = sha256_value(report)
    write_json(output_dir / "replay-report.json", report)
    return report
