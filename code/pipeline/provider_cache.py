#!/usr/bin/env python3
"""Deterministic cache and offline replay for retrieval stage remote providers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
from dotenv import dotenv_values
from openai import OpenAI


CACHE_SCHEMA_VERSION = "gate5-provider-cache-v1"
CACHE_FILE_SUFFIX = ".json"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_REMOTE_RERANK_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/"
    "text-rerank/text-rerank"
)


@dataclass(frozen=True)
class Gate5Provider:
    name: str
    key: str = field(repr=False)
    base_url: str
    rerank_endpoint: str
    capabilities: tuple[str, ...] = ("embedding", "rerank")


class ProviderCacheError(RuntimeError):
    """Base class for deterministic provider-cache failures."""


class ProviderCacheMiss(ProviderCacheError):
    def __init__(self, cache_key: str, operation: str):
        super().__init__(
            f"offline cache miss for {operation} request {cache_key}"
        )
        self.cache_key = cache_key
        self.operation = operation


class ProviderCallFailure(ProviderCacheError):
    def __init__(self, record: dict):
        self.record = record
        category = record.get("failure", {}).get("category", "provider_error")
        message = record.get("failure", {}).get("message", "provider call failed")
        super().__init__(f"{category}: {message}")


def resolve_gate5_dashscope(env_path: Path) -> Gate5Provider:
    """Resolve E3/R3 credentials while forcing the remote rerank endpoint."""

    env = dotenv_values(env_path)
    key = str(env.get("DASHSCOPE_API_KEY") or "").strip()
    if not key:
        raise ProviderCacheError("DASHSCOPE_API_KEY is missing")
    base_url = str(env.get("DASHSCOPE_BASE_URL") or DASHSCOPE_BASE_URL).strip()
    rerank_endpoint = str(
        env.get("DASHSCOPE_RERANK_ENDPOINT_REMOTE")
        or DASHSCOPE_REMOTE_RERANK_ENDPOINT
    ).strip()
    if not base_url or not rerank_endpoint:
        raise ProviderCacheError("DashScope remote endpoint configuration is incomplete")
    return Gate5Provider(
        name="dashscope",
        key=key,
        base_url=base_url.rstrip("/"),
        rerank_endpoint=rerank_endpoint,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def normalize_text(text: str) -> str:
    """Normalize semantically irrelevant formatting without changing content."""

    normalized = unicodedata.normalize("NFC", str(text))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    return normalized.strip()


def make_cache_key(
    operation: str,
    provider_name: str,
    model: str,
    parameters: dict,
    normalized_input: dict,
) -> str:
    request = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "operation": operation,
        "provider": provider_name,
        "model": model,
        "parameters": parameters,
        "input": normalized_input,
    }
    return sha256_json(request)


def _safe_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unknown"


def _jsonable_response(response: Any) -> Any:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "dict"):
        return response.dict()
    raise TypeError(f"cannot serialize provider response: {type(response).__name__}")


def parse_embedding_response(response: Any, expected_count: int) -> list[list[float]]:
    raw = _jsonable_response(response)
    rows = raw.get("data")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(
            f"embedding response expected {expected_count} rows, got "
            f"{len(rows) if isinstance(rows, list) else 'no'} rows"
        )
    ordered = sorted(rows, key=lambda item: int(item["index"]))
    if [int(item["index"]) for item in ordered] != list(range(expected_count)):
        raise ValueError("embedding response indexes are incomplete or duplicated")
    vectors: list[list[float]] = []
    dimension: int | None = None
    for item in ordered:
        vector = item.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise ValueError("embedding response contains an empty vector")
        parsed = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in parsed):
            raise ValueError("embedding response contains non-finite values")
        if dimension is None:
            dimension = len(parsed)
        elif len(parsed) != dimension:
            raise ValueError("embedding response vectors have inconsistent dimensions")
        vectors.append(parsed)
    return vectors


def parse_rerank_response(
    raw: Any,
    expected_count: int,
) -> list[dict]:
    results = raw.get("results") if isinstance(raw, dict) else None
    if results is None and isinstance(raw, dict):
        results = (raw.get("output") or {}).get("results")
    if not isinstance(results, list) or len(results) != expected_count:
        raise ValueError(
            f"rerank response expected {expected_count} rows, got "
            f"{len(results) if isinstance(results, list) else 'no'} rows"
        )
    parsed = [
        {
            "index": int(item["index"]),
            "relevance_score": float(item["relevance_score"]),
        }
        for item in results
    ]
    indexes = sorted(item["index"] for item in parsed)
    if indexes != list(range(expected_count)):
        raise ValueError("rerank response indexes are incomplete or duplicated")
    if not all(math.isfinite(item["relevance_score"]) for item in parsed):
        raise ValueError("rerank response contains non-finite scores")
    return parsed


def classify_failure(error: BaseException) -> str:
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return "timeout"
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code == 429:
        return "rate_limited"
    if status_code is not None:
        return "http_error"
    if isinstance(error, (ValueError, json.JSONDecodeError)):
        return "invalid_response"
    return "provider_error"


class Gate5ProviderCache:
    """Live-through cache with strict, network-free replay."""

    def __init__(self, root: Path, *, mode: str = "live", timeout: float = 60.0):
        if mode not in {"live", "offline"}:
            raise ValueError("mode must be 'live' or 'offline'")
        self.root = root
        self.mode = mode
        self.timeout = timeout

    def embed(
        self,
        provider: Any,
        model: str,
        texts: list[str],
        *,
        batch_size: int = 10,
    ) -> list[list[float]]:
        if "embedding" not in provider.capabilities:
            raise ValueError(f"{provider.name} does not provide embeddings")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        normalized = [normalize_text(text) for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(normalized), batch_size):
            batch = normalized[start : start + batch_size]
            parameters = {"encoding_format": "float"}
            record = self._execute(
                operation="embedding",
                provider=provider,
                model=model,
                parameters=parameters,
                normalized_input={"texts": batch},
                call=lambda: self._call_embedding(provider, model, batch),
                parse=lambda raw: {
                    "vectors": parse_embedding_response(raw, len(batch))
                },
            )
            vectors.extend(record["parsed_result"]["vectors"])
        return vectors

    def rerank(
        self,
        provider: Any,
        model: str,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[dict]:
        if "rerank" not in provider.capabilities or not provider.rerank_endpoint:
            raise ValueError(f"{provider.name} does not provide reranking")
        normalized_query = normalize_text(query)
        normalized_documents = [normalize_text(document) for document in documents]
        top_n = len(normalized_documents) if top_n is None else int(top_n)
        if top_n < 1 or top_n > len(normalized_documents):
            raise ValueError("top_n must be between 1 and document count")
        parameters = {
            "top_n": top_n,
            "return_documents": False,
        }
        record = self._execute(
            operation="rerank",
            provider=provider,
            model=model,
            parameters=parameters,
            normalized_input={
                "query": normalized_query,
                "documents": normalized_documents,
            },
            call=lambda: self._call_rerank(
                provider,
                model,
                normalized_query,
                normalized_documents,
                top_n,
            ),
            parse=lambda raw: {
                "results": parse_rerank_response(raw, top_n)
            },
        )
        return record["parsed_result"]["results"]

    def _call_embedding(self, provider: Any, model: str, texts: list[str]) -> Any:
        if self.mode == "offline":
            raise RuntimeError("offline mode attempted a network call")
        client = OpenAI(
            api_key=provider.key,
            base_url=provider.base_url,
            timeout=self.timeout,
            max_retries=0,
        )
        return client.embeddings.create(
            model=model,
            input=texts,
            encoding_format="float",
        )

    def _call_rerank(
        self,
        provider: Any,
        model: str,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> dict:
        if self.mode == "offline":
            raise RuntimeError("offline mode attempted a network call")
        response = httpx.post(
            provider.rerank_endpoint,
            headers={
                "Authorization": f"Bearer {provider.key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": {"query": query, "documents": documents},
                "query": query,
                "documents": documents,
                "parameters": {
                    "top_n": top_n,
                    "return_documents": False,
                },
                "top_n": top_n,
                "return_documents": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _execute(
        self,
        *,
        operation: str,
        provider: Any,
        model: str,
        parameters: dict,
        normalized_input: dict,
        call: Callable[[], Any],
        parse: Callable[[Any], dict],
    ) -> dict:
        cache_key = make_cache_key(
            operation,
            provider.name,
            model,
            parameters,
            normalized_input,
        )
        cache_path = self._cache_path(operation, provider.name, model, cache_key)
        if cache_path.exists():
            return self._load_record(
                cache_path,
                cache_key=cache_key,
                operation=operation,
                provider_name=provider.name,
                model=model,
                parameters=parameters,
                normalized_input=normalized_input,
            )
        if self.mode == "offline":
            raise ProviderCacheMiss(cache_key, operation)

        started_at = utc_now()
        try:
            raw = call()
            parsed = parse(raw)
        except Exception as error:
            response = getattr(error, "response", None)
            failure_record = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "status": "failed",
                "cache_key": cache_key,
                "operation": operation,
                "provider": provider.name,
                "model": model,
                "parameters": parameters,
                "normalized_input": normalized_input,
                "input_sha256": sha256_json(normalized_input),
                "started_at": started_at,
                "failed_at": utc_now(),
                "failure": {
                    "category": classify_failure(error),
                    "type": type(error).__name__,
                    "message": str(error),
                    "status_code": getattr(error, "status_code", None),
                    "response_body": getattr(response, "text", None),
                },
            }
            self._write_json(self._failure_path(cache_key), failure_record)
            raise ProviderCallFailure(failure_record) from error

        record = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "status": "passed",
            "cache_key": cache_key,
            "operation": operation,
            "provider": provider.name,
            "model": model,
            "parameters": parameters,
            "normalized_input": normalized_input,
            "input_sha256": sha256_json(normalized_input),
            "created_at": utc_now(),
            "raw_response": _jsonable_response(raw),
            "parsed_result": parsed,
            "result_sha256": sha256_json(parsed),
        }
        self._write_json(cache_path, record)
        return record

    def _load_record(
        self,
        path: Path,
        *,
        cache_key: str,
        operation: str,
        provider_name: str,
        model: str,
        parameters: dict,
        normalized_input: dict,
    ) -> dict:
        record = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "status": "passed",
            "cache_key": cache_key,
            "operation": operation,
            "provider": provider_name,
            "model": model,
            "parameters": parameters,
            "normalized_input": normalized_input,
        }
        mismatches = [
            key for key, value in expected.items() if record.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                f"cache record mismatch at {path}: {', '.join(mismatches)}"
            )
        if record.get("input_sha256") != sha256_json(normalized_input):
            raise ValueError(f"cache input hash mismatch at {path}")
        if record.get("result_sha256") != sha256_json(record.get("parsed_result")):
            raise ValueError(f"cache result hash mismatch at {path}")
        return record

    def _cache_path(
        self,
        operation: str,
        provider_name: str,
        model: str,
        cache_key: str,
    ) -> Path:
        return (
            self.root
            / operation
            / _safe_segment(provider_name)
            / _safe_segment(model)
            / f"{cache_key}{CACHE_FILE_SUFFIX}"
        )

    def _failure_path(self, cache_key: str) -> Path:
        return self.root / "failures" / f"{cache_key}{CACHE_FILE_SUFFIX}"

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
