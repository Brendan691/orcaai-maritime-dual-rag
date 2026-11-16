#!/usr/bin/env python3
"""Provider-neutral access for embeddings, rerankers, and chat models."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from dotenv import dotenv_values
from openai import OpenAI


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "model-providers.json"


class MissingProviderCredential(RuntimeError):
    pass


class UnsupportedCapability(RuntimeError):
    pass


@dataclass
class Provider:
    name: str
    key: str = field(repr=False)
    base_url: str
    config: dict
    rerank_endpoint: str | None = None

    @property
    def capabilities(self) -> list[str]:
        return list(self.config.get("capabilities", []))


def load_provider_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_provider(name: str, env_path: Path) -> Provider:
    providers = load_provider_config()["providers"]
    if name not in providers:
        raise KeyError(f"unknown provider: {name}")
    spec = providers[name]
    if name == "local":
        return Provider(
            name=name,
            key="",
            base_url="local",
            config=spec,
        )
    values = dotenv_values(env_path)
    key = str(values.get(spec["api_key_env"]) or "").strip()
    if not key:
        raise MissingProviderCredential(
            f"{spec['api_key_env']} is missing from {env_path}"
        )
    base_url = str(
        values.get(spec.get("base_url_env", ""))
        or spec.get("default_base_url")
        or ""
    ).strip()
    if not base_url:
        raise MissingProviderCredential(
            f"{spec.get('base_url_env', 'base URL')} is missing for {name}"
        )
    rerank_endpoint = None
    if "rerank" in spec.get("capabilities", []):
        rerank_endpoint = str(
            values.get(spec.get("rerank_endpoint_env", ""))
            or spec.get("default_rerank_endpoint")
            or ""
        ).strip() or None
    return Provider(
        name=name,
        key=key,
        base_url=base_url.rstrip("/"),
        config=spec,
        rerank_endpoint=rerank_endpoint,
    )


def openai_client(provider: Provider, timeout: float = 60.0) -> OpenAI:
    return OpenAI(
        api_key=provider.key,
        base_url=provider.base_url,
        timeout=timeout,
        max_retries=2,
    )


def embed_texts(
    provider: Provider,
    model: str,
    texts: list[str],
    *,
    batch_size: int = 10,
) -> list[list[float]]:
    if "embedding" not in provider.capabilities:
        raise UnsupportedCapability(f"{provider.name} does not provide embeddings")
    if provider.name == "local":
        return local_embed(provider, model, texts)
    client = openai_client(provider)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(
            model=model,
            input=texts[start : start + batch_size],
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)
    return vectors


def chat_text(
    provider: Provider,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.2,
    max_tokens: int = 1000,
) -> str:
    if "chat" not in provider.capabilities:
        raise UnsupportedCapability(f"{provider.name} does not provide chat")
    response = openai_client(provider).chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def parse_rerank_payload(payload: dict) -> list[dict]:
    if isinstance(payload.get("results"), list):
        results = payload["results"]
    else:
        results = (payload.get("output") or {}).get("results")
    if not isinstance(results, list):
        raise ValueError("rerank response has no results list")
    return [
        {
            "index": int(item["index"]),
            "relevance_score": float(item["relevance_score"]),
        }
        for item in results
    ]


def rerank_texts(
    provider: Provider,
    model: str,
    query: str,
    documents: list[str],
    *,
    timeout: float = 60.0,
) -> list[dict]:
    if "rerank" not in provider.capabilities or not provider.rerank_endpoint:
        if provider.name == "local":
            return local_rerank(provider, model, query, documents)
        raise UnsupportedCapability(f"{provider.name} does not provide reranking")
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
            "parameters": {"top_n": len(documents), "return_documents": False},
            "top_n": len(documents),
            "return_documents": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_rerank_payload(response.json())


def local_embed(provider: Provider, model: str, texts: list[str]) -> list[list[float]]:
    response = httpx.post(
        provider.config["embedding_url"],
        json={"texts": texts, "batch_size": 8},
        timeout=600.0,
    )
    response.raise_for_status()
    return response.json()["vectors"]


def local_rerank(
    provider: Provider,
    model: str,
    query: str,
    documents: list[str],
) -> list[dict]:
    response = httpx.post(
        provider.config["rerank_url"],
        json={"query": query, "documents": documents},
        timeout=600.0,
    )
    response.raise_for_status()
    return response.json()["results"]


def rerank_batch(
    provider: Provider,
    model: str,
    requests: list[tuple[str, list[str]]],
    *,
    request_batch_size: int = 1,
    on_batch_complete: Callable[[int, list[list[dict]], float], None] | None = None,
) -> list[list[dict]]:
    if not requests:
        return []
    if request_batch_size < 1:
        raise ValueError("request_batch_size must be at least 1")
    groups: list[list[dict]] = []
    for start in range(0, len(requests), request_batch_size):
        chunk = requests[start : start + request_batch_size]
        started = time.perf_counter()
        if provider.name == "local":
            response = httpx.post(
                provider.config["batch_rerank_url"],
                json={
                    "requests": [
                        {"query": query, "documents": documents}
                        for query, documents in chunk
                    ],
                    "batch_size": 32,
                },
                timeout=7200.0,
            )
            response.raise_for_status()
            chunk_groups = response.json()["groups"]
        else:
            chunk_groups = [
                rerank_texts(
                    provider,
                    model,
                    query,
                    documents,
                )
                for query, documents in chunk
            ]
        elapsed = time.perf_counter() - started
        if len(chunk_groups) != len(chunk):
            raise RuntimeError("批量中文重排返回数量不完整")
        groups.extend(chunk_groups)
        if on_batch_complete is not None:
            on_batch_complete(start, chunk_groups, elapsed)
    return groups
