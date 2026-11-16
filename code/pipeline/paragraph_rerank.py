#!/usr/bin/env python3
"""法规文本的条款/段落感知重排：子单元打分后回填父 chunk。"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import defaultdict


def _call_rerank(query: str, documents: list[str], endpoint: str) -> list[float]:
    body = json.dumps({"query": query, "documents": documents}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    payload = json.loads(urllib.request.urlopen(req, timeout=300).read())
    results = payload.get("output", {}).get("results") or payload.get("results") or []
    scores = [0.0] * len(documents)
    for item in results:
        scores[int(item["index"])] = float(item["relevance_score"])
    return scores


def rerank_documents(query: str, documents: list[str], endpoint: str) -> list[float]:
    """Chunk-only 基线：整块直接参与交叉编码器。"""
    return _call_rerank(query, documents, endpoint)


def split_units(text: str, max_chars: int = 320) -> list[str]:
    units: list[str] = []
    for block in re.split(r"\n\s*\n", text or ""):
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            units.append(block)
            continue
        parts = [part.strip() for part in re.split(r"(?<=[。！？；])", block) if part.strip()]
        group = ""
        for part in parts:
            if not group:
                group = part
            elif len(group) + len(part) <= max_chars:
                group += part
            else:
                units.append(group)
                group = part
        if group:
            units.append(group)
    if not units:
        units = [text.strip()[:max_chars]]
    return [unit for unit in units if unit]


def rerank_units(
    query: str,
    documents: list[str],
    endpoint: str,
    max_chars: int = 320,
) -> tuple[list[float], list[dict]]:
    unit_texts: list[str] = []
    unit_owners: list[int] = []
    for doc_index, document in enumerate(documents):
        for unit in split_units(document, max_chars=max_chars):
            unit_texts.append(unit)
            unit_owners.append(doc_index)
    if not unit_texts:
        return [0.0] * len(documents), []

    scores = _call_rerank(query, unit_texts, endpoint)
    by_doc: dict[int, float] = defaultdict(float)
    details = []
    for unit_index, score in enumerate(scores):
        owner = unit_owners[unit_index]
        by_doc[owner] = max(by_doc[owner], score)
        details.append({
            "document": owner,
            "unit": unit_texts[unit_index],
            "score": score,
        })
    return [float(by_doc.get(i, 0.0)) for i in range(len(documents))], details
