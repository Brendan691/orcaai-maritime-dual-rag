#!/usr/bin/env python3
"""Tokenization, normalization, and excerpt utilities for Chinese retrieval."""

from __future__ import annotations

import re

import numpy as np


_ZH_STOP = set("的了在是我有和就不人都一上也很到说要去你会着看好这那吗呢吧啊哦嗯")


def chinese_tokens(text: str) -> list[str]:
    runs = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[a-zA-Z][a-zA-Z0-9_-]*|\d+(?:[.-]\d+)*", text.lower())
    tokens = []
    for run in runs:
        if re.fullmatch(r"[a-z][a-z0-9_-]*|\d+(?:[.-]\d+)*", run):
            if len(run) > 1 or run.isdigit():
                tokens.append(run)
            continue
        if len(run) == 1:
            if run not in _ZH_STOP:
                tokens.append(run)
            continue
        for index in range(len(run) - 1):
            bigram = run[index : index + 2]
            if not (bigram[0] in _ZH_STOP and bigram[1] in _ZH_STOP):
                tokens.append(bigram)
    return tokens


def normalized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    span = high - low
    normalized = (values - low) / np.where(span < 1e-12, 1.0, span)
    return np.where(span < 1e-12, 0.5, normalized)


def chinese_focused_excerpt(query: str, text: str, window: int = 1800) -> str:
    if len(text) <= window:
        return text
    query_terms = [term for term in chinese_tokens(query) if len(term) >= 2]
    positions = [text.find(term) for term in query_terms]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - window // 4)
    end = min(len(text), start + window)
    if end - start < window:
        start = max(0, end - window)
    return text[start:end]
