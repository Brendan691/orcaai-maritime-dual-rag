#!/usr/bin/env python3
"""Local reranker service with explicit cross-encoder backends.

The service exposes the DashScope rerank response shape used by the rest of
the project. It supports:

- sequence-classification cross encoders such as BGE reranker v2;
- generative listwise cross encoders such as mxbai-rerank-large-v2.

The backend is selected from ``config.json``. There is no implicit classifier
head construction and therefore no randomly initialized ``score.weight`` path.
"""

from __future__ import annotations

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


BACKEND = None


def sigmoid(value: float) -> float:
    clipped = max(-30.0, min(30.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))


def validate_request(query: Any, documents: Any) -> tuple[str, list[str]]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(documents, list) or not documents:
        raise ValueError("documents must be a non-empty list")
    if not all(isinstance(document, str) and document.strip() for document in documents):
        raise ValueError("every document must be a non-empty string")
    return query.strip(), [document.strip() for document in documents]


def validate_scores(scores: Any, expected_count: int) -> list[float]:
    import numpy as np

    values = np.asarray(scores, dtype=float).reshape(-1).tolist()
    if len(values) != expected_count:
        raise RuntimeError(
            f"reranker returned {len(values)} scores for {expected_count} documents"
        )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("reranker returned a non-finite score")
    return values


def rank_scores(scores: list[float]) -> list[dict]:
    scored = [
        {
            "index": index,
            "relevance_score": round(sigmoid(score), 6),
        }
        for index, score in enumerate(scores)
    ]
    scored.sort(key=lambda item: (-item["relevance_score"], item["index"]))
    indexes = [item["index"] for item in scored]
    if len(indexes) != len(scores) or sorted(indexes) != list(range(len(scores))):
        raise RuntimeError("reranker returned duplicate or invalid document indexes")
    return scored


class SequenceClassificationBackend:
    """Backend for BERT-style models with a sequence-classification head."""

    name = "sequence-classification-cross-encoder"

    def __init__(self, model_path: Path, *, max_length: int, threads: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        self.model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            local_files_only=True,
            output_loading_info=True,
        )
        if loading_info.get("missing_keys"):
            raise RuntimeError(
                "reranker weights are incomplete: "
                + ", ".join(loading_info["missing_keys"][:5])
            )
        if getattr(self.model.config, "num_labels", None) != 1:
            raise RuntimeError(
                "sequence-classification reranker does not expose a single score head"
            )
        self.model.eval()

    def truncation_count(self, query: str, documents: list[str]) -> int:
        count = 0
        for document in documents:
            token_ids = self.tokenizer(
                query,
                document,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            if len(token_ids) > self.max_length:
                count += 1
        return count

    def score(self, query: str, documents: list[str]) -> list[float]:
        batch = self.tokenizer(
            [query] * len(documents),
            documents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self.torch.no_grad():
            logits = self.model(**batch).logits
        return validate_scores(logits, len(documents))


class GenerativeCrossEncoderBackend:
    """Backend for SentenceTransformers 5.x generative cross encoders."""

    name = "sentence-transformers-generative-cross-encoder"

    def __init__(self, model_path: Path, *, max_length: int, threads: int):
        import torch
        from sentence_transformers import CrossEncoder

        torch.set_num_threads(threads)
        self.max_length = max_length
        self.model = CrossEncoder(
            str(model_path),
            device="cpu",
            max_length=max_length,
            local_files_only=True,
        )
        self.tokenizer = self.model.tokenizer

    def truncation_count(self, query: str, documents: list[str]) -> int:
        count = 0
        for document in documents:
            token_ids = self.tokenizer(
                query,
                document,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            if len(token_ids) > self.max_length:
                count += 1
        return count

    def score(self, query: str, documents: list[str]) -> list[float]:
        scores = self.model.predict(
            [(query, document) for document in documents],
            batch_size=len(documents),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return validate_scores(scores, len(documents))


def load_backend(model_path: Path, *, max_length: int, threads: int):
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    if "Qwen2ForCausalLM" in architectures:
        return GenerativeCrossEncoderBackend(
            model_path,
            max_length=max_length,
            threads=threads,
        )
    return SequenceClassificationBackend(
        model_path,
        max_length=max_length,
        threads=threads,
    )


def rerank_payload(query: Any, documents: Any) -> dict:
    if BACKEND is None:
        raise RuntimeError("reranker model is not loaded")
    query_text, normalized_documents = validate_request(query, documents)
    scores = BACKEND.score(query_text, normalized_documents)
    ranked = rank_scores(scores)
    return {
        "output": {
            "results": ranked,
            "truncated_pairs": BACKEND.truncation_count(
                query_text,
                normalized_documents,
            ),
            "max_length": BACKEND.max_length,
            "backend": BACKEND.name,
        },
        "results": ranked,
    }


class Handler(BaseHTTPRequestHandler):
    timeout = 30

    def log_message(self, fmt, *args):
        return

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except TimeoutError:
            self._send(408, {"error": "request timeout"})
            self.close_connection = True

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("request body is empty")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            return body
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": f"bad request: {exc}"})
            return None

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(
                200,
                {
                    "status": "ok",
                    "model_loaded": BACKEND is not None,
                    "backend": getattr(BACKEND, "name", None),
                },
            )
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path not in {"/rerank", "/batch_rerank"}:
            self._send(404, {"error": "not found"})
            return
        body = self._read_body()
        if body is None:
            return
        try:
            if self.path == "/batch_rerank":
                requests = body.get("requests")
                if not isinstance(requests, list) or not requests:
                    raise ValueError("requests must be a non-empty list")
                groups = [
                    rerank_payload(item.get("query"), item.get("documents"))["results"]
                    for item in requests
                    if isinstance(item, dict)
                ]
                if len(groups) != len(requests):
                    raise ValueError("every batch item must be a JSON object")
                self._send(200, {"groups": groups})
                return
            payload = rerank_payload(body.get("query"), body.get("documents"))
            self._send(200, payload)
        except TimeoutError as exc:
            self._send(504, {"error": f"timeout: {exc}"})
        except (ValueError, TypeError) as exc:
            self._send(400, {"error": f"invalid request: {exc}"})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    global BACKEND
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    BACKEND = load_backend(
        args.model_path,
        max_length=args.max_length,
        threads=args.threads,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {
                "status": "listening",
                "url": f"http://{args.host}:{args.port}",
                "backend": BACKEND.name,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
