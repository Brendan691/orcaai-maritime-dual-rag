#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""robustness stage: thread-safe R1 reranker wrapper (engineering hardening version).

Problem fixed
-------------
The frozen answer stage/7/8 runtime (``local_reranker_server.SequenceClassificationBackend``)
shares one Hugging Face *fast* tokenizer instance across threads.  With
``transformers`` 5.8 the fast tokenizer mutates its Rust backend's
truncation/padding settings on every call (``enable_truncation`` /
``enable_padding`` / ``no_truncation`` / ``no_padding``).  Concurrent callers
therefore borrow the same Rust object mutably and PyO3 raises
``RuntimeError: Already borrowed``.  robustness stage observed this in two places:

* ``score()`` -> the frozen runner wraps it as ``FailClosed`` (D07, async);
* ``truncation_count()`` -> ``run_gate6_dev_retrieval.rerank_and_fuse`` catches
  every exception and writes ``truncated_pairs=None`` (D02, async round 10).

Fix
---
This module provides a new, versioned backend that keeps **one independent
tokenizer per thread** (``threading.local``) while sharing the frozen,
read-only R1 model weights.  Tokenization is deterministic and computes the
same token ids as the frozen backend, so candidate scores, ranking and evidence
are unchanged.  ``truncation_count`` is always computed; it never swallows an
exception and never returns ``None``.  If tokenization fails, the exception
propagates to the caller, which must turn it into an explicit error status
(see ``gate8r_output_validation.py``).

Boundaries
----------
* Frozen scripts are not imported-modifiable and are not changed.  This module
  only *uses* the frozen R1 weights at ``run_gate6_dev_retrieval.R1_MODEL_PATH``.
* The shared helper functions ``validate_scores`` and ``sigmoid`` are imported
  from the frozen runtime so that validation and score transforms stay
  bit-identical.
* Only sequence-classification cross encoders are supported.  The frozen R1
  model ``BAAI/bge-reranker-v2-m3`` is an
  ``XLMRobertaForSequenceClassification``; unsupported architectures fail
  loudly instead of silently changing behaviour.

Version: ``gate8r-threadsafe-rerank-v1-20261114``.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import local_reranker_server as frozen_runtime  # noqa: E402

HARDENING_VERSION = "gate8r-threadsafe-rerank-v1-20261114"
BACKEND_NAME = "sequence-classification-cross-encoder"


class UnsupportedRerankerArchitecture(RuntimeError):
    """Raised when the model is not a sequence-classification cross encoder."""


class ThreadSafeSequenceClassificationBackend:
    """Sequence-classification cross encoder with per-thread tokenizers.

    The PyTorch model is loaded once and shared.  It is used read-only under
    ``torch.no_grad()`` in ``eval()`` mode; it never mutates weights or buffers,
    so concurrent forward passes are safe.  Each calling thread lazily obtains
    its own ``AutoTokenizer`` instance, which removes all shared mutable state
    from the fast tokenizer path.
    """

    name = BACKEND_NAME
    hardening_version = HARDENING_VERSION
    tokenizer_isolation = "per-thread"

    def __init__(self, model_path: Path, *, max_length: int, threads: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if threads <= 0:
            raise ValueError("threads must be positive")

        torch.set_num_threads(threads)
        self.torch = torch
        self.model_path = Path(model_path)
        self.max_length = int(max_length)
        self.threads = int(threads)
        self._tokenizer_factory = lambda: AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )
        self._local = threading.local()
        self._diag_lock = threading.Lock()
        self._tokenizer_instances = 0
        self._score_calls = 0
        self._truncation_calls = 0

        self.model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path),
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

    @property
    def tokenizer(self):
        """Return this thread's tokenizer, creating it on first use."""
        tokenizer = getattr(self._local, "tokenizer", None)
        if tokenizer is None:
            tokenizer = self._tokenizer_factory()
            self._local.tokenizer = tokenizer
            with self._diag_lock:
                self._tokenizer_instances += 1
        return tokenizer

    def truncation_count(self, query: str, documents: list[str]) -> int:
        """Count document pairs truncated at ``max_length``.

        Deterministic and never ``None``: any tokenizer failure propagates.
        Uses only this thread's tokenizer instance.
        """
        with self._diag_lock:
            self._truncation_calls += 1
        count = 0
        tokenizer = self.tokenizer
        for document in documents:
            token_ids = tokenizer(
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
        """Score query-document pairs exactly as the frozen backend does."""
        with self._diag_lock:
            self._score_calls += 1
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
        return frozen_runtime.validate_scores(logits, len(documents))

    def diagnostics(self) -> dict:
        with self._diag_lock:
            return {
                "hardening_version": HARDENING_VERSION,
                "backend": self.name,
                "tokenizer_isolation": self.tokenizer_isolation,
                "tokenizer_instances": self._tokenizer_instances,
                "score_calls": self._score_calls,
                "truncation_calls": self._truncation_calls,
                "threads": self.threads,
                "max_length": self.max_length,
                "model_path": str(self.model_path),
            }


def load_threadsafe_backend(model_path: Path, *, max_length: int, threads: int):
    """Load the hardened backend for the frozen R1 model.

    Mirrors the frozen ``local_reranker_server.load_backend`` dispatch, but
    fails loudly for generative listwise architectures that this module does
    not harden.
    """
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"R1 model directory missing: {model_path}")
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    if "Qwen2ForCausalLM" in architectures:
        raise UnsupportedRerankerArchitecture(
            "the robustness stage hardening module supports sequence-classification cross "
            f"encoders only; model {model_path} declares {architectures}"
        )
    return ThreadSafeSequenceClassificationBackend(
        model_path,
        max_length=max_length,
        threads=threads,
    )
