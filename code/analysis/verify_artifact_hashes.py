#!/usr/bin/env python3
"""Pre-gate verification: measured artifact hashes == FILE-MAP records == gate report records.

Usage:
  .venv313/bin/python experiments/scripts/verify_artifact_hashes.py \
      --corpus experiments/data/raw/chinese-maritime-20261015-v12-FINAL-v9.jsonl \
      --chunks experiments/data/derived/chinese-maritime-20261015-v12-chunks.jsonl \
      --file-map experiments/results/corpus-chinese-20261015-v12-FROZEN-v9/FILE-MAP.json \
      --gate-report experiments/results/corpus-chinese-20261015-v12-FROZEN-v9/gate-1-report.json
Exit codes: 0 = all consistent; 1 = inconsistency found (gate pass must not be claimed).
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--file-map", type=Path, required=True)
    ap.add_argument("--gate-report", type=Path, required=True)
    a = ap.parse_args()
    fm = json.loads(a.file_map.read_text(encoding="utf-8"))
    gr = json.loads(a.gate_report.read_text(encoding="utf-8"))
    corpus_actual, chunks_actual = sha(a.corpus), sha(a.chunks)
    checks = [
        ("corpus: file == FILE-MAP", corpus_actual == fm.get("corpus_sha256"), corpus_actual[:16]),
        ("corpus: file == gate report", corpus_actual == (gr.get("inputs") or {}).get("corpus_sha256"), corpus_actual[:16]),
        ("chunks: file == FILE-MAP", chunks_actual == fm.get("chunks_sha256"), chunks_actual[:16]),
        ("chunks: file == gate report", chunks_actual == (gr.get("inputs") or {}).get("chunks_sha256"), chunks_actual[:16]),
    ]
    # Per-document hashes (FILE-MAP and corpus)
    rows = [json.loads(l) for l in a.corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    index = {r["doc_id"]: r for r in rows}
    per_doc_ok = all(
        hashlib.sha256(index[d["doc_id"]]["text"].encode()).hexdigest() == d["sha256"]
        for d in fm.get("per_document", []) if d["doc_id"] in index
    )
    checks.append(("per-document: corpus == FILE-MAP", per_doc_ok, f'{len(fm.get("per_document", []))} documents'))
    # Exact offset comparison (original slices)
    chunks = [json.loads(l) for l in a.chunks.read_text(encoding="utf-8").splitlines() if l.strip()]
    bad = sum(1 for c in chunks if index[c["doc_id"]]["text"][c["char_start"]:c["char_end"]] != c["text"])
    checks.append(("chunk offsets exact match (original slices)", bad == 0, f"mismatches {bad} / {len(chunks)} chunks"))

    width = max(len(n) for n, _, _ in checks)
    for name, ok, detail in checks:
        print(f"{name:<{width}}  {'PASS' if ok else 'FAIL'}  {detail}")
    failed = [n for n, ok, _ in checks if not ok]
    print("\nConclusion:", "All consistent; gate pass may be claimed" if not failed else f"Inconsistencies in {len(failed)} item(s): {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
