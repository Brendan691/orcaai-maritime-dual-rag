#!/usr/bin/env python3
"""门禁前置核验：产物实测哈希 == FILE-MAP 记录 == 门禁报告记录。

用法：
  .venv313/bin/python experiments/scripts/verify_artifact_hashes.py \
      --corpus experiments/data/raw/chinese-maritime-20261015-v12-FINAL-v9.jsonl \
      --chunks experiments/data/derived/chinese-maritime-20261015-v12-chunks.jsonl \
      --file-map experiments/results/corpus-chinese-20261015-v12-FROZEN-v9/FILE-MAP.json \
      --gate-report experiments/results/corpus-chinese-20261015-v12-FROZEN-v9/gate-1-report.json
退出码：0=全部一致；1=存在不一致（禁止声明门禁通过）。
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
        ("corpus: 文件 == FILE-MAP", corpus_actual == fm.get("corpus_sha256"), corpus_actual[:16]),
        ("corpus: 文件 == 门禁报告", corpus_actual == (gr.get("inputs") or {}).get("corpus_sha256"), corpus_actual[:16]),
        ("chunks: 文件 == FILE-MAP", chunks_actual == fm.get("chunks_sha256"), chunks_actual[:16]),
        ("chunks: 文件 == 门禁报告", chunks_actual == (gr.get("inputs") or {}).get("chunks_sha256"), chunks_actual[:16]),
    ]
    # 逐篇哈希（FILE-MAP 与语料）
    rows = [json.loads(l) for l in a.corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    index = {r["doc_id"]: r for r in rows}
    per_doc_ok = all(
        hashlib.sha256(index[d["doc_id"]]["text"].encode()).hexdigest() == d["sha256"]
        for d in fm.get("per_document", []) if d["doc_id"] in index
    )
    checks.append(("per-document: 语料 == FILE-MAP", per_doc_ok, f'{len(fm.get("per_document", []))} 篇'))
    # 偏移严格比对（原始切片）
    chunks = [json.loads(l) for l in a.chunks.read_text(encoding="utf-8").splitlines() if l.strip()]
    bad = sum(1 for c in chunks if index[c["doc_id"]]["text"][c["char_start"]:c["char_end"]] != c["text"])
    checks.append(("chunk 偏移严格比对（原始切片）", bad == 0, f"不符 {bad} / {len(chunks)} 块"))

    width = max(len(n) for n, _, _ in checks)
    for name, ok, detail in checks:
        print(f"{name:<{width}}  {'PASS' if ok else 'FAIL'}  {detail}")
    failed = [n for n, ok, _ in checks if not ok]
    print("\n结论:", "全部一致，可声明门禁通过" if not failed else f"存在 {len(failed)} 项不一致：{failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
