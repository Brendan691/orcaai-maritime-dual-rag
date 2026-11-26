#!/usr/bin/env python3
"""Generate local-only, web-only, and dual-source answers with bound citations.

DEPRECATED (2026-10-17): the generation settings in this script no longer match the frozen protocol.
It covers only three source conditions, uses document-level context and temperature 0.2, and cannot
reproduce the current official experimental configuration. Do not use it to produce new conclusions.

The script is retained because historical results were generated with it and must remain traceable.
New experiments should use the generation entry point for the current frozen configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI


def citation_id(prefix: str, identity: str) -> str:
    return f"{prefix}-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]


def evidence_block(index: int, source_type: str, title: str, text: str, identity: str, url: str = "") -> tuple[str, dict]:
    prefix = "L" if source_type == "local" else "W"
    stable_id = citation_id(prefix, identity)
    marker = f"[[{index}]]({stable_id})"
    evidence = {"index": index, "id": stable_id, "cite_marker": marker, "source_type": source_type, "title": title, "url": url, "text": text[:2500]}
    block = f"<source index=\"{index}\" id=\"{stable_id}\" source_type=\"{source_type}\" cite_marker=\"{marker}\" title=\"{title}\">\n{evidence['text']}\n</source>"
    return block, evidence


def main() -> None:
    print(
        "[DEPRECATED] The condition matrix and generation parameters in generate_dual_source_answers.py are frozen and invalid; "
        "this script is retained for historical reference only and must not be used to produce new conclusions.",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--web", type=Path, required=True)
    parser.add_argument("--source-env", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen-plus")
    args = parser.parse_args()

    env = dotenv_values(args.source_env)
    key = str(env.get("DASHSCOPE_API_KEY") or "").strip()
    base_url = str(env.get("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is missing")
    client = OpenAI(api_key=key, base_url=base_url)
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    documents = {item["doc_id"]: item for item in (json.loads(line) for line in args.documents.read_text(encoding="utf-8").splitlines() if line.strip())}
    queries = {item["query_id"]: item for item in (json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip())}
    retrieval = {row["query_id"]: row for row in (json.loads(line) for line in args.retrieval.read_text(encoding="utf-8").splitlines() if line.strip()) if row["split"] == "test" and row["method"] == "Orca-4D-tuned"}
    web_records = [json.loads(line) for line in args.web.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []

    for web_record in web_records:
        query_id = web_record["query_id"]
        local_blocks, local_evidence = [], []
        for index, doc_id in enumerate(retrieval[query_id]["ranked_doc_ids"][:5], start=1):
            document = documents[doc_id]
            block, evidence = evidence_block(index, "local", document["title"], document["text"], doc_id, document.get("url", ""))
            local_blocks.append(block)
            local_evidence.append(evidence)
        web_blocks, web_evidence = [], []
        for index, item in enumerate(web_record["results"][:5], start=1):
            block, evidence = evidence_block(index, "web", item.get("title", ""), item.get("content", ""), item.get("url", ""), item.get("url", ""))
            web_blocks.append(block)
            web_evidence.append(evidence)
        conditions = {
            "local_only": (local_blocks, local_evidence),
            "web_only": (web_blocks, web_evidence),
            "dual_source": (
                local_blocks + [block.replace(f"index=\"{index}\"", f"index=\"{index + len(local_blocks)}\"").replace(f"[[{index}]]", f"[[{index + len(local_blocks)}]]") for index, block in enumerate(web_blocks, start=1)],
                local_evidence + [{**item, "index": item["index"] + len(local_evidence), "cite_marker": item["cite_marker"].replace(f"[[{item['index']}]]", f"[[{item['index'] + len(local_evidence)}]]")} for item in web_evidence],
            ),
        }
        for condition, (blocks, evidence) in conditions.items():
            evidence = "\n\n".join(blocks)
            prompt = f"Evidence:\n{evidence}\n\nQuestion: {queries[query_id]['query']}\n\nUse only this evidence. At each supported sentence end, copy the source cite_marker exactly."
            try:
                response = client.chat.completions.create(model=args.model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], temperature=0.2, max_tokens=700)
                answer = (response.choices[0].message.content or "").strip()
                record_id = f"{query_id}-{condition}"
                raw = {"record_id": record_id, "question": queries[query_id]["query"], "condition": condition, "answer": answer, "evidence": evidence, "model": args.model, "prompt": prompt}
                (args.raw_dir / f"{record_id}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                results.append({**raw, "prompt": None})
            except Exception as exc:
                failures.append({"query_id": query_id, "condition": condition, "error": str(exc)})
            time.sleep(0.2)

    args.output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in results) + "\n", encoding="utf-8")
    print(json.dumps({"answers": len(results), "failures": failures}, ensure_ascii=False))


if __name__ == "__main__":
    main()
