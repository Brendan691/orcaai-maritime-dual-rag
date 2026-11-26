#!/usr/bin/env python3
"""Perform independent model-based citation binding on existing dual answers to produce dual_localized answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

from generate_answers import bind_citations_with_model
from qwen_config import qwen_credentials


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--source-env", type=Path, required=True)
    ap.add_argument("--base-prompt", type=Path, required=True)
    ap.add_argument("--citation-prompt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--query-id", default="")
    ap.add_argument("--query-ids", default="", help="comma-separated query ids")
    ap.add_argument("--model", default="qwen3.8-max-0902")
    a = ap.parse_args()

    rows = [
        json.loads(line)
        for line in a.answers.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    base_prompt_bytes = a.base_prompt.read_bytes()
    citation_prompt_bytes = a.citation_prompt.read_bytes()
    citation_prompts = json.loads(citation_prompt_bytes.decode("utf-8"))
    env = dotenv_values(a.source_env)
    model, api_key, base_url = qwen_credentials(env, a.model)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=1800.0,
        max_retries=2,
    )

    requested = {
        item.strip() for item in a.query_ids.split(",") if item.strip()
    }
    if a.query_id:
        requested.add(a.query_id)
    if not requested:
        requested = {row["query_id"] for row in rows}
    localized_by_query = {}
    for query_id in sorted(requested):
        dual = next(
            row for row in rows
            if row["query_id"] == query_id and row["condition"] == "dual"
        )
        started = time.perf_counter()
        try:
            annotated, annotation_stats, usage = bind_citations_with_model(
                client,
                model,
                citation_prompts,
                dual["query"],
                dual["answer"],
                dual["evidence"],
            )
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            annotated, annotation_stats, usage = "", {}, {}
            status = f"error:{type(exc).__name__}:{exc}"[:200]
        localized_by_query[query_id] = {
            **dual,
            "condition": "dual_localized",
            "answer": annotated,
            "usage": usage,
            "latency_s": round(time.perf_counter() - started, 2),
            "system_prompt_sha256": None,
            "generation_method": "model_citation_binding",
            "model": model,
            "source_answer_condition": "dual",
            "citation_annotation": annotation_stats,
            "citation_prompt_file": str(a.citation_prompt),
            "citation_prompt_version": citation_prompts.get("version"),
            "citation_prompt_sha256": sha256_bytes(citation_prompt_bytes),
            "base_prompt_sha256": sha256_bytes(base_prompt_bytes),
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for row in rows:
        if row["condition"] == "dual_localized":
            continue
        (a.out_dir / f'{row["query_id"]}-{row["condition"]}.json').write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        output_rows.append(row)
        if row["condition"] == "dual" and row["query_id"] in localized_by_query:
            localized = localized_by_query[row["query_id"]]
            (a.out_dir / f'{row["query_id"]}-dual_localized.json').write_text(
                json.dumps(localized, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            output_rows.append(localized)
    (a.out_dir / "answers.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "queries": len(localized_by_query),
        "statuses": {
            query_id: row["status"] for query_id, row in localized_by_query.items()
        },
        "out_dir": str(a.out_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
