#!/usr/bin/env python3
"""pilot stage：8 开发查询 × 4 条件 = 32 个答案（统一证据选择 + 冻结提示词 + temperature=0）。"""

from __future__ import annotations

import argparse, hashlib, html, json, math, re, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import dotenv_values
from openai import OpenAI

from paragraph_rerank import rerank_units
from qwen_config import qwen_credentials
from shared_policy import (
    FINAL_EVIDENCE_LIMIT,
    LOCAL_CANDIDATE_LIMIT,
    LOCAL_EVIDENCE_MAX_CHARS,
    WEB_CANDIDATE_LIMIT,
    WEB_EVIDENCE_MAX_CHARS,
    assess_conflict_prior,
    candidate_policy_snapshot,
    extract_year,
    interleave_source_candidates,
    query_intent_analyzer,
)

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "answer-generation.json"
CITATION_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "citation-binding.json"
WEB_PATH = Path("experiments/results/gate-p-readiness-20261017/web-snapshots-clean-v2.jsonl")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compose_system(prompts: dict, citation_rule: str) -> str:
    base = str(prompts["system"]).rstrip()
    insert_before = prompts.get("citation_insert_before")
    rule = citation_rule.strip()
    if not rule:
        return base
    if insert_before and insert_before in base:
        return base.replace(insert_before, f"{rule}\n\n{insert_before}", 1)
    return f"{base}\n\n{rule}"


def citation_id(index: int, evidence: dict) -> str:
    identity = (
        evidence.get("content_sha256")
        or evidence.get("chunk_id")
        or evidence.get("snapshot_id")
        or evidence.get("url")
        or evidence.get("id")
        or "evidence"
    )
    slug = sha256_bytes(str(identity).encode("utf-8"))[:10]
    line_suffix = ""
    if evidence.get("line_start") and evidence.get("line_end"):
        line_suffix = f'-L{evidence["line_start"]}-{evidence["line_end"]}'
    return f"C{index}-e{slug}{line_suffix}"


def evidence_meta(evidence: dict) -> str:
    parts = []
    if evidence.get("title"):
        parts.append(f'title="{html.escape(str(evidence["title"]), quote=True)}"')
    if evidence.get("date"):
        parts.append(f'date="{html.escape(str(evidence["date"]), quote=True)}"')
    if evidence.get("line_start") and evidence.get("line_end"):
        parts.append(
            f'line_range="{evidence["line_start"]}-{evidence["line_end"]}"'
        )
    return (" " + " ".join(parts)) if parts else ""


def text_grams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", re.sub(r"[*_#>`-]", "", text))
    return {compact[i : i + 2] for i in range(max(len(compact) - 1, 1))}


def evidence_overlap(claim: str, evidence: str) -> float:
    claim_grams = text_grams(claim)
    if not claim_grams:
        return 0.0
    return len(claim_grams & text_grams(evidence)) / len(claim_grams)


def strip_citation_markers(text: str) -> str:
    return re.sub(r" \[\[\d+\]\]\([^)]+\)", "", text)


def annotate_answer_citations(answer: str, evidence: list[dict], min_overlap: float = 0.10) -> tuple[str, dict]:
    lines = []
    annotated_sentences = 0
    for line in answer.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            lines.append(line)
            continue
        parts = re.findall(r"[^。！？；]+[。！？；]?", line)
        rendered = []
        for part in parts:
            if not part.strip():
                rendered.append(part)
                continue
            scored = [
                (evidence_overlap(part, item.get("text", "")), index)
                for index, item in enumerate(evidence, 1)
            ]
            score, best = max(scored, default=(0.0, 0))
            if score >= min_overlap and best:
                marker = evidence[best - 1].get("cite_marker") or f"[[{best}]]"
                body = part.rstrip()
                tail = part[len(body):]
                rendered.append(f"{body} {marker}{tail}")
                annotated_sentences += 1
            else:
                rendered.append(part)
        lines.append("".join(rendered))
    annotated = "\n".join(lines)
    if strip_citation_markers(annotated) != answer:
        raise ValueError("citation annotation changed answer body")
    return annotated, {
        "method": "engineering_evidence_overlap_v1",
        "min_overlap": min_overlap,
        "annotated_sentences": annotated_sentences,
        "body_identical_after_strip": True,
    }


def split_binding_sentences(answer: str) -> list[dict]:
    points = []
    offset = 0
    for line in answer.splitlines(keepends=True):
        for match in re.finditer(r"[^。！？；\n]+[。！？；]", line):
            raw = match.group()
            plain = re.sub(r"^\s*(?:#+\s*|[-*]\s+|\d+[.)]\s+)", "", raw).strip()
            plain = plain.replace("**", "")
            substantive = re.sub(r"[\s\W_]+", "", plain)
            if len(substantive) < 6:
                continue
            points.append({
                "id": len(points) + 1,
                "text": plain,
                "start": offset + match.start(),
                "end": offset + match.end(),
            })
        offset += len(line)
    return points


def parse_citation_bindings(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("citation binder returned no JSON object")
        payload = json.loads(text[start : end + 1])
    if isinstance(payload, dict):
        payload = payload.get("bindings")
    if not isinstance(payload, list):
        raise ValueError("citation binder payload is not a binding list")
    bindings = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("citation binding item is not an object")
        sentence = int(item["sentence"])
        markers = item.get("cite_markers") or []
        if not isinstance(markers, list) or not all(isinstance(marker, str) for marker in markers):
            raise ValueError("cite_markers must be a string list")
        unique_markers = list(dict.fromkeys(marker.strip() for marker in markers if marker.strip()))
        if len(unique_markers) > 4:
            raise ValueError("more than 4 citations for one sentence")
        if unique_markers:
            bindings.append({"sentence": sentence, "cite_markers": unique_markers})
    return bindings


def insert_citation_bindings(
    answer: str,
    points: list[dict],
    bindings: list[dict],
    evidence: list[dict],
) -> tuple[str, dict]:
    point_map = {point["id"]: point for point in points}
    allowed = {
        item.get("cite_marker")
        for item in evidence
        if item.get("cite_marker")
    }
    accepted = []
    for binding in bindings:
        sentence = binding["sentence"]
        if sentence not in point_map:
            raise ValueError(f"unknown sentence id: {sentence}")
        unknown = [marker for marker in binding["cite_markers"] if marker not in allowed]
        if unknown:
            raise ValueError(f"unknown cite_marker for sentence {sentence}: {unknown}")
        accepted.append({"sentence": sentence, "cite_markers": binding["cite_markers"]})

    annotated = answer
    for binding in sorted(accepted, key=lambda item: point_map[item["sentence"]]["end"], reverse=True):
        point = point_map[binding["sentence"]]
        marker_text = " " + " ".join(binding["cite_markers"])
        annotated = annotated[: point["end"]] + marker_text + annotated[point["end"] :]
    if strip_citation_markers(annotated) != answer:
        raise ValueError("citation binding changed answer body")
    return annotated, {
        "method": "model_citation_binding_v1",
        "sentences": len(points),
        "bound_sentences": len(accepted),
        "citations": sum(len(item["cite_markers"]) for item in accepted),
        "bindings": accepted,
        "body_identical_after_strip": True,
    }


def bind_citations_with_model(
    client: OpenAI,
    model: str,
    prompts: dict,
    query: str,
    answer: str,
    evidence: list[dict],
    *,
    batch_size: int = 8,
) -> tuple[str, dict, dict]:
    points = split_binding_sentences(answer)
    blocks = []
    for index, item in enumerate(evidence, 1):
        blocks.append(
            prompts["evidence_template"].format(
                index=index,
                citation_id=item["citation_id"],
                cite_marker=item["cite_marker"],
                meta=evidence_meta(item),
                text=item.get("text", "")[:1800],
            )
        )
    evidence_text = "\n\n".join(blocks)
    merged: dict[int, list[str]] = {}
    raw_responses = []
    prompt_hashes = []
    batch_usages = []
    usage_totals = {
        "batch_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for start in range(0, len(points), batch_size):
        batch_points = points[start : start + batch_size]
        sentence_text = "\n".join(
            prompts["sentence_template"].format(
                index=point["id"],
                text=html.escape(point["text"], quote=False),
            )
            for point in batch_points
        )
        user_prompt = prompts["user_template"].format(
            sentences=sentence_text,
            evidence=evidence_text,
            query=query,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=1600,
            response_format={"type": "json_object"},
            # Citation binding is a deterministic mapping task; chain-of-thought
            # adds latency without changing the frozen prompt or evidence.
            extra_body={"enable_thinking": False},
        )
        raw = (response.choices[0].message.content or "").strip()
        try:
            bindings = parse_citation_bindings(raw)
        except Exception as exc:
            raise ValueError(
                f"citation JSON parse failed in batch {start // batch_size + 1}: "
                f"{exc}; raw={raw[:4000]}"
            ) from exc
        allowed_ids = {point["id"] for point in batch_points}
        unknown_sentences = sorted(
            {
                int(item["sentence"])
                for item in bindings
                if int(item["sentence"]) not in allowed_ids
            }
        )
        if unknown_sentences:
            raise ValueError(
                f"citation binder returned sentence ids outside batch: {unknown_sentences}"
            )
        for item in bindings:
            sentence = int(item["sentence"])
            merged.setdefault(sentence, [])
            merged[sentence].extend(item["cite_markers"])
            merged[sentence] = list(dict.fromkeys(merged[sentence]))[:4]
        usage = response.usage.model_dump() if response.usage else {}
        batch_usages.append(usage)
        usage_totals["batch_count"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage_totals[key] += int(usage.get(key) or 0)
        raw_responses.append(raw)
        prompt_hashes.append(sha256_bytes(user_prompt.encode("utf-8")))

    bindings = [
        {"sentence": sentence, "cite_markers": markers}
        for sentence, markers in sorted(merged.items())
    ]
    annotated, stats = insert_citation_bindings(answer, points, bindings, evidence)
    stats["raw_responses"] = raw_responses
    stats["user_prompt_sha256"] = prompt_hashes
    stats["batches"] = usage_totals["batch_count"]
    stats["system_prompt_sha256"] = sha256_bytes(prompts["system"].encode("utf-8"))
    usage_totals["batches"] = batch_usages
    return annotated, stats, usage_totals


def validate_web_snapshots(web: dict[str, dict]) -> None:
    for query_id, row in web.items():
        for item in row.get("results", []):
            snapshot_id = item.get("snapshot_id")
            content = item.get("content")
            expected = item.get("content_sha256")
            if not snapshot_id or content is None or not expected:
                raise ValueError(f"{query_id}: web snapshot lacks id/content/hash")
            actual = sha256_bytes(str(content).encode("utf-8"))
            if actual != expected:
                raise ValueError(f"{query_id}: web snapshot hash mismatch for {snapshot_id}")


def rerank(q: str, docs: list[str], ep: str) -> list[float]:
    scores, _ = rerank_units(q, docs, ep)
    return scores


def gate_w(gate: dict, bm: float, rm: float, ag: float) -> float:
    s, c = gate["standardization"], gate["coefficients"]
    bz = (bm - s["baseline_margin"]["mean"]) / s["baseline_margin"]["scale"]
    rz = (rm - s["gte_margin"]["mean"]) / s["gte_margin"]["scale"]
    az = (ag - s["top1_agreement"]["mean"]) / s["top1_agreement"]["scale"]
    logit = c["intercept"] + c["baseline_margin"] * bz + c["gte_margin"] * rz + c["top1_agreement"] * az
    return 1 / (1 + math.exp(-max(-30, min(30, logit))))


def select(query: str, cands: list[dict], gate: dict, ep: str, e: int) -> tuple[list[dict], float]:
    scores = np.array([c["baseline"] for c in cands], dtype=float)
    sd = scores.std()
    z = (scores - scores.mean()) / sd if sd > 1e-9 else np.zeros_like(scores)
    rr = np.array(rerank(query, [c["text"] for c in cands], ep))
    rr_order = np.argsort(-rr)
    w = gate_w(gate, float(z.max() - np.sort(z)[-2]) if len(z) > 1 else 0.0,
               float(rr[rr_order[0]] - rr[rr_order[1]]) if len(rr) > 1 else 0.0,
               float(np.argmax(z) == rr_order[0]))
    base_rr = {i: 1.0 / (i + 1) for i in np.argsort(-z)}
    rr_rank = {int(idx): 1.0 / (r + 1) for r, idx in enumerate(rr_order)}
    fused = [(1 - w) * base_rr[i] + w * (0.4 * rr_rank[int(i)] + 0.6 * rr[int(i)]) for i in range(len(cands))]
    intent = query_intent_analyzer.analyze(query)
    adjusted = list(fused)
    if intent.conflict_sensitive:
        for index, candidate in enumerate(cands):
            assessment = assess_conflict_prior(
                title=str(candidate.get("title") or ""),
                text=str(candidate.get("text") or ""),
                url=str(candidate.get("url") or ""),
                published_at=candidate.get("date"),
                base_score=float(fused[index]),
                intent=intent,
            )
            adjusted[index] = assessment.final_score
            candidate["conflict_prior"] = assessment.as_dict()
    order = np.argsort(-np.array(adjusted))[:e]
    return [
        dict(
            cands[i],
            rerank=float(rr[i]),
            fused=float(fused[i]),
            prior_adjusted=float(adjusted[i]),
        )
        for i in order
    ], round(w, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--local", type=Path, required=True)
    ap.add_argument("--web", type=Path, default=WEB_PATH)
    ap.add_argument("--gate", type=Path, required=True)
    ap.add_argument("--source-env", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rerank-endpoint", default="http://127.0.0.1:8766/rerank")
    ap.add_argument("--model", default="qwen3.8-max-0902")
    ap.add_argument("--prompt-file", type=Path, default=PROMPT_PATH)
    ap.add_argument("--citation-prompt-file", type=Path, default=CITATION_PROMPT_PATH)
    ap.add_argument("--max-answer-tokens", type=int, default=2400)
    ap.add_argument("--slots", default="", help="comma-separated query slots")
    ap.add_argument("--kind", default="dev")
    ap.add_argument("--skip-localized", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    env = dotenv_values(a.source_env)
    model, api_key, base_url = qwen_credentials(env, a.model)
    prompt_bytes = a.prompt_file.read_bytes()
    prompts = json.loads(a.prompt_file.read_text(encoding="utf-8"))
    citation_prompt_bytes = a.citation_prompt_file.read_bytes()
    citation_prompts = json.loads(a.citation_prompt_file.read_text(encoding="utf-8"))
    NO_CITE = prompts["cite_off"]
    USER_TPL, EV_TPL = prompts["user_template"], prompts["evidence_template"]
    corpus_meta = {}
    for line in Path("experiments/data/raw/chinese-maritime-20261015-v12-FINAL-v9.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        host = re.sub(r"^https?://(www\.)?", "", d.get("url", "")).split("/")[0]
        inferred_year = extract_year(
            str(d.get("title") or ""),
            str(d.get("text") or ""),
            str(d.get("url") or ""),
            d.get("published_at"),
        )
        corpus_meta[d["doc_id"]] = {
            "pub": host,
            "date": str(d.get("published_at") or d.get("year") or inferred_year or ""),
            "title": str(d.get("title") or ""),
        }
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=1800.0,
        max_retries=2,
    )
    gate = json.loads(a.gate.read_text(encoding="utf-8"))
    local = {r["query_id"]: r for r in read_jsonl(a.local)}
    web = {r["query_id"]: r for r in read_jsonl(a.web)}
    validate_web_snapshots(web)
    slots = {slot.strip() for slot in a.slots.split(",") if slot.strip()}
    qs = [
        q for q in read_jsonl(a.queries)
        if (q["slot"] in slots if slots else q.get("kind") == a.kind)
    ]
    if a.limit:
        qs = qs[: a.limit]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in qs:
        qid, text = q["slot"], q["text"]
        loc = [{"id": h["chunk_id"], "doc_id": h["doc_id"], "text": "", "baseline": h["fused"],
                "char_start": h["char_start"], "char_end": h["char_end"]}
               for h in local[qid]["hits"][:LOCAL_CANDIDATE_LIMIT]]
        chunks = {c["chunk_id"]: c for c in read_jsonl(Path("experiments/data/derived/chinese-maritime-20261015-v12-chunks.jsonl"))}
        for c in loc:
            c["text"] = chunks[c["id"]]["content"]
            m = corpus_meta.get(c["doc_id"], {})
            c["publisher"], c["date"] = m.get("pub", ""), m.get("date", "")
            c["title"] = m.get("title", "")
            c["line_start"], c["line_end"] = chunks[c["id"]].get("line_start"), chunks[c["id"]].get("line_end")
        ws = [{"id": r["snapshot_id"], "snapshot_id": r["snapshot_id"], "url": r["url"],
               "text": r["content"], "baseline": r["score"] or 0.0,
               "query_variant": r.get("query_variant", ""),
               "variant_rule": r.get("variant_rule", ""),
               "title": r.get("title", ""),
               "publisher": re.sub(r"^https?://(www\.)?", "", r["url"]).split("/")[0],
               "date": "", "content_sha256": r["content_sha256"],
               "fetched_at": r.get("fetched_at", "")}
              for r in web[qid]["results"][:WEB_CANDIDATE_LIMIT]]
        conditions = {
            "local_only": (loc, NO_CITE),
            "web_only": (ws, NO_CITE),
            "dual": (
                [
                    candidate.item
                    for candidate in interleave_source_candidates(loc, ws)
                ],
                NO_CITE,
            ),
        }
        condition_rows = {}
        for cond, (cands, cite_rule) in conditions.items():
            t0 = time.perf_counter()
            chosen, w = select(text, cands, gate, a.rerank_endpoint, FINAL_EVIDENCE_LIMIT)
            blocks = []
            for i, c in enumerate(chosen, 1):
                cite_id = citation_id(i, c)
                cite_marker = f"[[{i}]]({cite_id})"
                c["citation_id"], c["cite_marker"] = cite_id, cite_marker
                text_limit = (
                    WEB_EVIDENCE_MAX_CHARS
                    if c.get("snapshot_id")
                    else LOCAL_EVIDENCE_MAX_CHARS
                )
                blocks.append(EV_TPL.format(
                    index=i,
                    citation_id=cite_id,
                    cite_marker=cite_marker,
                    meta=evidence_meta(c),
                    text=c["text"][:text_limit],
                ))
            prompt = USER_TPL.format(evidence="\n\n".join(blocks), query=text)
            system_prompt = compose_system(prompts, cite_rule)
            try:
                r = client.chat.completions.create(model=model,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": prompt}],
                    temperature=0, max_tokens=a.max_answer_tokens)
                answer = (r.choices[0].message.content or "").strip()
                usage = r.usage.model_dump() if r.usage else {}
                status = "ok"
            except Exception as exc:  # noqa: BLE001
                answer, usage, status = "", {}, f"error:{type(exc).__name__}:{exc}"[:200]
            row = {"query_id": qid, "condition": cond, "query": text, "answer": answer,
                   "care_weight": w, "status": status,
                   "model": model,
                   "latency_s": round(time.perf_counter() - t0, 2),
                   "usage": usage, "evidence": chosen,
                   "candidate_policy": candidate_policy_snapshot(),
                   "prompt_file": str(a.prompt_file),
                   "prompt_version": prompts.get("version"),
                   "prompt_sha256": sha256_bytes(prompt_bytes),
                   "system_prompt_sha256": sha256_bytes(system_prompt.encode("utf-8")),
                   "generation_method": "model",
                   "web_snapshot_file": str(a.web),
                   "created_at": datetime.now(timezone.utc).isoformat()}
            (a.out_dir / f"{qid}-{cond}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(row)
            condition_rows[cond] = row
            print(qid, cond, status, "%.1fs" % row["latency_s"], len(answer))
        if a.skip_localized:
            continue
        dual_row = condition_rows["dual"]
        if dual_row["status"] == "ok" and dual_row["answer"]:
            binder_start = time.perf_counter()
            try:
                annotated, annotation_stats, binder_usage = bind_citations_with_model(
                    client,
                    model,
                    citation_prompts,
                    text,
                    dual_row["answer"],
                    dual_row["evidence"],
                )
                localized_status = "ok"
            except Exception as exc:  # noqa: BLE001
                annotated = ""
                annotation_stats = {}
                binder_usage = {}
                localized_status = f"error:{type(exc).__name__}:{exc}"[:200]
            localized_row = {
                **dual_row,
                "condition": "dual_localized",
                "answer": annotated,
                "usage": binder_usage,
                "latency_s": round(time.perf_counter() - binder_start, 2),
                "system_prompt_sha256": None,
                "generation_method": "model_citation_binding",
                "model": model,
                "source_answer_condition": "dual",
                "citation_annotation": annotation_stats,
                "citation_prompt_file": str(a.citation_prompt_file),
                "citation_prompt_version": citation_prompts.get("version"),
                "citation_prompt_sha256": sha256_bytes(citation_prompt_bytes),
                "status": localized_status,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        else:
            localized_row = {
                **dual_row,
                "condition": "dual_localized",
                "answer": "",
                "usage": {},
                "latency_s": 0.0,
                "system_prompt_sha256": None,
                "generation_method": "model_citation_binding",
                "model": model,
                "source_answer_condition": "dual",
                "status": "error:dual_answer_unavailable",
                "citation_prompt_file": str(a.citation_prompt_file),
                "citation_prompt_version": citation_prompts.get("version"),
                "citation_prompt_sha256": sha256_bytes(citation_prompt_bytes),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        (a.out_dir / f"{qid}-dual_localized.json").write_text(
            json.dumps(localized_row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append(localized_row)
        print(qid, "dual_localized", localized_row["status"], "%.1fs" % localized_row["latency_s"], len(localized_row["answer"]))
    (a.out_dir / "answers.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"answers": len(rows), "errors": sum(1 for r in rows if r["status"] != "ok")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
