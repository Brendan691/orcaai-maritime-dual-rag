#!/usr/bin/env python3
"""Resolve citation markers in answers into paragraph-level anchors (offset, quote, context, content hash, status)."""

from __future__ import annotations

import argparse, hashlib, json, re
from pathlib import Path

SENT = re.compile(r"[^。！？；\n]+[。！？；]?(?:\s*(?:【\d+】|\[\[\d+\]\](?:\([^)\s]+\))?))*")
MARK = re.compile(r"【(\d+)】|\[\[(\d+)\]\](?:\([^)\s]+\))?")
VISIBLE = 5000
DEFAULT_WEB_SNAPSHOTS = Path(
    "experiments/results/gate-p-readiness-20261017/web-snapshots-clean-v2.jsonl"
)


def grams(t: str) -> set[str]:
    t = re.sub(r"\s+", "", t)
    return {t[i : i + 2] for i in range(max(len(t) - 1, 1))}


def support(claim: str, quote: str) -> float:
    a = grams(claim)
    return len(a & grams(quote)) / len(a) if a else 0.0


def load_web_snapshots(path: Path) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], dict]]:
    by_url, by_id = {}, {}
    if not path.exists():
        return by_url, by_id
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query_id = row["query_id"]
        for item in row.get("results", []):
            by_url[(query_id, item["url"])] = item
            if item.get("snapshot_id"):
                by_id[(query_id, item["snapshot_id"])] = item
    return by_url, by_id


def web_paragraph_spans(text: str, target_chars: int = 300) -> list[tuple[int, int]]:
    blocks = []
    pos = 0
    for block in re.split(r"\n\s*\n", text):
        if not block.strip():
            continue
        start = text.find(block, pos)
        lead = len(block) - len(block.lstrip())
        end = start + len(block.rstrip())
        blocks.append((start + lead, end))
        pos = start + len(block)
    if len(blocks) != 1 or "\n\n" in text:
        return blocks

    spans, group_start, group_end = [], 0, 0
    for match in re.finditer(r"[^。！？；]+[。！？；]?", text):
        if not match.group().strip():
            continue
        if group_start == group_end:
            group_start, group_end = match.start(), match.end()
        elif match.end() - group_start <= target_chars:
            group_end = match.end()
        else:
            spans.append((group_start, group_end))
            group_start, group_end = match.start(), match.end()
    if group_start != group_end:
        spans.append((group_start, group_end))
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--paragraphs", type=Path, required=True)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--web-snapshots", type=Path, default=DEFAULT_WEB_SNAPSHOTS)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--min-support", type=float, default=0.20)
    a = ap.parse_args()

    paras: dict[str, list[dict]] = {}
    for line in a.paragraphs.read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            paras.setdefault(p["doc_id"], []).append(p)
    docs = {}
    if a.corpus and a.corpus.exists():
        for line in a.corpus.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                docs[d["doc_id"]] = d.get("text", "")
    web_by_url, web_by_id = load_web_snapshots(a.web_snapshots)
    rows = [json.loads(l) for l in a.answers.read_text(encoding="utf-8").splitlines() if l.strip()]

    out, stats = [], {
        "citations": 0,
        "active": 0,
        "weak": 0,
        "orphaned": 0,
        "unbound": 0,
        "mapping_missing": 0,
    }
    for r in rows:
        for sent in SENT.findall(r["answer"]):
            for m in MARK.finditer(sent):
                n = int(m.group(1) or m.group(2))
                stats["citations"] += 1
                if n < 1 or n > len(r["evidence"]):
                    stats["unbound"] += 1
                    continue
                ev = r["evidence"][n - 1]
                claim = MARK.sub("", sent).strip()
                cands = []
                if ev.get("char_start") is not None:
                    cap = min(ev["char_end"], ev["char_start"] + VISIBLE)
                    for p in paras.get(ev.get("doc_id"), []):
                        if p["char_start"] < cap and p["char_end"] > ev["char_start"]:
                            cands.append(p)
                else:
                    url = ev.get("url") or ev.get("id", "")
                    snapshot_id = ev.get("snapshot_id")
                    snapshot = None
                    if snapshot_id:
                        snapshot = web_by_id.get((r["query_id"], str(snapshot_id)))
                    if snapshot is None and str(url).startswith("http"):
                        snapshot = web_by_url.get((r["query_id"], str(url)))
                    if snapshot is None:
                        stats["mapping_missing"] += 1
                        continue
                    snapshot_id = snapshot["snapshot_id"]
                    text = snapshot["content"][:VISIBLE]
                    url = snapshot["url"]
                    content_hash = snapshot["content_sha256"]
                    docs[snapshot_id] = text
                    for s, e in web_paragraph_spans(text):
                        quote = text[s:e]
                        cands.append({
                            "doc_id": snapshot_id,
                            "url": url,
                            "content_hash": content_hash,
                            "char_start": s,
                            "char_end": e,
                            "quoted_text": quote,
                            "prefix": text[max(0, s - 100):s],
                            "suffix": text[e:e + 100],
                            "line_start": text.count("\n", 0, s) + 1,
                            "line_end": text.count("\n", 0, max(s, e - 1)) + 1,
                        })
                if not cands:
                    stats["unbound"] += 1
                    continue
                best = max(cands, key=lambda p: support(claim, p["quoted_text"]))
                score = support(claim, best["quoted_text"])
                status = "ACTIVE" if score >= a.min_support else ("WEAK" if score >= 0.05 else "ORPHANED")
                if best["doc_id"] in docs:
                    doc = docs[best["doc_id"]]
                    if (doc[best["char_start"]:best["char_end"]] != best["quoted_text"]
                            or hashlib.sha256(doc.encode("utf-8")).hexdigest() != best["content_hash"]):
                        status = "ORPHANED"
                stats[{"ACTIVE": "active", "WEAK": "weak", "ORPHANED": "orphaned"}[status]] += 1
                out.append({"query_id": r["query_id"], "condition": r["condition"], "claim": claim,
                            "support": round(score, 3), "status": status,
                            "doc_id": best["doc_id"], "content_hash": best["content_hash"],
                            "snapshot_id": best["doc_id"] if ev.get("char_start") is None else None,
                            "char_start": best["char_start"], "char_end": best["char_end"],
                            "line_start": best.get("line_start"), "line_end": best.get("line_end"),
                            "quoted_text": best["quoted_text"],
                            "prefix": best.get("prefix", ""), "suffix": best.get("suffix", ""),
                            "url": best.get("url", ""),
                            "visible_chars": len(docs.get(best["doc_id"], ev.get("text", ""))[:VISIBLE]),
                            "evidence_kind": "local" if ev.get("char_start") is not None else "web"})
    a.output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    print(json.dumps({"anchors": len(out), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
