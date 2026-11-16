#!/usr/bin/env python3
"""Aggregate repeated citation judgments and calculate agreement."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


FIELDS = (
    "correctness",
    "faithfulness",
    "claims_total",
    "claims_supported",
    "claims_requiring_citation",
    "claims_with_valid_citation",
    "cited_claims_supported",
    "citations_total",
    "citations_valid",
)


def fleiss_kappa(groups: list[list[int]], categories: range = range(1, 6)) -> float | None:
    if not groups or any(len(group) < 2 for group in groups):
        return None
    category_list = list(categories)
    n_raters = len(groups[0])
    if any(len(group) != n_raters for group in groups):
        return None
    counts = np.asarray([[group.count(category) for category in category_list] for group in groups], dtype=float)
    agreement = ((counts * counts).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    proportions = counts.sum(axis=0) / counts.sum()
    expected = float((proportions * proportions).sum())
    if expected >= 1.0:
        return 1.0
    return float((agreement.mean() - expected) / (1.0 - expected))


def ratio(numerator: float, denominator: float, *, bounded: bool = False) -> float | None:
    if not denominator:
        return None
    value = numerator / denominator
    return min(1.0, max(0.0, value)) if bounded else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    answers = {item["record_id"]: item for item in (json.loads(line) for line in args.answers.read_text(encoding="utf-8").splitlines() if line.strip())}
    judgments = [json.loads(line) for line in args.judgments.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for judgment in judgments:
        grouped[judgment["record_id"]].append(judgment)

    records = []
    for record_id, items in grouped.items():
        answer = answers[record_id]
        found_ids = re.findall(r"\[\[\d+\]\]\(([^)]+)\)", answer.get("answer", ""))
        valid_ids = {item["id"] for item in answer.get("evidence", [])}
        aggregate = {field: float(np.median([float(item[field]) for item in items])) for field in FIELDS}
        constraint_violations = sum(
            (
                aggregate["claims_supported"] > aggregate["claims_total"],
                aggregate["claims_with_valid_citation"] > aggregate["claims_requiring_citation"],
                aggregate["cited_claims_supported"] > aggregate["claims_with_valid_citation"],
                aggregate["citations_valid"] > aggregate["citations_total"],
            )
        )
        aggregate.update(
            {
                "record_id": record_id,
                "condition": answer["condition"],
                "replicates": len(items),
                "citation_validity": ratio(aggregate["citations_valid"], aggregate["citations_total"], bounded=True),
                "citation_precision": ratio(aggregate["cited_claims_supported"], aggregate["claims_with_valid_citation"], bounded=True),
                "citation_recall": ratio(aggregate["claims_with_valid_citation"], aggregate["claims_requiring_citation"], bounded=True),
                "claim_support_rate": ratio(aggregate["claims_supported"], aggregate["claims_total"], bounded=True),
                "judge_constraint_violations": constraint_violations,
                "bound_citations_total": answer.get("bound_citations_total", len(found_ids)),
                "bound_citations_valid": answer.get("bound_citations_valid", sum(item in valid_ids for item in found_ids)),
            }
        )
        records.append(aggregate)

    conditions = sorted({record["condition"] for record in records})
    summary = {}
    for condition in conditions:
        selected = [record for record in records if record["condition"] == condition]
        summary[condition] = {}
        for field in ("correctness", "faithfulness", "citation_validity", "citation_precision", "citation_recall", "claim_support_rate"):
            values = [record[field] for record in selected if record[field] is not None]
            summary[condition][field] = float(np.mean(values)) if values else None
        total_bound = sum(record["bound_citations_total"] for record in selected)
        valid_bound = sum(record["bound_citations_valid"] for record in selected)
        summary[condition]["structural_bound_validity"] = ratio(valid_bound, total_bound)
        summary[condition]["judge_constraint_violations"] = int(sum(record["judge_constraint_violations"] for record in selected))
        summary[condition]["answers"] = len(selected)

    agreement = {}
    for field in ("correctness", "faithfulness"):
        groups = [[int(item[field]) for item in grouped[record_id]] for record_id in sorted(grouped) if len(grouped[record_id]) == 3]
        agreement[field] = {
            "fleiss_kappa": fleiss_kappa(groups),
            "complete_agreement_rate": float(np.mean([len(set(group)) == 1 for group in groups])) if groups else None,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "citation-summary.json").write_text(json.dumps({"summary": summary, "agreement": agreement, "records": records}, indent=2), encoding="utf-8")
    with (args.output_dir / "citation-main.csv").open("w", encoding="utf-8", newline="") as output:
        fields = ["condition", "answers", "correctness", "faithfulness", "citation_validity", "citation_precision", "citation_recall", "claim_support_rate", "structural_bound_validity", "judge_constraint_violations"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for condition in conditions:
            writer.writerow({"condition": condition, **summary[condition]})
    print(json.dumps({"summary": summary, "agreement": agreement}, ensure_ascii=False))


if __name__ == "__main__":
    main()
