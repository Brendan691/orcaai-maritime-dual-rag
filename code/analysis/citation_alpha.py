#!/usr/bin/env python3
"""Compute ordinal Krippendorff alpha for answer stage citation-quality overlap.

The script reuses ``krippendorff_alpha_ordinal`` from the project's existing
``analyze_human_eval.py`` implementation (read-only import).  It accepts the
JSON or CSV judgment tables exported by ``C01-20261110.html`` and
``C02-20261110.html``, restricts alpha to the pre-assigned overlap units, and
reports citation precision alongside the agreement statistics.

No external API and no network access are used.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SUPPORT_VALUES = {"full": 1.0, "partial": 0.5, "none": 0.0}
SCRIPT_DIR = Path(__file__).resolve().parent
ANALYZER_PATH = SCRIPT_DIR / "human_eval.py"


class AlphaError(ValueError):
    """Raised when a judgment file or the assignment is invalid."""


def load_analyzer():
    if not ANALYZER_PATH.exists():
        raise AlphaError(f"missing analyzer: {ANALYZER_PATH}")
    spec = importlib.util.spec_from_file_location(
        "gate6_analyze_human_eval", ANALYZER_PATH
    )
    if spec is None or spec.loader is None:
        raise AlphaError(f"cannot import analyzer: {ANALYZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "krippendorff_alpha_ordinal"):
        raise AlphaError("analyzer does not define krippendorff_alpha_ordinal")
    return module


def read_judgments(path: Path) -> tuple[str | None, list[dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            rater = payload.get("rater_id")
            rows = payload.get("judgments")
            if rows is None:
                rows = payload.get("rows")
            if not isinstance(rows, list):
                raise AlphaError(f"{path}: JSON object has no judgments list")
        elif isinstance(payload, list):
            rater = None
            rows = payload
        else:
            raise AlphaError(f"{path}: unsupported JSON root")
    elif suffix == ".csv":
        rater = None
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise AlphaError(f"{path}: expected .json or .csv")

    normalized = []
    seen = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise AlphaError(f"{path}: row {index} is not an object")
        judgment_id = str(row.get("judgment_id") or "").strip()
        if not judgment_id:
            raise AlphaError(f"{path}: row {index} has empty judgment_id")
        if judgment_id in seen:
            raise AlphaError(f"{path}: duplicate judgment_id {judgment_id}")
        seen.add(judgment_id)
        raw_level = row.get("support_level")
        level = None if raw_level is None else str(raw_level).strip()
        if level == "":
            level = None
        if level is not None and level not in SUPPORT_VALUES:
            raise AlphaError(
                f"{path}: judgment {judgment_id} has invalid support_level {level!r}"
            )
        normalized.append(
            {
                "judgment_id": judgment_id,
                "support_level": level,
                "note": "" if row.get("note") is None else str(row.get("note")),
            }
        )
    return rater, normalized


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_report(
    c01_path: Path, c02_path: Path, assignment_path: Path
) -> dict[str, Any]:
    c01_rater, c01_rows = read_judgments(c01_path)
    c02_rater, c02_rows = read_judgments(c02_path)
    if c01_rater is not None and c01_rater != "C01":
        raise AlphaError(f"{c01_path}: rater_id {c01_rater!r} is not C01")
    if c02_rater is not None and c02_rater != "C02":
        raise AlphaError(f"{c02_path}: rater_id {c02_rater!r} is not C02")

    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    overlap_ids = list(assignment.get("overlap_judgment_ids") or [])
    if not overlap_ids:
        raise AlphaError(f"{assignment_path}: no overlap_judgment_ids")
    assigned = {
        rater: set(assignment.get("raters", {}).get(rater, {}).get("judgment_ids", []))
        for rater in ("C01", "C02")
    }
    c01_map = {row["judgment_id"]: row for row in c01_rows}
    c02_map = {row["judgment_id"]: row for row in c02_rows}
    for rater, rows, allowed in (
        ("C01", c01_rows, assigned["C01"]),
        ("C02", c02_rows, assigned["C02"]),
    ):
        unexpected = sorted({row["judgment_id"] for row in rows} - allowed)
        if unexpected:
            raise AlphaError(f"{rater} file contains unassigned judgments: {unexpected[:10]}")

    overlap_units: list[list[float]] = []
    missing_overlap: list[str] = []
    for judgment_id in overlap_ids:
        left = c01_map.get(judgment_id, {}).get("support_level")
        right = c02_map.get(judgment_id, {}).get("support_level")
        if left is None or right is None:
            missing_overlap.append(judgment_id)
            continue
        overlap_units.append([SUPPORT_VALUES[left], SUPPORT_VALUES[right]])

    analyzer = load_analyzer()
    alpha = (
        float(analyzer.krippendorff_alpha_ordinal(overlap_units))
        if overlap_units
        else None
    )
    exact_agreement = (
        sum(1 for left, right in overlap_units if left == right) / len(overlap_units)
        if overlap_units
        else None
    )

    def precision(rows: list[dict[str, Any]]) -> dict[str, Any]:
        values = [
            SUPPORT_VALUES[row["support_level"]]
            for row in rows
            if row["support_level"] in SUPPORT_VALUES
        ]
        return {
            "judged": len(values),
            "missing": sum(1 for row in rows if row["support_level"] is None),
            "mean": mean(values),
            "levels": dict(
                sorted(
                    Counter(
                        row["support_level"] for row in rows if row["support_level"]
                    ).items()
                )
            ),
        }

    unit_values: dict[str, list[float]] = {}
    for row in c01_rows + c02_rows:
        if row["support_level"] in SUPPORT_VALUES:
            unit_values.setdefault(row["judgment_id"], []).append(
                SUPPORT_VALUES[row["support_level"]]
            )
    pooled_scores = [mean(values) for values in unit_values.values()]
    pooled_scores = [value for value in pooled_scores if value is not None]
    overlap_scores = [mean(pair) for pair in overlap_units]
    overlap_scores = [value for value in overlap_scores if value is not None]

    def distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
        return dict(
            sorted(
                Counter(
                    row["support_level"] for row in rows if row["support_level"]
                ).items()
            )
        )

    status = "completed" if overlap_units else "awaiting_judgments"
    return {
        "schema_version": "gate6-citation-quality-alpha-v1",
        "status": status,
        "inputs": {
            "c01": str(c01_path),
            "c02": str(c02_path),
            "assignment": str(assignment_path),
        },
        "value_mapping": SUPPORT_VALUES,
        "overlap_units_assigned": len(overlap_ids),
        "overlap_with_both": len(overlap_units),
        "missing_overlap": missing_overlap,
        "alpha_ordinal": alpha,
        "exact_agreement": exact_agreement,
        "citation_precision": {
            "C01": precision(c01_rows),
            "C02": precision(c02_rows),
            "overlap_unit_mean": {
                "units": len(overlap_scores),
                "mean": mean(overlap_scores),
            },
            "pooled_unit_mean": {
                "units_with_at_least_one_judgment": len(pooled_scores),
                "mean": mean(pooled_scores),
            },
        },
        "level_distributions": {
            "C01": distribution(c01_rows),
            "C02": distribution(c02_rows),
        },
        "note": (
            "Ordinal alpha uses only the pre-assigned overlap units and the "
            "existing analyze_human_eval.py implementation. Missing judgments "
            "are not imputed."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c01", type=Path, required=True)
    parser.add_argument("--c02", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    try:
        report = build_report(args.c01, args.c02, args.assignment)
    except AlphaError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


if __name__ == "__main__":
    main()