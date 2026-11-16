#!/usr/bin/env python3
"""Validate, merge, and analyze relevance-judgment ratings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from human_eval import (  # noqa: E402
    krippendorff_alpha_interval,
    krippendorff_alpha_nominal,
    krippendorff_alpha_ordinal,
)

RATERS = ("A01", "A02", "A03")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task_map(paths: dict[str, Path]) -> dict[str, dict[tuple[str, str], dict]]:
    result = {}
    for rater, path in paths.items():
        rows = read_jsonl(path)
        result[rater] = {
            (row["query_id"], row["doc_id"]): row
            for row in rows
        }
        if len(result[rater]) != len(rows):
            raise ValueError(f"duplicate task in {path}")
    return result


def validate_rating_file(
    path: Path,
    expected: dict[tuple[str, str], dict],
    rater: str,
    require_behavior_fields: bool = False,
    expected_page_id: str | None = None,
) -> tuple[list[dict], list[str]]:
    errors = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("rater") != rater:
        errors.append(f"{path}: rater mismatch")
    if expected_page_id is not None and payload.get("page_id") != expected_page_id:
        errors.append(
            f"{path}: page_id mismatch ({payload.get('page_id')} != {expected_page_id})"
        )
    session_elapsed = payload.get("session_elapsed_seconds")
    if require_behavior_fields and (
        isinstance(session_elapsed, bool)
        or not isinstance(session_elapsed, (int, float))
        or session_elapsed < 0
    ):
        errors.append(f"{path}: missing/invalid session_elapsed_seconds")
    rows = payload.get("rows") or []
    seen = set()
    for row in rows:
        key = (row.get("query_id"), row.get("doc_id"))
        if key not in expected:
            errors.append(f"{path}: unassigned task {key}")
            continue
        if key in seen:
            errors.append(f"{path}: duplicate task {key}")
            continue
        seen.add(key)
        relevance = row.get("relevance_0_3")
        confidence = row.get("confidence_1_3")
        if relevance not in (0, 1, 2, 3):
            errors.append(f"{path}: invalid relevance at {key}")
        if confidence not in (1, 2, 3):
            errors.append(f"{path}: invalid confidence at {key}")
        opened = row.get("opened")
        elapsed = row.get("elapsed_seconds")
        if require_behavior_fields and not isinstance(opened, bool):
            errors.append(f"{path}: missing/invalid opened at {key}")
        if opened is not None and not isinstance(opened, bool):
            errors.append(f"{path}: invalid opened at {key}")
        if require_behavior_fields and (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or elapsed < 0
        ):
            errors.append(f"{path}: missing/invalid elapsed_seconds at {key}")
        elif elapsed is not None and (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or elapsed < 0
        ):
            errors.append(f"{path}: invalid elapsed_seconds at {key}")
        if row.get("task_id") != expected[key].get("task_id"):
            errors.append(f"{path}: task_id mismatch at {key}")
    missing = set(expected) - seen
    if missing:
        errors.append(f"{path}: missing {len(missing)} assigned tasks")
    return rows, errors


def normalize_ratings(rows_by_rater: dict[str, list[dict]]) -> dict[tuple[str, str], list[dict]]:
    grouped = defaultdict(list)
    for rater, rows in rows_by_rater.items():
        for row in rows:
            grouped[(row["query_id"], row["doc_id"])].append(row)
    return grouped


def pair_expected_coverage(expected: dict[str, dict[tuple[str, str], dict]]) -> Counter:
    coverage = Counter()
    for rater in RATERS:
        coverage.update(expected[rater].keys())
    return coverage


def requires_adjudication(
    values: list[int],
    expected_count: int,
    has_ambiguity_flag: bool,
) -> bool:
    """Return whether a pair needs AR01 review under the assigned coverage."""
    if not values or len(values) < expected_count:
        return True
    if has_ambiguity_flag:
        return True
    if expected_count >= 3:
        return len(set(values)) == expected_count or max(values) - min(values) >= 2
    if expected_count == 2:
        return len(set(values)) == 2
    return False


def alpha_with_ci(grouped: dict[tuple[str, str], list[dict]], replicates: int = 10000, seed: int = 20261023) -> dict:
    units = [
        [int(row["relevance_0_3"]) for row in rows]
        for rows in grouped.values()
        if len(rows) >= 2
    ]
    coefficients = {
        "ordinal": krippendorff_alpha_ordinal,
        "interval_squared": krippendorff_alpha_interval,
        "nominal": krippendorff_alpha_nominal,
    }
    rng = np.random.default_rng(seed)
    boot = {name: [] for name in coefficients}
    if units:
        for _ in range(replicates):
            sample = [units[int(index)] for index in rng.integers(0, len(units), len(units))]
            for name, coefficient in coefficients.items():
                boot[name].append(coefficient(sample))
    results = {}
    for name, coefficient in coefficients.items():
        samples = boot[name]
        results[name] = {
            "alpha": coefficient(units),
            "ci95": (
                [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
                if samples
                else None
            ),
            "units": len(units),
        }
    return results


def agreement_summary(grouped: dict[tuple[str, str], list[dict]]) -> dict:
    units = [
        [int(row["relevance_0_3"]) for row in rows]
        for rows in grouped.values()
        if len(rows) >= 2
    ]
    return {
        "units": len(units),
        "exact": sum(len(set(values)) == 1 for values in units),
        "within_one": sum(max(values) - min(values) <= 1 for values in units),
    }


def behavior_summary(rows_by_rater: dict[str, list[dict]]) -> dict:
    output = {}
    for rater, rows in sorted(rows_by_rater.items()):
        confidence = Counter(int(row["confidence_1_3"]) for row in rows)
        triage = Counter(row.get("triage", "preview") for row in rows)
        opened_values = [row.get("opened") for row in rows if "opened" in row]
        elapsed_values = [
            float(row["elapsed_seconds"])
            for row in rows
            if "elapsed_seconds" in row
        ]
        output[rater] = {
            "rows": len(rows),
            "confidence_counts": {
                str(level): confidence.get(level, 0) for level in (1, 2, 3)
            },
            "triage_counts": {
                mode: triage.get(mode, 0)
                for mode in ("preview", "context", "full")
            },
            "opened_count": sum(value is True for value in opened_values)
            if len(opened_values) == len(rows)
            else None,
            "opened_rate": (
                sum(value is True for value in opened_values) / len(rows)
                if len(opened_values) == len(rows) and rows
                else None
            ),
            "elapsed_seconds_total": sum(elapsed_values)
            if len(elapsed_values) == len(rows)
            else None,
            "elapsed_seconds_mean": sum(elapsed_values) / len(rows)
            if len(elapsed_values) == len(rows) and rows
            else None,
            "elapsed_seconds_median": float(np.median(elapsed_values))
            if len(elapsed_values) == len(rows) and rows
            else None,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, action="append", required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="formal")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--require-behavior-fields", action="store_true")
    parser.add_argument("--review-manifest", type=Path)
    args = parser.parse_args()

    review_page_ids = {}
    if args.review_manifest:
        review_manifest = json.loads(args.review_manifest.read_text(encoding="utf-8"))
        if review_manifest.get("status") != "passed":
            raise SystemExit("review manifest status is not passed")
        review_page_ids = {
            rater: payload.get("page_id")
            for rater, payload in (review_manifest.get("pages") or {}).items()
        }
        if set(review_page_ids) != set(RATERS):
            raise SystemExit("review manifest must contain A01-A03")

    task_paths = {
        rater: args.tasks_dir / f"{rater}-{args.prefix}-tasks.jsonl"
        for rater in RATERS
    }
    expected = task_map(task_paths)
    ratings_by_rater = {}
    errors = []
    for path in args.ratings:
        rater = json.loads(path.read_text(encoding="utf-8")).get("rater")
        if rater not in RATERS:
            errors.append(f"{path}: unknown rater {rater}")
            continue
        rows, file_errors = validate_rating_file(
            path,
            expected[rater],
            rater,
            require_behavior_fields=args.require_behavior_fields,
            expected_page_id=review_page_ids.get(rater),
        )
        ratings_by_rater[rater] = rows
        errors.extend(file_errors)
    if set(ratings_by_rater) != set(RATERS):
        errors.append("all three rater files are required")
    if errors:
        raise SystemExit("\n".join(errors))

    grouped = normalize_ratings(ratings_by_rater)
    all_pairs = set().union(*(set(expected[r]) for r in RATERS))
    expected_coverage = pair_expected_coverage(expected)
    adjudication_rows = []
    adjudication_keys = set()
    for key in sorted(all_pairs):
        rows = grouped.get(key, [])
        values = [int(row["relevance_0_3"]) for row in rows]
        values.sort()
        needs = requires_adjudication(
            values,
            expected_coverage[key],
            any(bool(row.get("ambiguity_flag")) for row in rows),
        )
        if needs:
            adjudication_keys.add(key)
            adjudication_rows.append({
                "query_id": key[0],
                "doc_id": key[1],
                "A01": next((row["relevance_0_3"] for row in rows if row["rater_id"] == "A01"), ""),
                "A02": next((row["relevance_0_3"] for row in rows if row["rater_id"] == "A02"), ""),
                "A03": next((row["relevance_0_3"] for row in rows if row["rater_id"] == "A03"), ""),
                "range": max(values) - min(values) if values else "",
                "adjudicated_relevance_0_3": "",
                "rationale": "",
            })

    adjudicated = {}
    if args.adjudication:
        with args.adjudication.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                raw = (row.get("adjudicated_relevance_0_3") or "").strip()
                if raw:
                    value = int(raw)
                    if value not in (0, 1, 2, 3):
                        raise SystemExit(f"invalid adjudicated relevance: {row}")
                    adjudicated[(row["query_id"], row["doc_id"])] = {
                        "relevance": value,
                        "rationale": (row.get("rationale") or "").strip(),
                    }
    unknown_adjudications = set(adjudicated) - adjudication_keys
    if unknown_adjudications:
        raise SystemExit(f"adjudication contains unknown cases: {sorted(unknown_adjudications)}")
    for row in adjudication_rows:
        decision = adjudicated.get((row["query_id"], row["doc_id"]))
        if decision:
            row["adjudicated_relevance_0_3"] = decision["relevance"]
            row["rationale"] = decision["rationale"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.out_dir / f"{args.prefix}-merged-ratings.jsonl"
    merged_rows = []
    for key in sorted(grouped):
        for row in grouped[key]:
            merged_rows.append({
                "query_id": key[0],
                "doc_id": key[1],
                "rater_id": row["rater_id"],
                "relevance_0_3": int(row["relevance_0_3"]),
                "confidence_1_3": int(row["confidence_1_3"]),
                "ambiguity_flag": bool(row.get("ambiguity_flag")),
                "triage": row.get("triage", "preview"),
                "opened": row.get("opened"),
                "elapsed_seconds": row.get("elapsed_seconds"),
            })
    merged_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged_rows), encoding="utf-8")
    adjudication_path = args.out_dir / f"{args.prefix}-adjudication.csv"
    with adjudication_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(adjudication_rows[0]) if adjudication_rows else [
            "query_id", "doc_id", "A01", "A02", "A03", "range",
            "adjudicated_relevance_0_3", "rationale",
        ])
        writer.writeheader()
        writer.writerows(adjudication_rows)

    final_rows = []
    unresolved = []
    for key in sorted(all_pairs):
        values = [int(row["relevance_0_3"]) for row in grouped.get(key, [])]
        if key in adjudicated:
            value = adjudicated[key]["relevance"]
        else:
            counts = Counter(values)
            if len(counts) == 1:
                value = values[0]
            elif counts and max(counts.values()) >= 2:
                value = counts.most_common(1)[0][0]
            else:
                value = None
                unresolved.append(key)
        if value is not None:
            final_rows.append((key[0], 0, key[1], value))

    adjudication_complete = all(key in adjudicated for key in adjudication_keys)
    final_qrels_ready = not unresolved and adjudication_complete
    qrels_path = args.out_dir / (
        f"{args.prefix}-qrels.tsv"
        if final_qrels_ready
        else f"{args.prefix}-qrels-provisional.tsv"
    )
    if not unresolved:
        qrels_path.write_text(
            "".join(f"{query_id}\t{iteration}\t{doc_id}\t{relevance}\n"
                    for query_id, iteration, doc_id, relevance in final_rows),
            encoding="utf-8",
        )
    alpha_metrics = alpha_with_ci(grouped, args.bootstrap_replicates)
    report = {
        "status": "passed" if final_qrels_ready else "needs_adjudication",
        "raters": sorted(ratings_by_rater),
        "pairs": len(grouped),
        "expected_coverage": dict(sorted(Counter(expected_coverage.values()).items())),
        "adjudication_rows": len(adjudication_rows),
        "adjudication_complete": adjudication_complete,
        "unresolved_rows": len(unresolved),
        "final_qrels_ready": final_qrels_ready,
        "qrels_provisional": not final_qrels_ready,
        "alpha": alpha_metrics["ordinal"],
        "alpha_metrics": alpha_metrics,
        "agreement": agreement_summary(grouped),
        "behavior": behavior_summary(ratings_by_rater),
        "behavior_fields_required": args.require_behavior_fields,
        "outputs": {
            "merged": str(merged_path),
            "adjudication": str(adjudication_path),
            "qrels": str(qrels_path) if not unresolved else None,
        },
    }
    if review_page_ids:
        report["review_page_ids"] = review_page_ids
    report_path = args.out_dir / f"{args.prefix}-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
