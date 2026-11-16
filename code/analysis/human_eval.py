#!/usr/bin/env python3
"""Validate and analyze anonymized ratings from the blinded human study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


SCORE_FIELDS = (
    "relevance_1_5",
    "coverage_1_5",
    "traceability_1_5",
    "overall_1_5",
)
PRIMARY_FIELD = "overall_1_5"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def map_ratings_to_methods(rows: list[dict], blind_key: dict) -> list[dict]:
    mapping = blind_key.get("mapping", blind_key)
    mapped = []
    for row in rows:
        query_id = row["query_id"]
        label = row["candidate_label"]
        if query_id not in mapping or label not in mapping[query_id]:
            raise ValueError(f"unknown blinded candidate: {query_id}/{label}")
        output = {key: value for key, value in row.items() if key != "candidate_label"}
        output["method"] = mapping[query_id][label]
        mapped.append(output)
    return mapped


def _krippendorff_alpha(
    units: list[list[float]],
    distance,
) -> float:
    cleaned = [
        [float(value) for value in unit if value is not None and value != ""]
        for unit in units
    ]
    paired = [unit for unit in cleaned if len(unit) >= 2]
    values = [value for unit in paired for value in unit]
    if len(values) < 2 or len(set(values)) < 2:
        return 1.0

    observed_numerator = 0.0
    for unit in paired:
        denominator = len(unit) - 1
        for first, second in combinations(unit, 2):
            observed_numerator += 2.0 * distance(first, second) / denominator
    observed = observed_numerator / len(values)

    total = len(values)
    expected_numerator = 0.0
    for first, second in combinations(values, 2):
        expected_numerator += 2.0 * distance(first, second)
    expected = expected_numerator / (total * (total - 1))
    if expected <= 1e-15:
        return 1.0
    return float(1.0 - observed / expected)


def krippendorff_alpha_nominal(units: list[list[float]]) -> float:
    return _krippendorff_alpha(
        units,
        lambda first, second: 0.0 if first == second else 1.0,
    )


def krippendorff_alpha_interval(units: list[list[float]]) -> float:
    return _krippendorff_alpha(
        units,
        lambda first, second: (first - second) ** 2,
    )


def krippendorff_alpha_ordinal(units: list[list[float]]) -> float:
    cleaned = [
        [float(value) for value in unit if value is not None and value != ""]
        for unit in units
    ]
    paired = [unit for unit in cleaned if len(unit) >= 2]
    values = [value for unit in paired for value in unit]
    if len(values) < 2 or len(set(values)) < 2:
        return 1.0

    counts = Counter(values)
    midrank = {}
    cumulative = 0.0
    for value in sorted(counts):
        midrank[value] = cumulative + counts[value] / 2.0
        cumulative += counts[value]

    return _krippendorff_alpha(
        units,
        lambda first, second: (midrank[first] - midrank[second]) ** 2,
    )


def validate_ratings(
    rows: list[dict],
    study: dict,
    blind_key: dict,
) -> tuple[list[dict], list[dict]]:
    errors = []
    warnings = []
    expected_candidates = {
        query["query_id"]: {candidate["label"] for candidate in query["candidates"]}
        for query in study["queries"]
    }
    seen = set()
    valid = []
    for line_number, row in enumerate(rows, start=2):
        participant = (row.get("participant_id") or "").strip()
        query_id = (row.get("query_id") or "").strip()
        label = (row.get("candidate_label") or "").strip()
        if not participant or participant == "UNASSIGNED":
            errors.append(f"line {line_number}: participant_id is empty")
            continue
        if query_id not in expected_candidates:
            errors.append(f"line {line_number}: query_id {query_id!r} is not in the study")
            continue
        if label not in expected_candidates[query_id]:
            errors.append(f"line {line_number}: label {label!r} is invalid")
            continue
        key = (participant, query_id, label)
        if key in seen:
            errors.append(f"line {line_number}: duplicate rating {key}")
            continue
        seen.add(key)

        normalized = dict(row)
        for field in (*SCORE_FIELDS, "preference_rank"):
            raw = (row.get(field) or "").strip()
            if raw:
                try:
                    value = int(raw)
                except ValueError:
                    errors.append(f"line {line_number}: {field} must be an integer")
                    continue
                if field == "preference_rank":
                    if not 1 <= value <= study["methods_per_query"]:
                        errors.append(f"line {line_number}: preference_rank is out of range")
                elif not 1 <= value <= 5:
                    errors.append(f"line {line_number}: {field} is out of range")
                else:
                    normalized[field] = value
            else:
                normalized[field] = None
        valid.append(normalized)

    grouped = defaultdict(list)
    for row in valid:
        grouped[(row["participant_id"], row["query_id"])].append(row)
    for (participant, query_id), query_rows in grouped.items():
        ranks = [row["preference_rank"] for row in query_rows if row["preference_rank"]]
        if len(ranks) != len(set(ranks)):
            errors.append(
                f"participant {participant}, query {query_id}: duplicate preference ranks"
            )
        missing = [row["candidate_label"] for row in query_rows if row[PRIMARY_FIELD] is None]
        if missing:
            warnings.append(
                f"participant {participant}, query {query_id}: missing primary scores for "
                + ", ".join(missing)
            )
    try:
        map_ratings_to_methods(valid, blind_key)
    except ValueError as exc:
        errors.append(str(exc))
    return valid, [*errors, *[f"WARNING: {warning}" for warning in warnings]]


def query_level_method_scores(rows: list[dict]) -> tuple[dict, list[dict]]:
    scores = defaultdict(list)
    for row in rows:
        if row[PRIMARY_FIELD] is None:
            continue
        scores[(row["query_id"], row["method"])].append(float(row[PRIMARY_FIELD]))
    query_scores = {
        key: float(np.mean(values))
        for key, values in scores.items()
    }
    output = [
        {
            "query_id": query_id,
            "method": method,
            PRIMARY_FIELD: score,
        }
        for (query_id, method), score in sorted(query_scores.items())
    ]
    return query_scores, output


def paired_permutation_test(
    reference: np.ndarray,
    baseline: np.ndarray,
    repeats: int,
    seed: int,
) -> dict:
    differences = np.asarray(reference) - np.asarray(baseline)
    if len(differences) == 0:
        return {"difference": None, "ci95": [None, None], "p_value": None}
    observed = float(differences.mean())
    if np.allclose(differences, 0.0):
        return {"difference": 0.0, "ci95": [0.0, 0.0], "p_value": 1.0}
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(repeats, len(differences)))
    permuted = (signs * differences).mean(axis=1)
    p_value = (float(np.sum(np.abs(permuted) >= abs(observed) - 1e-15)) + 1.0) / (
        repeats + 1.0
    )
    return {
        "difference": observed,
        "ci95": [float(value) for value in np.quantile(permuted, [0.025, 0.975])],
        "p_value": p_value,
        "repeats": repeats,
    }


def bootstrap_mean_ci(values: list[float], repeats: int, seed: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repeats, len(array)))
    samples = array[indices].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {"mean": float(array.mean()), "ci95": [float(lower), float(upper)], "n": len(array)}


def alpha_units(rows: list[dict]) -> list[list[float]]:
    participants = sorted({row["participant_id"] for row in rows})
    ratings = defaultdict(dict)
    for row in rows:
        if row[PRIMARY_FIELD] is not None:
            ratings[(row["query_id"], row["candidate_label"])][row["participant_id"]] = row[
                PRIMARY_FIELD
            ]
    return [
        [unit[participant] for participant in participants if participant in unit]
        for unit in ratings.values()
    ]


def preference_rates(rows: list[dict]) -> dict:
    first = Counter()
    total = Counter()
    for row in rows:
        method = row["method"]
        total[method] += 1
        if row["preference_rank"] == 1:
            first[method] += 1
    return {
        method: {
            "first_place": first[method],
            "rated": total[method],
            "rate": first[method] / total[method] if total[method] else None,
        }
        for method in sorted(total)
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted = {}
    running_max = 0.0
    for index, (name, p_value) in enumerate(ordered):
        value = min(1.0, (count - index) * p_value)
        running_max = max(running_max, value)
        adjusted[name] = running_max
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()

    study_path = args.study_dir / "blind-study.json"
    key_path = args.study_dir / "blind-key.json"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    blind_key = json.loads(key_path.read_text(encoding="utf-8"))
    raw_rows = read_csv(args.ratings)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    valid, validation_messages = validate_ratings(raw_rows, study, blind_key)
    errors = [message for message in validation_messages if not message.startswith("WARNING:")]
    if errors:
        payload = {
            "status": "invalid",
            "errors": errors,
            "input_rows": len(raw_rows),
        }
        (args.output_dir / "analysis.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise SystemExit(json.dumps(payload, ensure_ascii=False))
    if not valid:
        if not args.allow_empty:
            raise SystemExit("ratings contain only the header; use --allow-empty for package checks")
        payload = {
            "status": "awaiting_ratings",
            "input_rows": 0,
            "study_id": study["study_id"],
            "expected_candidates": study["query_count"] * study["methods_per_query"],
            "message": "Template validated; no participant ratings are available.",
            "input_hashes": {
                "study": sha256(study_path),
                "blind_key": sha256(key_path),
                "ratings": sha256(args.ratings),
            },
        }
        (args.output_dir / "analysis.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False))
        return

    mapped = map_ratings_to_methods(valid, blind_key)
    _, query_rows = query_level_method_scores(mapped)
    by_method_query = defaultdict(dict)
    for row in query_rows:
        by_method_query[row["method"]][row["query_id"]] = row[PRIMARY_FIELD]
    methods = sorted(by_method_query)
    reference = "Orca-CARE"
    shared = sorted(
        set.intersection(
            *[
                set(by_method_query[method])
                for method in methods
            ]
        )
    )
    summary = {
        method: bootstrap_mean_ci(
            [by_method_query[method][query_id] for query_id in shared],
            args.bootstrap_repeats,
            args.seed + index,
        )
        for index, method in enumerate(methods)
    }
    comparisons = {}
    raw_p_values = {}
    if reference in by_method_query:
        for index, method in enumerate(methods):
            if method == reference:
                continue
            comparison = paired_permutation_test(
                np.asarray([by_method_query[reference][query_id] for query_id in shared]),
                np.asarray([by_method_query[method][query_id] for query_id in shared]),
                args.bootstrap_repeats,
                args.seed + 100 + index,
            )
            comparisons[method] = comparison
            if comparison["p_value"] is not None and method in {
                "GTE-raw",
                "Dense-MaxP",
                "Orca-4D-frozen",
            }:
                raw_p_values[method] = comparison["p_value"]
    for method, p_value in holm_adjust(raw_p_values).items():
        comparisons[method]["p_value_holm"] = p_value
    participants = sorted({row["participant_id"] for row in valid})
    payload = {
        "status": "completed",
        "study_id": study["study_id"],
        "participants": len(participants),
        "valid_ratings": len(valid),
        "complete_query_method_units": len(shared),
        "primary_metric": PRIMARY_FIELD,
        "scale": "1-5 ordinal",
        "missing_policy": (
            "No imputation. Primary comparisons use query-level means only when a method "
            "has at least one primary rating; paired tests use the intersection across methods."
        ),
        "agreement": {
            "coefficient": "Krippendorff alpha (ordinal)",
            "alpha": krippendorff_alpha_ordinal(alpha_units(valid)),
            "units": len(alpha_units(valid)),
        },
        "preference": preference_rates(mapped),
        "summary": summary,
        "comparisons_vs_CARE": comparisons,
        "validation": validation_messages,
        "input_hashes": {
            "study": sha256(study_path),
            "blind_key": sha256(key_path),
            "ratings": sha256(args.ratings),
        },
        "anonymized_rows": [
            {
                "participant_id": row["participant_id"],
                "query_id": row["query_id"],
                "method": row["method"],
                **{field: row[field] for field in (*SCORE_FIELDS, "preference_rank")},
            }
            for row in mapped
        ],
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "query-method-scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "method", PRIMARY_FIELD])
        writer.writeheader()
        writer.writerows(query_rows)
    print(
        json.dumps(
            {
                "status": "completed",
                "participants": len(participants),
                "valid_ratings": len(valid),
                "alpha": payload["agreement"]["alpha"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
