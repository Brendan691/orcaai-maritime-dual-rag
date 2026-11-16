#!/usr/bin/env python3
"""Frozen answer stage statistical estimators for synthetic-fixture implementation tests.

This module implements the D02-D12 rules frozen in
``experiments/results/gate6-d12-freeze-and-implementation-authorization-20261105``.
It does not read confirmation-set material. The bundle runner adds a separate
fail-closed synthetic-fixture guard around this module.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


QUERY_TYPES = (
    "complementary",
    "conflicting",
    "local_enough",
    "web_enough",
)
CONDITIONS = ("local", "web", "dual")
B_REGION_QUERY_TYPES = frozenset(("complementary", "conflicting"))
S_REGION_QUERY_TYPES = frozenset(("local_enough", "web_enough"))
RATER_IDS = tuple(f"B{index:02d}" for index in range(1, 8))

EXPECTED_QUERY_COUNT = 32
EXPECTED_ANSWERS = 96
EXPECTED_B_QUERIES = 16
EXPECTED_S_QUERIES = 16
EXPECTED_ANCHORS = 16
EXPECTED_RATERS = 7
MAIN_ALPHA = 0.05
TOST_BOUND_D = 0.5
TOST_ALPHA = 0.05
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_REPLICATES = 10_000
INTERACTION_PERMUTATION_SEED = 20260826
INTERACTION_PERMUTATION_REPLICATES = 100_000
ANCHOR_ALPHA_THRESHOLD = 0.67
LOO_SPEARMAN_THRESHOLD = 0.30
MAX_RATER_EXCLUSIONS = 2
MIN_RATERS_PER_ANSWER = 2
EXPLORATORY_ALPHA = 0.05
SOURCE_PAIRS = (("local", "web"), ("local", "dual"), ("web", "dual"))
EXPLORATORY_FAMILY_NAME = "within_query_type_source_contrasts"
EXPLORATORY_TEST_COUNT = len(QUERY_TYPES) * len(SOURCE_PAIRS)

C1_SINGLE_SOURCE = -1.0 / 3.0
C1_DUAL = 2.0 / 3.0
CONDITION_CODING = {
    "local": {"C1": C1_SINGLE_SOURCE, "C2": 1.0},
    "web": {"C1": C1_SINGLE_SOURCE, "C2": -1.0},
    "dual": {"C1": C1_DUAL, "C2": 0.0},
}


class Gate6StatisticsError(ValueError):
    """Raised when an input violates the frozen answer stage statistics contract."""


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or value is None:
        raise Gate6StatisticsError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Gate6StatisticsError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise Gate6StatisticsError(f"{field} must be a finite number")
    return number


def _percentile_ci(values: np.ndarray) -> list[float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    lower, upper = np.quantile(finite, [0.025, 0.975], method="linear")
    return [float(lower), float(upper)]


def _sample_sd(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return math.nan
    return float(np.std(array, ddof=1))


def exact_sign_flip_main(
    differences: Sequence[float],
    *,
    expected_n: int | None = None,
) -> dict[str, Any]:
    """D02/D05 paired sign-flip test.

    The p-value is ``count(mean(sign * D) >= observed_mean) / 2**n`` with no
    Monte Carlo correction and with ties counted. For the frozen full design
    ``n=16`` this is exactly 65,536 sign assignments.
    """

    values = np.asarray([_finite_float(value, "D_q") for value in differences], dtype=np.float64)
    n = int(values.size)
    if n < 2:
        return {
            "method": "exact_paired_sign_flip",
            "status": "not_estimable",
            "reason": "fewer than two complete paired queries",
            "n": n,
            "expected_n": expected_n,
        }
    if expected_n is not None and n != expected_n:
        raise Gate6StatisticsError(f"expected n={expected_n}, received n={n}")
    if n > 20:
        raise Gate6StatisticsError("exact sign-flip enumeration is limited to n<=20")

    observed = float(values.mean())
    bit_indexes = np.arange(n, dtype=np.uint32)
    assignments = (np.arange(2**n, dtype=np.uint32)[:, None] >> bit_indexes) & 1
    signs = 1.0 - 2.0 * assignments.astype(np.float64)
    permuted_means = signs @ values / float(n)
    exceedance_count = int(np.count_nonzero(permuted_means >= observed))
    sign_flip_count = int(2**n)
    sd = _sample_sd(values)
    return {
        "method": "exact_paired_sign_flip",
        "status": "estimated",
        "n": n,
        "expected_n": expected_n,
        "observed_mean_D": observed,
        "d_z": float(observed / sd) if sd > 0.0 else None,
        "sign_flip_count": sign_flip_count,
        "exceedance_count": exceedance_count,
        "p_raw": float(exceedance_count / sign_flip_count),
        "ties_retained": True,
        "monte_carlo_plus_one": False,
        "alternative": "greater",
    }


def exact_sign_flip_two_sided(
    differences: Sequence[float],
    *,
    expected_n: int | None = None,
) -> dict[str, Any]:
    """Exploratory paired two-sided sign-flip with ties retained."""

    values = np.asarray(
        [_finite_float(value, "paired difference") for value in differences],
        dtype=np.float64,
    )
    n = int(values.size)
    if expected_n is not None and n != expected_n:
        raise Gate6StatisticsError(f"expected n={expected_n}, received n={n}")
    if n < 2:
        return {
            "method": "exact_paired_sign_flip",
            "status": "not_estimable",
            "reason": "fewer than two paired queries",
            "n": n,
            "expected_n": expected_n,
            "alternative": "two-sided",
            "p_raw": None,
        }
    if n > 20:
        raise Gate6StatisticsError("exact sign-flip enumeration is limited to n<=20")

    observed = float(values.mean())
    bit_indexes = np.arange(n, dtype=np.uint32)
    assignments = (np.arange(2**n, dtype=np.uint32)[:, None] >> bit_indexes) & 1
    signs = 1.0 - 2.0 * assignments.astype(np.float64)
    permuted_means = signs @ values / float(n)
    exceedance_count = int(np.count_nonzero(np.abs(permuted_means) >= abs(observed)))
    sign_flip_count = int(2**n)
    return {
        "method": "exact_paired_sign_flip",
        "status": "estimated",
        "n": n,
        "expected_n": expected_n,
        "estimate": observed,
        "sign_flip_count": sign_flip_count,
        "exceedance_count": exceedance_count,
        "p_raw": float(exceedance_count / sign_flip_count),
        "ties_retained": True,
        "alternative": "two-sided",
    }


def _holm_adjusted_values(raw_p_values: Mapping[str, float]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for name, value in raw_p_values.items():
        number = _finite_float(value, f"p value for {name}")
        if not 0.0 <= number <= 1.0:
            raise Gate6StatisticsError(f"p value for {name} must be in [0, 1]")
        clean[name] = number
    if not clean:
        raise Gate6StatisticsError("Holm family must contain at least one test")
    ordered = sorted(clean.items(), key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for rank, (name, p_value) in enumerate(ordered):
        candidate = (family_size - rank) * p_value
        running = max(running, candidate)
        adjusted[name] = min(1.0, running)
    return adjusted


def exploratory_within_type_holm(population: Mapping[str, Any]) -> dict[str, Any]:
    """D08 exploratory Holm family of within-query-type source contrasts.

    This family is computed only from the twelve within-type pairs and is never
    pooled with the three-test confirmatory Holm family.
    """

    rows = list(population.get("query_rows") or [])
    tests: dict[str, dict[str, Any]] = {}
    family_inputs: dict[str, float] = {}
    for query_type in QUERY_TYPES:
        type_rows = [row for row in rows if row.get("query_type") == query_type]
        for first, second in SOURCE_PAIRS:
            label = f"{query_type}:{first}_vs_{second}"
            differences = [
                float(row[f"score_{first}"] - row[f"score_{second}"])
                for row in type_rows
                if row.get(f"score_{first}") is not None
                and row.get(f"score_{second}") is not None
            ]
            result = exact_sign_flip_two_sided(differences)
            family_inputs[label] = (
                1.0 if result["p_raw"] is None else float(result["p_raw"])
            )
            tests[label] = {
                "query_type": query_type,
                "comparison": f"{first}_vs_{second}",
                "direction": f"{first} - {second}",
                "status": result["status"],
                "n": result["n"],
                "estimate": result.get("estimate"),
                "method": result["method"],
                "alternative": result["alternative"],
                "sign_flip_count": result.get("sign_flip_count"),
                "exceedance_count": result.get("exceedance_count"),
                "raw_p": result["p_raw"],
                "holm_input_p": family_inputs[label],
                "exploratory": True,
            }
    adjusted = _holm_adjusted_values(family_inputs)
    for label, test in tests.items():
        test["adjusted_p"] = adjusted[label]
        test["reject_at_0_05"] = bool(
            test["raw_p"] is not None and adjusted[label] < EXPLORATORY_ALPHA
        )
    return {
        "family": EXPLORATORY_FAMILY_NAME,
        "exploratory": True,
        "independent_of_confirmatory_family": True,
        "alpha": EXPLORATORY_ALPHA,
        "test_count": len(tests),
        "expected_test_count": EXPLORATORY_TEST_COUNT,
        "tests": tests,
    }


def interaction_permutation(
    b_region: Sequence[float],
    s_region: Sequence[float],
    *,
    replicates: int = INTERACTION_PERMUTATION_REPLICATES,
    seed: int = INTERACTION_PERMUTATION_SEED,
    expected_group_size: int | None = EXPECTED_B_QUERIES,
) -> dict[str, Any]:
    """D03/D05 interaction-only Monte Carlo label permutation."""

    b_values = np.asarray([_finite_float(value, "B D_q") for value in b_region], dtype=np.float64)
    s_values = np.asarray([_finite_float(value, "S D_q") for value in s_region], dtype=np.float64)
    if replicates < 1:
        raise Gate6StatisticsError("replicates must be positive")
    if expected_group_size is not None and (
        b_values.size != expected_group_size or s_values.size != expected_group_size
    ):
        return {
            "method": "monte_carlo_region_label_permutation",
            "status": "not_estimable",
            "reason": (
                "frozen permutation requires "
                f"{expected_group_size} B and {expected_group_size} S queries; got "
                f"{b_values.size} and {s_values.size}"
            ),
            "n_B": int(b_values.size),
            "n_S": int(s_values.size),
            "expected_group_size": expected_group_size,
            "replicates": int(replicates),
            "seed": int(seed),
        }
    if b_values.size < 1 or s_values.size < 1:
        return {
            "method": "monte_carlo_region_label_permutation",
            "status": "not_estimable",
            "reason": "empty B or S region",
            "n_B": int(b_values.size),
            "n_S": int(s_values.size),
            "replicates": int(replicates),
            "seed": int(seed),
        }

    observed = float(b_values.mean() - s_values.mean())
    combined = np.concatenate((b_values, s_values))
    b_count = int(b_values.size)
    total = int(combined.size)
    rng = np.random.default_rng(seed)
    base_labels = np.broadcast_to(np.arange(total, dtype=np.int16), (int(replicates), total))
    permuted_labels = rng.permuted(base_labels, axis=1)
    permuted_values = np.take_along_axis(combined[None, :], permuted_labels, axis=1)
    permuted_interactions = permuted_values[:, :b_count].mean(axis=1) - permuted_values[
        :, b_count:
    ].mean(axis=1)
    exceedance_count = int(
        np.count_nonzero(np.abs(permuted_interactions) >= abs(observed))
    )
    p_value = float((1 + exceedance_count) / (int(replicates) + 1))

    pooled_sd = math.nan
    if b_values.size >= 2 and s_values.size >= 2:
        numerator = (b_values.size - 1) * float(np.var(b_values, ddof=1)) + (
            s_values.size - 1
        ) * float(np.var(s_values, ddof=1))
        denominator = b_values.size + s_values.size - 2
        pooled_sd = math.sqrt(numerator / denominator)

    return {
        "method": "monte_carlo_region_label_permutation",
        "status": "estimated",
        "n_B": int(b_values.size),
        "n_S": int(s_values.size),
        "expected_group_size": expected_group_size,
        "observed_interaction": observed,
        "d_pooled": float(observed / pooled_sd) if pooled_sd > 0.0 else None,
        "replicates": int(replicates),
        "seed": int(seed),
        "exceedance_count": exceedance_count,
        "monte_carlo_denominator": int(replicates) + 1,
        "p_raw": p_value,
        "alternative": "two-sided",
        "separate_within_group_sign_flips": False,
    }


def noncentral_t_tost(
    paired_differences: Sequence[float],
    *,
    bound_d: float = TOST_BOUND_D,
    alpha: float = TOST_ALPHA,
    expected_n: int | None = None,
) -> dict[str, Any]:
    """D04 paired noncentral-t TOST with the frozen ncp parameterization."""

    values = np.asarray(
        [_finite_float(value, "N_q") for value in paired_differences], dtype=np.float64
    )
    n = int(values.size)
    if expected_n is not None and n != expected_n:
        return {
            "method": "noncentral_t_tost",
            "status": "not_estimable",
            "reason": f"expected n={expected_n}, received n={n}",
            "n": n,
            "expected_n": expected_n,
            "bound_d": float(bound_d),
            "alpha_each_side": float(alpha),
            "p_lower": 1.0,
            "p_upper": 1.0,
            "p_raw": 1.0,
            "p_TOST": 1.0,
            "tost_not_rejected": True,
        }
    if n < 2:
        return {
            "method": "noncentral_t_tost",
            "status": "not_estimable",
            "reason": "fewer than two paired queries",
            "n": n,
            "expected_n": expected_n,
            "bound_d": float(bound_d),
            "alpha_each_side": float(alpha),
            "p_lower": 1.0,
            "p_upper": 1.0,
            "p_raw": 1.0,
            "p_TOST": 1.0,
            "tost_not_rejected": True,
        }

    mean_value = float(values.mean())
    sd = _sample_sd(values)
    df = n - 1
    if not math.isfinite(sd) or sd == 0.0:
        return {
            "method": "noncentral_t_tost",
            "status": "not_estimable",
            "reason": "S=0; frozen rule treats TOST as not rejected",
            "n": n,
            "expected_n": expected_n,
            "M": mean_value,
            "S": 0.0,
            "SE": 0.0,
            "d_z": None,
            "df": df,
            "bound_d": float(bound_d),
            "alpha_each_side": float(alpha),
            "Delta_raw": 0.0,
            "p_lower": 1.0,
            "p_upper": 1.0,
            "p_raw": 1.0,
            "p_TOST": 1.0,
            "tost_not_rejected": True,
            "equivalence_supported_uncorrected": False,
        }

    standard_error = float(sd / math.sqrt(n))
    t_observed = float(mean_value / standard_error)
    ncp = float(bound_d * math.sqrt(n))
    p_lower = float(1.0 - stats.nct.cdf(t_observed, df, -ncp))
    p_upper = float(stats.nct.cdf(t_observed, df, ncp))
    p_lower = min(1.0, max(0.0, p_lower))
    p_upper = min(1.0, max(0.0, p_upper))
    p_tost = max(p_lower, p_upper)
    t_critical_95 = float(stats.t.ppf(0.95, df))
    ci90_lower = float(mean_value - t_critical_95 * standard_error)
    ci90_upper = float(mean_value + t_critical_95 * standard_error)
    return {
        "method": "noncentral_t_tost",
        "status": "estimated",
        "n": n,
        "expected_n": expected_n,
        "M": mean_value,
        "S": sd,
        "SE": standard_error,
        "df": df,
        "d_z": float(mean_value / sd),
        "t_observed": t_observed,
        "bound_d": float(bound_d),
        "bound_raw_description": float(bound_d * sd),
        "Delta_raw": float(bound_d * sd),
        "alpha_each_side": float(alpha),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_raw": p_tost,
        "p_TOST": p_tost,
        "tost_not_rejected": not (p_lower < alpha and p_upper < alpha),
        "equivalence_supported_uncorrected": bool(p_lower < alpha and p_upper < alpha),
        "mean_regular_90ci": [ci90_lower, ci90_upper],
        "second_ordinary_t_test": False,
    }


def stratified_bootstrap(
    b_region: Sequence[float],
    s_region: Sequence[float],
    tost_differences: Sequence[float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    region_n: int = EXPECTED_B_QUERIES,
) -> dict[str, Any]:
    """D06 paired stratified percentile bootstrap for CIs only."""

    b_values = np.asarray([_finite_float(value, "B D_q") for value in b_region], dtype=np.float64)
    s_values = np.asarray([_finite_float(value, "S D_q") for value in s_region], dtype=np.float64)
    tost_values = np.asarray(
        [_finite_float(value, "TOST N_q") for value in tost_differences], dtype=np.float64
    )
    if replicates < 1:
        raise Gate6StatisticsError("bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {
        "method": "stratified_percentile_bootstrap",
        "seed": int(seed),
        "replicates_requested": int(replicates),
        "region_resample_n": int(region_n),
        "ci_level": 0.95,
        "creates_confirmatory_p_values": False,
        "failed_resamples_retained_and_reported": True,
    }

    def draw(values: np.ndarray) -> np.ndarray:
        indexes = rng.integers(0, values.size, size=(int(replicates), int(region_n)))
        return values[indexes]

    if b_values.size:
        main_samples = draw(b_values)
        main_means = main_samples.mean(axis=1)
        main_sds = main_samples.std(axis=1, ddof=1)
        main_dz = np.divide(
            main_means,
            main_sds,
            out=np.full_like(main_means, np.nan),
            where=main_sds > 0.0,
        )
        failed = np.flatnonzero(~np.isfinite(main_dz))
        result["main"] = {
            "mean": float(b_values.mean()),
            "mean_ci95": _percentile_ci(main_means),
            "d_z": float(b_values.mean() / _sample_sd(b_values))
            if _sample_sd(b_values) > 0.0
            else None,
            "d_z_ci95": _percentile_ci(main_dz),
            "failed_d_z_resamples": int(failed.size),
            "failed_d_z_replicate_indices_zero_based": failed.astype(int).tolist(),
        }
    else:
        result["main"] = {"status": "unavailable", "reason": "empty B region"}

    if s_values.size:
        s_samples = draw(s_values)
        s_means = s_samples.mean(axis=1)
        if b_values.size:
            interaction_samples = main_means - s_means
            b_sds = main_sds
            s_sds = s_samples.std(axis=1, ddof=1)
            pooled = np.sqrt((b_sds**2 + s_sds**2) / 2.0)
            interaction_dz = np.divide(
                interaction_samples,
                pooled,
                out=np.full_like(interaction_samples, np.nan),
                where=pooled > 0.0,
            )
            failed = np.flatnonzero(~np.isfinite(interaction_dz))
            observed_pooled = math.nan
            if b_values.size >= 2:
                observed_pooled = math.sqrt(
                    (
                        (b_values.size - 1) * float(np.var(b_values, ddof=1))
                        + (s_values.size - 1) * float(np.var(s_values, ddof=1))
                    )
                    / (b_values.size + s_values.size - 2)
                )
            observed_interaction = float(b_values.mean() - s_values.mean())
            result["interaction"] = {
                "estimate": observed_interaction,
                "ci95": _percentile_ci(interaction_samples),
                "d_pooled": float(observed_interaction / observed_pooled)
                if observed_pooled > 0.0
                else None,
                "d_pooled_ci95": _percentile_ci(interaction_dz),
                "failed_d_pooled_resamples": int(failed.size),
                "failed_d_pooled_replicate_indices_zero_based": failed.astype(int).tolist(),
            }
        else:
            result["interaction"] = {
                "status": "unavailable",
                "reason": "empty B region",
            }
    else:
        result["interaction"] = {"status": "unavailable", "reason": "empty S region"}

    if tost_values.size:
        tost_samples = draw(tost_values)
        tost_means = tost_samples.mean(axis=1)
        tost_sds = tost_samples.std(axis=1, ddof=1)
        tost_dz = np.divide(
            tost_means,
            tost_sds,
            out=np.full_like(tost_means, np.nan),
            where=tost_sds > 0.0,
        )
        failed = np.flatnonzero(~np.isfinite(tost_dz))
        tost_sd = _sample_sd(tost_values)
        result["tost"] = {
            "M": float(tost_values.mean()),
            "M_ci95": _percentile_ci(tost_means),
            "d_z": float(tost_values.mean() / tost_sd) if tost_sd > 0.0 else None,
            "d_z_ci95": _percentile_ci(tost_dz),
            "failed_d_z_resamples": int(failed.size),
            "failed_d_z_replicate_indices_zero_based": failed.astype(int).tolist(),
        }
    else:
        result["tost"] = {"status": "unavailable", "reason": "empty TOST query set"}

    return result


def holm_adjust(raw_p_values: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    """Holm step-down adjustment for the three-test confirmatory family."""

    if len(raw_p_values) != 3:
        raise Gate6StatisticsError("confirmatory Holm family must contain exactly three tests")
    adjusted = _holm_adjusted_values(raw_p_values)
    return {
        name: {
            "raw_p": float(raw_p_values[name]),
            "adjusted_p": adjusted[name],
            "reject_at_0_05": bool(adjusted[name] < 0.05),
        }
        for name in raw_p_values
    }


def krippendorff_alpha_ordinal(units: Sequence[Sequence[float]]) -> float:
    """Ordinal Krippendorff alpha using midranks across the full rating set."""

    cleaned = [
        [float(value) for value in unit if value is not None and value != ""]
        for unit in units
    ]
    paired = [unit for unit in cleaned if len(unit) >= 2]
    values = [value for unit in paired for value in unit]
    if len(values) < 2 or len(set(values)) < 2:
        return 1.0

    counts = Counter(values)
    midranks: dict[float, float] = {}
    cumulative = 0.0
    for value in sorted(counts):
        midranks[value] = cumulative + counts[value] / 2.0
        cumulative += counts[value]

    observed_numerator = 0.0
    for unit in paired:
        denominator = len(unit) - 1
        for first, second in combinations(unit, 2):
            rank_gap = midranks[first] - midranks[second]
            observed_numerator += 2.0 * rank_gap**2 / denominator
    observed = observed_numerator / len(values)

    total = len(values)
    expected_numerator = 0.0
    for first, second in combinations(values, 2):
        rank_gap = midranks[first] - midranks[second]
        expected_numerator += 2.0 * rank_gap**2
    expected = expected_numerator / (total * (total - 1))
    if expected <= 1e-15:
        return 1.0
    return float(1.0 - observed / expected)


def _spearman_against_other_median(
    rater_ids: Sequence[str], scores_by_rater: Mapping[str, Mapping[str, float]]
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for rater in rater_ids:
        own: list[float] = []
        other_medians: list[float] = []
        for anchor_id in sorted(scores_by_rater[rater]):
            own.append(float(scores_by_rater[rater][anchor_id]))
            others = [
                float(scores_by_rater[other][anchor_id])
                for other in rater_ids
                if other != rater
            ]
            other_medians.append(float(np.median(others)))
        if len(set(own)) < 2 or len(set(other_medians)) < 2:
            output[rater] = None
            continue
        value = stats.spearmanr(own, other_medians).statistic
        output[rater] = float(value) if np.isfinite(value) else None
    return output


def assess_anchor_reliability(
    anchor_ratings: Sequence[Mapping[str, Mapping[str, float]]],
    *,
    rater_ids: Sequence[str] = RATER_IDS,
    alpha_threshold: float = ANCHOR_ALPHA_THRESHOLD,
    spearman_threshold: float = LOO_SPEARMAN_THRESHOLD,
    max_exclusions: int = MAX_RATER_EXCLUSIONS,
) -> dict[str, Any]:
    """D12 anchor alpha with the frozen addendum-one exclusion order.

    Step one runs one LOO-Spearman pass over the initial seven-rater matrix.
    Step two computes ordinal alpha on the remaining raters. Step three triggers
    the one-shot rule two at most once when that alpha is below threshold. The
    LOO pass is never rerun after rule two, so the two rules cannot cycle.
    """

    if len(anchor_ratings) != EXPECTED_ANCHORS:
        raise Gate6StatisticsError(
            f"D12 requires exactly {EXPECTED_ANCHORS} anchors, got {len(anchor_ratings)}"
        )
    raters = tuple(rater_ids)
    if len(raters) != EXPECTED_RATERS or len(set(raters)) != len(raters):
        raise Gate6StatisticsError("D12 requires seven unique rater IDs")
    for index, anchor in enumerate(anchor_ratings):
        if set(anchor) != set(raters):
            raise Gate6StatisticsError(f"anchor {index} must contain one rating per rater")

    correctness_by_rater: dict[str, dict[str, float]] = {
        rater: {} for rater in raters
    }
    faithfulness_by_rater: dict[str, dict[str, float]] = {
        rater: {} for rater in raters
    }
    for anchor_index, anchor in enumerate(anchor_ratings):
        anchor_id = f"anchor-{anchor_index:02d}"
        for rater in raters:
            rating = anchor[rater]
            correctness_by_rater[rater][anchor_id] = _finite_float(
                rating.get("correctness"), "anchor correctness"
            )
            faithfulness_by_rater[rater][anchor_id] = _finite_float(
                rating.get("faithfulness"), "anchor faithfulness"
            )

    def alpha_for(excluded: Iterable[str], metric: str) -> float:
        excluded_set = set(excluded)
        source = correctness_by_rater if metric == "correctness" else faithfulness_by_rater
        kept = [rater for rater in raters if rater not in excluded_set]
        units = [
            [source[rater][f"anchor-{index:02d}"] for rater in kept]
            for index in range(len(anchor_ratings))
        ]
        return krippendorff_alpha_ordinal(units)

    initial_alpha = alpha_for((), "correctness")
    faithfulness_initial = alpha_for((), "faithfulness")
    rho = _spearman_against_other_median(raters, correctness_by_rater)
    removed: list[str] = []
    removal_records: list[dict[str, Any]] = []

    # Step 1: one LOO-Spearman pass over the initial correlations. The pass is
    # never rerun after rule two.
    spearman_candidates = sorted(
        (
            (value if value is not None else 1.0, rater)
            for rater, value in rho.items()
            if value is not None and value < spearman_threshold
        ),
        key=lambda item: (item[0], item[1]),
    )
    loo_spearman_removed: list[str] = []
    for _value, candidate in spearman_candidates:
        if len(removed) >= max_exclusions:
            break
        before = alpha_for(removed, "correctness")
        removed.append(candidate)
        loo_spearman_removed.append(candidate)
        removal_records.append(
            {
                "rater_id": candidate,
                "rule": "single_pass_loo_spearman_below_0_30",
                "loo_spearman_rho": rho.get(candidate),
                "alpha_before": before,
                "alpha_after": alpha_for(removed, "correctness"),
            }
        )

    # Steps 2-3: alpha on the remaining raters, then at most one rule-two trigger.
    alpha_after_loo = alpha_for(removed, "correctness")
    rule_two_triggered = alpha_after_loo < alpha_threshold
    rule_two_candidate: str | None = None
    rule_two_executed = False
    rule_two_blocked_by_exclusion_cap = False
    if rule_two_triggered:
        if len(removed) >= max_exclusions:
            rule_two_blocked_by_exclusion_cap = True
        else:
            best_rater: str | None = None
            best_alpha = alpha_after_loo
            for candidate in (rater for rater in raters if rater not in removed):
                trial_alpha = alpha_for((*removed, candidate), "correctness")
                if trial_alpha > best_alpha + 1e-15:
                    best_alpha = trial_alpha
                    best_rater = candidate
            if best_rater is not None:
                rule_two_candidate = best_rater
                before = alpha_after_loo
                removed.append(best_rater)
                rule_two_executed = True
                removal_records.append(
                    {
                        "rater_id": best_rater,
                        "rule": "one_shot_rule_two_maximize_alpha",
                        "loo_spearman_rho": rho.get(best_rater),
                        "alpha_before": before,
                        "alpha_after": alpha_for(removed, "correctness"),
                    }
                )

    alpha_after_rule_two = alpha_for(removed, "correctness")
    final_alpha = alpha_after_rule_two
    faithfulness_final = alpha_for(removed, "faithfulness")
    available_raters = [rater for rater in raters if rater not in removed]
    for index in range(len(anchor_ratings)):
        retained = sum(
            1
            for rater in available_raters
            if f"anchor-{index:02d}" in correctness_by_rater[rater]
        )
        if retained < MIN_RATERS_PER_ANSWER:
            raise Gate6StatisticsError(
                f"rater exclusion leaves anchor {index} with fewer than two ratings"
            )

    return {
        "method": "ordinal_krippendorff_alpha",
        "status": "estimated",
        "anchor_count": len(anchor_ratings),
        "initial_rater_count": len(raters),
        "initial_correctness_alpha": initial_alpha,
        "alpha_threshold": float(alpha_threshold),
        "loo_spearman_removed_raters": loo_spearman_removed,
        "alpha_after_loo_spearman": alpha_after_loo,
        "rule_two_threshold": float(alpha_threshold),
        "rule_two_triggered": rule_two_triggered,
        "rule_two_executed": rule_two_executed,
        "rule_two_blocked_by_exclusion_cap": rule_two_blocked_by_exclusion_cap,
        "rule_two_candidate": rule_two_candidate,
        "alpha_after_rule_two": alpha_after_rule_two,
        "loo_spearman_threshold": float(spearman_threshold),
        "loo_spearman_rho": rho,
        "removed_raters": removed,
        "removal_records": removal_records,
        "final_rater_count": len(available_raters),
        "final_available_raters": available_raters,
        "final_correctness_alpha": final_alpha,
        "faithfulness_alpha_initial": faithfulness_initial,
        "faithfulness_alpha_final": faithfulness_final,
        "faithfulness_triggers_exclusion": False,
        "total_exclusion_cap": int(max_exclusions),
        "min_raters_per_answer": MIN_RATERS_PER_ANSWER,
        "non_cycling_sequence": [
            "initial_seven_rater_alpha",
            "single_pass_loo_spearman",
            "one_shot_rule_two_without_loo_rerun",
        ],
        "exclusion_order": [
            "leave_one_out_spearman_rho_lt_0_30",
            "ordinal_krippendorff_alpha_on_remaining_raters",
            "one_shot_rule_two_if_alpha_lt_0_67",
        ],
    }


def validate_raters_per_answer(
    answers: Sequence[Mapping[str, Any]],
    excluded_raters: Sequence[str] = (),
) -> dict[str, Any]:
    """D12 fail-closed check that every answer keeps at least two valid raters."""

    excluded = set(excluded_raters)
    deficits: list[dict[str, Any]] = []
    for answer in answers:
        answer_id = str(answer.get("answer_id") or "").strip()
        valid_raters = 0
        ratings = answer.get("ratings") or []
        if not isinstance(ratings, list):
            ratings = []
        for rating in ratings:
            if not isinstance(rating, Mapping):
                continue
            rater_id = str(rating.get("rater_id") or "").strip()
            if not rater_id or rater_id in excluded:
                continue
            if rating.get("valid", True) is False:
                continue
            correctness = rating.get("correctness")
            if correctness is None or correctness == "":
                continue
            try:
                value = _finite_float(correctness, "correctness")
            except Gate6StatisticsError:
                continue
            if not 0.0 <= value <= 1.0:
                continue
            valid_raters += 1
        if valid_raters < MIN_RATERS_PER_ANSWER:
            deficits.append({"answer_id": answer_id, "valid_retained_raters": valid_raters})
    if deficits:
        summary = ", ".join(
            f"{deficit['answer_id']} ({deficit['valid_retained_raters']})"
            for deficit in deficits
        )
        raise Gate6StatisticsError(
            f"answer stage fail-closed: answers have fewer than "
            f"{MIN_RATERS_PER_ANSWER} valid raters after D12 exclusions: {summary}"
        )
    return {
        "status": "passed",
        "answers_checked": len(answers),
        "min_valid_retained_raters": MIN_RATERS_PER_ANSWER,
    }


def score_answers(
    answers: Sequence[Mapping[str, Any]],
    *,
    excluded_raters: Sequence[str] = (),
) -> dict[str, Any]:
    """Apply D11 generation, validity, and rater-average scoring rules."""

    excluded = set(excluded_raters)
    # D11 keeps every answer with fewer than two valid raters out of scoring;
    # this unconditional check turns that boundary into a hard failure instead.
    rater_retention_check = validate_raters_per_answer(answers, excluded)
    scored: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_ids: set[str] = set()
    for index, answer in enumerate(answers):
        answer_id = str(answer.get("answer_id") or "").strip()
        query_id = str(answer.get("query_id") or "").strip()
        condition = str(answer.get("condition") or "").strip()
        query_type = str(answer.get("query_type") or "").strip()
        if not answer_id or answer_id in seen_ids:
            raise Gate6StatisticsError(f"answer {index} has a missing or duplicate answer_id")
        seen_ids.add(answer_id)
        if not query_id or condition not in CONDITIONS or query_type not in QUERY_TYPES:
            raise Gate6StatisticsError(f"answer {answer_id} has invalid query metadata")
        key = (query_id, condition)
        if key in seen_keys:
            raise Gate6StatisticsError(f"duplicate answer for query/condition {key}")
        seen_keys.add(key)

        status = str(answer.get("generation_status") or "").strip()
        text = answer.get("answer_text")
        nonempty = isinstance(text, str) and bool(text.strip())
        failed = status != "success" or not nonempty
        ratings = answer.get("ratings") or []
        if not isinstance(ratings, list):
            raise Gate6StatisticsError(f"answer {answer_id} ratings must be a list")
        seen_raters: set[str] = set()
        valid_correctness: list[float] = []
        valid_faithfulness: list[float] = []
        missing_or_invalid_rating_records = 0
        excluded_rating_records = 0
        for rating_index, rating in enumerate(ratings):
            if not isinstance(rating, Mapping):
                raise Gate6StatisticsError(f"answer {answer_id} rating must be an object")
            rater_id = str(rating.get("rater_id") or "").strip()
            if not rater_id or rater_id in seen_raters:
                raise Gate6StatisticsError(
                    f"answer {answer_id} has a missing or duplicate rater_id"
                )
            seen_raters.add(rater_id)
            if rater_id in excluded:
                excluded_rating_records += 1
                continue
            if rating.get("valid", True) is False:
                missing_or_invalid_rating_records += 1
                continue
            correctness = rating.get("correctness")
            if correctness is None or correctness == "":
                missing_or_invalid_rating_records += 1
                continue
            try:
                correctness_value = _finite_float(
                    correctness, f"answer {answer_id} rating {rating_index} correctness"
                )
            except Gate6StatisticsError:
                missing_or_invalid_rating_records += 1
                continue
            if not 0.0 <= correctness_value <= 1.0:
                missing_or_invalid_rating_records += 1
                continue
            valid_correctness.append(correctness_value)
            faithfulness = rating.get("faithfulness")
            if faithfulness is not None and faithfulness != "":
                try:
                    faithfulness_value = _finite_float(
                        faithfulness, f"answer {answer_id} rating {rating_index} faithfulness"
                    )
                except Gate6StatisticsError:
                    faithfulness_value = None
                if faithfulness_value is not None and 0.0 <= faithfulness_value <= 1.0:
                    valid_faithfulness.append(faithfulness_value)

        at_least_two = bool(nonempty and len(valid_correctness) >= MIN_RATERS_PER_ANSWER)
        valid = bool(not failed and at_least_two)
        score = float(np.mean(valid_correctness)) if valid else None
        faithfulness_score = (
            float(np.mean(valid_faithfulness))
            if valid and len(valid_faithfulness) >= MIN_RATERS_PER_ANSWER
            else None
        )
        scored.append(
            {
                "answer_id": answer_id,
                "query_id": query_id,
                "query_type": query_type,
                "condition": condition,
                "generation_status": status,
                "F_qc": 1 if failed else 0,
                "nonempty_answer": nonempty,
                "valid_rating_count": len(valid_correctness),
                "excluded_rating_records": excluded_rating_records,
                "missing_or_invalid_rating_records": missing_or_invalid_rating_records,
                "at_least_two_valid_ratings": at_least_two,
                "V_qc": 1 if valid else 0,
                "score": score,
                "faithfulness_score": faithfulness_score,
            }
        )

    if len(scored) != EXPECTED_ANSWERS:
        raise Gate6StatisticsError(
            f"answer stage design requires {EXPECTED_ANSWERS} answer records; got {len(scored)}"
        )
    query_ids = {answer["query_id"] for answer in scored}
    if len(query_ids) != EXPECTED_QUERY_COUNT:
        raise Gate6StatisticsError(
            f"answer stage design requires {EXPECTED_QUERY_COUNT} queries; got {len(query_ids)}"
        )
    for query_id in query_ids:
        conditions = {
            answer["condition"] for answer in scored if answer["query_id"] == query_id
        }
        if conditions != set(CONDITIONS):
            raise Gate6StatisticsError(f"query {query_id} must have local/web/dual answers")

    counts = {
        "design_expected_answers": EXPECTED_ANSWERS,
        "successful_generation": sum(1 for answer in scored if answer["F_qc"] == 0),
        "failure_or_empty": sum(1 for answer in scored if answer["F_qc"] == 1),
        "at_least_two_valid_ratings": sum(
            1 for answer in scored if answer["at_least_two_valid_ratings"]
        ),
        "observed_answers": sum(1 for answer in scored if answer["V_qc"] == 1),
        "score_missing_exclusions": sum(
            1
            for answer in scored
            if answer["F_qc"] == 0 and answer["V_qc"] == 0
        ),
        "missing_or_invalid_rating_records": sum(
            answer["missing_or_invalid_rating_records"] for answer in scored
        ),
        "excluded_rater_rating_records": sum(
            answer["excluded_rating_records"] for answer in scored
        ),
    }
    return {"answers": scored, "counts": counts, "rater_retention_check": rater_retention_check}


def _analysis_score(answer: Mapping[str, Any], mode: str) -> float | None:
    if mode == "failure_zero" and answer["F_qc"] == 1:
        return 0.0
    return answer["score"]


def build_population(
    scored_answers: Sequence[Mapping[str, Any]], *, mode: str
) -> dict[str, Any]:
    """Construct D02/D03/D04/D07 inputs for one D11 analysis population."""

    if mode not in {"observed", "complete_case", "failure_zero"}:
        raise Gate6StatisticsError(f"unknown analysis population mode: {mode}")
    by_query: dict[str, dict[str, Mapping[str, Any]]] = {}
    query_types: dict[str, str] = {}
    for answer in scored_answers:
        by_query.setdefault(str(answer["query_id"]), {})[str(answer["condition"])] = answer
        query_types[str(answer["query_id"])] = str(answer["query_type"])

    complete_case_queries = {
        query_id
        for query_id, conditions in by_query.items()
        if all(conditions[condition]["V_qc"] == 1 for condition in CONDITIONS)
    }
    query_rows: list[dict[str, Any]] = []
    main_differences: list[float] = []
    main_query_ids: list[str] = []
    b_values: list[float] = []
    s_values: list[float] = []
    b_query_ids: list[str] = []
    s_query_ids: list[str] = []
    tost_differences: list[float] = []
    tost_query_ids: list[str] = []
    model_rows: list[dict[str, Any]] = []
    available_answers = 0
    failure_zero_filled = 0

    for query_id in sorted(by_query):
        query_type = query_types[query_id]
        region = 1 if query_type in B_REGION_QUERY_TYPES else 0
        conditions = by_query[query_id]
        scores = {
            condition: _analysis_score(conditions[condition], mode)
            for condition in CONDITIONS
        }
        if mode == "complete_case" and query_id not in complete_case_queries:
            scores = {condition: None for condition in CONDITIONS}
        available_answers += sum(value is not None for value in scores.values())
        if mode == "failure_zero":
            failure_zero_filled += sum(
                1
                for condition in CONDITIONS
                if conditions[condition]["F_qc"] == 1
            )

        all_three = all(scores[condition] is not None for condition in CONDITIONS)
        D_q = (
            float(scores["dual"] - (scores["local"] + scores["web"]) / 2.0)
            if all_three
            else None
        )
        N_q: float | None = None
        if query_type == "local_enough" and scores["dual"] is not None and scores["local"] is not None:
            N_q = float(scores["dual"] - scores["local"])
        elif query_type == "web_enough" and scores["dual"] is not None and scores["web"] is not None:
            N_q = float(scores["dual"] - scores["web"])

        main_included = bool(region == 1 and D_q is not None)
        interaction_included = bool(D_q is not None)
        if main_included:
            main_differences.append(float(D_q))
            main_query_ids.append(query_id)
        if interaction_included:
            if region == 1:
                b_values.append(float(D_q))
                b_query_ids.append(query_id)
            else:
                s_values.append(float(D_q))
                s_query_ids.append(query_id)
        if N_q is not None:
            tost_differences.append(float(N_q))
            tost_query_ids.append(query_id)
        if mode != "failure_zero":
            for condition in CONDITIONS:
                if scores[condition] is not None:
                    coding = CONDITION_CODING[condition]
                    model_rows.append(
                        {
                            "y": float(scores[condition]),
                            "query_id": query_id,
                            "condition": condition,
                            "region": region,
                            "C1": coding["C1"],
                            "C2": coding["C2"],
                        }
                    )
        query_rows.append(
            {
                "query_id": query_id,
                "query_type": query_type,
                "region": region,
                "score_local": scores["local"],
                "score_web": scores["web"],
                "score_dual": scores["dual"],
                "D_q": D_q,
                "TOST_N_q": N_q,
                "main_included_D02": main_included,
                "interaction_included_D03": interaction_included,
                "tost_included_D04": N_q is not None,
                "model_rows_included_D07": not (
                    mode == "failure_zero"
                ) and any(row["query_id"] == query_id for row in model_rows),
            }
        )

    model_queries = {row["query_id"] for row in model_rows}
    return {
        "mode": mode,
        "main_differences": main_differences,
        "main_query_ids": main_query_ids,
        "b_values": b_values,
        "s_values": s_values,
        "b_query_ids": b_query_ids,
        "s_query_ids": s_query_ids,
        "tost_differences": tost_differences,
        "tost_query_ids": tost_query_ids,
        "model_rows": model_rows,
        "query_rows": query_rows,
        "query_types": query_types,
        "complete_case_queries": sorted(complete_case_queries),
        "counts": {
            "analysis_available_answers": available_answers,
            "failure_zero_filled_answers": failure_zero_filled if mode == "failure_zero" else 0,
            "n_main": len(main_differences),
            "n_B": len(b_values),
            "n_S": len(s_values),
            "n_TOST": len(tost_differences),
            "n_model_answers": len(model_rows),
            "n_model_queries": len(model_queries),
            "n_complete_case_queries": len(complete_case_queries),
            "design_expected_main_queries": EXPECTED_B_QUERIES,
            "design_expected_B_queries": EXPECTED_B_QUERIES,
            "design_expected_S_queries": EXPECTED_S_QUERIES,
            "design_expected_TOST_queries": EXPECTED_S_QUERIES,
            "design_expected_model_answers": EXPECTED_ANSWERS,
            "design_expected_model_queries": EXPECTED_QUERY_COUNT,
        },
    }


def analyze_population(
    population: Mapping[str, Any],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    permutation_replicates: int = INTERACTION_PERMUTATION_REPLICATES,
) -> dict[str, Any]:
    main = exact_sign_flip_main(population["main_differences"])
    interaction = interaction_permutation(
        population["b_values"],
        population["s_values"],
        replicates=permutation_replicates,
    )
    tost = noncentral_t_tost(population["tost_differences"])
    bootstrap = stratified_bootstrap(
        population["b_values"],
        population["s_values"],
        population["tost_differences"],
        replicates=bootstrap_replicates,
    )
    if "main" in bootstrap and "mean_ci95" in bootstrap["main"]:
        main["mean_bootstrap_95ci"] = bootstrap["main"]["mean_ci95"]
        main["d_z_bootstrap_95ci"] = bootstrap["main"]["d_z_ci95"]
    if "interaction" in bootstrap and "ci95" in bootstrap["interaction"]:
        interaction["interaction_bootstrap_95ci"] = bootstrap["interaction"]["ci95"]
        interaction["d_pooled_bootstrap_95ci"] = bootstrap["interaction"]["d_pooled_ci95"]
    if "tost" in bootstrap and "M_ci95" in bootstrap["tost"]:
        tost["M_bootstrap_95ci"] = bootstrap["tost"]["M_ci95"]
        tost["d_z_bootstrap_95ci"] = bootstrap["tost"]["d_z_ci95"]
    return {
        "counts": population["counts"],
        "D02_main": main,
        "D03_interaction": interaction,
        "D04_TOST": tost,
        "D06_bootstrap": bootstrap,
        "D07_prepared": {
            "model_rows": len(population["model_rows"]),
            "model_queries": population["counts"]["n_model_queries"],
            "status": "prepared_for_R_not_executed_by_python_core",
            "failure_zero_branch": False,
        },
        "D08_exploratory_within_type": exploratory_within_type_holm(population),
    }


def analyze_synthetic_fixture(
    fixture: Mapping[str, Any],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    permutation_replicates: int = INTERACTION_PERMUTATION_REPLICATES,
) -> dict[str, Any]:
    """Run all frozen estimators and D11 populations on one synthetic fixture."""

    if fixture.get("fixture_kind") != "synthetic_gate6_statistics_fixture":
        raise Gate6StatisticsError("only synthetic answer stage fixtures are accepted")
    boundary = fixture.get("authorization") or {}
    for key in (
        "gate6_started",
        "confirmation_set_contacted",
        "confirmation_results_read",
        "confirmation_analysis_run",
        "qrels_v2_created",
        "qrels_v2_used",
    ):
        if boundary.get(key) is not False:
            raise Gate6StatisticsError(f"synthetic fixture boundary requires {key}=false")

    answers = fixture.get("answers") or []
    query_types = {str(answer["query_id"]): str(answer["query_type"]) for answer in answers}
    if len(query_types) != EXPECTED_QUERY_COUNT:
        raise Gate6StatisticsError("fixture must contain exactly 32 queries")
    type_counts = Counter(query_types.values())
    if any(type_counts[query_type] != 8 for query_type in QUERY_TYPES):
        raise Gate6StatisticsError("fixture must contain eight queries per query type")

    anchor_ids = list(fixture.get("anchor_answer_ids") or [])
    if len(anchor_ids) != EXPECTED_ANCHORS or len(set(anchor_ids)) != EXPECTED_ANCHORS:
        raise Gate6StatisticsError("fixture must contain 16 unique anchor answer IDs")
    answer_by_id = {str(answer["answer_id"]): answer for answer in answers}
    try:
        anchor_answers = [answer_by_id[anchor_id] for anchor_id in anchor_ids]
    except KeyError as exc:
        raise Gate6StatisticsError(f"unknown anchor answer ID: {exc.args[0]}") from exc
    anchor_ratings: list[dict[str, dict[str, float]]] = []
    for answer in anchor_answers:
        ratings = answer.get("ratings") or []
        rating_map = {str(rating["rater_id"]): rating for rating in ratings}
        anchor_ratings.append(
            {
                rater: {
                    "correctness": rating_map[rater]["correctness"],
                    "faithfulness": rating_map[rater]["faithfulness"],
                }
                for rater in RATER_IDS
            }
        )
    anchors = assess_anchor_reliability(anchor_ratings)
    scored = score_answers(answers, excluded_raters=anchors["removed_raters"])
    populations: dict[str, Any] = {}
    for mode in ("observed", "complete_case", "failure_zero"):
        population = build_population(scored["answers"], mode=mode)
        populations[mode] = analyze_population(
            population,
            bootstrap_replicates=bootstrap_replicates,
            permutation_replicates=permutation_replicates,
        )
        populations[mode]["query_inputs"] = population

    observed = populations["observed"]
    raw_p = {}
    for test_name, key in (
        ("main", "D02_main"),
        ("interaction", "D03_interaction"),
        ("TOST", "D04_TOST"),
    ):
        p_value = observed[key].get("p_raw")
        raw_p[test_name] = 1.0 if p_value is None else float(p_value)
    holm = holm_adjust(raw_p)
    observed["D08_Holm"] = {
        "family": ["main", "interaction", "TOST"],
        "test_count": 3,
        "TOST_counted_as_one": True,
        "alpha": 0.05,
        "tests": holm,
    }
    tost_raw_support = bool(
        observed["D04_TOST"].get("equivalence_supported_uncorrected", False)
    )
    observed["D08_Holm"]["conclusions"] = {
        "main": "supported"
        if holm["main"]["reject_at_0_05"]
        and (observed["D02_main"].get("observed_mean_D") or 0.0) > 0.0
        else "not_supported",
        "interaction": "detected"
        if holm["interaction"]["reject_at_0_05"]
        else "not_detected",
        "TOST": "equivalent_within_frozen_bound"
        if tost_raw_support and holm["TOST"]["reject_at_0_05"]
        else "insufficient_equivalence_evidence",
    }

    return {
        "schema_version": "gate6-statistics-results-v1",
        "implementation_status": "synthetic_fixture_verified_not_gate6_execution",
        "fixture": {
            "fixture_kind": fixture["fixture_kind"],
            "fixture_id": fixture.get("fixture_id"),
            "answer_count": len(answers),
            "anchor_count": len(anchor_ids),
            "synthetic_only": True,
        },
        "frozen_parameters": {
            "main_test": "exact paired sign-flip",
            "main_alternative": "greater",
            "main_alpha": MAIN_ALPHA,
            "interaction_test": "Monte Carlo region-label permutation",
            "interaction_alternative": "two-sided",
            "interaction_replicates": INTERACTION_PERMUTATION_REPLICATES,
            "interaction_seed": INTERACTION_PERMUTATION_SEED,
            "interaction_p_denominator": INTERACTION_PERMUTATION_REPLICATES + 1,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_ci": "95% percentile",
            "tost_bound_d": TOST_BOUND_D,
            "tost_alpha_each_side": TOST_ALPHA,
            "tost_p": "max(p_lower, p_upper)",
            "holm_family_size": 3,
            "exploratory_holm_family": EXPLORATORY_FAMILY_NAME,
            "exploratory_family_size": EXPLORATORY_TEST_COUNT,
            "exploratory_alpha": EXPLORATORY_ALPHA,
            "exploratory_independent_of_confirmatory_holm": True,
            "anchor_alpha_threshold": ANCHOR_ALPHA_THRESHOLD,
            "max_rater_exclusions": MAX_RATER_EXCLUSIONS,
        },
        "D11_scoring_counts": scored["counts"],
        "D12_anchors": anchors,
        "D12_global_rater_retention": scored["rater_retention_check"],
        "D11_populations": populations,
    }


__all__ = [
    "ANCHOR_ALPHA_THRESHOLD",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CONDITION_CODING",
    "CONDITIONS",
    "EXPECTED_ANCHORS",
    "EXPECTED_ANSWERS",
    "EXPECTED_B_QUERIES",
    "EXPECTED_QUERY_COUNT",
    "EXPECTED_RATERS",
    "EXPECTED_S_QUERIES",
    "EXPLORATORY_ALPHA",
    "EXPLORATORY_FAMILY_NAME",
    "EXPLORATORY_TEST_COUNT",
    "Gate6StatisticsError",
    "INTERACTION_PERMUTATION_REPLICATES",
    "INTERACTION_PERMUTATION_SEED",
    "MAIN_ALPHA",
    "MAX_RATER_EXCLUSIONS",
    "MIN_RATERS_PER_ANSWER",
    "QUERY_TYPES",
    "RATER_IDS",
    "SOURCE_PAIRS",
    "TOST_ALPHA",
    "TOST_BOUND_D",
    "analyze_population",
    "analyze_synthetic_fixture",
    "assess_anchor_reliability",
    "build_population",
    "exact_sign_flip_main",
    "exact_sign_flip_two_sided",
    "exploratory_within_type_holm",
    "holm_adjust",
    "interaction_permutation",
    "krippendorff_alpha_ordinal",
    "noncentral_t_tost",
    "score_answers",
    "stratified_bootstrap",
    "validate_raters_per_answer",
]
