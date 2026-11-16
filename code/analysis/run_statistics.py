#!/usr/bin/env python3
"""Build and analyze synthetic answer stage statistics fixtures only.

This runner is deliberately not a answer stage execution entry point. It rejects
fixtures unless their boundary metadata proves that answer stage, confirmation-set
contact, confirmation-result reads, and qrels v2 use are all false.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from stats_core import (
    BOOTSTRAP_REPLICATES,
    CONDITION_CODING,
    CONDITIONS,
    EXPECTED_ANSWERS,
    INTERACTION_PERMUTATION_REPLICATES,
    QUERY_TYPES,
    RATER_IDS,
    Gate6StatisticsError,
    analyze_synthetic_fixture,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DATE = "20261105"
# Evidence directories that are frozen and must never be rewritten in place.
IMMUTABLE_FROZEN_RESULT_DIRS = tuple(
    (REPO_ROOT / relative).resolve()
    for relative in (
        "experiments/results/gate3-5-pooling-20261023",
        "experiments/results/gate3-qrels-20261023",
        "experiments/results/gate4-split-20261102",
        "experiments/results/gate5-dev-qrels-20261102",
        "experiments/results/gate5-execution-v2",
        "experiments/results/gate6-contract-20261104",
        "experiments/results/gate6-d07-c1-coding-revision-20261105",
        "experiments/results/gate6-d12-freeze-and-implementation-authorization-20261105",
        "experiments/results/gate6-readiness-20261104",
        "experiments/results/gate6-statistics-addenda-20261109",
        "experiments/results/gate6-statistics-clarification-20261105",
        "experiments/results/gate6-statistics-implementation-20261105",
    )
)
R_SCRIPT = REPO_ROOT / "experiments/scripts/gate6_mixed_model.R"
R_PACKAGES = ("lme4", "lmerTest", "emmeans", "pbkrtest")
CORE_SCRIPT = REPO_ROOT / "experiments/scripts/gate6_statistics.py"
RUNNER_SCRIPT = Path(__file__).resolve()
FINALIZER_SCRIPT = REPO_ROOT / "experiments/scripts/finalize_gate6_d07_c1_revision.py"
FROZEN_PLAN = (
    REPO_ROOT
    / "experiments/results/gate6-statistics-addenda-20261109"
    / "FROZEN-STATISTICS-PLAN-2026-11-09.md"
)
FROZEN_PLAN_SHA256 = (
    "c7b7aa622e720badb462009026c60d1bbdcb607face2d8827ba7cc6fb2ad3505"
)
R_LIBRARY_ENV_VAR = "GATE6_R_LIB"
DEFAULT_R_LIBRARY = Path("/private/tmp/gate6-r-lib")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_plan(
    path: Path | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless the runner is bound to the authoritative frozen plan."""

    plan = FROZEN_PLAN if path is None else path
    expected = FROZEN_PLAN_SHA256 if expected_sha256 is None else expected_sha256
    if not plan.exists():
        raise Gate6StatisticsError(f"frozen statistics plan is missing: {plan}")
    actual = sha256_file(plan)
    if actual != expected:
        raise Gate6StatisticsError(
            f"frozen statistics plan hash mismatch: {plan} has {actual}, expected {expected}"
        )
    return {
        "path": plan.relative_to(REPO_ROOT).as_posix()
        if plan.resolve().is_relative_to(REPO_ROOT)
        else str(plan),
        "sha256": actual,
        "expected_sha256": expected,
        "status": "verified",
    }


def ensure_writable_output_dir(out_dir: Path) -> None:
    """Fail closed when a bundle would land in immutable frozen evidence."""

    target = out_dir.resolve()
    for frozen in IMMUTABLE_FROZEN_RESULT_DIRS:
        if target == frozen or target.is_relative_to(frozen):
            display = (
                frozen.relative_to(REPO_ROOT).as_posix()
                if frozen.is_relative_to(REPO_ROOT)
                else str(frozen)
            )
            raise Gate6StatisticsError(
                f"refusing to write into immutable frozen evidence directory: {display}"
            )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def clamp_score(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 6)


def build_synthetic_fixture() -> dict[str, Any]:
    """Create a deterministic 96-answer fixture with 16 seven-rater anchors."""

    type_sequence = [query_type for query_type in QUERY_TYPES for _ in range(8)]
    n_pattern = np.linspace(-0.16, 0.16, 8, dtype=np.float64)
    b_effect_pattern = 0.30 + 0.05 * np.sin(np.linspace(0.2, 2.8, 8, dtype=np.float64))
    jitter_four = [-0.03, -0.01, 0.01, 0.03]
    jitter_seven = [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06]
    queries: list[dict[str, Any]] = []
    answer_targets: dict[str, float] = {}
    for query_index, query_type in enumerate(type_sequence, start=1):
        query_id = f"Q{query_index:02d}"
        within_type = (query_index - 1) % 8
        offset = (within_type - 3.5) * 0.004
        if query_type in {"complementary", "conflicting"}:
            targets = {
                "local": 0.55 + offset,
                "web": 0.60 + offset,
                "dual": 0.575 + float(b_effect_pattern[within_type]),
            }
        elif query_type == "local_enough":
            local = 0.70 + offset
            targets = {
                "local": local,
                "web": 0.55 + offset,
                "dual": local + float(n_pattern[within_type]),
            }
        else:
            web = 0.70 + offset
            targets = {
                "local": 0.55 + offset,
                "web": web,
                "dual": web + float(n_pattern[-1 - within_type]),
            }
        queries.append(
            {
                "query_id": query_id,
                "query_type": query_type,
                "synthetic_query_text": f"Synthetic synthetic-only query {query_index:02d}",
            }
        )
        for condition in CONDITIONS:
            answer_id = f"{query_id}-{condition.upper()}"
            answer_targets[answer_id] = targets[condition]

    representative_answer_ids = {
        f"{query_id}-{condition.upper()}"
        for query_id in ("Q01", "Q09", "Q17", "Q25")
        for condition in CONDITIONS
    }
    representative_answer_ids.update(
        {
            "Q02-LOCAL",
            "Q02-DUAL",
            "Q10-WEB",
            "Q10-DUAL",
        }
    )
    anchor_ids = sorted(representative_answer_ids)
    if len(anchor_ids) != 16:
        raise AssertionError(f"synthetic anchor selection produced {len(anchor_ids)} anchors")

    answers: list[dict[str, Any]] = []
    for query in queries:
        query_id = query["query_id"]
        for condition in CONDITIONS:
            answer_id = f"{query_id}-{condition.upper()}"
            target = answer_targets[answer_id]
            is_anchor = answer_id in representative_answer_ids
            raters = RATER_IDS if is_anchor else RATER_IDS[:4]
            jitters = jitter_seven if is_anchor else jitter_four
            ratings = []
            for rater_index, rater_id in enumerate(raters):
                correctness = clamp_score(target + jitters[rater_index])
                faithfulness = clamp_score(
                    0.82
                    + 0.04 * math.sin((len(answer_id) + rater_index) / 5.0)
                    + jitters[rater_index] / 3.0
                )
                rating = {
                    "rater_id": rater_id,
                    "correctness": correctness,
                    "faithfulness": faithfulness,
                    "valid": True,
                }
                # Exercise D11 missing-rating handling while retaining at least
                # two valid raters and therefore V_qc=1.
                if answer_id in {"Q18-LOCAL", "Q26-DUAL"} and rater_id == "B04":
                    rating["correctness"] = None
                    rating["faithfulness"] = None
                    rating["valid"] = False
                ratings.append(rating)
            answers.append(
                {
                    "answer_id": answer_id,
                    "query_id": query_id,
                    "query_type": query["query_type"],
                    "condition": condition,
                    "generation_status": "success",
                    "answer_text": f"Synthetic answer body for {answer_id}.",
                    "synthetic_anchor_eligible": is_anchor,
                    "ratings": ratings,
                }
            )

    return {
        "schema_version": "gate6-synthetic-statistics-fixture-v1",
        "fixture_id": "GATE6-SYNTHETIC-STATISTICS-FIXTURE-20261105",
        "fixture_kind": "synthetic_gate6_statistics_fixture",
        "fixture_provenance": {
            "source": "deterministic Python generator",
            "contains_confirmation_queries": False,
            "contains_confirmation_evidence": False,
            "contains_confirmation_answers": False,
            "contains_confirmation_qrels": False,
            "contains_real_human_ratings": False,
        },
        "authorization": {
            "gate6_started": False,
            "confirmation_set_contacted": False,
            "confirmation_results_read": False,
            "confirmation_analysis_run": False,
            "qrels_v2_created": False,
            "qrels_v2_used": False,
        },
        "design": {
            "queries": 32,
            "answers": 96,
            "conditions": list(CONDITIONS),
            "query_types": list(QUERY_TYPES),
            "anchors": 16,
            "raters": list(RATER_IDS),
        },
        "queries": queries,
        "anchor_answer_ids": anchor_ids,
        "answers": answers,
    }


def mutate_fixture_for_missingness(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Return a synthetic stress fixture for D11 unit tests."""

    changed = copy.deepcopy(dict(fixture))
    by_id = {answer["answer_id"]: answer for answer in changed["answers"]}
    by_id["Q17-LOCAL"]["generation_status"] = "failed"
    by_id["Q17-LOCAL"]["answer_text"] = ""
    for rating in by_id["Q18-LOCAL"]["ratings"]:
        # B04 is already invalid in the base fixture; keep B01 and B02 valid so
        # the stress fixture satisfies the unconditional two-rater guard.
        if rating["rater_id"] == "B03":
            rating["correctness"] = None
            rating["faithfulness"] = None
            rating["valid"] = False
    by_id["Q25-DUAL"]["generation_status"] = "empty"
    by_id["Q25-DUAL"]["answer_text"] = "   "
    changed["fixture_id"] = "GATE6-SYNTHETIC-STATISTICS-MISSINGNESS-STRESS-20261105"
    return changed


def load_synthetic_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fixture_kind") != "synthetic_gate6_statistics_fixture":
        raise Gate6StatisticsError("only synthetic answer stage fixtures may be loaded")
    boundary = payload.get("authorization") or {}
    for key in (
        "gate6_started",
        "confirmation_set_contacted",
        "confirmation_results_read",
        "confirmation_analysis_run",
        "qrels_v2_created",
        "qrels_v2_used",
    ):
        if boundary.get(key) is not False:
            raise Gate6StatisticsError(f"{path}: authorization boundary requires {key}=false")
    provenance = payload.get("fixture_provenance") or {}
    for key in (
        "contains_confirmation_queries",
        "contains_confirmation_evidence",
        "contains_confirmation_answers",
        "contains_confirmation_qrels",
        "contains_real_human_ratings",
    ):
        if provenance.get(key) is not False:
            raise Gate6StatisticsError(f"{path}: fixture provenance requires {key}=false")
    return payload


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def build_denominator_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    scoring = results["D11_scoring_counts"]
    rows: list[dict[str, Any]] = []
    for mode, population in results["D11_populations"].items():
        counts = population["counts"]
        rows.append(
            {
                "analysis_population": mode,
                **scoring,
                **counts,
                "anchor_initial_correctness_alpha": results["D12_anchors"][
                    "initial_correctness_alpha"
                ],
                "anchor_final_correctness_alpha": results["D12_anchors"][
                    "final_correctness_alpha"
                ],
                "raters_removed": len(results["D12_anchors"]["removed_raters"]),
            }
        )
    return rows


def build_analysis_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, population in results["D11_populations"].items():
        main = population["D02_main"]
        interaction = population["D03_interaction"]
        tost = population["D04_TOST"]
        counts = population["counts"]
        table_counts = {
            **results["D11_scoring_counts"],
            "analysis_available_answers": counts["analysis_available_answers"],
            "complete_case_queries": counts["n_complete_case_queries"],
            "failure_zero_filled_answers": counts["failure_zero_filled_answers"],
            "n_main": counts["n_main"],
            "n_B": counts["n_B"],
            "n_S": counts["n_S"],
            "n_TOST": counts["n_TOST"],
            "n_model_answers": counts["n_model_answers"],
            "n_model_queries": counts["n_model_queries"],
        }
        rows.extend(
            [
                {
                    **table_counts,
                    "analysis_population": mode,
                    "analysis": "D02_main_exact_sign_flip",
                    "status": main.get("status"),
                    "n": main.get("n"),
                    "estimate": main.get("observed_mean_D"),
                    "effect_size": main.get("d_z"),
                    "ci95_lower": (main.get("mean_bootstrap_95ci") or [None, None])[0],
                    "ci95_upper": (main.get("mean_bootstrap_95ci") or [None, None])[1],
                    "p_raw": main.get("p_raw"),
                    "p_adjusted_holm": (
                        population.get("D08_Holm", {})
                        .get("tests", {})
                        .get("main", {})
                        .get("adjusted_p")
                    ),
                    "replicates_or_sign_flips": main.get("sign_flip_count"),
                    "seed": None,
                },
                {
                    **table_counts,
                    "analysis_population": mode,
                    "analysis": "D03_interaction_permutation",
                    "status": interaction.get("status"),
                    "n": interaction.get("n_B"),
                    "estimate": interaction.get("observed_interaction"),
                    "effect_size": interaction.get("d_pooled"),
                    "ci95_lower": (interaction.get("interaction_bootstrap_95ci") or [None, None])[0],
                    "ci95_upper": (interaction.get("interaction_bootstrap_95ci") or [None, None])[1],
                    "p_raw": interaction.get("p_raw"),
                    "p_adjusted_holm": (
                        population.get("D08_Holm", {})
                        .get("tests", {})
                        .get("interaction", {})
                        .get("adjusted_p")
                    ),
                    "replicates_or_sign_flips": interaction.get("replicates"),
                    "seed": interaction.get("seed"),
                },
                {
                    **table_counts,
                    "analysis_population": mode,
                    "analysis": "D04_noncentral_t_TOST",
                    "status": tost.get("status"),
                    "n": tost.get("n"),
                    "estimate": tost.get("M"),
                    "effect_size": tost.get("d_z"),
                    "ci95_lower": (tost.get("M_bootstrap_95ci") or [None, None])[0],
                    "ci95_upper": (tost.get("M_bootstrap_95ci") or [None, None])[1],
                    "p_raw": tost.get("p_raw"),
                    "p_adjusted_holm": (
                        population.get("D08_Holm", {})
                        .get("tests", {})
                        .get("TOST", {})
                        .get("adjusted_p")
                    ),
                    "replicates_or_sign_flips": tost.get("n"),
                    "seed": None,
                },
            ]
        )
        for label, test in population["D08_exploratory_within_type"]["tests"].items():
            rows.append(
                {
                    **table_counts,
                    "analysis_population": mode,
                    "analysis": f"D08_exploratory::{label}",
                    "status": test["status"],
                    "n": test["n"],
                    "estimate": test["estimate"],
                    "effect_size": None,
                    "ci95_lower": None,
                    "ci95_upper": None,
                    "p_raw": test["raw_p"],
                    "p_adjusted_holm": test["adjusted_p"],
                    "replicates_or_sign_flips": test.get("sign_flip_count"),
                    "seed": None,
                }
            )
    return rows


def write_r_input(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ("y", "query_id", "condition", "region", "C1", "C2")
    write_csv(path, rows, fields)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def configured_r_library() -> Path | None:
    configured = os.environ.get(R_LIBRARY_ENV_VAR)
    if configured:
        return Path(configured)
    return DEFAULT_R_LIBRARY if DEFAULT_R_LIBRARY.exists() else None


def probe_r_runtime() -> dict[str, Any]:
    rscript = shutil.which("Rscript")
    r_library = configured_r_library()
    if not rscript:
        return {
            "script_path": None,
            "library_path": None if r_library is None else str(r_library),
            "R_version": None,
            "packages": {name: {"available": False, "version": None} for name in R_PACKAGES},
            "probe_status": "Rscript_unavailable",
            "probe_error": None,
        }
    library_setup = (
        f".libPaths(c({json.dumps(str(r_library))}, .libPaths()));"
        if r_library is not None
        else ""
    )
    code = (
        library_setup
        + 'cat(sprintf("R_VERSION\\t%s\\n", as.character(getRversion())));'
        + 'for (p in c("lme4","lmerTest","emmeans","pbkrtest")) {'
        'ok <- requireNamespace(p, quietly=TRUE);'
        'v <- if (ok) as.character(packageVersion(p)) else "NA";'
        'cat(sprintf("PACKAGE\\t%s\\t%s\\t%s\\n", p, tolower(as.character(ok)), v))'
        '}'
    )
    try:
        completed = subprocess.run(
            [rscript, "-e", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "script_path": rscript,
            "library_path": None if r_library is None else str(r_library),
            "R_version": None,
            "packages": {name: {"available": False, "version": None} for name in R_PACKAGES},
            "probe_status": "probe_failed",
            "probe_error": str(exc),
        }
    r_version = None
    packages = {name: {"available": False, "version": None} for name in R_PACKAGES}
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) == 2 and fields[0] == "R_VERSION":
            r_version = fields[1]
        elif len(fields) == 4 and fields[0] == "PACKAGE" and fields[1] in packages:
            packages[fields[1]] = {
                "available": fields[2] == "true",
                "version": None if fields[3] == "NA" else fields[3],
            }
    ok = completed.returncode == 0 and all(
        package["available"] for package in packages.values()
    )
    return {
        "script_path": rscript,
        "library_path": None if r_library is None else str(r_library),
        "R_version": r_version,
        "packages": packages,
        "probe_status": "ready" if ok else "R_packages_missing",
        "probe_error": completed.stderr.strip() or None,
    }


def build_runtime_manifest(
    out_dir: Path,
    r_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if r_probe is None:
        r_probe = probe_r_runtime()
    return {
        "schema_version": "gate6-statistics-runtime-manifest-v1",
        "created_date": "2026-11-05",
        "status": "prepared_runtime_capture_pending",
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": {
                "numpy": package_version("numpy"),
                "scipy": package_version("scipy"),
                "pytest": package_version("pytest"),
            },
        },
        "r": {
            "required_for_D07": True,
            "Rscript_available": bool(r_probe["script_path"]),
            "Rscript_path": r_probe["script_path"],
            "library_path": r_probe["library_path"],
            "R_version_detected": r_probe["R_version"],
            "packages_detected": r_probe["packages"],
            "runtime_probe_status": r_probe["probe_status"],
            "runtime_probe_error": r_probe["probe_error"],
            "runtime_execution_status": "not_executed_probe_only",
            "target_versions_pending_runtime_verification": {
                "R": "4.6.2",
                "lme4": "2.1-0",
                "lmerTest": "3.2-1",
                "emmeans": "2.0.4",
                "pbkrtest": "0.5.5",
            },
            "version_capture_required_before_implementation_freeze": True,
            "session_info_output": "R-SESSION-INFO-20261105.txt",
            "script_path": "experiments/scripts/gate6_mixed_model.R",
            "script_sha256": sha256_file(R_SCRIPT) if R_SCRIPT.exists() else None,
            "expected_command": (
                "Rscript experiments/scripts/gate6_mixed_model.R "
                "--input <analysis-input.csv> --output-dir <mixed-model-output-dir>"
            ),
            "model": {
                "engine": "lme4::lmer",
                "fitting": "REML",
                "df": "lmerTest Satterthwaite",
                "contrast_engine": "emmeans",
                "formula": "y ~ region*(C1+C2)+(1|query_id)",
                "random_slopes": False,
                "C1": {
                    "local": -1.0 / 3.0,
                    "web": -1.0 / 3.0,
                    "dual": 2.0 / 3.0,
                },
                "C2": {"local": 1.0, "web": -1.0, "dual": 0.0},
                "preset_contrasts": {
                    "main": "beta_C1 + beta_region:C1 = dual - (local+web)/2 "
                    "at region=1",
                    "interaction": "beta_region:C1 = "
                    "(dual-(local+web)/2 at region=1) - "
                    "(dual-(local+web)/2 at region=0)",
                },
            },
        },
        "analysis_inputs": {
            "observed": "R-INPUT/OBSERVED-20261105.csv",
            "complete_case": "R-INPUT/COMPLETE-CASE-20261105.csv",
            "failure_zero_model_branch": "not_run_by_frozen_D11_rule",
        },
        "boundary": {
            "synthetic_fixtures_only": True,
            "confirmation_set_contacted": False,
            "confirmation_results_read": False,
            "gate6_started": False,
        },
    }


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}g}"


def build_report(results: Mapping[str, Any], runtime: Mapping[str, Any]) -> str:
    anchors = results["D12_anchors"]
    scoring = results["D11_scoring_counts"]
    observed = results["D11_populations"]["observed"]
    main = observed["D02_main"]
    interaction = observed["D03_interaction"]
    tost = observed["D04_TOST"]
    holm = observed.get("D08_Holm", {})
    exploratory = observed["D08_exploratory_within_type"]
    exploratory_rows = "\n".join(
        f"| {label} | {test['n']} | {format_float(test['estimate'])} | "
        f"{format_float(test['raw_p'])} | {format_float(test['adjusted_p'])} | "
        f"{str(test['exploratory']).lower()} |"
        for label, test in exploratory["tests"].items()
    )
    counts = observed["counts"]
    r_status = (
        f"{runtime['r']['runtime_execution_status']}; "
        f"probe={runtime['r']['runtime_probe_status']}"
    )
    return f"""# answer stage 统计实现（合成夹具，2026-11-05）

状态：`synthetic_fixture_verified_not_gate6_execution`。本目录只包含统计实现、
确定性合成夹具和测试证据。answer stage 未启动，确认集未接触、未读取、未分析，qrels v2
未创建或使用；这些事实由夹具 provenance 和运行器 fail-closed 检查共同绑定。

## 冻结口径

- D02：16 条 B 区查询的 `D_q=dual-(local+web)/2`；单侧精确配对 sign-flip，
  完整设计穷举 `2^16=65536` 次，`p=count(T_b>=T_obs)/65536`，保留等号，无 MC `+1`。
- D03：合并 B/S 的 32 个 `D_q` 后随机置换区域标签，每次保留 B=16、S=16；
  100,000 次，seed `20260826`，`p=(1+count(|I_b|>=|I_obs|))/100001`。
- D04：`N_q=dual-local` 或 `dual-web` 的 16 条单源足够查询；非中心 t 双单侧 TOST，
  界值 `delta=0.5`、每侧 alpha 0.05，`p_TOST=max(p_lower,p_upper)`。
- D06：seed `20260825`，B/S 分层按查询有放回重抽样，每区 16 条，10,000 次，
  95% percentile CI；Bootstrap 不生成确认性 p 值。
- D08：主对比、交互、TOST 共 3 个确认性检验；TOST 计 1 个，报告 raw 与 Holm adjusted p。
  类内来源两两对比组成独立的 12 检验探索性 Holm 族，同时报告 raw 与 Holm adjusted p，
  全部标注 `exploratory=true`，不与确认性族混合校正。
- D11：observed-data、严格完整案例和“仅 `F_qc=1` 记 0”三套分析；评分缺失不补 0，
  D07 不运行失败记 0 分支。
- D12：先执行一次 leave-one-out Spearman ρ<0.30 剔除，再在剩余评分员上计算 16 个锚点的
  ordinal Krippendorff alpha；若该 alpha < 0.67 则规则②至多触发一次，且不回退重跑 LOO；
  总剔除上限 2，全部答案剔除后至少保留 2 名有效评分员（fail-closed）。

## 合成夹具

- 设计答案：{results['fixture']['answer_count']}（期望 {EXPECTED_ANSWERS}）。
- 锚点：{results['fixture']['anchor_count']}。
- 成功生成：{scoring['successful_generation']}；失败/空：{scoring['failure_or_empty']}。
- 达到至少 2 个有效评分：{scoring['at_least_two_valid_ratings']}。
- observed-data 答案：{scoring['observed_answers']}。
- 缺失/无效评分记录：{scoring['missing_or_invalid_rating_records']}；评分缺失排除：
  {scoring['score_missing_exclusions']}。
- 主夹具无生成失败；失败记 0 与不足 2 评分的边界另有合成压力夹具单元测试。

## 主分析（observed-data）

| 分析 | n | 估计 | 效应量 | 95% CI | raw p | Holm p | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| D02 主对比 | {main.get('n')} | {format_float(main.get('observed_mean_D'))} | {format_float(main.get('d_z'))} | [{format_float((main.get('mean_bootstrap_95ci') or [None,None])[0])}, {format_float((main.get('mean_bootstrap_95ci') or [None,None])[1])}] | {format_float(main.get('p_raw'))} | {format_float(holm.get('tests',{}).get('main',{}).get('adjusted_p'))} | {holm.get('conclusions',{}).get('main')} |
| D03 交互 | B={interaction.get('n_B')}, S={interaction.get('n_S')} | {format_float(interaction.get('observed_interaction'))} | {format_float(interaction.get('d_pooled'))} | [{format_float((interaction.get('interaction_bootstrap_95ci') or [None,None])[0])}, {format_float((interaction.get('interaction_bootstrap_95ci') or [None,None])[1])}] | {format_float(interaction.get('p_raw'))} | {format_float(holm.get('tests',{}).get('interaction',{}).get('adjusted_p'))} | {holm.get('conclusions',{}).get('interaction')} |
| D04 TOST | {tost.get('n')} | M={format_float(tost.get('M'))}; S={format_float(tost.get('S'))} | d_z={format_float(tost.get('d_z'))} | [{format_float((tost.get('M_bootstrap_95ci') or [None,None])[0])}, {format_float((tost.get('M_bootstrap_95ci') or [None,None])[1])}] | {format_float(tost.get('p_raw'))} | {format_float(holm.get('tests',{}).get('TOST',{}).get('adjusted_p'))} | {holm.get('conclusions',{}).get('TOST')} |

D04 附加报告：`p_lower={format_float(tost.get('p_lower'))}`，
`p_upper={format_float(tost.get('p_upper'))}`，`Delta_raw={format_float(tost.get('Delta_raw'))}`，
M 的常规 90% CI 为 `{tost.get('mean_regular_90ci')}`。

## D08 类内探索性 Holm 族（独立于确认性族）

| 对比 | n | 估计 | raw p | Holm adjusted p | exploratory |
|---|---:|---:|---:|---:|---|
{exploratory_rows}

家庭大小固定为 {exploratory['test_count']}；全部为类内来源两两对比，使用双侧精确
sign-flip，等号保留且不加 Monte Carlo `+1`；未与 3 个确认性检验混合校正。

## 分母与敏感性

| 分析人群 | n_main | n_B | n_S | n_TOST | n_model_answers | n_model_queries | 完整案例查询 | 失败记 0 答案 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {mode} | {pop['counts']['n_main']} | {pop['counts']['n_B']} | "
        f"{pop['counts']['n_S']} | {pop['counts']['n_TOST']} | "
        f"{pop['counts']['n_model_answers']} | {pop['counts']['n_model_queries']} | "
        f"{pop['counts']['n_complete_case_queries']} | "
        f"{pop['counts']['failure_zero_filled_answers']} |"
        for mode, pop in results["D11_populations"].items()
    ) + f"""

## D12 锚点

- 执行顺序：先 LOO Spearman（阈值 {anchors['loo_spearman_threshold']}），再在剩余评分员上
  计算 ordinal alpha（阈值 {anchors['rule_two_threshold']}），最后至多一次规则②；LOO 不回退重跑。
- 七人 initial correctness alpha：{format_float(anchors['initial_correctness_alpha'])}。
- LOO Spearman 剔除：{', '.join(anchors['loo_spearman_removed_raters']) or 'none'}；
  剔除后 alpha：{format_float(anchors['alpha_after_loo_spearman'])}。
- 规则②触发：{str(anchors['rule_two_triggered']).lower()}；执行：
  {str(anchors['rule_two_executed']).lower()}；候选：
  {anchors['rule_two_candidate'] or 'none'}。
- 总剔除评分员：{', '.join(anchors['removed_raters']) or 'none'}。
- 最终 correctness alpha：{format_float(anchors['final_correctness_alpha'])}；有效评分员：
  {anchors['final_rater_count']}。
- 全部答案（含零剔除路径）≥2 名有效评分员校验：
  `{results['D12_global_rater_retention']['status']}`。
- Faithfulness alpha（只报告）：初次 {format_float(anchors['faithfulness_alpha_initial'])}，
  最终 {format_float(anchors['faithfulness_alpha_final'])}；`triggers_exclusion=false`。

## D07 与运行时

已准备 `experiments/scripts/gate6_mixed_model.R`、observed/complete-case R 输入 CSV 和
runtime manifest。R 运行状态：`{r_status}`。脚本固定使用
`lmerTest::lmer`、Satterthwaite、`emmeans` 预设对比、REML=TRUE、
`y ~ region*(C1+C2)+(1|query_id)`，不设随机斜率；C1 采用
`local=-1/3, web=-1/3, dual=+2/3`，因此
`L_main=beta_C1+beta_region:C1` 与 `dual-(local+web)/2` 直接一致，
不再使用 `1.5` 倍换算。不收敛或奇异时标记，不切换模型；R/package 版本由实际
`sessionInfo()` 与 runtime manifest 绑定。

## 边界

- 本报告不得写成 answer stage 已运行或“全部预注册”。
- Bootstrap 仅作 CI；D07 仅敏感性；确认性推断只来自 D02/D03/D04 的 D08 Holm 族。
- 失败记 0 只把 `F_qc=1` 记 0；评分缺失、少于 2 个有效评分及剔除造成的缺失保持缺失。
- 未因内容、条件、效应方向或 p 值排除任何合成答案。
"""


def bundle_files(out_dir: Path, exclude: Iterable[str] = ()) -> list[Path]:
    excluded = set(exclude)
    return sorted(
        path
        for path in out_dir.rglob("*")
        if path.is_file()
        and path.name != "._"
        and not path.name.startswith("._")
        and path.relative_to(out_dir).as_posix() not in excluded
    )


def write_implementation_hashes(out_dir: Path) -> Path:
    path = out_dir / "IMPLEMENTATION-HASHES-20261105.json"
    files: dict[str, str] = {}
    for source in (
        CORE_SCRIPT,
        RUNNER_SCRIPT,
        FINALIZER_SCRIPT,
        R_SCRIPT,
        REPO_ROOT / "experiments/tests/test_gate6_statistics.py",
        REPO_ROOT / "experiments/tests/test_run_gate6_statistics.py",
        REPO_ROOT / "experiments/tests/test_finalize_gate6_d07_c1_revision.py",
        FROZEN_PLAN,
    ):
        if source.exists():
            files[source.relative_to(REPO_ROOT).as_posix()] = sha256_file(source)
    generated = {}
    for output in bundle_files(
        out_dir,
        exclude={path.name, "SHA256SUMS-20261105.txt"},
    ):
        generated[output.relative_to(out_dir).as_posix()] = sha256_file(output)
    payload = {
        "schema_version": "gate6-statistics-implementation-hashes-v1",
        "created_date": "2026-11-05",
        "status": "prepared_not_reviewed_not_frozen",
        "gate6_readiness_B1_07": "open",
        "source_files": files,
        "generated_files": generated,
    }
    write_json(path, payload)
    return path


def write_sha_manifest(out_dir: Path) -> Path:
    path = out_dir / "SHA256SUMS-20261105.txt"
    rows = []
    for output in bundle_files(out_dir, exclude={path.name}):
        rows.append(f"{sha256_file(output)}  {output.relative_to(out_dir).as_posix()}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def write_bundle(
    out_dir: Path,
    *,
    fixture_path: Path | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    permutation_replicates: int = INTERACTION_PERMUTATION_REPLICATES,
) -> dict[str, Any]:
    ensure_writable_output_dir(out_dir)
    plan_binding = verify_frozen_plan()
    r_probe = probe_r_runtime()
    if r_probe["probe_status"] != "ready":
        raise Gate6StatisticsError(
            "R runtime required for D07 is unavailable: "
            f"probe_status={r_probe['probe_status']}; "
            f"Rscript={r_probe['script_path']}; library={r_probe['library_path']}; "
            f"packages={r_probe['packages']}; error={r_probe['probe_error']}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if fixture_path is None:
        fixture = build_synthetic_fixture()
        fixture_path = out_dir / "SYNTHETIC-FIXTURE-20261105.json"
        write_json(fixture_path, fixture)
    else:
        fixture = load_synthetic_fixture(fixture_path)
        destination = out_dir / "SYNTHETIC-FIXTURE-20261105.json"
        if fixture_path.resolve() != destination.resolve():
            shutil.copyfile(fixture_path, destination)
            fixture_path = destination

    results = analyze_synthetic_fixture(
        fixture,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
    )
    results["artifact_paths"] = {
        "fixture": "SYNTHETIC-FIXTURE-20261105.json",
        "report": "GATE6-STATISTICS-REPORT-20261105.md",
        "runtime_manifest": "RUNTIME-MANIFEST-20261105.json",
        "observed_r_input": "R-INPUT/OBSERVED-20261105.csv",
        "complete_case_r_input": "R-INPUT/COMPLETE-CASE-20261105.csv",
    }
    results["frozen_plan_binding"] = plan_binding
    results["r_runtime_preflight"] = {
        "probe_status": r_probe["probe_status"],
        "Rscript_path": r_probe["script_path"],
        "library_path": r_probe["library_path"],
        "R_version": r_probe["R_version"],
        "packages": r_probe["packages"],
    }
    write_json(out_dir / "GATE6-STATISTICS-RESULTS-20261105.json", results)

    denominator_rows = build_denominator_rows(results)
    denominator_fields = list(denominator_rows[0])
    write_csv(
        out_dir / "TABLES/DENOMINATORS-20261105.csv",
        denominator_rows,
        denominator_fields,
    )
    analysis_rows = build_analysis_rows(results)
    analysis_fields = list(analysis_rows[0])
    write_csv(
        out_dir / "TABLES/ANALYSES-20261105.csv",
        analysis_rows,
        analysis_fields,
    )

    query_fields = (
        "query_id",
        "query_type",
        "region",
        "score_local",
        "score_web",
        "score_dual",
        "D_q",
        "TOST_N_q",
        "main_included_D02",
        "interaction_included_D03",
        "tost_included_D04",
        "model_rows_included_D07",
    )
    for mode, population in results["D11_populations"].items():
        query_rows = population["query_inputs"]["query_rows"]
        write_csv(
            out_dir / f"TABLES/QUERY-LEVEL-{mode.upper().replace('_', '-')}-20261105.csv",
            query_rows,
            query_fields,
        )
        if mode != "failure_zero":
            write_r_input(
                out_dir
                / f"R-INPUT/{mode.upper().replace('_', '-')}-20261105.csv",
                population["query_inputs"]["model_rows"],
            )

    runtime = build_runtime_manifest(out_dir, r_probe)
    write_json(out_dir / "RUNTIME-MANIFEST-20261105.json", runtime)
    report = build_report(results, runtime)
    (out_dir / "GATE6-STATISTICS-REPORT-20261105.md").write_text(report, encoding="utf-8")
    write_implementation_hashes(out_dir)
    write_sha_manifest(out_dir)
    return results


def finalize_bundle(out_dir: Path) -> None:
    ensure_writable_output_dir(out_dir)
    verify_frozen_plan()
    if not (out_dir / "GATE6-STATISTICS-RESULTS-20261105.json").exists():
        raise Gate6StatisticsError(f"{out_dir} does not contain a statistics result bundle")
    write_implementation_hashes(out_dir)
    write_sha_manifest(out_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="New bundle directory; immutable frozen evidence directories are refused.",
    )
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument(
        "--permutation-replicates", type=int, default=INTERACTION_PERMUTATION_REPLICATES
    )
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.finalize_only:
            finalize_bundle(args.out_dir)
            return 0
        write_bundle(
            args.out_dir,
            fixture_path=args.fixture,
            bootstrap_replicates=args.bootstrap_replicates,
            permutation_replicates=args.permutation_replicates,
        )
    except Gate6StatisticsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

