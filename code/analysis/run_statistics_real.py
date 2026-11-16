#!/usr/bin/env python3
"""answer stage real-data statistics adapter — executes the frozen pipeline exactly once.

This script assembles the frozen linked dataset into the input schema of the
frozen statistical library and runs the SAME orchestration as
``gate6_statistics.analyze_synthetic_fixture``. The only differences:
1) the synthetic-fixture guard is replaced by real-dataset validation;
2) ratings on the 1–5 blind-eval scale are mapped to the library's 0–1 scale
   with the affine map ``s = (r - 1) / 4``.

No statistical rule, threshold, seed, replicate count, denominator rule or
exclusion rule is changed. All confirmatory statistics are invariant to positive
affine rescaling; only descriptive raw units and R coefficients depend on the
rating-scale mapping, which is recorded in the bundle.

Fail-closed:
- frozen plan and frozen implementation SHA-256s must match the registry;
- the linked dataset must satisfy the 96/32/16 design and 32/32/32, 8/8/8/8 splits;
- the R runtime required by D07 must be present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path("/Volumes/Lenovo/26大创/OrcaAI-EI论文项目")
sys.path.insert(0, str(REPO / "experiments/scripts"))

import run_statistics as runner  # noqa: E402
from stats_core import (  # noqa: E402
    ANCHOR_ALPHA_THRESHOLD,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONDITIONS,
    EXPECTED_ANCHORS,
    EXPECTED_ANSWERS,
    EXPECTED_QUERY_COUNT,
    EXPLORATORY_ALPHA,
    EXPLORATORY_FAMILY_NAME,
    EXPLORATORY_TEST_COUNT,
    INTERACTION_PERMUTATION_REPLICATES,
    INTERACTION_PERMUTATION_SEED,
    MAIN_ALPHA,
    MAX_RATER_EXCLUSIONS,
    QUERY_TYPES,
    RATER_IDS,
    TOST_ALPHA,
    TOST_BOUND_D,
    Gate6StatisticsError,
    analyze_population,
    assess_anchor_reliability,
    build_population,
    holm_adjust,
    score_answers,
)

DATE = "20261113"
EXPECTED_IMPLEMENTATION_SHA256 = {
    "experiments/scripts/gate6_statistics.py": "893198b19aaa1650064e8e950c6cbdbcd2bbd780285b9d9aa0d93323db9ac0e6",
    "experiments/scripts/run_gate6_statistics.py": "b89e4b0658d586e8e8e3739d721ca062f453186374a904b7153f3a849df746a4",
    "experiments/scripts/gate6_mixed_model.R": "34511f8392d5c426e8659c87acc0c2965822ab0218973f99d49a6979170ec970",
}
DEFAULT_RECEIPT = REPO / f"experiments/results/gate6-linked-dataset-receipt-{DATE}/RECEIPT-{DATE}.json"


def sha256_file(path: Path) -> str:
    return runner.sha256_file(path)


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def verify_frozen_implementation() -> dict[str, str]:
    runner.verify_frozen_plan()
    verified: dict[str, str] = {}
    for rel, expected in EXPECTED_IMPLEMENTATION_SHA256.items():
        path = REPO / rel
        if not path.exists():
            fail(f"frozen implementation missing: {rel}")
        actual = sha256_file(path)
        if actual != expected:
            fail(f"frozen implementation hash mismatch: {rel} has {actual}, expected {expected}")
        verified[rel] = actual
    return verified


def load_linked_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"linked dataset missing: {path}")
    ds = json.loads(path.read_text(encoding="utf-8"))
    if ds.get("schema_version") != "gate6-linked-analysis-dataset-v1-20261113":
        fail("unexpected linked dataset schema")
    boundary = ds.get("boundary") or {}
    if boundary.get("derived_from_confirmation_set") is not True:
        fail("linked dataset must be derived from the confirmation set")
    if boundary.get("original_ratings_unmodified") is not True:
        fail("linked dataset must assert unmodified original ratings")
    if boundary.get("blind_key_copied_into_repo") is not False:
        fail("linked dataset boundary violated: blind key copied into repo")
    counts = ds.get("counts") or {}
    if counts.get("answers") != EXPECTED_ANSWERS or counts.get("queries") != EXPECTED_QUERY_COUNT:
        fail(f"linked dataset counts wrong: {counts}")
    if counts.get("anchors") != EXPECTED_ANCHORS:
        fail("linked dataset must contain 16 anchors")
    if counts.get("conditions") != {"local_only": 32, "web_only": 32, "dual": 32}:
        fail(f"condition counts wrong: {counts.get('conditions')}")
    if any(counts.get("query_types", {}).get(t) != 8 for t in QUERY_TYPES):
        fail(f"query type counts wrong: {counts.get('query_types')}")
    answers = ds.get("answers") or []
    if len(answers) != EXPECTED_ANSWERS:
        fail("linked dataset must contain exactly 96 answers")
    anchors = list(ds.get("anchor_answer_slots") or [])
    if len(anchors) != EXPECTED_ANCHORS or len(set(anchors)) != EXPECTED_ANCHORS:
        fail("linked dataset must list 16 unique anchor slots")
    seen_ids = set()
    by_slot = {a["answer_slot_id"]: a for a in answers}
    for a in answers:
        if a["answer_id"] in seen_ids:
            fail(f"duplicate answer_id {a['answer_id']}")
        seen_ids.add(a["answer_id"])
        if a["query_type"] not in QUERY_TYPES or a["condition"] not in CONDITIONS:
            fail(f"{a['answer_id']}: invalid query_type/condition")
        if not str(a.get("answer_text") or "").strip():
            fail(f"{a['answer_id']}: empty answer text")
        rids = [r["rater_id"] for r in a["ratings"]]
        if len(rids) != len(set(rids)) or not set(rids) <= set(RATER_IDS):
            fail(f"{a['answer_id']}: invalid rater list")
        for r in a["ratings"]:
            for key in ("correctness_1to5", "faithfulness_1to5"):
                value = r.get(key)
                if not isinstance(value, int) or not 1 <= value <= 5:
                    fail(f"{a['answer_id']}/{r['rater_id']}: {key} must be a 1–5 integer")
        expected_raters = 7 if a["answer_slot_id"] in anchors else 4
        if len(rids) != expected_raters:
            fail(f"{a['answer_id']}: expected {expected_raters} raters, found {len(rids)}")
    for slot in anchors:
        a = by_slot.get(slot)
        if a is None:
            fail(f"anchor slot {slot} not present")
        if set(r["rater_id"] for r in a["ratings"]) != set(RATER_IDS):
            fail(f"anchor {slot} must be rated by all seven raters")
    return ds


def normalize_scale(value_1to5: int) -> float:
    return (float(value_1to5) - 1.0) / 4.0


def build_library_answers(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    answers = []
    for a in dataset["answers"]:
        answers.append(
            {
                "answer_id": a["answer_id"],
                "answer_slot_id": a["answer_slot_id"],
                "query_id": a["query_id"],
                "query_type": a["query_type"],
                "condition": a["condition"],
                "generation_status": a["generation_status"],
                "answer_text": a["answer_text"],
                "ratings": [
                    {
                        "rater_id": r["rater_id"],
                        "correctness": normalize_scale(r["correctness_1to5"]),
                        "faithfulness": normalize_scale(r["faithfulness_1to5"]),
                        "valid": True,
                    }
                    for r in a["ratings"]
                ],
            }
        )
    return answers


def run_orchestration(
    answers: Sequence[Mapping[str, Any]],
    anchor_answer_slots: Sequence[str],
    *,
    bootstrap_replicates: int,
    permutation_replicates: int,
) -> dict[str, Any]:
    """Verbatim orchestration of analyze_synthetic_fixture minus the synthetic guard.

    ``answers`` use the frozen library schema and must carry ``answer_slot_id``.
    ``anchor_answer_slots`` lists the 16 anchor slot IDs (rated by all seven raters).
    """

    query_types = {str(answer["query_id"]): str(answer["query_type"]) for answer in answers}
    if len(query_types) != EXPECTED_QUERY_COUNT:
        raise Gate6StatisticsError("dataset must contain exactly 32 queries")
    type_counts = Counter(query_types.values())
    if any(type_counts[t] != 8 for t in QUERY_TYPES):
        raise Gate6StatisticsError("dataset must contain eight queries per query type")

    anchor_ids = list(anchor_answer_slots)
    if len(anchor_ids) != EXPECTED_ANCHORS or len(set(anchor_ids)) != EXPECTED_ANCHORS:
        raise Gate6StatisticsError("dataset must contain 16 unique anchor answer slots")
    by_slot = {str(a["answer_slot_id"]): a for a in answers}
    try:
        anchor_answers = [by_slot[slot] for slot in anchor_ids]
    except KeyError as exc:
        raise Gate6StatisticsError(f"unknown anchor slot: {exc.args[0]}") from exc

    anchor_ratings: list[dict[str, dict[str, float]]] = []
    for answer in anchor_answers:
        rating_map = {str(r["rater_id"]): r for r in answer.get("ratings", [])}
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
    raw_p: dict[str, float] = {}
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
    tost_raw_support = bool(observed["D04_TOST"].get("equivalence_supported_uncorrected", False))
    observed["D08_Holm"]["conclusions"] = {
        "main": "supported"
        if holm["main"]["reject_at_0_05"]
        and (observed["D02_main"].get("observed_mean_D") or 0.0) > 0.0
        else "not_supported",
        "interaction": "detected" if holm["interaction"]["reject_at_0_05"] else "not_detected",
        "TOST": "equivalent_within_frozen_bound"
        if tost_raw_support and holm["TOST"]["reject_at_0_05"]
        else "insufficient_equivalence_evidence",
    }

    return {
        "schema_version": "gate6-statistics-results-v1",
        "implementation_status": "gate6_execution_real_data",
        "fixture": {
            "fixture_kind": "gate6_linked_real_dataset",
            "fixture_id": "GATE6-LINKED-DATASET-20261113",
            "answer_count": len(answers),
            "anchor_count": len(anchor_ids),
            "synthetic_only": False,
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
            "rating_scale_mapping": "s = (r - 1) / 4 (1–5 integer ratings -> 0–1 scores)",
            "bootstrap_replicates_used": bootstrap_replicates,
            "interaction_replicates_used": permutation_replicates,
        },
        "D11_scoring_counts": scored["counts"],
        "D12_anchors": anchors,
        "D12_global_rater_retention": scored["rater_retention_check"],
        "D11_populations": populations,
    }


def build_report(
    results: Mapping[str, Any],
    *,
    dataset_meta: Mapping[str, Any],
    r_status: str,
) -> str:
    fmt = runner.format_float
    anchors = results["D12_anchors"]
    scoring = results["D11_scoring_counts"]
    observed = results["D11_populations"]["observed"]
    main = observed["D02_main"]
    interaction = observed["D03_interaction"]
    tost = observed["D04_TOST"]
    holm = observed.get("D08_Holm", {})
    exploratory = observed["D08_exploratory_within_type"]
    exploratory_rows = "\n".join(
        f"| {label} | {test['n']} | {fmt(test['estimate'])} | "
        f"{fmt(test['raw_p'])} | {fmt(test['adjusted_p'])} | "
        f"{str(test['exploratory']).lower()} |"
        for label, test in exploratory["tests"].items()
    )
    denominator_rows = "\n".join(
        f"| {mode} | {pop['counts']['n_main']} | {pop['counts']['n_B']} | "
        f"{pop['counts']['n_S']} | {pop['counts']['n_TOST']} | "
        f"{pop['counts']['n_model_answers']} | {pop['counts']['n_model_queries']} | "
        f"{pop['counts']['n_complete_case_queries']} | "
        f"{pop['counts']['failure_zero_filled_answers']} |"
        for mode, pop in results["D11_populations"].items()
    )
    return f"""# answer stage 正式统计（确认集，2026-11-13）

状态：`gate6_execution_real_data`。本目录是 answer stage 主实验的**唯一一次**正式统计，
输入为冻结的确认集连接数据集（仓库外），运行冻结统计实现
（`gate6_statistics.py` SHA-256 `{EXPECTED_IMPLEMENTATION_SHA256['experiments/scripts/gate6_statistics.py']}`）。
确认集按 B2 流程正式使用一次；qrels v2 未创建、未使用；仲裁结论不覆盖原始评分。

## 输入与口径

- 连接数据集：`{dataset_meta['dataset_path_class']}`，SHA-256 `{dataset_meta['sha256']}`。
- 评分员：B01–B07；答案 96（32 查询 × 3 条件）；锚点 16（全员必评）。
- 评分量表映射：盲评 1–5 整数按仿射映射 `s=(r-1)/4` 转为 0–1；确认性检验对正仿射
  变换不变，仅描述单位与 R 系数受影响。
- D02/D03/D04 与 D06/D08 口径完全沿用冻结计划（见 `frozen_parameters`）。

## 评分计数（D11）

- 成功生成：{scoring['successful_generation']}；失败/空：{scoring['failure_or_empty']}。
- 达到至少 2 个有效评分：{scoring['at_least_two_valid_ratings']}。
- observed-data 答案：{scoring['observed_answers']}。
- 缺失/无效评分记录：{scoring['missing_or_invalid_rating_records']}；评分缺失排除：
  {scoring['score_missing_exclusions']}。

## 主分析（observed-data）

| 分析 | n | 估计 | 效应量 | 95% CI | raw p | Holm p | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| D02 主对比 | {main.get('n')} | {fmt(main.get('observed_mean_D'))} | {fmt(main.get('d_z'))} | [{fmt((main.get('mean_bootstrap_95ci') or [None,None])[0])}, {fmt((main.get('mean_bootstrap_95ci') or [None,None])[1])}] | {fmt(main.get('p_raw'))} | {fmt(holm.get('tests',{}).get('main',{}).get('adjusted_p'))} | {holm.get('conclusions',{}).get('main')} |
| D03 交互 | B={interaction.get('n_B')}, S={interaction.get('n_S')} | {fmt(interaction.get('observed_interaction'))} | {fmt(interaction.get('d_pooled'))} | [{fmt((interaction.get('interaction_bootstrap_95ci') or [None,None])[0])}, {fmt((interaction.get('interaction_bootstrap_95ci') or [None,None])[1])}] | {fmt(interaction.get('p_raw'))} | {fmt(holm.get('tests',{}).get('interaction',{}).get('adjusted_p'))} | {holm.get('conclusions',{}).get('interaction')} |
| D04 TOST | {tost.get('n')} | M={fmt(tost.get('M'))}; S={fmt(tost.get('S'))} | d_z={fmt(tost.get('d_z'))} | [{fmt((tost.get('M_bootstrap_95ci') or [None,None])[0])}, {fmt((tost.get('M_bootstrap_95ci') or [None,None])[1])}] | {fmt(tost.get('p_raw'))} | {fmt(holm.get('tests',{}).get('TOST',{}).get('adjusted_p'))} | {holm.get('conclusions',{}).get('TOST')} |

D04 附加：`p_lower={fmt(tost.get('p_lower'))}`，`p_upper={fmt(tost.get('p_upper'))}`，
`Delta_raw={fmt(tost.get('Delta_raw'))}`，M 的常规 90% CI 为 `{tost.get('mean_regular_90ci')}`。

## D08 类内探索性 Holm 族（独立于确认性族）

| 对比 | n | 估计 | raw p | Holm adjusted p | exploratory |
|---|---:|---:|---:|---:|---|
{exploratory_rows}

家庭大小固定为 {exploratory['test_count']}；全部标注 `exploratory=true`，不与 3 个确认性检验混合校正。

## 分母与敏感性

| 分析人群 | n_main | n_B | n_S | n_TOST | n_model_answers | n_model_queries | 完整案例查询 | 失败记 0 答案 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{denominator_rows}

## D12 锚点

- 执行顺序：先 LOO Spearman（阈值 {anchors['loo_spearman_threshold']}），再在剩余评分员上
  计算 ordinal alpha（阈值 {anchors['rule_two_threshold']}），最后至多一次规则②；LOO 不回退重跑。
- 七人 initial correctness alpha：{fmt(anchors['initial_correctness_alpha'])}。
- LOO Spearman 剔除：{', '.join(anchors['loo_spearman_removed_raters']) or 'none'}；
  剔除后 alpha：{fmt(anchors['alpha_after_loo_spearman'])}。
- 规则②触发：{str(anchors['rule_two_triggered']).lower()}；执行：
  {str(anchors['rule_two_executed']).lower()}；候选：{anchors['rule_two_candidate'] or 'none'}。
- 总剔除评分员：{', '.join(anchors['removed_raters']) or 'none'}。
- 最终 correctness alpha：{fmt(anchors['final_correctness_alpha'])}；有效评分员：
  {anchors['final_rater_count']}。
- ≥2 名有效评分员校验：`{results['D12_global_rater_retention']['status']}`。
- Faithfulness alpha（只报告）：初次 {fmt(anchors['faithfulness_alpha_initial'])}，
  最终 {fmt(anchors['faithfulness_alpha_final'])}；`triggers_exclusion=false`。

## D07 混合模型（敏感性）

R 运行状态：`{r_status}`。模型固定为 `lmerTest::lmer`、Satterthwaite、`emmeans` 预设对比、
REML=TRUE、`y ~ region*(C1+C2)+(1|query_id)`；C1 为 `local=-1/3, web=-1/3, dual=+2/3`。
不收敛或奇异时标记，不切换模型。R 版本与包版本由运行时捕获绑定。

## 边界

- 本结果为唯一一次正式分析；不得重跑、换参数或据 p 值调整。
- Bootstrap 仅作 CI；D07 仅敏感性；确认性推断只来自 D02/D03/D04 的 D08 Holm 族。
- 论文不得写成“全部预注册”；生成模型不可逐字复现须如实披露。
"""


def write_implementation_hashes(out_dir: Path, *, extra: Mapping[str, str]) -> Path:
    files: dict[str, str] = {}
    for rel in EXPECTED_IMPLEMENTATION_SHA256:
        files[rel] = sha256_file(REPO / rel)
    for rel in (
        "experiments/scripts/run_gate6_statistics_real.py",
        "experiments/scripts/build_gate6_linked_dataset.py",
    ):
        path = REPO / rel
        if path.exists():
            files[rel] = sha256_file(path)
    files.update(extra)
    generated = {}
    for output in runner.bundle_files(out_dir, exclude={"IMPLEMENTATION-HASHES-20261113.json", f"SHA256SUMS-{DATE}.txt"}):
        generated[output.relative_to(out_dir).as_posix()] = sha256_file(output)
    payload = {
        "schema_version": "gate6-statistics-implementation-hashes-v1",
        "created_date": "2026-11-13",
        "status": "gate6_execution_real_data",
        "source_files": files,
        "generated_files": generated,
    }
    path = out_dir / "IMPLEMENTATION-HASHES-20261113.json"
    runner.write_json(path, payload)
    return path


def run_r(input_csv: Path, out_subdir: Path) -> dict[str, Any]:
    rscript = shutil.which("Rscript")
    if not rscript:
        return {"status": "Rscript_unavailable"}
    out_subdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    lib = runner.configured_r_library()
    if lib is not None:
        env["R_LIBS"] = str(lib)
    completed = subprocess.run(
        [rscript, str(runner.R_SCRIPT), "--input", str(input_csv), "--output-dir", str(out_subdir)],
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
    )
    (out_subdir / "R-STDOUT.txt").write_text(completed.stdout, encoding="utf-8")
    (out_subdir / "R-STDERR.txt").write_text(completed.stderr, encoding="utf-8")
    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout_chars": len(completed.stdout),
        "stderr_tail": completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else None,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Linked dataset JSON (outside repo)")
    parser.add_argument("--out-dir", type=Path, required=True, help="New bundle directory")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--permutation-replicates", type=int, default=INTERACTION_PERMUTATION_REPLICATES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    implementation = verify_frozen_implementation()
    dataset = load_linked_dataset(args.dataset)
    dataset_sha = sha256_file(args.dataset)
    answers = build_library_answers(dataset)
    results = run_orchestration(
        answers,
        dataset["anchor_answer_slots"],
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_replicates=args.permutation_replicates,
    )

    runner.ensure_writable_output_dir(args.out_dir)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    # R inputs
    for mode in ("observed", "complete_case"):
        runner.write_r_input(
            out_dir / f"R-INPUT/{mode.upper().replace('_', '-')}-20261113.csv",
            results["D11_populations"][mode]["query_inputs"]["model_rows"],
        )

    r_results = {
        mode: run_r(
            out_dir / f"R-INPUT/{mode.upper().replace('_', '-')}-20261113.csv",
            out_dir / f"R-OUTPUT/{mode.upper().replace('_', '-')}",
        )
        for mode in ("observed", "complete_case")
    }
    r_status = "; ".join(f"{mode}={res['status']}" for mode, res in r_results.items())
    results["D07_R_execution"] = r_results

    # dataset metadata (no mapping) and boundary
    receipt = json.loads(args.receipt.read_text(encoding="utf-8")) if args.receipt.exists() else {}
    results["artifact_paths"] = {
        "linked_dataset": "outside_repository",
        "receipt": str(args.receipt.relative_to(REPO)) if args.receipt.is_relative_to(REPO) else str(args.receipt),
        "report": "GATE6-STATISTICS-REPORT-20261113.md",
        "runtime_manifest": "RUNTIME-MANIFEST-20261113.json",
        "observed_r_input": "R-INPUT/OBSERVED-20261113.csv",
        "complete_case_r_input": "R-INPUT/COMPLETE-CASE-20261113.csv",
    }
    results["linked_dataset"] = {
        "sha256": dataset_sha,
        "counts": dataset["counts"],
        "dataset_path_class": "outside_repository_restricted",
        "receipt_sha256": sha256_file(args.receipt) if args.receipt.exists() else None,
        "receipt_matches_dataset": bool(receipt.get("dataset_sha256") == dataset_sha),
    }
    results["frozen_implementation"] = implementation
    results["execution"] = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "one_shot": True,
        "python": sys.version,
        "platform": platform.platform(),
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_replicates": args.permutation_replicates,
    }

    runner.write_json(out_dir / "GATE6-STATISTICS-RESULTS-20261113.json", results)

    denominator_rows = runner.build_denominator_rows(results)
    runner.write_csv(out_dir / "TABLES/DENOMINATORS-20261113.csv", denominator_rows, list(denominator_rows[0]))
    analysis_rows = runner.build_analysis_rows(results)
    runner.write_csv(out_dir / "TABLES/ANALYSES-20261113.csv", analysis_rows, list(analysis_rows[0]))
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
        runner.write_csv(
            out_dir / f"TABLES/QUERY-LEVEL-{mode.upper().replace('_', '-')}-20261113.csv",
            population["query_inputs"]["query_rows"],
            query_fields,
        )

    runtime = runner.build_runtime_manifest(out_dir)
    runtime["created_date"] = "2026-11-13"
    runtime["status"] = "gate6_execution_real_data"
    runtime["r"]["runtime_execution_status"] = r_status
    runner.write_json(out_dir / "RUNTIME-MANIFEST-20261113.json", runtime)

    report = build_report(results, dataset_meta=results["linked_dataset"], r_status=r_status)
    (out_dir / "GATE6-STATISTICS-REPORT-20261113.md").write_text(report, encoding="utf-8")

    extra = {
        "linked_dataset_sha256": dataset_sha,
        "receipt_sha256": results["linked_dataset"]["receipt_sha256"] or "missing",
    }
    write_implementation_hashes(out_dir, extra=extra)
    rows = []
    for output in runner.bundle_files(out_dir, exclude={f"SHA256SUMS-{DATE}.txt"}):
        rows.append(f"{sha256_file(output)}  {output.relative_to(out_dir).as_posix()}")
    (out_dir / f"SHA256SUMS-{DATE}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"OK bundle: {out_dir}")
    print(f"   observed: D02 p={results['D11_populations']['observed']['D02_main'].get('p_raw')}, "
          f"D03 p={results['D11_populations']['observed']['D03_interaction'].get('p_raw')}, "
          f"TOST p={results['D11_populations']['observed']['D04_TOST'].get('p_raw')}")
    print(f"   D12: alpha={results['D12_anchors'].get('final_correctness_alpha')}, "
          f"removed={results['D12_anchors'].get('removed_raters')}")
    print(f"   R: {r_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())