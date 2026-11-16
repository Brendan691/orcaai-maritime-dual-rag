#!/usr/bin/env python3
"""Recompute the paper's confirmatory statistics from the public per-query scores."""
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "derived" / "query-scores.csv"

rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
B_TYPES = {"complementary", "conflicting"}
b = np.array([float(r["dual_minus_single_mean"]) for r in rows if r["query_type"] in B_TYPES])
s = np.array([float(r["dual_minus_single_mean"]) for r in rows if r["query_type"] not in B_TYPES])
tost = np.array([float(r["dual_minus_paired_single"]) for r in rows if r["equivalence_included"].lower() == "true"])

# main contrast: exact paired sign-flip (2^16 enumeration, ties retained)
n = b.size
bits = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
signs = 2.0 * bits - 1.0
means = signs @ b / n
p_main = float(np.count_nonzero(means >= b.mean())) / (1 << n)

# interaction: 100,000 region-label permutations, seed 20260826, two-sided
rng = np.random.default_rng(20260826)
obs = float(b.mean() - s.mean())
combined = np.concatenate([b, s])
labels = rng.permuted(np.broadcast_to(np.arange(combined.size), (100_000, combined.size)), axis=1)
vals = np.take_along_axis(combined[None, :], labels, axis=1)
perm = vals[:, : b.size].mean(axis=1) - vals[:, b.size :].mean(axis=1)
p_inter = float((1 + np.count_nonzero(np.abs(perm) >= abs(obs))) / 100_001)

# equivalence: noncentral t TOST (bound d=0.5, n=16)
n4 = tost.size
m4, sd4 = float(tost.mean()), float(tost.std(ddof=1))
dz4 = m4 / sd4
t4 = dz4 * math.sqrt(n4)
p_low = float(1 - stats.nct.cdf(t4, df=n4 - 1, nc=-0.5 * math.sqrt(n4)))
p_up = float(stats.nct.cdf(t4, df=n4 - 1, nc=0.5 * math.sqrt(n4)))
p_eq = max(p_low, p_up)

expected = {
    "main contrast estimate": (float(b.mean()), 0.271763, 1e-6),
    "main contrast p (exact sign-flip)": (p_main, 6.10e-05, 5e-7),
    "interaction estimate": (obs, 0.154297, 1e-6),
    "interaction p (permutation)": (p_inter, 0.01912, 5e-5),
    "equivalence mean difference": (m4, -0.008371, 1e-6),
    "equivalence d": (dz4, -0.0767, 5e-4),
    "equivalence p (TOST)": (p_eq, 0.044977, 5e-5),
}
print(f"{'quantity':34s} {'recomputed':>14s} {'paper':>12s}  match")
all_ok = True
for name, (got, want, tol) in expected.items():
    ok = abs(got - want) <= tol
    all_ok &= ok
    print(f"{name:34s} {got:14.6f} {want:12.6f}  {'OK' if ok else 'MISMATCH'}")
sys.exit(0 if all_ok else 1)
