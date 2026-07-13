#!/usr/bin/env python3
"""
Statistical-hygiene helper: produces paper-ready strings for accuracy +
Wilson 95% CI and bounded p-values. Reviewers dislike `p = 1.69 × 10⁻⁸⁰`
style precision because it implies more discrimination than the test
actually provides; clipping at `p < 10⁻¹⁰` is the standard fix.

Usage as a module:

    from scripts.format_paper_stats import wilson_ci_str, mcnemar_p_str
    print(wilson_ci_str(3343, 3429))      # → "0.9749 [0.9691, 0.9796]"
    print(mcnemar_p_str(266, 0))          # → "p < 10^{-10}"

Or as a script reading the pooled CSV:

    python scripts/format_paper_stats.py \\
        --pooled artifacts/eval_joined_pooled_20260202_192420.csv \\
        --mode-a ADAPTIVERAG --mode-b RAG_BASE
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Tuple


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_ci_str(k: int, n: int) -> str:
    p = k / n if n else 0.0
    lo, hi = wilson_ci(k, n)
    return f"{p:.4f} [{lo:.4f}, {hi:.4f}]"


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = 0.0
    for i in range(k + 1):
        tail += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2 * tail)


def mcnemar_p_str(b: int, c: int, floor_exp: int = -10) -> str:
    """Returns 'p < 10^{-N}' when the exact p is below that threshold, else
    standard scientific notation. Default floor of 10^{-10}."""
    p = mcnemar_exact_p(b, c)
    if p == 0.0 or p < 10 ** floor_exp:
        return f"p < 10^{{{floor_exp}}}"
    if p < 0.001:
        return f"p = {p:.2e}"
    return f"p = {p:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled", type=Path, required=True)
    ap.add_argument("--mode-a", required=True)
    ap.add_argument("--mode-b", required=True)
    args = ap.parse_args()

    a_success: dict = {}
    b_success: dict = {}
    with args.pooled.open() as fh:
        for r in csv.DictReader(fh):
            key = (r["id"], r["domain"])
            if r["mode"] == args.mode_a:
                a_success[key] = int(r["success"])
            elif r["mode"] == args.mode_b:
                b_success[key] = int(r["success"])

    common = set(a_success) & set(b_success)
    k_a = sum(a_success[k] for k in common)
    k_b = sum(b_success[k] for k in common)
    n = len(common)
    a_only = sum(1 for k in common if a_success[k] and not b_success[k])
    b_only = sum(1 for k in common if b_success[k] and not a_success[k])

    print(f"n common = {n}")
    print(f"{args.mode_a}: {wilson_ci_str(k_a, n)}")
    print(f"{args.mode_b}: {wilson_ci_str(k_b, n)}")
    print(f"McNemar: {args.mode_a}-only={a_only}  {args.mode_b}-only={b_only}  {mcnemar_p_str(a_only, b_only)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
