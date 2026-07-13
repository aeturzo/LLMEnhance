#!/usr/bin/env python3
"""
Aggregate baseline CSVs into a paper-ready comparison table.

For each baseline CSV produced by scripts/run_baselines.py, computes:

  * pooled accuracy + Wilson 95% confidence interval
  * by-type accuracy (logic / open / recall)
  * McNemar paired test against ADAPTIVERAG on the (id, domain) pairs that
    appear in BOTH the baseline CSV and the pooled CSV
  * total cost (USD) and call count
  * outputs both a Markdown table and a LaTeX table for direct paper inclusion

Usage
-----

    python scripts/aggregate_baselines.py \
        --pooled artifacts/eval_joined_pooled_20260202_192420.csv \
        --baseline artifacts/eval_GPT4O_LONGCTX_full.csv \
        --baseline artifacts/eval_LINC_full.csv \
        --baseline artifacts/eval_LOGIC_LM_full.csv \
        --out-md artifacts/baseline_comparison.md \
        --out-tex docs/paper/tables/baseline_comparison.tex
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

GPT4O_MINI_INPUT_USD_PER_M = 0.15
GPT4O_MINI_OUTPUT_USD_PER_M = 0.60

# ---------------------------------------------------------------------------
# Wilson 95% CI
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# McNemar exact two-sided
# ---------------------------------------------------------------------------
def mcnemar_exact_two_sided(b: int, c: int) -> float:
    """Returns the exact two-sided binomial p-value for n = b+c successes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # tail probability P(X <= k | n, 0.5)
    tail = 0.0
    for i in range(k + 1):
        tail += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2 * tail)


def format_p_value(p_value: float, latex: bool = False) -> str:
    if p_value == 0.0 or p_value < 1e-10:
        return r"$<10^{-10}$" if latex else "p < 1e-10"
    if p_value < 0.01:
        return f"{p_value:.2e}"
    return f"{p_value:.3f}"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
def load_eval_csv(path: Path, mode_filter: str = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if mode_filter and r.get("mode") != mode_filter:
                continue
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Stats summary
# ---------------------------------------------------------------------------
def stats_for_baseline(baseline_rows: List[Dict[str, str]],
                       pooled_rows: List[Dict[str, str]]) -> Dict[str, object]:
    # Only score rows where gold is available
    scored = [r for r in baseline_rows if (r.get("expected_contains") or "").strip()]
    n = len(scored)
    k = sum(int(r.get("success", 0)) for r in scored)
    acc = k / n if n else 0.0
    ci_lo, ci_hi = wilson_ci(k, n)

    # By type
    by_type: Dict[str, Tuple[int, int, float]] = {}
    for qtype in ("logic", "open", "recall"):
        sub = [r for r in scored if r.get("type") == qtype]
        if not sub:
            continue
        sn = len(sub)
        sk = sum(int(r.get("success", 0)) for r in sub)
        by_type[qtype] = (sk, sn, sk / sn if sn else 0.0)

    # McNemar vs ADAPTIVERAG (only on (id, domain) pairs in both)
    adapt_success = {(r["id"], r["domain"]): int(r["success"]) for r in pooled_rows}
    base_success = {(r["id"], r["domain"]): int(r["success"]) for r in scored}
    common = set(adapt_success) & set(base_success)
    a_only = sum(1 for k in common if adapt_success[k] and not base_success[k])
    b_only = sum(1 for k in common if base_success[k] and not adapt_success[k])
    both = sum(1 for k in common if adapt_success[k] and base_success[k])
    neither = sum(1 for k in common if not adapt_success[k] and not base_success[k])
    p_value = mcnemar_exact_two_sided(a_only, b_only)

    # Cost. Prefer summing per-row tokens so consolidated shard outputs report
    # the true total rather than the last shard's running total.
    tokens_in = sum(int(r.get("cost_tokens_in") or 0) for r in baseline_rows)
    tokens_out = sum(int(r.get("cost_tokens_out") or 0) for r in baseline_rows)
    total_cost = (
        tokens_in * GPT4O_MINI_INPUT_USD_PER_M / 1_000_000
        + tokens_out * GPT4O_MINI_OUTPUT_USD_PER_M / 1_000_000
    )
    if total_cost == 0.0:
        last_cost = baseline_rows[-1].get("cost_usd_running") if baseline_rows else "0"
        try:
            total_cost = float(last_cost)
        except Exception:
            total_cost = 0.0
    return {
        "n_scored": n,
        "k_correct": k,
        "accuracy": acc,
        "wilson_lo": ci_lo,
        "wilson_hi": ci_hi,
        "by_type": by_type,
        "mcnemar_common_pairs": len(common),
        "mcnemar_adaptive_only": a_only,
        "mcnemar_baseline_only": b_only,
        "mcnemar_both": both,
        "mcnemar_neither": neither,
        "mcnemar_p": p_value,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_cost_usd": total_cost,
        "n_calls": len(baseline_rows),
    }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def render_markdown(results: Dict[str, Dict[str, object]]) -> str:
    lines = ["# Baseline comparison vs ADAPTIVERAG", ""]
    lines.append("Pooled accuracy with Wilson 95% CI, McNemar paired exact two-sided p-value vs ADAPTIVERAG on (id, domain) pairs in both CSVs.")
    lines.append("")
    lines.append("| Baseline | n scored | Accuracy | 95% CI | McNemar pairs | Adaptive-only | Baseline-only | p (vs ADAPTIVERAG) | Calls | Cost USD |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---|---:|---:|")
    for name, s in results.items():
        ci = f"[{s['wilson_lo']:.4f}, {s['wilson_hi']:.4f}]"
        p_str = format_p_value(float(s["mcnemar_p"]))
        lines.append(
            f"| {name} | {s['n_scored']} | {s['accuracy']:.4f} | {ci} | "
            f"{s['mcnemar_common_pairs']} | {s['mcnemar_adaptive_only']} | {s['mcnemar_baseline_only']} | "
            f"{p_str} | {s['n_calls']} | ${s['total_cost_usd']:.2f} |"
        )
    lines.append("")
    lines.append("## By-type accuracy")
    lines.append("")
    lines.append("| Baseline | logic | open | recall |")
    lines.append("|---|---|---|---|")
    for name, s in results.items():
        cells = []
        for qtype in ("logic", "open", "recall"):
            v = s["by_type"].get(qtype)
            if v is None:
                cells.append("n/a")
            else:
                k, n, acc = v
                cells.append(f"{acc:.4f} ({k}/{n})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_latex(results: Dict[str, Dict[str, object]]) -> str:
    lines = []
    lines.append(r"\begin{tabular}{lrcccrcr}")
    lines.append(r"\toprule")
    lines.append(r"Baseline & $n$ & Accuracy & 95\% CI & McNemar pairs & ADAPTIVERAG-only & Baseline-only & $p$ \\")
    lines.append(r"\midrule")
    for name, s in results.items():
        ci = f"[{s['wilson_lo']:.4f}, {s['wilson_hi']:.4f}]"
        p_str = format_p_value(float(s["mcnemar_p"]), latex=True)
        latex_name = name.replace("_", r"\_")
        lines.append(
            f"{latex_name} & {s['n_scored']} & {s['accuracy']:.4f} & {ci} & "
            f"{s['mcnemar_common_pairs']} & {s['mcnemar_adaptive_only']} & "
            f"{s['mcnemar_baseline_only']} & {p_str} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled", type=Path, required=True,
                    help="The paper's pooled eval CSV (contains ADAPTIVERAG slice).")
    ap.add_argument("--baseline", action="append", required=True,
                    help="Path to a baseline CSV; pass multiple times to compare multiple modes.")
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-tex", type=Path, default=None)
    args = ap.parse_args()

    pooled_rows = load_eval_csv(args.pooled, mode_filter="ADAPTIVERAG")
    print(f"[aggregate] ADAPTIVERAG rows in pooled CSV: {len(pooled_rows)}")

    results: Dict[str, Dict[str, object]] = {}
    for bpath in args.baseline:
        bpath = Path(bpath)
        bname = bpath.stem.replace("eval_", "").replace("_full", "")
        rows = load_eval_csv(bpath)
        if not rows:
            print(f"[aggregate] SKIP {bpath}: empty")
            continue
        mode = rows[0].get("mode") or bname
        stats = stats_for_baseline(rows, pooled_rows)
        results[mode] = stats
        print(f"[aggregate] {mode}: n_scored={stats['n_scored']} acc={stats['accuracy']:.4f} "
              f"McNemar {format_p_value(float(stats['mcnemar_p']))} on {stats['mcnemar_common_pairs']} pairs "
              f"cost=${stats['total_cost_usd']:.2f}")

    md = render_markdown(results)
    tex = render_latex(results)
    print()
    print(md)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md)
        print(f"Markdown written: {args.out_md}")
    if args.out_tex:
        args.out_tex.parent.mkdir(parents=True, exist_ok=True)
        args.out_tex.write_text(tex)
        print(f"LaTeX written:    {args.out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
