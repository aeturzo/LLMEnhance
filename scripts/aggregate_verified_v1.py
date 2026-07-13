#!/usr/bin/env python3
"""
Aggregate the verified-subset comparison: ADAPTIVERAG vs the three external
LLM baselines, all evaluated on the SAME objectively-verified release-clean
rows (artifacts/release_clean_verified_v1.json).

Unlike scripts/aggregate_baselines.py (which compared baselines against the
archived paper-split pooled CSV by id alone), this script:

  * pairs every mode on identical (id, domain) verified keys;
  * runs McNemar ADAPTIVERAG-vs-each-baseline on those identical rows;
  * reports accuracy + Wilson 95% CI, by-type and by-domain;
  * folds in the verified-subset exclusion counts for the paper's methods note.

Inputs
------
  artifacts/verified_v1/eval_ADAPTIVERAG_verified_v1_full.csv   (the one paid rerun)
  artifacts/verified_v1/eval_GPT4O_LONGCTX_verified_v1.csv      (filtered, no rerun)
  artifacts/verified_v1/eval_LINC_verified_v1.csv               (filtered, no rerun)
  artifacts/verified_v1/eval_LOGIC_LM_verified_v1.csv           (filtered, no rerun)
  artifacts/release_clean_verified_v1_report.md                 (exclusion counts)

Outputs
-------
  artifacts/verified_v1/baseline_comparison_verified_v1.md
  docs/paper/tables/baseline_comparison_verified_v1.tex

Usage
-----
  python scripts/aggregate_verified_v1.py
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_DIR = REPO_ROOT / "artifacts" / "verified_v1"
REPORT_MD = REPO_ROOT / "artifacts" / "release_clean_verified_v1_report.md"

ADAPTIVERAG_CSV = VERIFIED_DIR / "eval_ADAPTIVERAG_verified_v1_full.csv"
BASELINE_CSVS = {
    "GPT4O_LONGCTX": VERIFIED_DIR / "eval_GPT4O_LONGCTX_verified_v1.csv",
    "LINC": VERIFIED_DIR / "eval_LINC_verified_v1.csv",
    "LOGIC_LM": VERIFIED_DIR / "eval_LOGIC_LM_verified_v1.csv",
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, 2 * tail)


def format_p(p: float) -> str:
    if p == 0.0 or p < 1e-10:
        return "p < 1e-10"
    if p < 0.001:
        return f"p = {p:.2e}"
    return f"p = {p:.3f}"


def load_success(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            out[(r["id"], r["domain"])] = r
    return out


def acc_block(rows: List[Dict[str, str]]) -> Dict[str, object]:
    scored = [r for r in rows if (r.get("expected_contains") or "").strip()]
    n = len(scored)
    k = sum(int(r.get("success", 0)) for r in scored)
    lo, hi = wilson_ci(k, n)
    by_type: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    by_domain: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in scored:
        kt, nt = by_type[r["type"]]
        by_type[r["type"]] = (kt + int(r["success"]), nt + 1)
        kd, nd = by_domain[r["domain"]]
        by_domain[r["domain"]] = (kd + int(r["success"]), nd + 1)
    return {
        "n": n, "k": k, "acc": (k / n if n else 0.0),
        "ci": (lo, hi), "by_type": dict(by_type), "by_domain": dict(by_domain),
    }


def read_exclusions(path: Path) -> str:
    if not path.exists():
        return "(verified-subset report not found)"
    text = path.read_text()
    m = re.search(r"## Exclusions by reason.*?(?=\n## )", text, flags=re.DOTALL)
    return m.group(0).strip() if m else "(exclusion section not found)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptiverag", type=Path, default=ADAPTIVERAG_CSV)
    ap.add_argument("--out-md", type=Path,
                    default=VERIFIED_DIR / "baseline_comparison_verified_v1.md")
    ap.add_argument("--out-tex", type=Path,
                    default=REPO_ROOT / "docs" / "paper" / "tables" / "baseline_comparison_verified_v1.tex")
    args = ap.parse_args()

    adapt = load_success(args.adaptiverag)
    baselines = {name: load_success(path) for name, path in BASELINE_CSVS.items()}

    if not adapt:
        print(f"[aggregate] ADAPTIVERAG CSV not found / empty: {args.adaptiverag}")
        print("[aggregate] Run the verified ADAPTIVERAG full pass first (Step 6).")
        return 1

    # Common verified keys across ALL modes
    common = set(adapt)
    for b in baselines.values():
        if b:
            common &= set(b)
    print(f"[aggregate] common verified (id,domain) keys across all modes: {len(common)}")

    # Per-mode accuracy on the common keys
    modes: Dict[str, Dict[str, object]] = {}
    for name, table in [("ADAPTIVERAG", adapt), *baselines.items()]:
        rows = [table[k] for k in common if k in table]
        modes[name] = acc_block(rows)

    # McNemar ADAPTIVERAG vs each baseline on identical rows
    mcnemar: Dict[str, Dict[str, object]] = {}
    for name, table in baselines.items():
        if not table:
            continue
        a_only = b_only = both = neither = 0
        for k in common:
            av = int(adapt[k]["success"])
            bv = int(table[k]["success"])
            if av and not bv:
                a_only += 1
            elif bv and not av:
                b_only += 1
            elif av and bv:
                both += 1
            else:
                neither += 1
        mcnemar[name] = {
            "adaptiverag_only": a_only, "baseline_only": b_only,
            "both": both, "neither": neither,
            "p": mcnemar_exact_two_sided(a_only, b_only),
        }

    # ---- Markdown -------------------------------------------------------
    L: List[str] = []
    L.append("# Verified-subset comparison — ADAPTIVERAG vs external baselines")
    L.append("")
    L.append(f"All modes evaluated on the **same {len(common)} objectively-verified "
             f"`(id, domain)` rows** from `release_clean_verified_v1`. External "
             f"baseline CSVs were filtered (no rerun); only ADAPTIVERAG was rerun.")
    L.append("")
    L.append("## Accuracy")
    L.append("")
    L.append("| Mode | n | Accuracy | Wilson 95% CI | logic | open | recall |")
    L.append("|---|---:|---:|---|---:|---:|---:|")
    for name in ["ADAPTIVERAG", "GPT4O_LONGCTX", "LINC", "LOGIC_LM"]:
        s = modes.get(name)
        if not s:
            continue
        lo, hi = s["ci"]
        def tcell(t: str) -> str:
            v = s["by_type"].get(t)
            return f"{v[0]/v[1]:.4f}" if v and v[1] else "n/a"
        L.append(f"| {name} | {s['n']} | {s['acc']:.4f} | [{lo:.4f}, {hi:.4f}] | "
                 f"{tcell('logic')} | {tcell('open')} | {tcell('recall')} |")
    L.append("")
    L.append("## By domain")
    L.append("")
    L.append("| Mode | battery | lexmark | viessmann |")
    L.append("|---|---:|---:|---:|")
    for name in ["ADAPTIVERAG", "GPT4O_LONGCTX", "LINC", "LOGIC_LM"]:
        s = modes.get(name)
        if not s:
            continue
        def dcell(d: str) -> str:
            v = s["by_domain"].get(d)
            return f"{v[0]/v[1]:.4f}" if v and v[1] else "n/a"
        L.append(f"| {name} | {dcell('battery')} | {dcell('lexmark')} | {dcell('viessmann')} |")
    L.append("")
    L.append("## McNemar — ADAPTIVERAG vs each baseline (identical verified rows)")
    L.append("")
    L.append("| Baseline | ADAPTIVERAG-only | Baseline-only | Both | Neither | p |")
    L.append("|---|---:|---:|---:|---:|---|")
    for name, mc in mcnemar.items():
        L.append(f"| {name} | {mc['adaptiverag_only']} | {mc['baseline_only']} | "
                 f"{mc['both']} | {mc['neither']} | {format_p(float(mc['p']))} |")
    L.append("")
    L.append("## Verified-subset exclusions (objective, model-independent)")
    L.append("")
    L.append(read_exclusions(REPORT_MD))
    L.append("")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(L))

    # ---- LaTeX ----------------------------------------------------------
    T: List[str] = []
    T.append(r"\begin{tabular}{lrcccc}")
    T.append(r"\toprule")
    T.append(r"Mode & $n$ & Accuracy & 95\% CI & McNemar vs ADAPTIVERAG \\")
    T.append(r"\midrule")
    for name in ["ADAPTIVERAG", "GPT4O_LONGCTX", "LINC", "LOGIC_LM"]:
        s = modes.get(name)
        if not s:
            continue
        lo, hi = s["ci"]
        if name == "ADAPTIVERAG":
            mc_str = "--"
        else:
            mc = mcnemar.get(name, {})
            mc_str = format_p(float(mc.get("p", 1.0))).replace("p = ", "").replace("p < ", "$<$")
        L_name = name.replace("_", r"\_")
        T.append(f"{L_name} & {s['n']} & {s['acc']:.4f} & "
                 f"[{lo:.4f}, {hi:.4f}] & {mc_str} \\\\")
    T.append(r"\bottomrule")
    T.append(r"\end{tabular}")
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(T))

    # ---- Console --------------------------------------------------------
    print("\n".join(L))
    print(f"\n[aggregate] wrote {args.out_md}")
    print(f"[aggregate] wrote {args.out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
