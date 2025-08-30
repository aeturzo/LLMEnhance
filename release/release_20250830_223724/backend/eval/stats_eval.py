#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backend/eval/stats_eval.py

Paired bootstrap over queries to compare modes (e.g., RL vs MEMSYM, RL vs ROUTER).
Also computes effect sizes (Cliff's delta and Cohen's d over per-query diffs).

Inputs:
  - One or more eval_joined_*.csv (from run_eval_all.py)

Outputs:
  - Prints a clean table to console
  - Writes artifacts/stats_{stamp}.txt

Usage:
  python -m backend.eval.stats_eval artifacts/eval_joined_*.csv
  # or just:
  python -m backend.eval.stats_eval
  (it will pick the newest eval_joined_*.csv under ./artifacts)
"""
from __future__ import annotations
import argparse, glob, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

ART = Path("artifacts")

PAIRS_DEFAULT = [
    ("RL", "MEMSYM"),
    ("RL", "ROUTER"),
    ("RL", "ADAPTIVERAG"),
]

def _latest_joined() -> List[Path]:
    files = sorted(ART.glob("eval_joined_*.csv"))
    return files[-1:] if files else []

def _load(paths: List[Path]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__file"] = p.name
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        raise SystemExit("No eval_joined CSVs found.")
    return pd.concat(dfs, ignore_index=True)

def _pivot_by_query(df: pd.DataFrame, modes: List[str]) -> pd.DataFrame:
    """
    Return a DataFrame indexed by 'id' with columns success_{mode}.
    Only keep queries that exist for all requested modes.
    """
    df2 = df[["id","mode","success"]].copy()
    df2["mode"] = df2["mode"].astype(str)
    piv = df2.pivot_table(index="id", columns="mode", values="success", aggfunc="max")
    keep = [m for m in modes if m in piv.columns]
    piv = piv[keep].dropna()  # only queries with all modes present
    piv.columns = [f"success_{c}" for c in piv.columns]
    return piv

def _paired_bootstrap(x: np.ndarray, y: np.ndarray, B: int = 10000, rng: np.random.Generator | None = None) -> Tuple[float, Tuple[float,float], float]:
    """
    x,y: arrays of per-query success (0/1). We test mean(x) - mean(y).
    Returns: (diff, (ci_lo, ci_hi), p_two_sided)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(x)
    base = float(x.mean() - y.mean())
    # bootstrap over queries
    diffs = np.empty(B, dtype="float64")
    idx = np.arange(n)
    for b in range(B):
        sample = rng.choice(idx, size=n, replace=True)
        diffs[b] = x[sample].mean() - y[sample].mean()
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided p-value: probability mass on the opposite side of zero
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return base, (float(ci_lo), float(ci_hi)), float(p)

def _cliffs_delta(d: np.ndarray) -> float:
    """
    Cliff's delta on per-query differences (x - y) for binary success.
    delta in [-1,1]; 0 means no effect.
    """
    # For binary success, d is in {-1,0,1}. Compute P(x>y) - P(x<y).
    gt = (d > 0).mean()
    lt = (d < 0).mean()
    return float(gt - lt)

def _cohens_d(d: np.ndarray) -> float:
    """Cohen's d on per-query differences d = x - y."""
    mu = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    return float(mu / (sd + 1e-12))

def compare_modes(df_joined: pd.DataFrame, pairs: List[Tuple[str,str]]) -> str:
    out_lines = []
    out_lines.append("Statistical comparison (paired over queries)")
    out_lines.append("pair\tN\tacc_A\tacc_B\tdiff\t95%CI\tp(two-sided)\tCliffΔ\tCohen_d")
    for a,b in pairs:
        piv = _pivot_by_query(df_joined, [a,b])
        if piv.empty:
            out_lines.append(f"{a} vs {b}\t0\t-\t-\t-\t-\t-\t-\t-")
            continue
        x = piv[f"success_{a}"].to_numpy(dtype="float64")
        y = piv[f"success_{b}"].to_numpy(dtype="float64")
        d = x - y
        n = len(x)
        diff, ci, p = _paired_bootstrap(x, y, B=10000)
        cliffs = _cliffs_delta(d)
        cohen = _cohens_d(d)
        out_lines.append(
            f"{a} vs {b}\t{n}\t{x.mean():.3f}\t{y.mean():.3f}\t{diff:.3f}\t[{ci[0]:.3f},{ci[1]:.3f}]\t{p:.4f}\t{cliffs:.3f}\t{cohen:.3f}"
        )
    return "\n".join(out_lines)

def main(paths: List[str]) -> Path:
    if not paths:
        csvs = _latest_joined()
    else:
        csvs = [Path(p) for p in paths]
    df = _load(csvs)
    txt = compare_modes(df, PAIRS_DEFAULT)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ART.mkdir(exist_ok=True, parents=True)
    out = ART / f"stats_{stamp}.txt"
    out.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\nWrote {out}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="eval_joined_*.csv paths (optional)")
    args = ap.parse_args()
    main(args.paths)
