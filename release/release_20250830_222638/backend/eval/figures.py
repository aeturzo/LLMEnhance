#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backend/eval/figures.py

Builds camera-ready assets:
- fig_pareto.png from pareto_*.csv
- fig_selective.png from selective_*.csv
- tables/ablation_by_type.csv (accuracy per mode × type)
- tables/summary_latest.csv (latest eval_summary)
- (best-effort) fig_ablation.png bar chart

Usage:
  python -m backend.eval.figures
"""
from __future__ import annotations
import glob, os, time
from pathlib import Path
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # plotting optional

ART = Path("artifacts")
TAB = Path("tables")

def _latest(glob_pat: str) -> Path | None:
    files = sorted(ART.glob(glob_pat))
    return files[-1] if files else None

def _read_csv(p: Path | None) -> pd.DataFrame:
    if not p or not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

def fig_pareto():
    p = _latest("pareto_*.csv")
    df = _read_csv(p)
    if df.empty or plt is None:
        return None
    plt.figure()
    plt.plot(df["alpha"], df["accuracy"], marker="o")
    plt.xlabel("alpha (cost penalty)")
    plt.ylabel("Accuracy (RL)")
    plt.title("Pareto: Accuracy vs Cost Penalty (RL)")
    out = ART / "fig_pareto.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    return out

def fig_selective():
    p = _latest("selective_*.csv")
    df = _read_csv(p)
    if df.empty or plt is None:
        return None
    plt.figure()
    plt.plot(df["coverage"], df["risk"], marker="o")
    plt.xlabel("Coverage (fraction answered)")
    plt.ylabel("Risk (error rate among answered)")
    plt.title("Selective Risk Curve")
    out = ART / "fig_selective.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    return out

def table_ablation_by_type():
    p = _latest("eval_joined_*.csv")
    df = _read_csv(p)
    if df.empty:
        return None
    # Accuracy per type × mode
    t = (df.groupby(["type","mode"])["success"]
           .mean()
           .reset_index()
           .pivot(index="type", columns="mode", values="success")
           .fillna(0.0))
    TAB.mkdir(exist_ok=True, parents=True)
    out = TAB / "ablation_by_type.csv"
    t.to_csv(out)
    return out

def table_summary_latest():
    p = _latest("eval_summary_*.csv")
    df = _read_csv(p)
    if df.empty:
        return None
    TAB.mkdir(exist_ok=True, parents=True)
    out = TAB / "summary_latest.csv"
    df.to_csv(out, index=False)
    return out

def fig_ablation():
    p = TAB / "ablation_by_type.csv"
    if not p.exists() or plt is None:
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    df = df.set_index("type")
    plt.figure(figsize=(8,4))
    # plot only a few key modes if there are many
    keep_modes = [c for c in df.columns if c in ("BASE","MEM","SYM","MEMSYM","ROUTER","ADAPTIVERAG","RL")]
    (df[keep_modes] if keep_modes else df).plot(kind="bar")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.title("Ablation by Query Type")
    plt.legend(loc="lower right", ncol=4, fontsize=8)
    out = ART / "fig_ablation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    return out

def main():
    made = []
    if (x := fig_pareto()): made.append(x)
    if (x := fig_selective()): made.append(x)
    if (x := table_summary_latest()): made.append(x)
    if (x := table_ablation_by_type()): made.append(x)
    if (x := fig_ablation()): made.append(x)
    print("Generated:", ", ".join(str(m) for m in made) if made else "nothing (missing inputs or matplotlib)")

if __name__ == "__main__":
    main()
