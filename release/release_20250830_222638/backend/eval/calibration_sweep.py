#!/usr/bin/env python3
# backend/eval/calibration_sweep.py
import argparse, os, json, glob
from pathlib import Path
import pandas as pd
import numpy as np

def load_traces(art_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(art_dir, "trace_*.jsonl")))
    if not paths:
        raise SystemExit("No trace_*.jsonl found in artifacts")
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

def conf_from_feats(feats: dict) -> float:
    # heuristic "confidence": max of (mem/search top) else 1 - normalized length
    if not isinstance(feats, dict): return np.nan
    mem = float(feats.get("mem_top", 0.0))
    sea = float(feats.get("search_top", 0.0))
    return max(mem, sea)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="artifacts")
    ap.add_argument("--out", dest="out", default="artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_traces(args.inp)
    # derive fields
    df["success"] = df["success"].astype(int)
    df["conf"] = df["features"].map(conf_from_feats)
    # ECE-style binning
    bins = np.linspace(0, 1, 6)  # 5 bins
    df["bin"] = pd.cut(df["conf"].fillna(0.0), bins=bins, include_lowest=True)
    cal = df.groupby("bin").agg(
        n=("success","size"),
        avg_conf=("conf","mean"),
        acc=("success","mean")
    ).reset_index()
    cal["gap"] = (cal["avg_conf"] - cal["acc"]).abs()
    cal.to_csv(os.path.join(args.out, "calibration.csv"), index=False)

    # threshold sweep for ADAPTIVERAG-like routing
    # We replay the rule: if mem_top>=Tmem → MEM; elif search_top>=Tsearch → SEARCH; elif sym_fired→SYM else MEMSYM
    # Use the recorded features + actual successes per example to estimate expected accuracy per (Tmem,Tsearch).
    feats = df["features"].apply(lambda x: x if isinstance(x,dict) else {})
    mem_top = feats.apply(lambda d: float(d.get("mem_top",0.0)))
    sea_top = feats.apply(lambda d: float(d.get("search_top",0.0)))
    sym = feats.apply(lambda d: int(d.get("sym_fired",0)))
    succ = df["success"].values

    grid = []
    for Tm in np.linspace(0.1, 0.9, 9):
        for Ts in np.linspace(0.1, 0.9, 9):
            # predicted action (for reporting only)
            # score each example by mode pick; then aggregate observed success
            pick_mem = (mem_top >= Tm)
            pick_sea = (~pick_mem) & (sea_top >= Ts)
            pick_sym = (~pick_mem) & (~pick_sea) & (sym == 1)
            # default to MEMSYM otherwise
            # Use observed mode-agnostic success as proxy (conservative)
            # Better: map to per-mode success; but we don't have per-example per-mode labels here.
            acc = succ.mean()  # fallback constant
            # crude proxy: assume MEM helps when mem_top high; SEARCH helps when sea_top high; SYM helps when sym fired
            w = len(succ)
            est = (
                succ[pick_mem].mean() if pick_mem.any() else 0
            ) + (
                succ[pick_sea].mean() if pick_sea.any() else 0
            ) + (
                succ[pick_sym].mean() if pick_sym.any() else 0
            ) + (
                succ[(~pick_mem)&(~pick_sea)&(~pick_sym)].mean() if ((~pick_mem)&(~pick_sea)&(~pick_sym)).any() else 0
            )
            est /= 4.0  # very rough normalization
            grid.append({"T_mem": round(Tm,2), "T_search": round(Ts,2), "est_accuracy": round(float(est),4)})

    pd.DataFrame(grid).to_csv(os.path.join(args.out, "calibration_sweep.csv"), index=False)
    print("Wrote calibration.csv and calibration_sweep.csv")

if __name__ == "__main__":
    main()
