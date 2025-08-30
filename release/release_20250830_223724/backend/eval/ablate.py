#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation summaries (robust to NaNs / missing columns)

Outputs (to --out):
  - ablation_acc_by_mode.csv
  - ablation_by_type.csv               (if 'type' present)
  - ablation_costs.csv                 (avg steps & latency per mode)
  - ablation_by_domain_mode.csv        (if 'domain' present)
"""
from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np
import pandas as pd

def load_joined(art_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(art_dir, "eval_*.csv")))
    if not paths:
        raise SystemExit(f"No eval_*.csv found under {art_dir}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source"] = Path(p).name
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Ensure required columns exist
    for col, default in [
        ("mode", ""), ("type", "open"), ("id", ""), ("query", ""), ("product", ""), ("session", ""),
        ("success", 0), ("steps", 0), ("latency_ms", 0.0)
    ]:
        if col not in df.columns:
            df[col] = default

    # Clean types
    df["mode"] = df["mode"].astype(str)
    df["type"] = df["type"].fillna("open").astype(str)

    # success -> int {0,1}, safely
    df["success"] = (
        pd.to_numeric(df["success"], errors="coerce")
          .replace([np.inf, -np.inf], np.nan)
          .fillna(0)
          .astype(int)
    )

    # steps -> numeric
    df["steps"] = (
        pd.to_numeric(df["steps"], errors="coerce")
          .replace([np.inf, -np.inf], np.nan)
          .fillna(0)
          .astype(int)
    )

    # latency -> numeric
    df["latency_ms"] = (
        pd.to_numeric(df["latency_ms"], errors="coerce")
          .replace([np.inf, -np.inf], np.nan)
          .fillna(0.0)
          .astype(float)
    )

    # De-dup across multiple runs: keep the last version (by source sort order)
    keys = ["id", "mode", "query", "product", "session"]
    df = df.sort_values(["__source", "latency_ms"]).drop_duplicates(subset=keys, keep="last")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", default="artifacts")
    ap.add_argument("--out", dest="out", default="artifacts")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    df = load_joined(args.inp)

    # Accuracy by mode
    acc_mode = df.groupby("mode", as_index=False)["success"].mean().rename(columns={"success":"accuracy"})
    acc_mode = acc_mode.sort_values("accuracy", ascending=False)
    acc_mode.to_csv(os.path.join(args.out, "ablation_acc_by_mode.csv"), index=False)

    # Accuracy by mode × type
    if "type" in df.columns:
        ab_by_type = df.pivot_table(index="mode", columns="type", values="success", aggfunc="mean").reset_index()
        ab_by_type.to_csv(os.path.join(args.out, "ablation_by_type.csv"), index=False)

    # Cost proxies
    agg = df.groupby("mode", as_index=False).agg(
        n=("success","size"),
        acc=("success","mean"),
        avg_steps=("steps","mean"),
        avg_latency_ms=("latency_ms","mean"),
    ).sort_values("acc", ascending=False)
    agg.to_csv(os.path.join(args.out, "ablation_costs.csv"), index=False)

    # Optional: by-domain if present
    if "domain" in df.columns:
        by_dom = df.groupby(["domain","mode"], as_index=False).agg(
            n=("success","size"),
            acc=("success","mean"),
            avg_steps=("steps","mean"),
            avg_latency_ms=("latency_ms","mean"),
        ).sort_values(["domain","acc"], ascending=[True, False])
        by_dom.to_csv(os.path.join(args.out, "ablation_by_domain_mode.csv"), index=False)

    print("Wrote:",
          os.path.join(args.out, "ablation_acc_by_mode.csv"),
          os.path.join(args.out, "ablation_by_type.csv") if "type" in df.columns else "(no type col)",
          os.path.join(args.out, "ablation_costs.csv"),
          os.path.join(args.out, "ablation_by_domain_mode.csv") if "domain" in df.columns else "(no domain col)")

if __name__ == "__main__":
    main()
