#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, pandas as pd
from pathlib import Path

def main(sel_csv: str, out_dir: str, target_cov: float):
    p = Path(sel_csv); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(p)
    rows=[]
    for mode, g in df.groupby("mode"):
        g = g.copy()
        g["gap"] = (g["coverage"] - target_cov).abs()
        best = g.sort_values(["gap","risk","threshold"]).iloc[0]
        rows.append({"mode": mode,
                     "target_coverage": target_cov,
                     "threshold": float(best["threshold"]),
                     "coverage": float(best["coverage"]),
                     "accuracy": float(best["accuracy"])})
    fp = out / f"thresholds_cov{int(target_cov*100)}.csv"
    pd.DataFrame(rows).to_csv(fp, index=False)
    print(f"Wrote {fp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)  # artifacts/selective_*.csv
    ap.add_argument("--out", default="tables")
    ap.add_argument("--cov", type=float, default=0.5)
    a = ap.parse_args()
    main(a.csv, a.out, a.cov)
