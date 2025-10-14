#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, pandas as pd
from pathlib import Path
from math import sqrt

def wilson(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    rad = z * sqrt((p*(1-p)/n) + (z*z/(4*n*n))) / denom
    return (max(0.0, center - rad), min(1.0, center + rad))

def main(joined_csv: str, out_dir: str):
    df = pd.read_csv(joined_csv)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for (mode, typ), g in df.groupby(["mode", "type"]):
        n = len(g); acc = g["success"].mean() if n else 0.0
        lo, hi = wilson(acc, n)
        rows.append({"mode": mode, "type": typ, "n": n,
                     "accuracy": round(acc, 4),
                     "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)})
    (out/"acc_ci.csv").write_text(pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")
    print(f"Wrote {out/'acc_ci.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", required=True)  # artifacts/eval_joined_*.csv
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    main(a.joined, a.out)
