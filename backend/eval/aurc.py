#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, pandas as pd, numpy as np
from pathlib import Path

def aurc(df_mode: pd.DataFrame) -> float:
    df = df_mode.sort_values("coverage")
    cov = df["coverage"].to_numpy()
    risk = (1.0 - df["accuracy"]).to_numpy()
    if len(cov) < 2:
        return float("nan")
    return float(np.trapz(risk, cov))

def main(csv_path: str, out_dir: str):
    p = Path(csv_path); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(p)
    rows=[]
    for mode, g in df.groupby("mode"):
        rows.append({"mode": mode, "AURC": round(aurc(g), 6)})
    (out/"aurc.csv").write_text(pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")
    print(f"Wrote {out/'aurc.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)  # artifacts/selective_*.csv
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    main(a.csv, a.out)
