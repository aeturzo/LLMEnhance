#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, pandas as pd
from pathlib import Path
from math import comb

def mcnemar_exact_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k+1)) / (2**n)
    p = 2 * tail
    return 1.0 if p > 1.0 else float(p)

def main(joined: str, out_path: str, A: str, B: str):
    df = pd.read_csv(joined)
    da = df[df["mode"] == A][["id","type","success"]].rename(columns={"success":"sa"})
    db = df[df["mode"] == B][["id","type","success"]].rename(columns={"success":"sb"})
    d = pd.merge(da, db, on=["id","type"], how="inner")
    b = int(((d["sa"]==1) & (d["sb"]==0)).sum())
    c = int(((d["sa"]==0) & (d["sb"]==1)).sum())
    p = mcnemar_exact_two_sided(b, c)
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"mode_A":A,"mode_B":B,"b(A=1,B=0)":b,"c(A=0,B=1)":c,"p_value":p}])\
      .to_csv(out, index=False)
    print(f"Wrote {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", required=True)
    ap.add_argument("--out", default="tables/mcnemar.csv")
    ap.add_argument("--A", default="ADAPTIVERAG")
    ap.add_argument("--B", default="RAG_BASE")
    a = ap.parse_args()
    main(a.joined, a.out, a.A, a.B)
