# backend/eval/selective.py
import argparse, glob, os
import pandas as pd
from pathlib import Path

def main(artifacts: str, out: str):
    arts = Path(artifacts)
    outp = Path(out); outp.mkdir(parents=True, exist_ok=True)
    # pick the newest joined file
    joined = sorted(arts.glob("eval_joined_*.csv"))[-1]
    df = pd.read_csv(joined)
    df = df.dropna(subset=["confidence"]).copy()

    rows = []
    for mode, g in df.groupby("mode"):
        for t in [round(x, 2) for x in [i/100 for i in range(0, 101, 2)]]:
            picked = g[g["confidence"] >= t]
            cov = len(picked) / max(len(g), 1)
            acc = picked["success"].mean() if len(picked) else 0.0
            rows.append({"mode": mode, "threshold": t, "coverage": cov, "accuracy": acc, "n": len(picked)})

    out_csv = outp / f"selective_{joined.stem.split('_')[-1]}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    main(args.artifacts, args.out)
