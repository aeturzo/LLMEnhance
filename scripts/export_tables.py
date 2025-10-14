#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from pathlib import Path

SPEC = [
    ("artifacts", "eval_summary_*.csv", "eval_summary"),
    ("tables",    "acc_ci.csv",         "acc_ci"),
    ("tables",    "aurc.csv",           "aurc"),
    ("tables",    "sym_coverage.csv",   "sym_coverage"),
    ("tables",    "mcnemar.csv",        "mcnemar"),
    ("tables",    "thresholds_*.csv",   "thresholds"),  # optional, if present
]

def _latest(dir_path: str, pattern: str):
    p = Path(dir_path)
    files = sorted(p.glob(pattern), key=lambda x: x.stat().st_mtime)
    return files[-1] if files else None

def main(out_dir: str = "docs/paper/tables"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for folder, pattern, stem in SPEC:
        fp = _latest(folder, pattern)
        if not fp:
            print(f"[export] missing {folder}/{pattern} — skipping")
            continue

        df = pd.read_csv(fp)
        csv_fp = out / f"{stem}.csv"
        tex_fp = out / f"{stem}.tex"

        csv_fp.write_text(df.to_csv(index=False), encoding="utf-8")
        tex_fp.write_text(df.to_latex(index=False), encoding="utf-8")
        print(f"[export] wrote {csv_fp} and {tex_fp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/paper/tables", help="Output directory for exported tables")
    args = ap.parse_args()
    main(args.out)
