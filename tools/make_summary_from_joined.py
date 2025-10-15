# tools/make_summary_from_joined.py
# -*- coding: utf-8 -*-
"""
Recompute the paper's eval summary from all joined CSVs.
- Pools across domains
- Robust to schema variants (success/correct/is_correct; mode/run_mode; latency/steps)
- Writes: docs/paper/tables/eval_summary.csv (+ optional .tex)

Usage:
  python tools/make_summary_from_joined.py
  # or customize:
  python tools/make_summary_from_joined.py --glob "artifacts/eval_joined_*.csv" --outdir docs/paper/tables --write-tex
"""

import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import sys


def _to01(v):
    if isinstance(v, str):
        v = v.strip().lower()
        if v in {"1", "true", "t", "yes", "y"}:
            return 1.0
        if v in {"0", "false", "f", "no", "n", ""}:
            return 0.0
    if v is True:
        return 1.0
    if v is False or v is None:
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _write_tex(df: pd.DataFrame, out_path: Path):
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    with out_path.open("w", encoding="utf-8") as w:
        w.write(f"\\begin{{tabular}}{{{align}}}\n\\toprule\n")
        w.write(" & ".join(cols) + " \\\\\n\\midrule\n")
        for _, row in df.iterrows():
            w.write(" & ".join(str(row[c]) for c in cols) + " \\\\\n")
        w.write("\\bottomrule\n\\end{tabular}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        default="artifacts/eval_joined_*_calibrated.csv",
        help="Glob for joined CSVs (fallback to artifacts/eval_joined_*.csv if none match).",
    )
    ap.add_argument(
        "--outdir",
        default="docs/paper/tables",
        help="Directory to write eval_summary.(csv|tex).",
    )
    ap.add_argument("--write-tex", action="store_true", help="Also write a simple TeX table.")
    ap.add_argument(
        "--round",
        type=int,
        default=4,
        help="Round float columns to this many decimals in the CSV.",
    )
    args = ap.parse_args()

    # Load all joined files
    files = sorted(glob.glob(args.glob))
    if not files:
        files = sorted(glob.glob("artifacts/eval_joined_*.csv"))
    if not files:
        print("[make_summary] No artifacts/eval_joined*.csv found. Run evals first.", file=sys.stderr)
        sys.exit(1)

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"[make_summary] WARN: failed to read {f}: {e}", file=sys.stderr)
    if not dfs:
        print("[make_summary] No readable joined CSVs.", file=sys.stderr)
        sys.exit(1)

    joined = pd.concat(dfs, ignore_index=True)

    # Flexible column mapping
    mode_col = None
    for c in ("mode", "run_mode"):
        if c in joined.columns:
            mode_col = c
            break

    acc_col = None
    for c in ("success", "correct", "is_correct"):
        if c in joined.columns:
            acc_col = c
            break

    lat_col = None
    for c in ("latency_ms", "latency"):
        if c in joined.columns:
            lat_col = c
            break

    steps_col = None
    for c in ("steps", "n_steps"):
        if c in joined.columns:
            steps_col = c
            break

    if mode_col is None or acc_col is None:
        print(f"[make_summary] Needed cols missing; have: {joined.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    # Normalize accuracy to 0/1
    joined["_acc"] = joined[acc_col].apply(_to01)

    # Group and aggregate
    grp = joined.groupby(mode_col, dropna=False)
    eval_summary = pd.DataFrame({
        "accuracy": grp["_acc"].mean(),
        "avg_latency_ms": grp[lat_col].mean() if lat_col else np.nan,
        "avg_steps": grp[steps_col].mean() if steps_col else np.nan,
        "median_latency_ms": grp[lat_col].median() if lat_col else np.nan,
        "mode": grp.size().index,
        "n": grp.size().values,
    }).reset_index(drop=True)

    # Order columns for the paper
    eval_summary = eval_summary[["accuracy", "avg_latency_ms", "avg_steps", "median_latency_ms", "mode", "n"]]

    # Optional rounding for nicer CSV
    float_cols = ["accuracy", "avg_latency_ms", "avg_steps", "median_latency_ms"]
    for c in float_cols:
        if c in eval_summary.columns:
            eval_summary[c] = eval_summary[c].astype(float).round(args.round)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "eval_summary.csv"
    eval_summary.to_csv(csv_path, index=False)
    print(f"[make_summary] wrote {csv_path} with per-mode n and accuracy:")
    try:
        print(eval_summary.sort_values("n", ascending=False).to_string(index=False))
    except Exception:
        pass

    if args.write_tex:
        tex_path = outdir / "eval_summary.tex"
        _write_tex(eval_summary, tex_path)
        print(f"[make_summary] wrote {tex_path}")


if __name__ == "__main__":
    main()
