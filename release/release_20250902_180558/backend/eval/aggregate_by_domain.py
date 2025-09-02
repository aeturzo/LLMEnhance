#!/usr/bin/env python3
"""
Aggregate eval_* CSVs by domain and mode, optionally keeping only the latest run per domain.

Assumes your eval CSVs include a 'domain' column (as per the run_eval_all.py patch).
If 'mode' is missing in rows, it will derive from the filename.

Outputs:
  tables/summary_by_domain_mode.csv
  tables/best_mode_by_domain.csv
"""
from __future__ import annotations
import argparse, glob, os, re
from pathlib import Path
from typing import List, Dict
import pandas as pd
from statistics import median

RUN_RE = re.compile(r"eval_(?P<mode>[^_]+)_(?P<runid>\d{8}_\d{6})\.csv$")

def parse_file_meta(p: str) -> dict:
    m = RUN_RE.search(os.path.basename(p))
    if not m:
        return {"mode_from_file": None, "run_id": None}
    return {"mode_from_file": m.group("mode"), "run_id": m.group("runid")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--latest-per-domain", action="store_true",
                    help="Keep rows only from the most recent run_id per domain.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.artifacts, "eval_*.csv")))
    if not paths:
        raise SystemExit(f"No eval_*.csv in {args.artifacts}")

    frames: List[pd.DataFrame] = []
    for p in paths:
        meta = parse_file_meta(p)
        df = pd.read_csv(p)
        if "mode" not in df.columns and meta["mode_from_file"]:
            df["mode"] = meta["mode_from_file"]
        df["run_id"] = meta["run_id"]
        df["src_file"] = os.path.basename(p)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    if "domain" not in df.columns:
        raise SystemExit("Missing 'domain' column in eval CSVs. Patch run_eval_all.py to include it, then re-run.")

    # Optionally keep only latest run per domain
    if args.latest_per_domain:
        latest_by_domain: Dict[str, str] = {}
        for dom, sub in df.groupby("domain"):
            # pick the lexicographically largest run_id (YYYYMMDD_HHMMSS)
            rid = sorted(sub["run_id"].dropna().unique())
            latest_by_domain[dom] = rid[-1] if rid else None
        mask = df.apply(lambda r: (latest_by_domain.get(r["domain"]) is None) or (r["run_id"] == latest_by_domain[r["domain"]]), axis=1)
        df = df[mask].copy()

    # Aggregate per domain x mode
    def agg(group: pd.DataFrame) -> pd.Series:
        lats = group["latency_ms"].astype(float).tolist() if "latency_ms" in group.columns else []
        steps = group["steps"].astype(float).tolist() if "steps" in group.columns else []
        acc = group["success"].astype(float).mean() if "success" in group.columns else float("nan")
        return pd.Series({
            "n": len(group),
            "accuracy": round(acc, 4),
            "avg_latency_ms": round(sum(lats)/len(lats), 2) if lats else None,
            "median_latency_ms": round(median(lats), 2) if lats else None,
            "avg_steps": round(sum(steps)/len(steps), 2) if steps else None,
            "run_ids": ",".join(sorted(group["run_id"].dropna().unique().tolist())),
        })

    summary = df.groupby(["domain","mode"], as_index=False).apply(agg)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary_by_domain_mode.csv"
    summary.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    # Best mode per domain by accuracy
    best = summary.sort_values(["domain","accuracy"], ascending=[True, False]).groupby("domain", as_index=False).first()
    best_path = out_dir / "best_mode_by_domain.csv"
    best.to_csv(best_path, index=False)
    print(f"Wrote {best_path}")

if __name__ == "__main__":
    main()
