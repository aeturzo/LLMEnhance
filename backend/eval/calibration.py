#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Expected Calibration Error (ECE) per mode from eval_joined_*.csv files.

Outputs:
  - tables/calibration.csv                    (mode,ece,n,bins,k)
  - artifacts/fig_calibration_<mode>.png      (per-mode reliability diagrams)
  - artifacts/fig_calibration_overall.png     (all modes overlay)

Usage:
  python backend/eval/calibration.py --artifacts artifacts --tables tables --bins 10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_all_joined(artifacts_dir: Path) -> pd.DataFrame:
    files = sorted(artifacts_dir.glob("eval_joined_*.csv"))
    if not files:
        raise SystemExit("No artifacts/eval_joined_*.csv found. Run eval first.")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass
    if not dfs:
        raise SystemExit("Could not read any eval_joined_*.csv files.")
    df = pd.concat(dfs, ignore_index=True)
    return df


def _choose_conf_column(df: pd.DataFrame) -> Optional[str]:
    """Pick the numeric confidence-like column to use."""
    candidates = ["confidence", "prob", "p_correct", "router_conf", "score", "conf"]
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return c
    return None


def _ece_quantile(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """
    Quantile-binned ECE (avoids empty bins).
    scores ∈ [0,1], labels ∈ {0,1}.
    """
    n = len(scores)
    if n == 0:
        return float("nan")
    # sort by confidence
    order = np.argsort(scores)
    s = scores[order]
    y = labels[order]
    # assign quantile bins
    k_eff = int(min(k, n))
    bins = np.floor(np.linspace(0, n, num=k_eff + 1)).astype(int)
    ece = 0.0
    for i in range(k_eff):
        lo, hi = bins[i], bins[i + 1]
        if hi <= lo:
            continue
        sb = s[lo:hi]
        yb = y[lo:hi]
        acc = float(np.mean(yb)) if len(yb) else 0.0
        conf = float(np.mean(sb)) if len(sb) else 0.0
        ece += abs(acc - conf) * (len(sb) / n)
    return float(ece)


def _reliability_points(scores: np.ndarray, labels: np.ndarray, k: int = 10):
    """Return (bin_confidence, bin_accuracy) points for plotting."""
    n = len(scores)
    if n == 0:
        return [], []
    order = np.argsort(scores)
    s = scores[order]
    y = labels[order]
    k_eff = int(min(k, n))
    bins = np.floor(np.linspace(0, n, num=k_eff + 1)).astype(int)
    xs, ys = [], []
    for i in range(k_eff):
        lo, hi = bins[i], bins[i + 1]
        if hi <= lo:
            continue
        sb = s[lo:hi]
        yb = y[lo:hi]
        if len(sb):
            xs.append(float(np.mean(sb)))
            ys.append(float(np.mean(yb)))
    return xs, ys


def _draw_reliability(scores: np.ndarray, labels: np.ndarray, title: str, out_png: Path, k: int = 10):
    xs, ys = _reliability_points(scores, labels, k)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", label="ideal")
    if xs:
        plt.plot(xs, ys, marker="o", label="empirical")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png.as_posix(), bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts", help="folder containing eval_joined_*.csv")
    ap.add_argument("--tables", default="tables", help="folder to write calibration.csv")
    ap.add_argument("--bins", type=int, default=10, help="number of bins (quantile ECE)")
    args = ap.parse_args()

    artifacts = Path(args.artifacts)
    tables = Path(args.tables)
    tables.mkdir(parents=True, exist_ok=True)

    df = _load_all_joined(artifacts)

    # Ensure required columns
    if "mode" not in df.columns or "success" not in df.columns:
        raise SystemExit("Joined CSV missing required columns ('mode','success').")

    conf_col = _choose_conf_column(df)
    if conf_col is None:
        # Write placeholder consistent with pipeline expectations
        out = pd.DataFrame(
            [{"mode": m, "ece": None, "n": 0, "bins": 0, "k": args.bins, "note": "No confidence logged"}]
            for m in sorted(df["mode"].astype(str).unique())
        )
        out.to_csv(tables / "calibration.csv", index=False)
        print(f"Wrote {tables / 'calibration.csv'} (no confidence column found)")
        return

    # Clean & clip
    d = df.copy()
    d["success"] = pd.to_numeric(d["success"], errors="coerce").fillna(0.0).astype(float)
    d["confidence"] = pd.to_numeric(d[conf_col], errors="coerce")
    d = d.dropna(subset=["confidence"])
    d["confidence"] = d["confidence"].clip(0.0, 1.0)

    if d.empty:
        out = pd.DataFrame(
            [{"mode": m, "ece": None, "n": 0, "bins": 0, "k": args.bins, "note": "No confidence logged"}]
            for m in sorted(df["mode"].astype(str).unique())
        )
        out.to_csv(tables / "calibration.csv", index=False)
        print(f"Wrote {tables / 'calibration.csv'} (no numeric confidence values)")
        return

    rows: List[Dict[str, Any]] = []
    all_scores, all_labels = [], []

    for mode, grp in d.groupby("mode", dropna=False):
        scores = grp["confidence"].to_numpy(dtype=float)
        labels = grp["success"].to_numpy(dtype=float)
        n = len(scores)
        if n == 0:
            rows.append({"mode": str(mode), "ece": None, "n": 0, "bins": 0, "k": args.bins})
            continue

        val_ece = _ece_quantile(scores, labels, k=args.bins)
        rows.append({
            "mode": str(mode),
            "ece": round(float(val_ece), 6),
            "n": int(n),
            "bins": int(min(args.bins, n)),
            "k": int(args.bins),
        })

        # Per-mode reliability plot
        _draw_reliability(scores, labels, f"{mode} reliability", artifacts / f"fig_calibration_{mode}.png", k=args.bins)

        # Accumulate for overall
        all_scores.append(scores)
        all_labels.append(labels)

    # Overall overlay (optional)
    if all_scores:
        try:
            s_all = np.concatenate(all_scores)
            y_all = np.concatenate(all_labels)
            _draw_reliability(s_all, y_all, "All modes (overlay)", artifacts / "fig_calibration_overall.png", k=args.bins)
        except Exception:
            pass

    out_df = pd.DataFrame(rows).sort_values("mode")
    out_df.to_csv(tables / "calibration.csv", index=False)
    print(f"Wrote {tables / 'calibration.csv'} rows={len(out_df)}")

    # Optional: dump a small JSON for debugging provenance
    meta = {
        "artifacts_used": sorted([p.name for p in artifacts.glob("eval_joined_*.csv")]),
        "confidence_column": conf_col,
        "bins": int(args.bins),
    }
    with (artifacts / "calibration_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {(artifacts / 'calibration_meta.json').as_posix()}")


if __name__ == "__main__":
    main()
