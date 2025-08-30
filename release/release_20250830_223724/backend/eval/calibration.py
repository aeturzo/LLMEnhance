#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration utility:
- Scans a directory for eval_*.csv and trace_*.jsonl
- Computes ECE (Expected Calibration Error) and draws reliability diagrams
- Writes:
    artifacts/fig_calibration_<basename>.png           (one per input file)
    artifacts/calibration_metrics.json                 (file-level metrics)
    artifacts/calibration_by_mode.json                 (mode-level metrics from traces, if available)

Usage:
  python backend/eval/calibration.py --in artifacts --out artifacts
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any

import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


# ------------------------ helpers ------------------------

CONF_COLS = [
    "confidence", "prob", "p_correct", "router_conf", "score", "conf"
]

def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x.astype(float), 0.0, 1.0)

def ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Simple ECE with equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(scores, bins) - 1
    e = 0.0
    n = len(scores)
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        acc = labels[m].mean()
        conf = scores[m].mean()
        e += (m.sum() / n) * abs(acc - conf)
    return float(e)

def reliability_points(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Tuple[List[float], List[float]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(scores, bins) - 1
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if np.any(m):
            xs.append(float(scores[m].mean()))
            ys.append(float(labels[m].mean()))
    return xs, ys

def find_conf_col(df: pd.DataFrame) -> str | None:
    for c in CONF_COLS:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def draw_reliability(scores: np.ndarray, labels: np.ndarray, title: str, out_png: Path, n_bins: int = 10) -> None:
    xs, ys = reliability_points(scores, labels, n_bins=n_bins)
    plt.figure(figsize=(5, 5))
    plt.plot([0,1],[0,1], linestyle="--", label="ideal")
    plt.plot(xs, ys, marker="o", label="empirical")
    plt.xlabel("Confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png.as_posix(), bbox_inches="tight")
    plt.close()

# ------------------------ traces support ------------------------

def _proxy_conf_from_trace(row: Dict[str, Any]) -> float:
    """
    Derive a reasonable confidence proxy from a trace row when no explicit confidence is present.
    Priority:
      1) row["confidence"] if present
      2) max(features.mem_top, features.search_top) if present
      3) transform of 'cost' (lower cost -> higher confidence)
      4) fallback 0.5
    """
    # 1) direct
    val = row.get("confidence")
    if isinstance(val, (int, float)):
        return float(np.clip(val, 0.0, 1.0))
    # 2) features
    feats = row.get("features") or {}
    mt = feats.get("mem_top")
    st = feats.get("search_top")
    cand = [v for v in [mt, st] if isinstance(v, (int, float))]
    if cand:
        return float(np.clip(max(cand), 0.0, 1.0))
    # 3) from cost (monotone decreasing)
    cost = row.get("cost")
    if isinstance(cost, (int, float)):
        # Smooth monotone map: conf = 1 / (1 + exp((cost - 2.0)))
        try:
            conf = 1.0 / (1.0 + math.e ** (float(cost) - 2.0))
        except Exception:
            conf = 1.0 / (1.0 + float(cost))
        return float(np.clip(conf, 0.0, 1.0))
    # 4) fallback
    return 0.5

def load_trace_points(path: Path) -> List[Tuple[float, float, str]]:
    """
    Returns list of (confidence, success, mode) from a trace_*.jsonl file.
    """
    out: List[Tuple[float, float, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            success = row.get("success")
            if success is None:
                # Try: derive success if expected info exists (rare for traces)
                continue
            try:
                s = float(success)
            except Exception:
                continue
            conf = _proxy_conf_from_trace(row)
            mode = str(row.get("mode", "") or "")
            out.append((conf, s, mode))
    return out

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="indir",  default="artifacts", help="input folder with eval_*.csv / trace_*.jsonl")
    ap.add_argument("--out", dest="outdir", default="artifacts", help="output folder for figs/metrics")
    ap.add_argument("--bins", type=int, default=10, help="number of bins for reliability/ECE")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_files: Dict[str, Dict[str, Any]] = {}
    by_mode_accum: Dict[str, Dict[str, List[float]]] = {}  # mode -> {"scores":[...], "labels":[...]}

    # -------- eval_*.csv (only if a confidence-like column exists) --------
    for p_str in sorted(glob.glob(os.path.join(indir.as_posix(), "eval_*.csv"))):
        p = Path(p_str)
        try:
            df = pd.read_csv(p)
        except Exception as e:
            metrics_files[p.name] = {"error": f"read-failed: {e}"}
            continue

        conf_col = find_conf_col(df)
        if conf_col is None:
            metrics_files[p.name] = {"skipped": "no-confidence-column", "n": int(len(df))}
            continue

        if "success" not in df.columns:
            metrics_files[p.name] = {"skipped": "no-success-column", "n": int(len(df))}
            continue

        scores = clip01(df[conf_col].to_numpy())
        labels = df["success"].astype(float).to_numpy()
        val_ece = ece(scores, labels, n_bins=args.bins)

        fig_path = outdir / f"fig_calibration_{p.stem}.png"
        draw_reliability(scores, labels, title=p.name, out_png=fig_path, n_bins=args.bins)

        metrics_files[p.name] = {
            "type": "eval_csv",
            "n": int(len(df)),
            "ece": float(val_ece),
            "conf_col": conf_col,
            "figure": fig_path.name,
        }

    # -------- trace_*.jsonl (derive proxy confidence) --------
    for p_str in sorted(glob.glob(os.path.join(indir.as_posix(), "trace_*.jsonl"))):
        p = Path(p_str)
        try:
            pts = load_trace_points(p)
        except Exception as e:
            metrics_files[p.name] = {"error": f"read-failed: {e}"}
            continue

        if not pts:
            metrics_files[p.name] = {"skipped": "empty-or-no-success"}
            continue

        scores = clip01(np.array([c for (c, s, m) in pts], dtype=float))
        labels = np.array([s for (c, s, m) in pts], dtype=float)
        val_ece = ece(scores, labels, n_bins=args.bins)

        fig_path = outdir / f"fig_calibration_{p.stem}.png"
        draw_reliability(scores, labels, title=p.name + " (proxy)", out_png=fig_path, n_bins=args.bins)

        metrics_files[p.name] = {
            "type": "trace_jsonl",
            "n": int(len(scores)),
            "ece": float(val_ece),
            "conf_col": "proxy(features.mem_top/search_top|cost)",
            "figure": fig_path.name,
        }

        # accumulate per mode as well
        for c, s, m in pts:
            if not m:
                continue
            acc = by_mode_accum.setdefault(m, {"scores": [], "labels": []})
            acc["scores"].append(float(np.clip(c, 0.0, 1.0)))
            acc["labels"].append(float(s))

    # -------- write metrics --------
    metrics_path = outdir / "calibration_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_files, f, indent=2)
    print(f"Wrote {metrics_path.as_posix()}")

    # mode-level aggregation (if we had traces)
    if by_mode_accum:
        by_mode_out: Dict[str, Dict[str, Any]] = {}
        for mode, acc in by_mode_accum.items():
            s = np.array(acc["scores"], dtype=float)
            y = np.array(acc["labels"], dtype=float)
            by_mode_out[mode] = {
                "n": int(len(s)),
                "ece": float(ece(s, y, n_bins=args.bins)),
            }
        by_mode_path = outdir / "calibration_by_mode.json"
        with by_mode_path.open("w", encoding="utf-8") as f:
            json.dump(by_mode_out, f, indent=2)
        print(f"Wrote {by_mode_path.as_posix()}")

if __name__ == "__main__":
    main()
