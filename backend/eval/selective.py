#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build selective-risk curves (coverage vs accuracy, risk).
Prefers calibrated confidence if available.

Outputs:
- artifacts/selective_<joined_stem>.csv (traceable)
- artifacts/selective_calibrated.csv   (stable, for downstream)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np


def _latest(art: Path, pattern: str) -> Optional[Path]:
    cand = sorted(art.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0] if cand else None


def _pick_joined(artifacts: str) -> Tuple[Path, str]:
    """
    Only consider EVAL JOINED files, never the selective output itself.
    Priority:
      1) eval_joined_*_calibrated.csv  (use confidence_cal)
      2) eval_joined_*.csv             (use confidence)
    """
    art = Path(artifacts)

    cal = _latest(art, "eval_joined_*_calibrated.csv")
    if cal is not None:
        return cal, "confidence_cal"

    raw = _latest(art, "eval_joined_*.csv")
    if raw is not None:
        return raw, "confidence"

    raise FileNotFoundError(
        f"No eval joined CSV found under {art} "
        "(expected eval_joined_*_calibrated.csv or eval_joined_*.csv)"
    )


def _build_selective(df: pd.DataFrame, conf_col: str, step: float = 0.02) -> pd.DataFrame:
    # If the expected confidence column is missing but the other exists, auto-fallback.
    if conf_col not in df.columns:
        alt = "confidence" if conf_col == "confidence_cal" else "confidence_cal"
        if alt in df.columns:
            conf_col = alt
        else:
            raise SystemExit(f"Missing confidence column '{conf_col}' and '{alt}' in joined CSV")

    df = df.copy()
    # coerce types
    df["success"] = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int)
    df[conf_col] = pd.to_numeric(df[conf_col], errors="coerce")
    df = df.dropna(subset=[conf_col])

    thresholds = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    rows = []

    for mode, g in df.groupby("mode"):
        g = g.copy()
        g_conf = g[conf_col]
        N = len(g_conf)
        for t in thresholds:
            picked = g[g_conf >= t]
            n = len(picked)
            cov = n / N if N else 0.0
            acc = picked["success"].mean() if n else 0.0
            risk = 1.0 - acc if n else 1.0
            rows.append({
                "mode": mode,
                "threshold": float(t),
                "coverage": float(cov),
                "accuracy": float(acc),
                "risk": float(risk),
                "n": int(n),
            })

    return pd.DataFrame(rows)


def main(artifacts: str, out_dir: str):
    joined, conf_col = _pick_joined(artifacts)
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(joined)
    sel = _build_selective(df, conf_col=conf_col, step=0.02)

    # Write both a traceable and a stable filename
    stem_out = outp / f"selective_{joined.stem}.csv"
    compat_out = outp / "selective_calibrated.csv"

    sel.to_csv(stem_out, index=False)
    sel.to_csv(compat_out, index=False)
    print(f"Wrote {stem_out}")
    print(f"Wrote {compat_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    main(args.artifacts, args.out)
