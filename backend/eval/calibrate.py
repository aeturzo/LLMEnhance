#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified calibrator:
- ECE + reliability diagrams by delegating to backend.eval.calibration (your existing script)
- Optional per-mode isotonic maps + calibrated CSV when --joined is provided
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

def _run_ece(artifacts: str, tables: str, bins: int) -> None:
    """
    Try to call your existing ECE script (backend.eval.calibration) in a robust way,
    regardless of its exact main() signature.
    """
    try:
        from . import calibration as cal
    except Exception as e:
        print(f"[calibrate] Skipping ECE step (no backend.eval.calibration): {e}")
        return

    # Try function-style main with kwargs, then positional, then no-arg CLI style.
    for call in (
        lambda: cal.main(artifacts=artifacts, tables=tables, bins=bins),
        lambda: cal.main(artifacts, tables, bins),
        lambda: cal.main(),
    ):
        try:
            call()
            print(f"[calibrate] ECE done via backend.eval.calibration (artifacts={artifacts}, tables={tables}, bins={bins})")
            return
        except TypeError:
            continue
        except SystemExit:
            # If their script uses argparse and calls sys.exit on completion, treat as success
            print(f"[calibrate] ECE script exited normally.")
            return
        except Exception as e:
            print(f"[calibrate] ECE call variant failed: {e}")
    print("[calibrate] Could not run ECE script; continuing without it.")

def _fit_isotonic_per_mode(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build per-mode isotonic maps from 'confidence' -> 'success'.
    Returns dict: mode -> DataFrame with columns [x,y] for x in [0,1] grid.
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception as e:
        print(f"[calibrate] scikit-learn not available; skipping isotonic calibration: {e}")
        return {}

    maps: Dict[str, pd.DataFrame] = {}
    for mode, g in df.dropna(subset=["confidence"]).groupby("mode"):
        xs = pd.to_numeric(g["confidence"], errors="coerce").dropna()
        ys = pd.to_numeric(g["success"], errors="coerce").dropna()
        # Need at least 2 distinct x and both 0/1 present
        if xs.nunique() < 2 or set(ys.unique()) <= {0} or set(ys.unique()) <= {1}:
            grid = pd.Series([i/100 for i in range(101)])
            maps[str(mode)] = pd.DataFrame({"x": grid, "y": grid})
            continue
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(xs.to_numpy(), ys.to_numpy())
        grid = pd.Series([i/100 for i in range(101)])
        maps[str(mode)] = pd.DataFrame({"x": grid, "y": ir.predict(grid)})
    return maps

def _apply_maps(df: pd.DataFrame, maps: Dict[str, pd.DataFrame]) -> pd.Series:
    import numpy as np
    def cal_one(mode: str, c) -> Optional[float]:
        if c in (None, "", "None"):
            return ""
        try:
            c = float(c)
        except Exception:
            return ""
        m = maps.get(mode)
        if m is None:  # identity if no map
            return c
        xs = m["x"].to_numpy()
        ys = m["y"].to_numpy()
        return float(np.interp(c, xs, ys))
    return pd.Series([cal_one(str(m), c) for m, c in zip(df["mode"], df["confidence"])])

def _run_isotonic(joined_csv: str, out_dir: str) -> None:
    inp = Path(joined_csv)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    if not inp.exists():
        raise FileNotFoundError(f"Joined CSV not found: {inp}")

    df = pd.read_csv(inp)
    for col in ("mode", "success", "confidence"):
        if col not in df.columns:
            raise SystemExit(f"[calibrate] {inp.name} missing column: {col}")

    # Standardize empties and coerce
    df["confidence"] = df["confidence"].replace({"": None, "None": None})
    maps = _fit_isotonic_per_mode(df)
    # Write maps
    for mode, m in maps.items():
        fp = outp / f"cal_map_{mode}.csv"
        m.to_csv(fp, index=False)
        print(f"[calibrate] wrote {fp}")

    if maps:
        df["confidence_cal"] = _apply_maps(df, maps)
    else:
        df["confidence_cal"] = df["confidence"]  # identity if no maps

    out_cal = outp / f"{inp.stem}_calibrated.csv"
    df.to_csv(out_cal, index=False)
    print(f"[calibrate] wrote {out_cal}")

def main():
    ap = argparse.ArgumentParser(description="Unified calibration runner")
    # ECE / reliability args (your existing script)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--bins", type=int, default=10)
    # Isotonic map args (for calibrated CSV)
    ap.add_argument("--joined", default=None, help="artifacts/eval_joined_*.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--skip_ece", action="store_true", help="skip calling backend.eval.calibration")
    args = ap.parse_args()

    if not args.skip_ece:
        _run_ece(args.artifacts, args.tables, args.bins)

    if args.joined:
        _run_isotonic(args.joined, args.out)
    else:
        print("[calibrate] No --joined provided; ECE only.")

if __name__ == "__main__":
    main()
