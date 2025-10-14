# backend/eval/sym_coverage.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute symbolic coverage & precision from traces produced by run_eval_all.py.

Inputs:
  - EITHER --trace artifacts/trace_<STAMP>.jsonl
  - OR     --joined artifacts/eval_joined_<STAMP>.csv  (we derive the trace path)

Outputs:
  - tables/sym_coverage.csv  with rows per mode + ALL (RL excluded by default)

Definitions:
  coverage = (# examples where a SYM step fired) / (total examples for that mode)
  precision = mean(success) over examples where SYM fired
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _sym_fired(steps: Any) -> bool:
    """
    True if any step has source == 'SYM'.
    'steps' may be a list[dict] or a JSON string; be defensive.
    """
    if steps is None:
        return False
    if isinstance(steps, str):
        try:
            steps = json.loads(steps)
        except Exception:
            return False
    if not isinstance(steps, list):
        return False
    for s in steps:
        try:
            if (s.get("source") or "").upper() == "SYM":
                return True
        except Exception:
            continue
    return False


def _derive_trace_from_joined(joined_csv: Path) -> Path:
    """
    eval_joined_YYYYMMDD_HHMMSS[(_anything)](.csv) -> trace_YYYYMMDD_HHMMSS.jsonl
    - Strips optional '_calibrated' or any suffix after the timestamp.
    - If pattern not matched, returns newest trace_* in the same dir (or raises).
    """
    stem = joined_csv.stem  # e.g., eval_joined_20250905_222657_calibrated
    m = re.search(r"eval_joined_(\d{8}_\d{6})", stem)
    if m:
        ts = m.group(1)
        return joined_csv.parent / f"trace_{ts}.jsonl"
    # fallback: newest trace in same dir
    traces = sorted(joined_csv.parent.glob("trace_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if traces:
        return traces[0]
    raise FileNotFoundError("Could not derive trace file; pass --trace explicitly.")


def _resolve_trace_path(joined: str | None, trace: str | None) -> Path:
    """
    Resolve a concrete trace path:
    - If --trace provided, use it (with a calibrated->uncalibrated fallback).
    - Else derive from --joined (strip _calibrated), else newest in artifacts/.
    """
    if trace:
        t = Path(trace)
        if t.exists():
            return t
        # If a calibrated suffix slipped in, try removing it
        t_alt = Path(str(t).replace("_calibrated", ""))
        if t_alt != t and t_alt.exists():
            return t_alt
        # Final fallback: newest trace in same dir or artifacts/
        for base in (t.parent, Path("artifacts")):
            cand = sorted(base.glob("trace_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if cand:
                return cand[0]
        raise FileNotFoundError(f"Trace not found: {t}")

    if joined:
        d = _derive_trace_from_joined(Path(joined))
        if d.exists():
            return d
        d_alt = Path(str(d).replace("_calibrated", ""))
        if d_alt != d and d_alt.exists():
            return d_alt
        # newest in same dir or artifacts/
        for base in (Path(joined).parent, Path("artifacts")):
            cand = sorted(base.glob("trace_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if cand:
                return cand[0]
        raise FileNotFoundError(f"Could not find a trace file near {joined}")

    # Last resort: newest in artifacts/
    cand = sorted(Path("artifacts").glob("trace_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cand:
        return cand[0]
    raise FileNotFoundError("No trace file found in artifacts/. Provide --trace or --joined.")


def main(joined: str | None, trace: str | None, out_dir: str, include_rl: bool) -> None:
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    trace_path = _resolve_trace_path(joined, trace)

    rows = _load_jsonl(trace_path)
    if not rows:
        raise SystemExit(f"No rows in {trace_path}")

    df = pd.DataFrame(rows)
    need_cols = {"mode", "success", "steps"}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{trace_path.name} missing columns: {missing}")

    # Normalize types
    df["mode"] = df["mode"].astype(str)
    df["success"] = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int)
    df["sym_fired"] = df["steps"].apply(_sym_fired)

    if not include_rl:
        df = df[df["mode"] != "RL"]

    # Per-mode stats
    out_rows: List[Dict[str, Any]] = []
    for mode, g in df.groupby("mode"):
        n_total = len(g)
        g_fired = g[g["sym_fired"]]
        fired_n = len(g_fired)
        coverage = fired_n / n_total if n_total else 0.0
        precision = g_fired["success"].mean() if fired_n else 0.0
        out_rows.append({
            "mode": mode,
            "n_total": n_total,
            "fired_n": fired_n,
            "coverage": round(float(coverage), 6),
            "precision": round(float(precision), 6),
        })

    # Overall across modes (excluding RL by default)
    n_total = len(df)
    g_fired = df[df["sym_fired"]]
    fired_n = len(g_fired)
    coverage = fired_n / n_total if n_total else 0.0
    precision = g_fired["success"].mean() if fired_n else 0.0
    out_rows.append({
        "mode": "ALL" + ("" if not include_rl else "+RL"),
        "n_total": n_total,
        "fired_n": fired_n,
        "coverage": round(float(coverage), 6),
        "precision": round(float(precision), 6),
    })

    out_fp = outp / "sym_coverage.csv"
    pd.DataFrame(out_rows).to_csv(out_fp, index=False)
    print(f"Wrote {out_fp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", default=None, help="artifacts/eval_joined_<STAMP>.csv (trace inferred)")
    ap.add_argument("--trace", default=None, help="artifacts/trace_<STAMP>.jsonl")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--include_rl", action="store_true", help="include RL rows in the ALL aggregate")
    a = ap.parse_args()
    main(a.joined, a.trace, a.out, a.include_rl)
