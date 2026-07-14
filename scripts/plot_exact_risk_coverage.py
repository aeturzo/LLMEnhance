#!/usr/bin/env python3
"""Plot exact pooled risk--coverage curves with publication-facing names."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISPLAY_NAMES = {
    "ADAPTIVERAG": "COMPASS",
    "RAG_BASE": "RAG-Base",
    "MEMSYM": "Mem+Sym",
    "SYM_ONLY": "Sym-Only",
    "RL": "RL",
    "ROUTER": "Router",
}


def exact_curve(frame: pd.DataFrame, confidence_col: str) -> pd.DataFrame:
    """Return one point per unique threshold, keeping tied scores together."""
    conf = pd.to_numeric(frame[confidence_col], errors="coerce")
    correct = pd.to_numeric(frame["correct"], errors="coerce")
    valid = conf.notna() & correct.notna()
    work = pd.DataFrame({"confidence": conf[valid], "correct": correct[valid]})
    work = work.sort_values("confidence", ascending=False, kind="mergesort")
    if work.empty:
        return pd.DataFrame(columns=["threshold", "coverage", "risk", "n"])

    grouped = work.groupby("confidence", sort=False)["correct"].agg(["size", "sum"])
    accepted = grouped["size"].cumsum()
    correct_cum = grouped["sum"].cumsum()
    out = pd.DataFrame(
        {
            "threshold": grouped.index.astype(float),
            "coverage": accepted / len(work),
            "risk": 1.0 - correct_cum / accepted,
            "n": accepted.astype(int),
        }
    ).reset_index(drop=True)
    return out


def curve_auc(curve: pd.DataFrame) -> float:
    if curve.empty:
        return float("nan")
    coverage = np.r_[0.0, curve["coverage"].to_numpy(float)]
    risk = np.r_[float(curve.iloc[0]["risk"]), curve["risk"].to_numpy(float)]
    return float(np.trapz(risk, coverage))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--aurc-csv", type=Path, required=True)
    parser.add_argument("--confidence-col", default="confidence_cal")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    required = {"mode", "correct", args.confidence_col}
    missing = required - set(data.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")

    curves: list[pd.DataFrame] = []
    aurc_rows: list[dict[str, object]] = []
    for internal, display in DISPLAY_NAMES.items():
        frame = data[data["mode"] == internal]
        if frame.empty:
            continue
        curve = exact_curve(frame, args.confidence_col)
        curve.insert(0, "series", display)
        curves.append(curve)
        aurc_rows.append(
            {"series": display, "internal_mode": internal, "n": len(frame), "aurc": curve_auc(curve)}
        )

    if not curves:
        raise SystemExit("none of the requested modes were present")
    combined = pd.concat(curves, ignore_index=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.csv, index=False)
    pd.DataFrame(aurc_rows).to_csv(args.aurc_csv, index=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(5.25, 3.0), constrained_layout=False)
    for display, curve in combined.groupby("series", sort=False):
        x = np.r_[0.0, curve["coverage"].to_numpy(float)]
        y = np.r_[float(curve.iloc[0]["risk"]), curve["risk"].to_numpy(float)]
        ax.plot(x, y, linewidth=1.35, label=display)
    ax.set(xlabel="Coverage", ylabel="Risk (1 - accuracy)", xlim=(0, 1), ylim=(0, 1))
    ax.grid(True, alpha=0.22, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.subplots_adjust(right=0.75, bottom=0.18)
    fig.savefig(args.pdf, bbox_inches="tight")
    plt.close(fig)

    compass = next((r for r in aurc_rows if r["series"] == "COMPASS"), None)
    if compass is not None:
        print(f"COMPASS exact AURC={float(compass['aurc']):.6f} n={compass['n']}")
    print(f"wrote {args.pdf}, {args.csv}, and {args.aurc_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
