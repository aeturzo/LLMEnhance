#!/usr/bin/env python3
"""Generate reproducible pooled and Open Food Facts calibration diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAME_MAP = {"ADAPTIVERAG": "COMPASS", "RAG_BASE": "RAG-Base", "MEMSYM": "Mem+Sym", "SYM_ONLY": "Sym-Only"}


def equal_mass_bins(scores: np.ndarray, labels: np.ndarray, bins: int) -> list[tuple[float, float, int]]:
    order = np.argsort(scores, kind="mergesort")
    chunks = np.array_split(order, min(bins, len(order)))
    return [(float(scores[idx].mean()), float(labels[idx].mean()), len(idx)) for idx in chunks if len(idx)]


def equal_mass_ece(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    n = len(scores)
    return float(sum(abs(acc - conf) * count / n for conf, acc, count in equal_mass_bins(scores, labels, bins)))


def fixed_ece(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    membership = np.minimum(np.digitize(scores, edges[1:-1], right=False), bins - 1)
    result = 0.0
    for idx in range(bins):
        keep = membership == idx
        if keep.any():
            result += float(keep.mean()) * abs(float(labels[keep].mean()) - float(scores[keep].mean()))
    return result


def bootstrap_ece_ci(
    scores: np.ndarray, labels: np.ndarray, bins: int, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    values = np.empty(samples, dtype=float)
    n = len(scores)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        values[i] = equal_mass_ece(scores[idx], labels[idx], bins)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def clean(frame: pd.DataFrame, confidence_col: str, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    scores = pd.to_numeric(frame[confidence_col], errors="coerce")
    labels = pd.to_numeric(frame[label_col], errors="coerce")
    valid = scores.notna() & labels.notna()
    return scores[valid].clip(0, 1).to_numpy(float), labels[valid].to_numpy(float)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooled", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--calibration-source", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    pooled = pd.read_csv(args.pooled)
    off = pd.read_csv(args.off)
    calibration_source = pd.read_csv(args.calibration_source)
    calibration_size = len(calibration_source)
    calibration_per_mode = calibration_source.groupby("mode").size().to_dict()
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    diagrams: list[tuple[str, np.ndarray, np.ndarray]] = []

    for mode, frame in pooled.groupby("mode", sort=True):
        scores, labels = clean(frame, "confidence_cal", "correct")
        low, high = bootstrap_ece_ci(scores, labels, args.bins, args.bootstrap, rng)
        rows.append(
            {
                "dataset": "pooled",
                "mode": NAME_MAP.get(str(mode), str(mode)),
                "internal_mode": mode,
                "n_eval": len(scores),
                "n_calibration": int(calibration_per_mode.get(mode, 0)),
                "calibration_scope": "pooled across development domains; evaluated per domain and pooled",
                "brier": float(np.mean((scores - labels) ** 2)),
                "ece_equal_mass": equal_mass_ece(scores, labels, args.bins),
                "ece_ci_low": low,
                "ece_ci_high": high,
                "ece_fixed_width": fixed_ece(scores, labels, args.bins),
            }
        )
        if mode in {"ADAPTIVERAG", "RAG_BASE", "MEMSYM"}:
            diagrams.append((NAME_MAP.get(str(mode), str(mode)), scores, labels))

    scores, labels = clean(off, "confidence", "correct")
    low, high = bootstrap_ece_ci(scores, labels, args.bins, args.bootstrap, rng)
    rows.append(
        {
            "dataset": "Open Food Facts",
            "mode": "COMPASS-OFF",
            "internal_mode": str(off["mode"].iloc[0]) if "mode" in off and len(off) else "OFF",
            "n_eval": len(scores),
            "n_calibration": 0,
            "calibration_scope": "no post-hoc fit; native confidence evaluated on held-out OFF rows",
            "brier": float(np.mean((scores - labels) ** 2)),
            "ece_equal_mass": equal_mass_ece(scores, labels, args.bins),
            "ece_ci_low": low,
            "ece_ci_high": high,
            "ece_fixed_width": fixed_ece(scores, labels, args.bins),
        }
    )
    diagrams.append(("COMPASS-OFF", scores, labels))

    result = pd.DataFrame(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.csv, index=False)
    tex_cols = ["dataset", "mode", "n_eval", "n_calibration", "brier", "ece_equal_mass", "ece_ci_low", "ece_ci_high"]
    args.tex.write_text(result[tex_cols].to_latex(index=False, float_format=lambda x: f"{x:.4f}"), encoding="utf-8")

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.2), sharex=True, sharey=True)
    for ax, (title, diag_scores, diag_labels) in zip(axes.flat, diagrams):
        points = equal_mass_bins(diag_scores, diag_labels, args.bins)
        ax.plot([0, 1], [0, 1], "--", color="0.55", linewidth=0.8)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", markersize=3, linewidth=1.1)
        ax.set_title(title)
        ax.grid(True, alpha=0.2, linewidth=0.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("Mean confidence")
    for ax in axes[:, 0]:
        ax.set_ylabel("Empirical accuracy")
    fig.tight_layout()
    fig.savefig(args.pdf, bbox_inches="tight")
    plt.close(fig)

    compass = result[(result["dataset"] == "pooled") & (result["internal_mode"] == "ADAPTIVERAG")].iloc[0]
    reproduced = float(compass["ece_fixed_width"])
    if not np.isclose(reproduced, 0.5246800816564595, atol=1e-9):
        raise SystemExit(f"sanity check failed: expected COMPASS fixed-bin ECE 0.524680, got {reproduced:.12f}")
    print(f"reproduced COMPASS fixed-bin ECE={reproduced:.6f}; calibration source n={calibration_size}")
    print(f"wrote {args.pdf}, {args.csv}, and {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
