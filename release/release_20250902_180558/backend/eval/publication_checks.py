#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
publication_checks.py (tabulate-free, robust)
- Loads artifacts/eval_*.csv
- Repairs 'success' if missing (literal, case-insensitive containment)
- Deduplicates by (id,mode,query,product,session)
- Accuracies overall & by type
- Counts (per-domain if present)
- McNemar exact p-values vs BASE (requires scipy; if missing, install scipy)

Writes:
  tables/pub_summary.md
  tables/acc_overall.csv
  tables/acc_by_type.csv
  tables/counts.csv
  tables/mcnemar.csv
  tables/calibration.csv (placeholder)
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd

OUTDIR = Path("tables"); OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- tiny markdown writer (no tabulate needed) ----------
def df_to_md(df: pd.DataFrame) -> str:
    d = df.copy()
    if isinstance(d.index, pd.MultiIndex) or d.index.name is not None:
        d = d.reset_index()
    cols = [str(c) for c in d.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in d.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

# ---------- helpers ----------
def _to_str(x):
    return "" if pd.isna(x) else str(x)

def _derive_success_literal(answer: str, expected) -> int:
    a = _to_str(answer).lower()
    if isinstance(expected, (list, tuple, set)):
        toks = [_to_str(t).lower() for t in expected if _to_str(t) != ""]
        if not toks:
            return 0
        return int(all(t in a for t in toks))
    e = _to_str(expected).lower()
    if e == "":
        return 0
    return int(e in a)

def load_all_eval() -> pd.DataFrame:
    files = sorted(glob.glob("artifacts/eval_*.csv"))
    if not files:
        raise SystemExit("No artifacts/eval_*.csv found.")
    dfs = []
    for p in files:
        df = pd.read_csv(p)
        df["__source"] = Path(p).name
        for col, default in [
            ("mode", ""), ("type", "open"), ("id", ""), ("query", ""), ("product", ""), ("session", ""),
            ("answer", ""), ("expected_contains", "")
        ]:
            if col not in df.columns:
                df[col] = default
        dfs.append(df)
    big = pd.concat(dfs, ignore_index=True)

    # success repair/compute
    if "success" not in big.columns:
        big["success"] = np.nan
    mask = ~pd.to_numeric(big["success"], errors="coerce").apply(np.isfinite)
    if mask.any():
        sub = big.loc[mask, ["answer", "expected_contains"]]
        derived_vals = [_derive_success_literal(a, e) for a, e in zip(sub["answer"], sub["expected_contains"])]
        big.loc[mask, "success"] = derived_vals
    big["success"] = (
        pd.to_numeric(big["success"], errors="coerce")
          .replace([np.inf, -np.inf], 0)
          .fillna(0)
          .astype(int)
    )

    big["mode"] = big["mode"].astype(str)
    big["type"] = big["type"].fillna("open").astype(str)
    return big

def dedup(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["id","mode","query","product","session"]
    return df.sort_values("__source").drop_duplicates(subset=keys, keep="last")

def accuracy_tables(df: pd.DataFrame):
    acc_overall = df.groupby("mode", as_index=False)["success"].mean().rename(columns={"success":"accuracy"})
    acc_by_type = df.groupby(["mode","type"], as_index=False)["success"].mean().rename(columns={"success":"accuracy"})
    acc_overall.to_csv(OUTDIR/"acc_overall.csv", index=False)
    acc_by_type.to_csv(OUTDIR/"acc_by_type.csv", index=False)
    return acc_overall, acc_by_type

def counts_table(df: pd.DataFrame):
    base_keys = ["id","query","product","session"]
    unique_examples = df.drop_duplicates(subset=base_keys)
    if "domain" in df.columns:
        counts = unique_examples.groupby(["domain","type"]).size().reset_index(name="n")
    else:
        counts = unique_examples.groupby(["type"]).size().reset_index(name="n")
    counts.to_csv(OUTDIR/"counts.csv", index=False)
    return counts

def mcnemar_exact(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.stats import binomtest
        have_scipy = True
    except Exception:
        have_scipy = False
        binomtest = None  # type: ignore
    key = ["id","query","product","session"]
    base = df[df["mode"]=="BASE"].set_index(key)["success"]
    rows = []
    for m in sorted(df["mode"].unique()):
        if m == "BASE":
            continue
        alt = df[df["mode"]==m].set_index(key)["success"]
        paired = pd.concat([base, alt], axis=1, keys=["base","alt"]).dropna()
        if paired.empty:
            rows.append({"mode": m, "b":0, "c":0, "n_paired":0, "p_value":1.0, "note":"no pairs"})
            continue
        b = int(((paired["base"]==1) & (paired["alt"]==0)).sum())
        c = int(((paired["base"]==0) & (paired["alt"]==1)).sum())
        n = b + c
        if n == 0:
            p = 1.0
        elif have_scipy:
            p = float(binomtest(min(b,c), n=n, p=0.5, alternative="two-sided").pvalue)
        else:
            # symmetric exact two-sided for p=0.5 using tail sum
            from math import comb
            tail = min(b, c)
            denom = 2.0 ** n
            p = 2.0 * sum(comb(n, i) / denom for i in range(0, tail + 1))
            p = float(min(1.0, p))
        rows.append({"mode": m, "b": b, "c": c, "n_paired": n, "p_value": round(p, 6)})
    out = pd.DataFrame(rows).sort_values("mode")
    out.to_csv(OUTDIR/"mcnemar.csv", index=False)
    return out

def simple_calibration_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"mode": m, "ece": np.nan, "note": "No confidence logged; run backend/eval/calibration.py for real ECE"}
            for m in sorted(df["mode"].unique())]
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR/"calibration.csv", index=False)
    return out

def write_md(acc_overall, acc_by_type, counts, mcnemar_df):
    lines = []
    lines.append("# Publication Checks Summary\n")
    if "domain" in counts.columns:
        lines.append("## Counts per domain × type\n")
        pivot = counts.pivot_table(index="domain", columns="type", values="n", fill_value=0)
        lines.append(df_to_md(pivot.reset_index()))
    else:
        lines.append("## Counts per type (domain not present in eval CSVs)\n")
        pivot = counts.pivot_table(index=None, columns="type", values="n", aggfunc="sum", fill_value=0)
        lines.append(df_to_md(pivot.reset_index()))
        lines.append("\n> Tip: add a `domain` column in run_eval_all.py rows for domain-aware aggregation.\n")
    lines.append("\n## Accuracy (overall)\n")
    lines.append(df_to_md(acc_overall.sort_values("accuracy", ascending=False)))
    lines.append("\n## Accuracy by type\n")
    lines.append(df_to_md(acc_by_type.pivot_table(index="mode", columns="type", values="accuracy").reset_index()))
    lines.append("\n## McNemar vs BASE (exact, two-sided)\n")
    lines.append(df_to_md(mcnemar_df))
    (OUTDIR/"pub_summary.md").write_text("\n\n".join(lines), encoding="utf-8")

def main():
    df = load_all_eval()
    df = dedup(df)
    acc_overall, acc_by_type = accuracy_tables(df)
    counts = counts_table(df)
    mcn = mcnemar_exact(df)
    simple_calibration_placeholder(df)
    write_md(acc_overall, acc_by_type, counts, mcn)
    print("Wrote:",
          OUTDIR/"acc_overall.csv",
          OUTDIR/"acc_by_type.csv",
          OUTDIR/"counts.csv",
          OUTDIR/"mcnemar.csv",
          OUTDIR/"calibration.csv",
          OUTDIR/"pub_summary.md")

if __name__ == "__main__":
    main()
