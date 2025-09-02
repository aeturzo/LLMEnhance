#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
publication_checks.py (tabulate-free, robust)
- Loads artifacts/eval_*.csv
- Repairs 'success' if missing (literal, case-insensitive containment)
- Deduplicates by (id,mode,query,product,session)
- Accuracies overall & by type
- Counts (per-domain if present)
- McNemar exact p-values vs BASE (requires scipy; if missing, installs? no — we fail soft)

Writes:
  tables/pub_summary.md
  tables/acc_overall.csv
  tables/acc_by_type.csv
  tables/counts.csv
  tables/mcnemar.csv
  tables/calibration.csv   (now real ECE if confidence exists; otherwise placeholder)
"""
from __future__ import annotations
import glob, math
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

# ---------- success repair (substring contains) ----------
def _to_str(x): return "" if x is None else str(x)

def repair_success(row) -> int:
    # if success exists, keep it
    if "success" in row and pd.notna(row["success"]):
        try:
            return int(row["success"])
        except Exception:
            pass
    expected = row.get("expected_contains")
    answer = row.get("answer", "")
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

    # ensure success present
    if "success" not in big.columns:
        big["success"] = big.apply(repair_success, axis=1)
    else:
        # coerce to 0/1 ints; if NaN, repair
        base_success = pd.to_numeric(big["success"], errors="coerce")
        need = base_success.isna()
        big.loc[need, "success"] = big[need].apply(repair_success, axis=1)
        big["success"] = pd.to_numeric(big["success"], errors="coerce").fillna(0).astype(int)

    # dedup, prefer later sources
    keys = ["id","mode","query","product","session"]
    return big.sort_values("__source").drop_duplicates(subset=keys, keep="last")

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
    for mode in sorted(df["mode"].unique()):
        if mode == "BASE": continue
        cur = df[df["mode"]==mode].set_index(key)["success"]
        both = base.index.intersection(cur.index)
        if len(both) == 0:
            rows.append({"mode": mode, "n": 0, "p_two_sided": np.nan, "note": "no overlap"})
            continue
        b = base.loc[both].astype(int); c = cur.loc[both].astype(int)
        n01 = int(((b==0) & (c==1)).sum())
        n10 = int(((b==1) & (c==0)).sum())
        n = n01 + n10
        if not have_scipy or n == 0:
            rows.append({"mode": mode, "n": n, "p_two_sided": np.nan, "note": "scipy missing or no discordant pairs"})
            continue
        p = float(binomtest(k=min(n01,n10), n=n, p=0.5, alternative="two-sided").pvalue)
        rows.append({"mode": mode, "n": n, "p_two_sided": p})
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR/"mcnemar.csv", index=False)
    return out

# -------- Calibration (new) --------
_CONF_CANDS = ["confidence","prob","p_correct","router_conf","score","conf"]

def _choose_conf_col(df: pd.DataFrame) -> str | None:
    for c in _CONF_CANDS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return c
    return None

def _ece_quantile(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    n = len(scores)
    if n == 0:
        return float("nan")
    order = np.argsort(scores)
    s = scores[order]; y = labels[order]
    k_eff = int(min(k, n))
    edges = np.floor(np.linspace(0, n, k_eff + 1)).astype(int)
    ece = 0.0
    for i in range(k_eff):
        lo, hi = edges[i], edges[i+1]
        if hi <= lo: 
            continue
        sb = s[lo:hi]; yb = y[lo:hi]
        acc = float(np.mean(yb)) if len(yb) else 0.0
        conf = float(np.mean(sb)) if len(sb) else 0.0
        ece += abs(acc - conf) * (len(sb) / n)
    return float(ece)

def calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    # If an existing tables/calibration.csv already has numeric ece, don't overwrite
    out_path = OUTDIR / "calibration.csv"
    if out_path.exists():
        try:
            tmp = pd.read_csv(out_path)
            if "ece" in tmp.columns and pd.to_numeric(tmp["ece"], errors="coerce").notna().any():
                return tmp
        except Exception:
            pass

    conf_col = _choose_conf_col(df)
    if conf_col is None:
        rows = [{"mode": m, "ece": np.nan, "note": "No confidence logged; run backend/eval/calibration.py for real ECE"}
                for m in sorted(df["mode"].unique())]
        out = pd.DataFrame(rows)
        out.to_csv(out_path, index=False)
        return out

    d = df.copy()
    d["success"] = pd.to_numeric(d["success"], errors="coerce").fillna(0.0).astype(float)
    d["__conf__"] = pd.to_numeric(d[conf_col], errors="coerce")
    d = d.dropna(subset=["__conf__"])
    d["__conf__"] = d["__conf__"].clip(0.0, 1.0)

    rows = []
    for mode, grp in d.groupby("mode", dropna=False):
        s = grp["__conf__"].to_numpy(dtype=float)
        y = grp["success"].to_numpy(dtype=float)
        n = len(s)
        if n == 0:
            rows.append({"mode": str(mode), "ece": np.nan, "n": 0, "bins": 0, "k": 10})
            continue
        val = _ece_quantile(s, y, k=10)
        rows.append({"mode": str(mode), "ece": round(float(val), 6), "n": int(n), "bins": int(min(10, n)), "k": 10})

    out = pd.DataFrame(rows).sort_values("mode")
    out.to_csv(out_path, index=False)
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
    acc_overall, acc_by_type = accuracy_tables(df)
    counts = counts_table(df)
    mcn = mcnemar_exact(df)
    calibration_table(df)  # <— compute real ECE if possible (or write placeholder)
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
