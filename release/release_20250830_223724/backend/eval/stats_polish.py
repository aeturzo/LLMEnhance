#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_polish.py — Bootstrap CIs, effect sizes, and per-domain summaries.

Outputs (under tables/):
  - stats_ci_overall.csv
  - stats_ci_by_type.csv
  - stats_ci_by_domain_mode.csv
  - stats_effects_rl_vs.csv      (RL vs each baseline: Δacc, Cohen's h)
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
TAB = ROOT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

def latest_joined():
    paths = sorted(ART.glob("eval_joined_*.csv"))
    if not paths:
        raise SystemExit("No artifacts/eval_joined_*.csv found. Run your pipeline first.")
    return paths[-1]

def map_id_to_domain():
    m = {}
    for dom in ["battery","lexmark","viessmann"]:
        p = ROOT / "tests" / dom / "tests.jsonl"
        if p.exists():
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    j = json.loads(line)
                    if "id" in j:
                        m[j["id"]] = dom
    return m

def bootstrap_ci(acc_series: np.ndarray, B: int = 2000, alpha: float = 0.05):
    n = len(acc_series)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    # acc_series = {0,1}
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(B):
        samp = rng.choice(acc_series, size=n, replace=True)
        boots.append(samp.mean())
    boots = np.array(boots)
    return (float(acc_series.mean()),
            float(np.quantile(boots, alpha/2)),
            float(np.quantile(boots, 1 - alpha/2)))

def cohens_h(p1: float, p2: float):
    # Cohen's h for proportions: 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))
    from math import asin, sqrt
    if any(np.isnan([p1,p2])):
        return np.nan
    return 2*asin(sqrt(p1)) - 2*asin(sqrt(p2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", default=None, help="Path to eval_joined_*.csv (default: latest)")
    args = ap.parse_args()

    joined = Path(args.joined) if args.joined else latest_joined()
    df = pd.read_csv(joined)
    # Coerce types
    df["success"] = df["success"].astype(int)
    df["mode"] = df["mode"].astype(str)
    # Domain mapping from tests ids
    id2dom = map_id_to_domain()
    df["domain"] = df["id"].map(id2dom).fillna("unknown")

    # ---- Overall CIs by mode ----
    rows = []
    for mode, g in df.groupby("mode"):
        m, lo, hi = bootstrap_ci(g["success"].to_numpy())
        rows.append({"mode": mode, "n": len(g), "acc": round(m,4), "ci_lo": round(lo,4), "ci_hi": round(hi,4)})
    pd.DataFrame(rows).sort_values("mode").to_csv(TAB/"stats_ci_overall.csv", index=False)

    # ---- By type ----
    rows = []
    for (mode, typ), g in df.groupby(["mode","type"]):
        m, lo, hi = bootstrap_ci(g["success"].to_numpy())
        rows.append({"mode": mode, "type": typ, "n": len(g), "acc": round(m,4), "ci_lo": round(lo,4), "ci_hi": round(hi,4)})
    pd.DataFrame(rows).sort_values(["type","mode"]).to_csv(TAB/"stats_ci_by_type.csv", index=False)

    # ---- By domain & mode ----
    rows = []
    for (dom, mode), g in df.groupby(["domain","mode"]):
        m, lo, hi = bootstrap_ci(g["success"].to_numpy())
        rows.append({"domain": dom, "mode": mode, "n": len(g), "acc": round(m,4), "ci_lo": round(lo,4), "ci_hi": round(hi,4)})
    pd.DataFrame(rows).sort_values(["domain","mode"]).to_csv(TAB/"stats_ci_by_domain_mode.csv", index=False)

    # ---- Effects: RL vs baselines ----
    baselines = ["BASE","MEM","SYM","MEMSYM","ROUTER","ADAPTIVERAG"]
    rows = []
    rl = df[df["mode"]=="RL"]
    for base in baselines:
        bdf = df[df["mode"]==base]
        # Overall
        p_rl = rl["success"].mean() if len(rl) else np.nan
        p_b  = bdf["success"].mean() if len(bdf) else np.nan
        rows.append({"scope":"overall", "compare":f"RL_vs_{base}", "n_rl":len(rl), "n_b":len(bdf),
                     "acc_rl":round(p_rl,4), "acc_b":round(p_b,4),
                     "delta": round(p_rl - p_b, 4), "cohens_h": round(cohens_h(p_rl,p_b),4)})
        # Per domain
        for dom in sorted(df["domain"].unique()):
            rl_d = rl[rl["domain"]==dom]; b_d = bdf[bdf["domain"]==dom]
            p_rl = rl_d["success"].mean() if len(rl_d) else np.nan
            p_b  = b_d["success"].mean() if len(b_d) else np.nan
            rows.append({"scope":dom, "compare":f"RL_vs_{base}", "n_rl":len(rl_d), "n_b":len(b_d),
                        "acc_rl":round(p_rl,4) if pd.notna(p_rl) else np.nan,
                        "acc_b":round(p_b,4) if pd.notna(p_b) else np.nan,
                        "delta": round((p_rl - p_b), 4) if all(pd.notna([p_rl,p_b])) else np.nan,
                        "cohens_h": round(cohens_h(p_rl,p_b),4) if all(pd.notna([p_rl,p_b])) else np.nan})
    pd.DataFrame(rows).to_csv(TAB/"stats_effects_rl_vs.csv", index=False)

    print("Wrote:", "tables/stats_ci_overall.csv",
          "tables/stats_ci_by_type.csv",
          "tables/stats_ci_by_domain_mode.csv",
          "tables/stats_effects_rl_vs.csv")

if __name__ == "__main__":
    main()
