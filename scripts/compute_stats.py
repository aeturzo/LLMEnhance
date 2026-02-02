#!/usr/bin/env python3
"""
Compute aggregate stats from eval_joined_*.csv:
- Accuracy with Wilson 95% CI (overall, by-domain, by-type)
- McNemar vs baseline (default RAG_BASE)
- Risk-coverage and AURC from confidence/confidence_cal

Outputs (under tables/):
  acc_ci.csv
  mcnemar.csv
  risk_coverage.csv
  aurc.csv
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def _is_calibrated(fp: Path) -> bool:
    return "_calibrated" in fp.name


def _select_joined_files(artifacts: Path) -> list[Path]:
    """
    Prefer a single pooled eval_joined_pooled_*.csv if present.
    Otherwise, use all per-domain eval_joined_*.csv (excluding pooled and calibrated).
    """
    pooled = sorted(
        [p for p in artifacts.glob("eval_joined_pooled_*.csv") if not _is_calibrated(p)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pooled:
        return [pooled[0]]

    files = sorted(
        [
            p for p in artifacts.glob("eval_joined_*.csv")
            if not _is_calibrated(p) and "pooled" not in p.name
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files


def load_all_eval_joined(artifacts: Path, joined: str | None = None) -> pd.DataFrame:
    if joined:
        fp = Path(joined)
        if not fp.exists():
            raise FileNotFoundError(f"Joined CSV not found: {fp}")
        df = pd.read_csv(fp)
        df["__file"] = fp.name
        return df

    files = _select_joined_files(artifacts)
    if not files:
        raise FileNotFoundError(f"No eval_joined_*.csv found in {artifacts}")
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        df["__file"] = fp.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def acc_ci(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    # by mode, domain, type
    for (mode, domain, typ), g in df.groupby(["mode", "domain", "type"]):
        n = len(g)
        k = int(g["correct"].sum())
        p, lo, hi = wilson(k, n)
        rows.append({"mode": mode, "domain": domain, "type": typ, "n": n, "acc": p, "ci_lo": lo, "ci_hi": hi})
    # by mode, domain (all types)
    for (mode, domain), g in df.groupby(["mode", "domain"]):
        n = len(g); k = int(g["correct"].sum()); p, lo, hi = wilson(k, n)
        rows.append({"mode": mode, "domain": domain, "type": "all", "n": n, "acc": p, "ci_lo": lo, "ci_hi": hi})
    # by mode, type (pooled domains)
    for (mode, typ), g in df.groupby(["mode", "type"]):
        n = len(g); k = int(g["correct"].sum()); p, lo, hi = wilson(k, n)
        rows.append({"mode": mode, "domain": "pooled", "type": typ, "n": n, "acc": p, "ci_lo": lo, "ci_hi": hi})
    # pooled overall
    for mode, g in df.groupby("mode"):
        n = len(g); k = int(g["correct"].sum()); p, lo, hi = wilson(k, n)
        rows.append({"mode": mode, "domain": "pooled", "type": "all", "n": n, "acc": p, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def cost_by_mode(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["cost_retrieval_calls", "cost_rule_checks", "cost_tokens_in", "cost_tokens_out", "latency_ms", "n_steps"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    rows = []
    for (mode, domain), g in df.groupby(["mode", "domain"]):
        row = {"mode": mode, "domain": domain}
        for c in cols:
            row[f"avg_{c}"] = float(pd.to_numeric(g[c], errors="coerce").fillna(0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def mcnemar(df: pd.DataFrame, baseline: str = "RAG_BASE") -> pd.DataFrame:
    rows: List[dict] = []
    modes = sorted(df["mode"].unique())
    if baseline not in modes:
        return pd.DataFrame()

    def per_domain(domain_df: pd.DataFrame, dom_label: str):
        base = domain_df[domain_df["mode"] == baseline][["id", "correct"]].set_index("id")
        for mode in modes:
            if mode == baseline:
                continue
            other = domain_df[domain_df["mode"] == mode][["id", "correct"]].set_index("id")
            joined = base.join(other, lsuffix="_base", rsuffix="_other", how="inner")
            b = int(((joined["correct_other"] == 1) & (joined["correct_base"] == 0)).sum())
            c = int(((joined["correct_other"] == 0) & (joined["correct_base"] == 1)).sum())
            n = b + c
            pval = 1.0
            if n > 0:
                from math import comb
                tail = sum(comb(n, k) for k in range(0, min(b, c) + 1))
                pval = min(1.0, 2 * tail / (2 ** n))
            rows.append({"domain": dom_label, "mode": mode, "baseline": baseline, "b_mode_gt_base": b, "c_base_gt_mode": c, "p_value": pval})

    # by-domain
    for dom, g in df.groupby("domain"):
        per_domain(g, dom)
    # pooled
    per_domain(df, "pooled")
    return pd.DataFrame(rows)


def risk_coverage(df: pd.DataFrame, conf_col: str = "confidence_cal", step: float = 0.02) -> pd.DataFrame:
    rows: List[dict] = []
    df[conf_col] = pd.to_numeric(df[conf_col], errors="coerce").fillna(0.0)
    df["correct"] = df["correct"].astype(int)
    for (mode, dom), g in df.groupby(["mode", "domain"]):
        conf = g[conf_col]
        suc = g["correct"]
        N = len(conf)
        thresholds = np.arange(0.0, 1.0 + 1e-9, step)
        for t in thresholds:
            mask = conf >= t
            n = int(mask.sum())
            cov = n / N if N else 0.0
            acc = suc[mask].mean() if n else 0.0
            risk = 1.0 - acc if n else 1.0
            rows.append({"mode": mode, "domain": dom, "threshold": float(t), "coverage": cov, "accuracy": acc, "risk": risk, "n": n})
    # pooled across domains
    for mode, g in df.groupby("mode"):
        conf = g[conf_col]; suc = g["correct"]; N = len(conf)
        thresholds = np.arange(0.0, 1.0 + 1e-9, step)
        for t in thresholds:
            mask = conf >= t
            n = int(mask.sum())
            cov = n / N if N else 0.0
            acc = suc[mask].mean() if n else 0.0
            risk = 1.0 - acc if n else 1.0
            rows.append({"mode": mode, "domain": "pooled", "threshold": float(t), "coverage": cov, "accuracy": acc, "risk": risk, "n": n})
    return pd.DataFrame(rows)


def aurc_from_rc(rc: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for (mode, dom), g in rc.groupby(["mode", "domain"]):
        g = g.sort_values("coverage")
        if g.empty:
            continue
        cov = g["coverage"].to_numpy()
        risk = g["risk"].to_numpy()
        auc = np.trapz(risk, cov)
        rows.append({"mode": mode, "domain": dom, "aurc": float(auc)})
    return pd.DataFrame(rows)


def effect_sizes(rc: pd.DataFrame, aurc_df: pd.DataFrame, acc_df: pd.DataFrame, df_eval: pd.DataFrame, baseline: str = "RAG_BASE") -> pd.DataFrame:
    rows: List[dict] = []
    cov_targets = [0.3, 0.5, 0.7]
    alpha = 0.3  # cost weight

    # cost proxy: mean n_steps per domain/mode, normalized by max within domain
    cost_means = df_eval.groupby(["domain", "mode"])["n_steps"].mean().reset_index().rename(columns={"n_steps": "cost"})
    norm_costs = []
    for dom, g in cost_means.groupby("domain"):
        max_c = max(float(g["cost"].max()), 1.0)
        for _, row in g.iterrows():
            norm_costs.append({"domain": dom, "mode": row["mode"], "cost_norm": float(row["cost"]) / max_c})
    cost_df = pd.DataFrame(norm_costs)

    # helper to get risk at threshold
    def get_risk(mode: str, dom: str, thr: float) -> float:
        g = rc[(rc["mode"] == mode) & (rc["domain"] == dom) & (rc["threshold"].round(4) == round(thr, 4))]
        if g.empty:
            return 1.0
        return float(g.iloc[0]["risk"])

    # helper acc full coverage
    def get_acc(mode: str, dom: str) -> float:
        g = acc_df[(acc_df["mode"] == mode) & (acc_df["domain"] == dom) & (acc_df["type"] == "all")]
        if g.empty:
            return 0.0
        return float(g.iloc[0]["acc"])

    # aurc lookup
    def get_aurc(mode: str, dom: str) -> float:
        g = aurc_df[(aurc_df["mode"] == mode) & (aurc_df["domain"] == dom)]
        if g.empty:
            return 0.0
        return float(g.iloc[0]["aurc"])

    domains = rc["domain"].unique()
    for dom in domains:
        modes = rc[rc["domain"] == dom]["mode"].unique()
        if baseline not in modes:
            continue
        for mode in modes:
            if mode == baseline:
                continue
            row = {"domain": dom, "mode": mode, "baseline": baseline}
            # risk reduction at targets
            for t in cov_targets:
                base_risk = get_risk(baseline, dom, t)
                mode_risk = get_risk(mode, dom, t)
                row[f"risk_reduction@{t}"] = base_risk - mode_risk
            # aurc delta
            row["aurc_delta"] = get_aurc(baseline, dom) - get_aurc(mode, dom)
            # utility delta
            acc_base = get_acc(baseline, dom)
            acc_mode = get_acc(mode, dom)
            cost_base = float(cost_df[(cost_df["domain"] == dom) & (cost_df["mode"] == baseline)]["cost_norm"].fillna(0.0).max())
            cost_mode = float(cost_df[(cost_df["domain"] == dom) & (cost_df["mode"] == mode)]["cost_norm"].fillna(0.0).max())
            util_base = acc_base - alpha * cost_base
            util_mode = acc_mode - alpha * cost_mode
            row["utility_delta"] = util_mode - util_base
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--baseline", default="RAG_BASE")
    ap.add_argument("--joined", default=None, help="use a specific eval_joined CSV")
    args = ap.parse_args()

    art = Path(args.artifacts)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_eval_joined(art, joined=args.joined)
    # normalize
    for col in ("confidence", "confidence_raw", "confidence_cal"):
        if col not in df.columns:
            df[col] = df.get("confidence", 0.0)
    if "domain" not in df.columns:
        df["domain"] = os.getenv("DPP_DOMAIN", "battery")
    if "type" not in df.columns:
        df["type"] = "open"
    if "n_steps" not in df.columns:
        df["n_steps"] = 0
    if "latency_ms" not in df.columns:
        df["latency_ms"] = 0.0

    acc_df = acc_ci(df)
    rc_df = risk_coverage(df, conf_col="confidence_cal" if "confidence_cal" in df.columns else "confidence")
    aurc_df = aurc_from_rc(rc_df)
    mc_df = mcnemar(df, baseline=args.baseline)
    eff_df = effect_sizes(rc_df, aurc_df, acc_df, df, baseline=args.baseline)
    cost_df = cost_by_mode(df)

    acc_df.to_csv(out_dir / "acc_ci.csv", index=False)
    rc_df.to_csv(out_dir / "risk_coverage.csv", index=False)
    aurc_df.to_csv(out_dir / "aurc.csv", index=False)
    if not mc_df.empty:
        mc_df.to_csv(out_dir / "mcnemar.csv", index=False)
    if not eff_df.empty:
        eff_df.to_csv(out_dir / "effect_sizes.csv", index=False)
    if not cost_df.empty:
        cost_df.to_csv(out_dir / "cost_by_mode.csv", index=False)

    print(f"Wrote acc_ci.csv ({len(acc_df)})")
    print(f"Wrote risk_coverage.csv ({len(rc_df)})")
    print(f"Wrote aurc.csv ({len(aurc_df)})")
    if not mc_df.empty:
        print(f"Wrote mcnemar.csv ({len(mc_df)})")
    if not eff_df.empty:
        print(f"Wrote effect_sizes.csv ({len(eff_df)})")
    if not cost_df.empty:
        print(f"Wrote cost_by_mode.csv ({len(cost_df)})")


if __name__ == "__main__":
    main()
