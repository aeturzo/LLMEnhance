#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backend/eval/report.py

Generates a one-page HTML report from artifacts:
- eval_summary_*.csv (scoreboards)
- eval_joined_*.csv (per-question rows)
- pareto_*.csv (accuracy–cost sweep)
- robustness_*.csv (Day 9: paraphrase/noise & memory scaling)
- faithfulness_*.csv (Day 5: knockouts & ablations)
- trace_*.jsonl (pulls a few SYM traces: rules & triples used)

Output: artifacts/report_{stamp}.html
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
from pathlib import Path
import pandas as pd

def _glob(root: Path, pattern: str):
    return sorted(root.rglob(pattern))

def _read_csvs(paths):
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__file"] = p.name
            dfs.append(df)
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def _html_table(df: pd.DataFrame, max_rows=50) -> str:
    if df is None or df.empty:
        return "<em>No data</em>"
    return df.head(max_rows).to_html(index=False, escape=False)

def _load_sym_traces(paths, limit=6):
    rows = []
    for p in paths:
        try:
            for i, line in enumerate(open(p, "r", encoding="utf-8")):
                if i > 2000:
                    break
                d = json.loads(line)
                st = d.get("sym_trace")
                if st:
                    triples = st.get("triples") if isinstance(st, dict) else None
                    rules = st.get("rules_fired") if isinstance(st, dict) else None
                    rows.append({
                        "file": p.name,
                        "id": d.get("id"),
                        "mode": d.get("mode"),
                        "query": d.get("query"),
                        "answer": d.get("answer"),
                        "triples": triples,
                        "rules": rules,
                    })
                    if len(rows) >= limit:
                        return rows
        except Exception:
            pass
    return rows

def main(artifacts_dir: str = "artifacts") -> Path:
    ROOT = Path(".").resolve()
    ART = (ROOT / artifacts_dir).resolve()
    ART.mkdir(exist_ok=True, parents=True)

    # Load
    eval_summary = _glob(ART, "eval_summary_*.csv")
    eval_joined  = _glob(ART, "eval_joined_*.csv")
    pareto       = _glob(ART, "pareto_*.csv")
    robust       = _glob(ART, "robustness_*.csv")
    faith        = _glob(ART, "faithfulness_*.csv")
    traces       = _glob(ART, "trace_*.jsonl")

    df_summary = _read_csvs(eval_summary)
    df_joined  = _read_csvs(eval_joined)
    df_pareto  = _read_csvs(pareto)
    df_robust  = _read_csvs(robust)
    df_faith   = _read_csvs(faith)

    # Scoreboard (sorted by file then mode)
    summary_view = pd.DataFrame()
    if not df_summary.empty:
        summary_view = (df_summary
                        .sort_values(["__file","mode"])
                        .reset_index(drop=True))

    # Failures (a small sample)
    fail_samples = pd.DataFrame()
    if not df_joined.empty:
        fails = df_joined[df_joined["success"] == 0].copy()
        cols = [c for c in ["id","type","mode","query","expected_contains","answer","latency_ms","steps","__file"] if c in fails.columns]
        fail_samples = fails[cols].head(20)

    # Symbolic answer samples
    sym_samples = pd.DataFrame()
    if not df_joined.empty:
        sj = df_joined[df_joined["answer"].fillna("").str.lower().str.startswith("symbolic:")].copy()
        cols = [c for c in ["id","type","mode","query","answer","latency_ms","steps","__file"] if c in sj.columns]
        sym_samples = sj[cols].groupby("__file", as_index=False).head(5)

    # Pareto view
    if "alpha" in df_pareto.columns:
        try:
            df_pareto["alpha"] = df_pareto["alpha"].astype(float)
        except Exception:
            pass
        df_pareto = df_pareto.sort_values(["__file","alpha"])

    # Robustness view
    robust_view = df_robust.sort_values(["setting","param","mode","__file"]) if not df_robust.empty else df_robust

    # Faithfulness agg
    faith_view = pd.DataFrame()
    if not df_faith.empty:
        for col in ["delta_after_knockout","max_rule_ablation_drop"]:
            if col in df_faith.columns:
                df_faith[col] = pd.to_numeric(df_faith[col], errors="coerce").fillna(0.0)
        try:
            faith_view = (df_faith
                          .groupby("__file")
                          .agg({"delta_after_knockout":"mean","max_rule_ablation_drop":"mean"})
                          .reset_index())
        except Exception:
            faith_view = df_faith.head(20)

    # Sym traces (rules & triples)
    sym_rows = _load_sym_traces(traces, limit=6)
    sym_evidence_html = "<em>No traces found</em>"
    if sym_rows:
        cards = []
        for r in sym_rows:
            triples_html = "<ul>" + "".join(f"<li>{t}</li>" for t in (r.get("triples") or [])) + "</ul>" if r.get("triples") else "<em>n/a</em>"
            rules_html   = "<ul>" + "".join(f"<li>{t}</li>" for t in (r.get("rules") or [])) + "</ul>" if r.get("rules") else "<em>n/a</em>"
            cards.append(f"""
              <div class='card'>
                <div class='sub'>trace: {r['file']} — id={r['id']} — mode={r['mode']}</div>
                <div><strong>Q:</strong> {r['query']}</div>
                <div><strong>Answer:</strong> {r['answer']}</div>
                <div class='sub2'><strong>Rules fired:</strong> {rules_html}</div>
                <div class='sub2'><strong>Triples used:</strong> {triples_html}</div>
              </div>
            """)
        sym_evidence_html = "\n".join(cards)

    # HTML
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>DPP System Report — {stamp}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin:24px; color:#222}}
h1,h2,h3{{margin:0 0 8px 0}}
h1{{font-size:24px}} h2{{font-size:20px; margin-top:24px}} h3{{font-size:16px; margin-top:18px}}
.sub{{color:#666; font-size:12px; margin-top:4px; margin-bottom:8px}}
.sub2{{color:#444; font-size:13px; margin-top:8px; margin-bottom:8px}}
table{{border-collapse:collapse; margin:8px 0; width:100%}}
th,td{{border:1px solid #ddd; padding:6px 8px; font-size:13px; vertical-align:top}}
th{{background:#f7f7f7; text-align:left}}
.card{{border:1px solid #eee; padding:12px; border-radius:8px; margin:10px 0; box-shadow:0 1px 2px rgba(0,0,0,0.04)}}
code{{background:#f6f8fa; padding:2px 4px; border-radius:4px}}
em{{color:#666}}
.section{{margin-bottom:24px}}
</style>
</head>
<body>
  <h1>DPP System Report</h1>
  <div class="sub">Generated {stamp}. Aggregates latest eval, Pareto, robustness, faithfulness, and traces.</div>

  <div class="section">
    <h2>Scoreboard (per mode)</h2>
    <div class="sub">From eval_summary_*.csv (all recent files).</div>
    {_html_table(summary_view)}
  </div>

  <div class="section">
    <h2>Example symbolic answers</h2>
    <div class="sub">Answer rows starting with <code>Symbolic:</code> (up to 5 per file).</div>
    {_html_table(sym_samples, max_rows=20)}
  </div>

  <div class="section">
    <h2>Interesting failures</h2>
    <div class="sub">A small sample across modes (for quick inspection).</div>
    {_html_table(fail_samples, max_rows=20)}
  </div>

  <div class="section">
    <h2>RL cost–accuracy sweep (Pareto)</h2>
    <div class="sub">Higher α → cheaper actions; check accuracy stays high.</div>
    {_html_table(df_pareto, max_rows=100)}
  </div>

  <div class="section">
    <h2>Robustness</h2>
    <div class="sub">AUGMENT = paraphrase/noise/number shifts; MEMSCALE = memory size.</div>
    {_html_table(robust_view, max_rows=150)}
  </div>

  <div class="section">
    <h2>Faithfulness (causal drops)</h2>
    <div class="sub">Mean score drop after memory knockout / max rule ablation per file.</div>
    {_html_table(faith_view, max_rows=100)}
  </div>

  <div class="section">
    <h2>Symbolic traces</h2>
    {sym_evidence_html}
  </div>

  <div class="section">
    <h3>File origins</h3>
    <ul>
      <li>Summaries: {", ".join(p.name for p in eval_summary) or "–"}</li>
      <li>Joined: {", ".join(p.name for p in eval_joined) or "–"}</li>
      <li>Pareto: {", ".join(p.name for p in pareto) or "–"}</li>
      <li>Robustness: {", ".join(p.name for p in robust) or "–"}</li>
      <li>Faithfulness: {", ".join(p.name for p in faith) or "–"}</li>
      <li>Traces: {", ".join(p.name for p in traces[:6]) + (" …" if len(traces)>6 else "") or "–"}</li>
    </ul>
  </div>
</body>
</html>
"""
    out = ART / f"report_{stamp}.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()
    main(args.artifacts)
