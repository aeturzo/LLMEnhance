#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_v2.py — Per-domain explainability HTML:
- pulls latest trace_*.jsonl
- filters rows by test IDs that belong to the domain
- shows successes/failures for each mode with features, steps, sym_trace
- NEW: re-scores each row with strict validators (recall/open, optional yes/no logic)
       and displays both "trace success" and "validator success" side-by-side.

Usage:
  python backend/eval/report_v2.py --domain battery --artifacts artifacts --out artifacts/report_battery.html
"""
from __future__ import annotations
import argparse, glob, json, html, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]

# --- Validators (strict) ----------------------------------------------------
# These are used ONLY for display here; your CSV scorer should also import these.
try:
    from backend.eval.validators import (
        recall_canonical,
        open_with_citation,
        logic_yesno,
    )
except Exception:
    # Soft fallback if validators aren't in place yet
    def recall_canonical(answer, gold_value, aliases=None):
        a = (answer or "").strip().lower()
        g = (gold_value or "").strip().lower()
        return bool(g) and (g in a)

    def open_with_citation(answer, span):
        a = (answer or "").strip().lower()
        s = (span or "").strip().lower()
        return bool(s) and (s in a)

    def logic_yesno(answer, gold_label):
        a = (answer or "").strip().lower()
        g = (gold_label or "").strip().lower()
        return (a == g) or a.startswith(g + " ")

_WS = re.compile(r"\s+")

def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip().lower())


def latest_trace(artifacts: Path) -> Path:
    paths = sorted(artifacts.glob("trace_*.jsonl"))
    if not paths:
        raise SystemExit("No artifacts/trace_*.jsonl found. Run eval first.")
    return paths[-1]


def load_ids_for_domain(domain: str) -> set[str]:
    p = ROOT / "tests" / domain / "tests.jsonl"
    if not p.exists():
        return set()
    ids = set()
    for line in p.open(encoding="utf-8"):
        if not line.strip(): continue
        j = json.loads(line)
        if "id" in j: ids.add(j["id"])
    return ids


def score_with_validators(row: dict) -> int:
    """
    Strict scoring used for display:
      - recall: canonical/aliases exact (lenient contains in canonical space)
      - open: gold span must appear in the answer
      - logic: if clearly yes/no, use logic_yesno; else fallback to substring
    """
    t = row.get("type")
    ans = row.get("answer") or ""
    exp = row.get("expected_contains") or ""
    meta = row.get("meta") or {}
    aliases = meta.get("aliases") or []

    if t == "recall":
        return 1 if recall_canonical(ans, exp, aliases) else 0

    if t == "open":
        return 1 if open_with_citation(ans, exp) else 0

    if t == "logic":
        # If we can tell it's a yes/no gold, use strict yes/no; else fallback to substring.
        gold_label = (meta.get("gold_label") or exp or "").strip().lower()
        if gold_label in {"yes", "no"}:
            return 1 if logic_yesno(ans, gold_label) else 0
        return 1 if (exp and exp.lower() in (ans or "").lower()) else 0

    return 0


def to_html_block(ex):
    esc = lambda s: html.escape(str(s))
    feats = ex.get("features", {})
    steps = ex.get("steps", [])
    symt  = ex.get("sym_trace")
    ev    = ex.get("sources", [])
    conf  = ex.get("confidence", None)

    # Re-score with validators for display (separate from the trace's success)
    v_succ = score_with_validators(ex)
    t_succ = int(ex.get("success", 0))

    # color border by validator result
    border = "#10b981" if v_succ else "#ef4444"

    lines = []
    lines.append(
        f"<div class='card' style='border-color:{border}'>"
        f"<div class='hdr'>"
        f"<b>ID</b> {esc(ex.get('id'))} | "
        f"<b>Mode</b> {esc(ex.get('mode'))} | "
        f"<b>Trace success</b> {t_succ} | "
        f"<b>Validator</b> {v_succ} | "
        f"<b>Latency</b> {esc(ex.get('latency_ms'))} ms"
        + (f" | <b>Conf</b> {esc(conf)}" if conf is not None else "")
        + "</div>"
    )
    lines.append(f"<div class='q'><b>Type:</b> {esc(ex.get('type'))}</div>")
    lines.append(f"<div class='q'><b>Query:</b> {esc(ex.get('query'))}</div>")
    if ex.get("product"):
        lines.append(f"<div class='q'><b>Product:</b> {esc(ex.get('product'))}</div>")
    if ex.get("expected_contains"):
        lines.append(f"<div class='q'><b>Expected contains:</b> {esc(ex.get('expected_contains'))}</div>")
    lines.append(f"<div class='ans'><b>Answer:</b> {esc(ex.get('answer'))}</div>")
    if feats:
        lines.append("<details><summary><b>Features</b></summary><pre>"+esc(json.dumps(feats, indent=2))+"</pre></details>")
    if steps:
        lines.append("<details><summary><b>Steps</b></summary><pre>"+esc(json.dumps(steps, indent=2))+"</pre></details>")
    if symt:
        lines.append("<details><summary><b>Symbolic trace</b></summary><pre>"+esc(json.dumps(symt, indent=2))+"</pre></details>")
    if ev:
        lines.append("<details><summary><b>Sources</b></summary><pre>"+esc(json.dumps(ev, indent=2))+"</pre></details>")
    lines.append("</div>")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["battery","lexmark","viessmann"])
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8, help="Examples per bucket")
    args = ap.parse_args()

    artifacts = Path(args.artifacts)
    trace_fp = latest_trace(artifacts)
    dom_ids = load_ids_for_domain(args.domain)

    rows = []
    with trace_fp.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            j = json.loads(line)
            if j.get("id") in dom_ids:
                # compute validator success once and store for bucketing
                j["_validator_success"] = score_with_validators(j)
                rows.append(j)

    if not rows:
        raise SystemExit(f"No trace rows matched domain={args.domain} (check tests/{args.domain}/tests.jsonl)")

    # Bu
