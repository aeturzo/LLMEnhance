#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_v2.py — Per-domain explainability HTML:
- pulls latest trace_*.jsonl
- filters rows by test IDs that belong to the domain
- shows successes/failures for each mode with features, steps, sym_trace

Usage:
  python backend/eval/report_v2.py --domain battery --artifacts artifacts --out artifacts/report_battery.html
"""
from __future__ import annotations
import argparse, glob, json, html
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]

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

def to_html_block(ex):
    esc = lambda s: html.escape(str(s))
    feats = ex.get("features", {})
    steps = ex.get("steps", [])
    symt  = ex.get("sym_trace")
    ev    = ex.get("sources", [])
    lines = []
    lines.append(f"<div class='card'><div class='hdr'><b>ID</b> {esc(ex.get('id'))} | <b>Mode</b> {esc(ex.get('mode'))} | <b>Success</b> {int(ex.get('success',0))} | <b>Latency</b> {esc(ex.get('latency_ms'))} ms</div>")
    lines.append(f"<div class='q'><b>Query:</b> {esc(ex.get('query'))}</div>")
    if ex.get("product"): lines.append(f"<div class='q'><b>Product:</b> {esc(ex.get('product'))}</div>")
    if ex.get("expected_contains"): lines.append(f"<div class='q'><b>Expected contains:</b> {esc(ex.get('expected_contains'))}</div>")
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
                rows.append(j)

    if not rows:
        raise SystemExit(f"No trace rows matched domain={args.domain} (check tests/{args.domain}/tests.jsonl)")

    # Buckets: (mode, success)
    buckets = defaultdict(list)
    for r in rows:
        key = (r.get("mode","?"), int(r.get("success",0)))
        buckets[key].append(r)

    # Assemble HTML
    parts = []
    parts.append("""<html><head><meta charset="utf-8">
<style>
body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;line-height:1.4}
h1,h2{margin:0 0 8px 0}
.card{border:1px solid #e5e7eb;border-radius:12px;padding:12px;margin:10px 0;background:#fff}
.hdr{font-size:13px;color:#444;margin-bottom:6px}
.q{margin:4px 0}
.ans{margin:6px 0;background:#f8fafc;padding:8px;border-radius:8px}
summary{cursor:pointer}
</style></head><body>""")
    parts.append(f"<h1>Explainability report — {args.domain}</h1>")
    parts.append(f"<p>Trace file: <code>{trace_fp.name}</code>. Examples per bucket: {args.k}.</p>")

    for mode in sorted({m for (m,_) in buckets.keys()}):
        for succ in (1,0):
            key = (mode, succ)
            exs = buckets.get(key, [])[:args.k]
            if not exs: continue
            parts.append(f"<h2>{mode} — {'SUCCESS' if succ else 'FAIL'}</h2>")
            for e in exs:
                parts.append(to_html_block(e))

    parts.append("</body></html>")
    out = Path(args.out)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
