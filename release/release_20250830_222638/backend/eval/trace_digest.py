#!/usr/bin/env python3
# backend/eval/trace_digest.py
import argparse, os, json, glob, random
from pathlib import Path
import pandas as pd

def load_traces(art_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(art_dir, "trace_*.jsonl")))
    if not paths:
        raise SystemExit("No trace_*.jsonl found in artifacts")
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["success"] = df["success"].astype(int)
    return df

def pick_examples(df, mode, success, k=4):
    sub = df[(df["mode"]==mode) & (df["success"]==success)]
    if len(sub)==0: return []
    return sub.sample(n=min(k,len(sub)), random_state=42).to_dict(orient="records")

def to_md(ex):
    steps = ex.get("steps") or []
    sources = ex.get("sources") or []
    sym_trace = ex.get("sym_trace")
    md = []
    md.append(f"### ID {ex.get('id')} | Mode {ex.get('mode')} | Success {ex.get('success')}")
    md.append(f"**Query:** {ex.get('query')}")
    prod = ex.get("product")
    if prod: md.append(f"**Product:** {prod}")
    md.append(f"**Answer:** {ex.get('answer')}")
    md.append(f"**Latency:** {ex.get('latency_ms')} ms, **Steps:** {len(steps)}")
    if steps:
        md.append("**Steps:**")
        for s in steps:
            src = s.get("source")
            md.append(f"- {src}: { {k:v for k,v in s.items() if k!='source'} }")
    if sources:
        md.append("**Sources (top):**")
        for s in sources[:3]:
            md.append(f"- {s.get('type')} | score={s.get('score')} | {str(s.get('snippet'))[:120]}...")
    if sym_trace:
        md.append("**Symbolic trace:**")
        md.append(f"- rules_fired: {sym_trace.get('rules_fired')}")
        md.append(f"- inferred: {sym_trace.get('inferred')}")
    md.append("")
    return "\n".join(md)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="artifacts")
    ap.add_argument("--out", dest="out", default="artifacts/qualitative.md")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    df = load_traces(args.inp)
    modes = sorted(df["mode"].unique())
    lines = ["# Qualitative Digest", ""]
    for m in modes:
        lines.append(f"## Mode: {m}")
        succs = pick_examples(df, m, 1, k=args.k)
        fails = pick_examples(df, m, 0, k=args.k)
        if succs:
            lines.append("#### Success cases")
            for ex in succs: lines.append(to_md(ex))
        if fails:
            lines.append("#### Failure cases")
            for ex in fails: lines.append(to_md(ex))
        lines.append("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
