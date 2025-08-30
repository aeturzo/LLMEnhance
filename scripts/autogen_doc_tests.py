#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate open + numeric-recall QAs from domain docs.

Search order for documents (first existing wins and is merged):
  tests/<dom>/seed_docs.jsonl
  tests/<dom>/seed_mem.jsonl
  tests/dpp_<dom>/seed_docs.jsonl
  tests/dpp_<dom>/seed_mem.jsonl
  tests/dpp_rl/seed_docs.jsonl        (battery)
  tests/dpp_rl/seed_mem.jsonl         (battery)
  tests/<dom>/docs/*.txt|*.md         (ingested if available)

Lines may use keys: "text" or "content" or "doc" or "memory"
"""
from __future__ import annotations
import argparse, json, random, re, glob, os
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
NUM_UNIT = re.compile(r"(\b\d+(?:\.\d+)?\s?(?:mAh|Ah|V|W|Watt|Wh|kg|g|mm|cm|°C|%)\b)", re.I)
SENT_SPLIT = re.compile(r"(?<=[\.\?!])\s+")

PATTERNS = [
    (re.compile(r"\b([A-Z][\w\-]+)\s+has\s+([\w\-\s%°C]+)\b", re.I), "What does {prod} have?"),
    (re.compile(r"\b([A-Z][\w\-]+)\s+supports\s+([\w\-\s]+)\b", re.I), "What feature does {prod} support?"),
    (re.compile(r"\b([A-Z][\w\-]+)\s+is\s+compatible\s+with\s+([\w\-\s]+)\b", re.I), "What is {prod} compatible with?"),
    (re.compile(r"\b([A-Z][\w\-]+)\s+weighs\s+([\d\.\s]*(?:kg|g))\b", re.I), "What is the weight of {prod}?"),
    (re.compile(r"\b([A-Z][\w\-]+)\s+voltage\s+(?:is|:)\s*([\d\.\s]*V)\b", re.I), "What is the voltage of {prod}?"),
]

def collect_docs(dom: str) -> List[str]:
    candidates = [
        ROOT / "tests" / dom / "seed_docs.jsonl",
        ROOT / "tests" / dom / "seed_mem.jsonl",
        ROOT / "tests" / f"dpp_{dom}" / "seed_docs.jsonl",
        ROOT / "tests" / f"dpp_{dom}" / "seed_mem.jsonl",
    ]
    if dom == "battery":
        candidates += [
            ROOT / "tests" / "dpp_rl" / "seed_docs.jsonl",
            ROOT / "tests" / "dpp_rl" / "seed_mem.jsonl",
        ]

    texts: List[str] = []
    # JSONL sources
    for p in candidates:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    j = json.loads(line)
                    t = j.get("text") or j.get("content") or j.get("doc") or j.get("memory") or ""
                    if t: texts.append(str(t))

    # Raw docs ingested (if any)
    docs_dir = ROOT / "tests" / dom / "docs"
    for ext in ("*.txt","*.md"):
        for p in docs_dir.glob(ext):
            try:
                texts.append(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return texts

def gen_from_sentence(s: str) -> List[Dict]:
    out = []
    s = s.strip()
    # numeric unit extraction → recall/open
    m = NUM_UNIT.search(s)
    if m:
        val = m.group(1).strip()
        prod = next((w for w in s.split() if re.match(r"^[A-Z][\w\-]+$", w)), None)
        if prod:
            out.append({"type":"recall","query":f"What is the specification value mentioned for {prod}?",
                        "expected_contains":val,"session":"s1"})
    for rex, qtpl in PATTERNS:
        m = rex.search(s)
        if m:
            prod = m.group(1).strip()
            val  = m.group(2).strip()
            out.append({"type":"open","query":qtpl.format(prod=prod),
                        "expected_contains":val,"session":"s1"})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["battery","lexmark","viessmann"])
    ap.add_argument("--n_open", type=int, default=600)
    ap.add_argument("--n_recall", type=int, default=200)
    args = ap.parse_args()

    out_dir = ROOT / "tests" / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tests.jsonl"

    existing, seen = [], set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            j = json.loads(line)
            existing.append(j)
            seen.add((j.get("query",""), j.get("expected_contains","")))

    texts = collect_docs(args.domain)
    sents = []
    for t in texts:
        sents.extend([s for s in SENT_SPLIT.split(t) if s.strip()])

    candidates = []
    for s in sents:
        for ex in gen_from_sentence(s):
            key = (ex["query"], ex["expected_contains"])
            if key not in seen:
                candidates.append(ex)
                seen.add(key)

    random.shuffle(candidates)
    nums   = [c for c in candidates if NUM_UNIT.search(c["expected_contains"])]
    opens  = [c for c in candidates if c not in nums]

    take_rec  = nums[:args.n_recall]
    take_open = opens[:args.n_open]

    rows = existing + [
        {**ex, "id": ex.get("id") or f"{args.domain}.doc.{ex['type']}.{i:05d}"}
        for i, ex in enumerate(take_rec + take_open, start=len(existing)+1)
    ]

    with out_path.open("w", encoding="utf-8") as f:
        for j in rows:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    print(json.dumps({
        "domain": args.domain,
        "doc_sources": len(texts),
        "added_doc_recall": len(take_rec),
        "added_open": len(take_open),
        "total": len(rows),
        "out": out_path.as_posix()
    }, indent=2))

if __name__ == "__main__":
    main()
