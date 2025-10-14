#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create deterministic dev/test splits from tests.jsonl (+ optional tests_synth.jsonl)
Hash by id to avoid randomness. Default split: 30% dev, 70% test.
Outputs: tests/<domain>/split_dev.jsonl and split_test.jsonl
"""

from __future__ import annotations
import json, hashlib
from pathlib import Path
from typing import List, Dict

def load_jsonl(p: Path) -> List[Dict]:
    if not p.exists(): return []
    rows=[]
    for line in p.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line: continue
        try: rows.append(json.loads(line))
        except: pass
    return rows

def bucket(key: str, dev_ratio: float = 0.3) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < dev_ratio else "test"

def write_jsonl(p: Path, rows: List[Dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main(domain: str, dev_ratio: float = 0.3):
    base = Path("tests")/domain
    srcs = [base/"tests.jsonl", base/"tests_synth.jsonl"]
    rows=[]
    for s in srcs: rows += load_jsonl(s)
    if not rows: raise SystemExit(f"No tests found for {domain}")

    dev, test = [], []
    for r in rows:
        key = str(r.get("id") or r.get("question") or r.get("query"))
        (dev if bucket(key, dev_ratio)=="dev" else test).append(r)

    write_jsonl(base/"split_dev.jsonl", dev)
    write_jsonl(base/"split_test.jsonl", test)
    print(f"{domain}: dev={len(dev)} test={len(test)} (total {len(rows)})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--dev_ratio", type=float, default=0.3)
    a = ap.parse_args()
    main(a.domain, a.dev_ratio)
