#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingest tests/<domain>/docs/*.txt|*.md into tests/<domain>/seed_docs.jsonl
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["battery","lexmark","viessmann"])
    args = ap.parse_args()
    ddir = ROOT / "tests" / args.domain / "docs"
    out  = ROOT / "tests" / args.domain / "seed_docs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("*.txt","*.md"):
        paths += list(ddir.glob(ext))
    if not paths:
        print(f"No raw docs found in {ddir}")
        return
    with out.open("w", encoding="utf-8") as f:
        for p in paths:
            try:
                t = p.read_text(encoding="utf-8")
                t = " ".join(t.split())
                if t:
                    f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            except Exception:
                pass
    print(f"Wrote {out} from {len(paths)} files")

if __name__ == "__main__":
    main()
