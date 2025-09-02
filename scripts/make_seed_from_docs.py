#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create seed_docs.jsonl from tests/<domain>/docs/*.txt|*.md for memory/RAG.

Usage:
  python scripts/make_seed_from_docs.py --domain battery --session s1 --min 300 --max 900
"""
from __future__ import annotations
import argparse, pathlib, json, re

def chunks(text: str, min_len=600, max_len=1200):
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    cur = ""
    for p in paras:
        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= max_len:
            cur = f"{cur}\n\n{p}"
        else:
            # emit if long enough, otherwise keep growing
            if len(cur) >= min_len:
                yield cur
                cur = p
            else:
                cur = f"{cur}\n\n{p}"
    if cur:
        yield cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=("battery","lexmark","viessmann"))
    ap.add_argument("--session", default="s1")
    ap.add_argument("--min", type=int, default=300, help="min chunk length (chars)")
    ap.add_argument("--max", type=int, default=900, help="max chunk length (chars)")
    args = ap.parse_args()

    docs_dir = pathlib.Path("tests")/args.domain/"docs"
    out_path = pathlib.Path("tests")/args.domain/"seed_docs.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n=0
    with out_path.open("w", encoding="utf-8") as out:
        for ext in ("*.txt","*.md"):
            for p in sorted(docs_dir.glob(ext)):
                txt = p.read_text(encoding="utf-8", errors="ignore")
                for ck in chunks(txt, args.min, args.max):
                    out.write(json.dumps({"session": args.session, "text": ck, "source": p.name})+"\n")
                    n+=1
    print(f"[seed-from-docs] Wrote {out_path} with {n} chunks")

if __name__ == "__main__":
    main()
