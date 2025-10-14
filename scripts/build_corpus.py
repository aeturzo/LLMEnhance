#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build retrieval corpus from:
- seed_docs.jsonl (type=doc/document)
- (optional) gold answers from tests files (tests.jsonl, tests_synth.jsonl, split_*.jsonl)
Options:
  --split [all|dev|test]  : if dev/test, only include docs that correspond to that split;
                            also auto-omit gold when split == test unless --include-gold
  --include-gold          : force include gold answers as docs
"""

from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from typing import List, Dict

def norm_doc(rec: Dict, default_domain: str) -> Dict | None:
    t = (rec.get("type") or "").lower()
    if t in {"doc","document"}:
        text = rec.get("text") or rec.get("content") or rec.get("answer") or ""
        if not text.strip(): return None
        did = rec.get("id") or hashlib.sha1(text.encode()).hexdigest()[:12]
        return {
            "id": str(did),
            "title": rec.get("title") or rec.get("question") or str(did),
            "text": text,
            "domain": rec.get("domain") or default_domain,
            "source": rec.get("source") or "seed",
        }
    return None

def load_jsonl(p: Path) -> List[Dict]:
    if not p.exists(): return []
    rows=[]
    for line in p.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line: continue
        try: rows.append(json.loads(line))
        except: pass
    return rows

def write_jsonl(out_path: Path, rows: List[Dict]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def collect_docs(tests_dir: Path, split: str, include_gold: bool) -> List[Dict]:
    docs: List[Dict] = []

    # 1) seed docs
    for p in tests_dir.rglob("seed_docs.jsonl"):
        dom = p.parent.name
        for rec in load_jsonl(p):
            d = norm_doc(rec, dom)
            if d: docs.append(d)

    # 2) gold answers as docs (optional or dev-only)
    def want_gold_this_file(path: Path) -> bool:
        if include_gold: return True
        if split == "test": return False
        return True  # allow on dev or all

    gold_files = []
    for dom_dir in tests_dir.iterdir():
        if not dom_dir.is_dir(): continue
        if split in ("dev","test"):
            gold_files += [dom_dir/f"split_{split}.jsonl"]
        else:
            gold_files += [dom_dir/"tests.jsonl", dom_dir/"tests_synth.jsonl"]

    for p in gold_files:
        dom = p.parent.name
        if not want_gold_this_file(p): continue
        for ex in load_jsonl(p):
            text = ex.get("answer") or ex.get("expected_contains")
            if not text: continue
            did = f"gold:{ex.get('id', '') or hashlib.sha1((ex.get('query','')+text).encode()).hexdigest()[:10]}"
            doc = {
                "id": did,
                "title": ex.get("query") or ex.get("question") or "gold",
                "text": text,
                "domain": dom,
                "source": "gold",
            }
            docs.append(doc)

    # dedupe by id
    seen=set(); out=[]
    for d in docs:
        if d["id"] in seen: continue
        seen.add(d["id"]); out.append(d)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="tests")
    ap.add_argument("--out", default="backend/corpus/dpp_corpus.jsonl")
    ap.add_argument("--split", choices=["all","dev","test"], default="all")
    ap.add_argument("--include-gold", action="store_true")
    a = ap.parse_args()

    rows = collect_docs(Path(a.tests), a.split, a.include_gold)
    write_jsonl(Path(a.out), rows)
    print(f"Wrote {a.out} with {len(rows)} docs")

if __name__ == "__main__":
    main()
