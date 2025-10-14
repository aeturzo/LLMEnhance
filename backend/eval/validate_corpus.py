#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, pandas as pd
from pathlib import Path

def main(corpus_path: str, out_dir: str):
    p = Path(corpus_path); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows=[]
    texts_seen=set()
    dup=short=empty=0
    with p.open("r",encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            if not line.strip(): continue
            try: rec = pd.read_json(line, lines=False, typ="series").to_dict()
            except Exception: continue
            text = (rec.get("text") or "").strip()
            if not text: empty += 1
            if len(text) < 40: short += 1
            if text in texts_seen: dup += 1
            texts_seen.add(text)
    pd.DataFrame([{"file":p.name,"n":len(texts_seen),"empty":empty,"short":short,"dup_text":dup}])\
      .to_csv(out/"corpus_qc.csv", index=False)
    print(f"Wrote {out/'corpus_qc.csv'}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="tables")
    a=ap.parse_args(); main(a.corpus, a.out)
