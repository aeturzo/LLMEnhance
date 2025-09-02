#!/usr/bin/env python3
import json, sys, re
from pathlib import Path

REQ = {"id","query","type","session","expected_contains"}

def norm(s): return re.sub(r"\s+"," ", (s or "").strip().lower())

def repair_row(obj, domain, counters):
    # already good?
    if all(k in obj for k in REQ):
        return obj

    # try to map from our earlier schema { "q": ..., "gold": {...} }
    q = obj.get("q") or obj.get("query")
    gold = obj.get("gold") or {}
    exp = gold.get("span") or gold.get("value") or (gold.get("aliases") or [None])[0]
    if not q or not exp:
        return None  # cannot repair reliably

    # infer type if missing
    t = obj.get("type")
    if not t:
        t = "open" if (gold.get("span") and not re.search(r"\d", str(gold.get("span")))) else "recall"

    # build repaired row
    counters.setdefault(domain, 0)
    rid = obj.get("id") or f"docfix-{domain}-{counters[domain]:06d}"
    counters[domain] += 1

    repaired = {
        "id": rid,
        "type": t,
        "session": obj.get("session") or "s_docs",
        "product": obj.get("product") or obj.get("meta",{}).get("source_id") or f"{domain}_seed",
        "query": q,
        "expected_contains": exp,
    }
    # keep anything else around under 'meta'
    meta = dict(obj)
    for k in list(repaired.keys()): meta.pop(k, None)
    if meta: repaired["meta"] = meta
    return repaired

def process(path: Path):
    out = []
    bad, repaired, kept = 0, 0, 0
    counters = {}
    domain = path.parts[-2]
    with path.open("r", encoding="utf-8") as f:
        for i, ln in enumerate(f, 1):
            ln = ln.strip()
            if not ln: continue
            try:
                obj = json.loads(ln)
            except:
                bad += 1
                continue
            if all(k in obj for k in REQ):
                out.append(obj); kept += 1
            else:
                row = repair_row(obj, domain, counters)
                if row is None:
                    bad += 1
                else:
                    out.append(row); repaired += 1
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(json.dumps({
        "file": str(path),
        "kept": kept,
        "repaired": repaired,
        "dropped": bad,
        "total": len(out)
    }))

if __name__ == "__main__":
    roots = [Path(p) for p in sys.argv[1:]]
    for p in roots:
        process(p)
