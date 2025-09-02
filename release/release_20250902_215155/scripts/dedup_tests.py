#!/usr/bin/env python3
import json, sys, re
from pathlib import Path

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    yield json.loads(ln)
                except:
                    pass

def write_jsonl(p: Path, rows):
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main(paths):
    for path_str in paths:
        p = Path(path_str)
        seen = set()
        kept, dropped = 0, 0
        out = []
        for obj in load_jsonl(p):
            q = obj.get("query") or obj.get("q") or ""
            key = norm(q)
            if not key:
                # keep empty-query rows (rare) to avoid losing KB logic items by mistake
                out.append(obj); kept += 1
                continue
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            out.append(obj); kept += 1
        write_jsonl(p, out)
        print(json.dumps({
            "file": str(p),
            "kept": kept,
            "dropped_dupes_by_query": dropped,
            "total_after": len(out)
        }))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: dedup_tests.py tests/<domain>/tests.jsonl [...]")
        sys.exit(1)
    main(sys.argv[1:])
