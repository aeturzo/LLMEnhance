#!/usr/bin/env python3
import json, sys, re
from pathlib import Path
RE_BAD = re.compile(r"\.\.\.|\\u2026")
REQ = {"id","query","type","session","expected_contains"}
ok = True
for p in map(Path, sys.argv[1:]):
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            t = line.strip()
            if not t: continue
            if RE_BAD.search(t):
                print(f"[BAD] {p}:{i} contains ellipsis placeholder")
                ok = False
                continue
            try:
                obj = json.loads(t)
            except Exception as e:
                print(f"[BAD] {p}:{i} invalid JSON: {e}")
                ok = False; continue
            miss = REQ - set(obj)
            if miss:
                print(f"[BAD] {p}:{i} missing keys: {sorted(miss)}"); ok = False
            if obj.get("type") not in {"recall","logic","open"}:
                print(f"[WARN] {p}:{i} unusual type: {obj.get('type')}")
sys.exit(1 if not ok else 0)
