import argparse, json, sys
from pathlib import Path

EVAL_TYPES = {"logic", "recall", "open"}
EVAL_BASENAMES = {"tests.jsonl", "combined.jsonl"}  # only these are "evaluation"
REQUIRED = {"id", "type", "domain", "question", "answer"}

def should_check(p: Path) -> bool:
    # Only lint evaluation files (skip seed_docs, seed_mem, episodes, mem, queries, corpora, etc.)
    return p.name in EVAL_BASENAMES

def check_file(p: Path) -> bool:
    ok = True
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                print(f"[ERROR] {p}:{i} empty line")
                ok = False
                continue
            try:
                o = json.loads(line)
            except Exception as e:
                print(f"[ERROR] {p}:{i} not JSON: {e}")
                ok = False
                continue
            missing = REQUIRED - set(o)
            if missing:
                print(f"[ERROR] {p}:{i} missing {missing}")
                ok = False
            t = o.get("type")
            if t not in EVAL_TYPES:
                print(f"[ERROR] {p}:{i} bad type={t}")
                ok = False
            if not o.get("domain"):
                print(f"[ERROR] {p}:{i} empty domain")
                ok = False
    return ok

def check_dir(root: Path) -> bool:
    ok = True
    for p in sorted(root.rglob("*.jsonl")):
        if should_check(p):
            ok = check_file(p) and ok
    return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tests", nargs="+")
    args = ap.parse_args()
    all_ok = True
    for r in args.root:
        all_ok = check_dir(Path(r)) and all_ok
    sys.exit(0 if all_ok else 1)
