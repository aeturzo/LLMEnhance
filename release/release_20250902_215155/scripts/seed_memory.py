#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path

# --- ensure repo root on sys.path so "backend" imports work when run as a script ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services import memory_service  # after sys.path patch

def _read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                yield json.loads(t)

def _extract_text(row: dict) -> str:
    return row.get("text") or row.get("memory") or row.get("content") or row.get("doc") or ""

def _add_mem(session: str, text: str, meta: dict):
    # try common APIs (your codebase uses add_memory)
    if hasattr(memory_service, "add_memory"):
        memory_service.add_memory(session, text)
    elif hasattr(memory_service, "add"):
        memory_service.add(session_id=session, content=text, metadata=meta)
    elif hasattr(memory_service, "insert"):
        memory_service.insert(session_id=session, content=text, metadata=meta)
    elif hasattr(memory_service, "index_texts"):
        memory_service.index_texts(session_id=session, texts=[text])
    else:
        raise RuntimeError("memory_service has no add/insert/index_texts method")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["battery","lexmark","viessmann","textiles","dpp_rl"], default=None)
    ap.add_argument("--file", type=str, default=None)
    ap.add_argument("--session", type=str, default="s1")
    args = ap.parse_args()

    # resolve seed file
    if args.file:
        path = Path(args.file)
    elif args.domain:
        candidates = [
            Path(f"tests/{args.domain}/seed_mem.jsonl"),
            Path(f"tests/{args.domain}/seed_docs.jsonl"),
        ]
        path = next((p for p in candidates if p.exists()), Path("tests/dpp_rl/seed_mem.jsonl"))
    else:
        path = Path("tests/dpp_rl/seed_mem.jsonl")

    if not path.exists():
        raise SystemExit(f"[seed_memory] Seed file not found: {path}")

    n = 0
    for row in _read_jsonl(path):
        txt = _extract_text(row)
        if not txt:
            continue
        meta = {k: v for k, v in row.items() if k not in {"id","text","memory","content","doc"}}
        _add_mem(args.session, txt, meta)
        n += 1
    print(f"[seed_memory] Added {n} memories from {path}")

if __name__ == "__main__":
    main()
