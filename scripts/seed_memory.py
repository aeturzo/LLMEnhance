#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seed episodic memory from tests/dpp_rl/seed_mem.jsonl
Each line: {"session": "s1", "memory": "text to remember"}
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services import memory_service  # noqa: E402

SEED = ROOT / "tests" / "dpp_rl" / "seed_mem.jsonl"

def main() -> None:
    if not SEED.exists():
        print(f"[seed_memory] No seed file at {SEED} (skipping).")
        return
    n = 0
    with SEED.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            memory_service.add_memory(row.get("session", "default"), row["memory"])
            n += 1
    # optional: rebuild embeddings if backend dim changed
    try:
        memory_service.reindex()
    except Exception:
        pass
    print(f"[seed_memory] Added {n} memories from {SEED}")

if __name__ == "__main__":
    main()
