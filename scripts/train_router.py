#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the supervised Router on latest eval_joined_*.csv + Day-2 features.
Writes: artifacts/policy_router.json
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure repo root is on sys.path so 'backend' is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.policy_router import RouterModel, MODEL_PATH  # noqa: E402


def main() -> None:
    model = RouterModel.train()
    out = model.save(MODEL_PATH)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
