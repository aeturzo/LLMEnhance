#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_router.py

Trains the supervised router from the latest eval_joined_*.csv
and writes artifacts/policy_router.json. Prints training stats.
Supports optional --artifacts DIR or ARTIFACTS_DIR env var.
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse

# Ensure repo root on sys.path so "backend" imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.policy_router import RouterModel, MODEL_PATH  # type: ignore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=None, help="Path to artifacts dir with eval_joined_*.csv")
    args = ap.parse_args()

    model, stats = RouterModel.train(args.artifacts)
    print(f"Wrote {MODEL_PATH}")
    print(f"Loaded from: {stats.loaded_from}")
    print(f"Unique queries: {stats.n_queries}")
    print(f"Training rows: {stats.n_train}")
    print(f"Label counts: {stats.label_counts}")

if __name__ == "__main__":
    main()
