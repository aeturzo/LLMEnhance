#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pareto sweep for RL (accuracy vs cost)

What it does
------------
- Loads the canonical test set (tests/dpp_rl/tests.jsonl).
- For alpha in [0.0 ... 1.0], sets env RL_ALPHA and calls /solve_rl on each test.
- Computes:
    accuracy = mean( success_contains )
    avg_cost = mean( episode_cost(steps) )
    avg_steps = mean( len(steps) )
- Writes artifacts/pareto_{stamp}.csv with columns:
    alpha, accuracy, avg_cost, avg_steps, n, mode='RL'
- Attempts to write artifacts/pareto_{stamp}.png (scatter: cost vs accuracy).

Notes
-----
- If your /solve_rl endpoint doesn’t yet return a "steps" array, we
  infer a single-step source (SYM/MEM/SEARCH/BASE) from the answer text
  as a fallback so costs aren’t zero.
"""

from __future__ import annotations

import os
import csv
import json
import time
import pathlib
from typing import List, Dict, Any

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.policy_costs import episode_cost

ROOT = pathlib.Path(__file__).parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True, parents=True)
DATA_CANON = ROOT / "tests" / "dpp_rl" / "tests.jsonl"


def set_global_seed(seed: int = 42) -> None:
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def load_tests() -> List[Dict[str, Any]]:
    if not DATA_CANON.exists():
        raise SystemExit(f"Missing canonical tests at: {DATA_CANON}")
    with open(DATA_CANON, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def success_contains(expected: str | None, answer: str | None) -> int:
    if not expected:
        return 0
    return 1 if str(expected).lower() in (answer or "").lower() else 0


def post_solve_rl(client: TestClient, payload: dict) -> dict:
    resp = client.post("/solve_rl", json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"/solve_rl HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def infer_steps_if_missing(answer: str | None) -> List[Dict[str, Any]]:
    """
    Fallback heuristic: if RL didn't return steps, infer a single module
    from the answer surface form so cost isn't zero.
    """
    a = (answer or "").lower()
    if a.startswith("symbolic:") or "symbolic:" in a:
        return [{"source": "SYM"}]
    if a.startswith("memory:"):
        return [{"source": "MEM"}]
    if a.startswith("search:") or "search:" in a:
        return [{"source": "SEARCH"}]
    # default
    return [{"source": "BASE"}]


if __name__ == "__main__":
    set_global_seed(42)

    client = TestClient(app)
    tests = load_tests()
    rid = time.strftime("%Y%m%d_%H%M%S")

    alphas = [round(a, 2) for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    rows: List[Dict[str, Any]] = []

    for alpha in alphas:
        os.environ["RL_ALPHA"] = str(alpha)

        correct_flags: List[int] = []
        costs: List[float] = []
        n_steps: List[int] = []

        for ex in tests:
            payload = {
                "query": ex["query"],
                "product": ex.get("product"),
                "session": ex.get("session", "s1"),
            }
            out = post_solve_rl(client, payload)
            ans = out.get("answer", "")
            steps = out.get("steps", [])

            # Fallback: infer one step if RL didn't return any
            if not steps:
                steps = infer_steps_if_missing(ans)

            correct_flags.append(success_contains(ex.get("expected_contains"), ans))
            costs.append(episode_cost(steps))
            n_steps.append(len(steps or []))

        n = max(1, len(correct_flags))
        row = {
            "alpha": alpha,
            "accuracy": sum(correct_flags) / n,
            "avg_cost": sum(costs) / n,
            "avg_steps": sum(n_steps) / n,
            "n": n,
            "mode": "RL",
        }
        rows.append(row)

    # Write CSV
    fp = ART / f"pareto_{rid}.csv"
    with fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "accuracy", "avg_cost", "avg_steps", "n", "mode"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {fp}")

    # Try to plot
    try:
        import matplotlib.pyplot as plt
        xs = [r["avg_cost"] for r in rows]
        ys = [r["accuracy"] for r in rows]
        plt.figure()
        plt.scatter(xs, ys)
        for r in rows:
            plt.annotate(f"{r['alpha']}", (r["avg_cost"], r["accuracy"]))
        plt.xlabel("Average cost")
        plt.ylabel("Accuracy")
        plt.title("Pareto (RL accuracy vs cost)")
        png = ART / f"pareto_{rid}.png"
        plt.savefig(png, dpi=160, bbox_inches="tight")
        print(f"Wrote {png}")
    except Exception as e:
        print(f"[WARN] matplotlib not available or plotting failed: {e}")
