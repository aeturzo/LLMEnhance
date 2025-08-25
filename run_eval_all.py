#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 1+2+4 — Experiment spine + traces (with SYM trace + RL cost/reward)
- Reads canonical tests from tests/dpp_rl/tests.jsonl (preferred), else tests/combined_eval/combined.jsonl.
- Evaluates BASE, MEM, SYM, MEMSYM via /solve (mode in JSON), and RL via /solve_rl on the SAME tests.
- Deterministic seeding for random/numpy/torch.
- Writes:
    artifacts/eval_{MODE}_{stamp}.csv
    artifacts/eval_joined_{stamp}.csv
    artifacts/eval_summary_{stamp}.csv
    artifacts/trace_{stamp}.jsonl   (features + full steps incl. sym_trace, RL cost/reward)
- Acceptance: no HTTP 4xx/5xx; joined rows == (#tests × 5 modes).
"""
from __future__ import annotations

import os
import csv
import json
import pathlib
import time
import random
from typing import List, Dict, Any

from fastapi.testclient import TestClient

# project imports
from backend.config.run_modes import RunMode
from backend.services.symbolic_reasoning_service import build_reasoner
from backend.services import memory_service
from backend.services.policy_costs import episode_cost   # <-- Day 4
from backend.main import app

# -------- Paths --------
ROOT = pathlib.Path(__file__).parent
DATA_CANON = ROOT / "tests" / "dpp_rl" / "tests.jsonl"                    # canonical spec
DATA_COMBINED = ROOT / "tests" / "combined_eval" / "combined.jsonl"       # fallback
DATA_MEMSEED = ROOT / "tests" / "dpp_rl" / "seed_mem.jsonl"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True, parents=True)

# -------- Determinism --------
def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # type: ignore
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False     # type: ignore
    except Exception:
        pass

# -------- Utilities --------
def seed_memory() -> None:
    if not DATA_MEMSEED.exists():
        return
    with open(DATA_MEMSEED, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                memory_service.add_memory(row.get("session", "default"), row["memory"])

def load_jsonl(path: pathlib.Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_tests() -> List[Dict[str, Any]]:
    """
    Preferred schema (tests/dpp_rl/tests.jsonl):
      {"id","query","product","session","type","expected_contains"}

    Fallback (tests/combined_eval/combined.jsonl):
      expects {"id","query","product?","session?","type?","gold":{"contains":"..."}}
    """
    if DATA_CANON.exists():
        rows = load_jsonl(DATA_CANON)
        for i, ex in enumerate(rows, 1):
            for k in ("id", "query", "session", "type", "expected_contains"):
                if k not in ex:
                    raise KeyError(f"Missing '{k}' in canonical tests on line {i}")
        return rows

    combined = load_jsonl(DATA_COMBINED)
    out = []
    for ex in combined:
        gold = ex.get("gold") or {}
        contains = gold.get("contains")
        if not contains:
            continue
        out.append({
            "id": ex.get("id") or f"c_{len(out)+1}",
            "query": ex["query"],
            "product": ex.get("product"),
            "session": ex.get("session", "s1"),
            "type": ex.get("type", "open"),
            "expected_contains": contains,
        })
    if not out:
        raise SystemExit("No usable tests found. Please add tests/dpp_rl/tests.jsonl.")
    print(f"[WARN] Using fallback from combined.jsonl → {len(out)} tests")
    return out

def success_contains(expected: str | None, answer: str | None) -> int:
    if not expected:
        return 0
    return 1 if str(expected).lower() in (answer or "").lower() else 0

# ---- (Day-2) lightweight policy features (graceful if file not present) ----
def extract_features_safe(query: str, product: str | None, session: str) -> Dict[str, float | int]:
    try:
        from backend.services.policy_features import extract_features  # type: ignore
        return extract_features(query, product, session) or {}
    except Exception:
        has_num = int(any(ch.isdigit() for ch in query))
        return {
            "len_query": len(query),
            "has_number": has_num,
            "has_product": int(bool(product)),
            "mem_top": 0.0,
            "mem_max3": 0.0,
            "search_top": 0.0,
            "search_max3": 0.0,
            "sym_fired": 0,
        }

def guess_action_from_steps(steps: List[Dict[str, Any]] | None) -> str:
    steps = steps or []
    sources = [s.get("source") for s in steps if isinstance(s, dict)]
    if "MEM" in sources and "SYM" in sources:
        return "MEMSYM"
    if "SYM" in sources:
        return "SYM"
    if "MEM" in sources:
        return "MEM"
    if "SEARCH" in sources:
        return "SEARCH"
    return "BASE"

def guess_action_from_answer(answer: str | None) -> str:
    a = (answer or "").lower()
    if a.startswith("memory:"):
        return "MEM"
    if a.startswith("search:") or "search:" in a:
        return "SEARCH"
    if a.startswith("symbolic:") or "symbolic:" in a or "rule:" in a:
        return "SYM"
    return "BASE"

def post_solve(client: TestClient, payload: dict) -> dict:
    resp = client.post("/solve", json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"/solve HTTP {resp.status_code}: {resp.text}")
    return resp.json()

def post_solve_rl(client: TestClient, payload: dict) -> dict:
    resp = client.post("/solve_rl", json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"/solve_rl HTTP {resp.status_code}: {resp.text}")
    return resp.json()

# -------- Main --------
if __name__ == "__main__":
    set_global_seed(42)
    app.state.reasoner = build_reasoner(run_owl_rl=True)
    seed_memory()
    client = TestClient(app)

    rid = time.strftime("%Y%m%d_%H%M%S")
    tests = load_tests()
    trace_fp = ART / f"trace_{rid}.jsonl"

    # 1) Classic modes (BASE, MEM, SYM, MEMSYM)
    CLASSIC = (RunMode.BASE, RunMode.MEM, RunMode.SYM, RunMode.MEMSYM)
    per_mode_paths: List[pathlib.Path] = []

    for mode in CLASSIC:
        rows: List[Dict[str, Any]] = []
        for ex in tests:
            payload = {
                "session": ex["session"],
                "session_id": ex.get("session"),
                "query": ex["query"],
                "product": ex.get("product"),
                "mode": mode.value,
            }
            t0 = time.time()
            out = post_solve(client, payload)
            t1 = time.time()
            ans = out.get("answer", "")
            succ = success_contains(ex.get("expected_contains"), ans)
            steps = out.get("steps", [])

            # features for trace
            feats = extract_features_safe(ex["query"], ex.get("product"), ex["session"])

            # capture SYM sub-trace if present
            sym_step = None
            for s in steps or []:
                if isinstance(s, dict) and s.get("source") == "SYM":
                    sym_step = s
                    break

            # (optional) compute classic episode cost for observability
            cost = episode_cost(steps)

            # persist trace (includes full steps and sym_trace if present)
            trace_row = {
                "id": ex["id"],
                "mode": mode.value,
                "query": ex["query"],
                "product": ex.get("product"),
                "session": ex["session"],
                "features": feats,
                "chosen_action": mode.value,      # classic: fixed by design
                "success": int(succ),
                "reward": float(succ),            # classic: no alpha penalty
                "cost": round(cost, 4),
                "alpha": 0.0,
                "latency_ms": round((t1 - t0) * 1000.0, 2),
                "answer": ans,
                "steps": steps,
                "sym_trace": sym_step.get("sym_trace") if sym_step else None,
            }
            with trace_fp.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trace_row) + "\n")

            rows.append({
                "id": ex["id"],
                "type": ex.get("type"),
                "query": ex["query"],
                "product": ex.get("product"),
                "session": ex.get("session"),
                "expected_contains": ex.get("expected_contains"),
                "mode": mode.value,
                "latency_ms": round((t1 - t0) * 1000.0, 2),
                "steps": len(steps or []),
                "success": int(succ),
                "answer": ans,
            })
        fp = ART / f"eval_{mode.value}_{rid}.csv"
        with fp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        per_mode_paths.append(fp)
        print(f"Wrote {fp}")

    # 2) RL on the same tests
    rows: List[Dict[str, Any]] = []
    for ex in tests:
        payload = {
            "query": ex["query"],
            "product": ex.get("product"),
            "session": ex.get("session", "default"),
        }
        t0 = time.time()
        out = post_solve_rl(client, payload)
        t1 = time.time()
        ans = out.get("answer", "")
        succ = success_contains(ex.get("expected_contains"), ans)
        steps = out.get("steps", [])

        # features for trace
        feats = extract_features_safe(ex["query"], ex.get("product"), ex.get("session", "default"))

        # infer chosen action from steps (fallback to answer text)
        action_guess = guess_action_from_steps(steps) or guess_action_from_answer(ans)

        # Day 4: cost-aware reward
        alpha = float(os.getenv("RL_ALPHA", "0.0"))
        cost = episode_cost(steps)
        reward = float(succ) - alpha * (cost / 2.0)  # simple normalization

        # capture SYM sub-trace if present
        sym_step = None
        for s in steps or []:
            if isinstance(s, dict) and s.get("source") == "SYM":
                sym_step = s
                break

        # persist trace
        trace_row = {
            "id": ex["id"],
            "mode": "RL",
            "query": ex["query"],
            "product": ex.get("product"),
            "session": ex.get("session", "default"),
            "features": feats,
            "chosen_action": action_guess,
            "success": int(succ),
            "reward": round(reward, 4),
            "cost": round(cost, 4),
            "alpha": alpha,
            "latency_ms": round((t1 - t0) * 1000.0, 2),
            "answer": ans,
            "steps": steps,
            "sym_trace": sym_step.get("sym_trace") if sym_step else None,
        }
        with trace_fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace_row) + "\n")

        rows.append({
            "id": ex["id"],
            "type": ex.get("type"),
            "query": ex["query"],
            "product": ex.get("product"),
            "session": ex.get("session"),
            "expected_contains": ex.get("expected_contains"),
            "mode": "RL",
            "latency_ms": round((t1 - t0) * 1000.0, 2),
            "steps": len(steps or []),
            "success": int(succ),
            "answer": ans,
        })
    fp_rl = ART / f"eval_RL_{rid}.csv"
    with fp_rl.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    per_mode_paths.append(fp_rl)
    print(f"Wrote {fp_rl}")

    # 3) Join & summarize
    joined_rows: List[Dict[str, Any]] = []
    for p in per_mode_paths:
        with p.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                row["latency_ms"] = float(row["latency_ms"])
                row["steps"] = int(row["steps"])
                row["success"] = int(row["success"])
                joined_rows.append(row)

    joined_fp = ART / f"eval_joined_{rid}.csv"
    with joined_fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(joined_rows[0].keys()))
        w.writeheader(); w.writerows(joined_rows)
    print(f"Wrote {joined_fp}")

    # summary
    from statistics import median
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for r in joined_rows:
        by_mode.setdefault(r["mode"], []).append(r)
    summary_rows: List[Dict[str, Any]] = []
    for mode, rows_m in by_mode.items():
        n = len(rows_m)
        acc = sum(r["success"] for r in rows_m) / max(n, 1)
        lats = [r["latency_ms"] for r in rows_m]
        steps_ct = [r["steps"] for r in rows_m]
        summary_rows.append({
            "mode": mode,
            "n": n,
            "accuracy": round(acc, 4),
            "avg_latency_ms": round(sum(lats) / max(n, 1), 2),
            "median_latency_ms": round(median(lats), 2),
            "avg_steps": round(sum(steps_ct) / max(n, 1), 2),
        })
    summary_fp = ART / f"eval_summary_{rid}.csv"
    with summary_fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"Wrote {summary_fp}")

    # 4) Acceptance check
    expected_rows = len(load_tests()) * 5  # BASE, MEM, SYM, MEMSYM, RL
    actual_rows = len(joined_rows)
    if actual_rows != expected_rows:
        raise SystemExit(f"❌ Acceptance failed: expected {expected_rows} joined rows, got {actual_rows}")
    print(f"✅ Acceptance OK: {actual_rows} rows (tests × 5 modes)")
    print(f"📝 Traces: {trace_fp}")
