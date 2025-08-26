#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 1+2+4+6+8 — Experiment spine (multi-domain) + traces
- Domains: battery (default), textiles, viessmann, lexmark
- Modes: BASE, MEM, SYM, MEMSYM, ROUTER, ADAPTIVERAG, RL
- Artifacts:
    artifacts/eval_{MODE}_{stamp}.csv
    artifacts/eval_joined_{stamp}.csv
    artifacts/eval_summary_{stamp}.csv
    artifacts/trace_{stamp}.jsonl
- Acceptance: joined rows == (#tests × 7 modes)
"""
from __future__ import annotations

import os
import csv
import json
import pathlib
import time
import random
import argparse
from typing import List, Dict, Any

from fastapi.testclient import TestClient

from backend.config.run_modes import RunMode
from backend.services.symbolic_reasoning_service import build_reasoner
from backend.services import memory_service
from backend.services.policy_costs import episode_cost
from backend.main import app

ROOT = pathlib.Path(__file__).parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True, parents=True)


# ---------------- Determinism ----------------
def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
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


# ---------------- IO helpers ----------------
def seed_memory(seed_file: pathlib.Path) -> None:
    """Load per-domain memory seeds if present."""
    if not seed_file.exists():
        print(f"[seed_memory] No seed file at {seed_file} (skipping)")
        return
    with seed_file.open("r", encoding="utf-8") as f:
        n = 0
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            memory_service.add_memory(row.get("session", "default"), row["memory"])
            n += 1
    print(f"[seed_memory] Added {n} memories from {seed_file}")


def load_jsonl(path: pathlib.Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def tests_path_for_domain(domain: str) -> pathlib.Path:
    if domain == "textiles":
        return ROOT / "tests" / "dpp_textiles" / "tests.jsonl"
    if domain == "viessmann":
        return ROOT / "tests" / "dpp_viessmann" / "tests.jsonl"
    if domain == "lexmark":
        return ROOT / "tests" / "dpp_lexmark" / "tests.jsonl"
    # default battery
    return ROOT / "tests" / "dpp_rl" / "tests.jsonl"


def seed_path_for_domain(domain: str) -> pathlib.Path:
    """
    Prefer seed_docs.jsonl; fallback to seed_mem.jsonl in the domain folder.
    """
    folder = {
        "textiles":  ROOT / "tests" / "dpp_textiles",
        "viessmann": ROOT / "tests" / "dpp_viessmann",
        "lexmark":   ROOT / "tests" / "dpp_lexmark",
        "battery":   ROOT / "tests" / "dpp_rl",
    }.get(domain, ROOT / "tests" / "dpp_rl")
    p1 = folder / "seed_docs.jsonl"
    p2 = folder / "seed_mem.jsonl"
    return p1 if p1.exists() else p2


def load_tests(domain: str) -> List[Dict[str, Any]]:
    """
    Canonical schema:
      {"id","query","product","session","type","expected_contains"}
    Fallback: tests/combined_eval/combined.jsonl with gold.contains
    """
    canonical = tests_path_for_domain(domain)
    if canonical.exists():
        rows = load_jsonl(canonical)
        for i, ex in enumerate(rows, 1):
            for k in ("id", "query", "session", "type", "expected_contains"):
                if k not in ex:
                    raise KeyError(f"Missing '{k}' in tests on line {i}")
        return rows

    # Fallback legacy
    combined = ROOT / "tests" / "combined_eval" / "combined.jsonl"
    out: List[Dict[str, Any]] = []
    for ex in load_jsonl(combined):
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
        raise SystemExit("No usable tests found.")
    print(f"[WARN] Using fallback combined.jsonl → {len(out)} tests")
    return out


def success_contains(expected: str | None, answer: str | None) -> int:
    if not expected:
        return 0
    return 1 if str(expected).lower() in (answer or "").lower() else 0


# ---- Day-2 features (robust if module missing) ----
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
    src = [s.get("source") for s in steps if isinstance(s, dict)]
    if "MEM" in src and "SYM" in src:
        return "MEMSYM"
    if "SYM" in src:
        return "SYM"
    if "MEM" in src:
        return "MEM"
    if "SEARCH" in src:
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


# ---------------- Main ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--domain",
        default=os.environ.get("DPP_DOMAIN", "battery"),
        choices=("battery", "textiles", "viessmann", "lexmark"),
        help="evaluation domain",
    )
    args = ap.parse_args()
    os.environ["DPP_DOMAIN"] = args.domain  # visible to services

    set_global_seed(42)

    # Domain-aware seeding
    seed_file = seed_path_for_domain(args.domain)
    seed_memory(seed_file)

    # Build reasoner once; reused by SYM via app.state.reasoner
    app.state.reasoner = build_reasoner(run_owl_rl=True, domain=args.domain)
    client = TestClient(app)

    rid = time.strftime("%Y%m%d_%H%M%S")
    tests = load_tests(args.domain)
    trace_fp = ART / f"trace_{rid}.jsonl"

    # Classic + supervised baselines
    CLASSIC = (
        RunMode.BASE,
        RunMode.MEM,
        RunMode.SYM,
        RunMode.MEMSYM,
        RunMode.ROUTER,
        RunMode.ADAPTIVERAG,
    )
    per_mode_paths: List[pathlib.Path] = []

    # ----- Classic / Router / AdaptiveRAG -----
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

            feats = extract_features_safe(ex["query"], ex.get("product"), ex["session"])
            sym_step = next((s for s in steps or [] if isinstance(s, dict) and s.get("source") == "SYM"), None)
            cost = episode_cost(steps)

            # trace line
            trace_row = {
                "id": ex["id"],
                "mode": mode.value,
                "query": ex["query"],
                "product": ex.get("product"),
                "session": ex["session"],
                "features": feats,
                "chosen_action": mode.value,
                "success": int(succ),
                "reward": float(succ),   # classic: no alpha penalty
                "cost": round(cost, 4),
                "alpha": 0.0,
                "latency_ms": round((t1 - t0) * 1000.0, 2),
                "answer": ans,
                "steps": steps,
                "sym_trace": (sym_step or {}).get("sym_trace"),
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
            w.writeheader()
            w.writerows(rows)
        per_mode_paths.append(fp)
        print(f"Wrote {fp}")

    # ----- RL on the same tests -----
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

        feats = extract_features_safe(ex["query"], ex.get("product"), ex.get("session", "default"))
        action_guess = guess_action_from_steps(steps) or guess_action_from_answer(ans)

        alpha = float(os.getenv("RL_ALPHA", "0.0"))
        cost = episode_cost(steps)
        reward = float(succ) - alpha * (cost / 2.0)  # simple normalization

        sym_step = next((s for s in steps or [] if isinstance(s, dict) and s.get("source") == "SYM"), None)

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
            "sym_trace": (sym_step or {}).get("sym_trace"),
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
        w.writeheader()
        w.writerows(rows)
    per_mode_paths.append(fp_rl)
    print(f"Wrote {fp_rl}")

    # ----- Join & summarize -----
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
        w.writeheader()
        w.writerows(joined_rows)
    print(f"Wrote {joined_fp}")

    from statistics import median
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for r in joined_rows:
        by_mode.setdefault(r["mode"], []).append(r)

    summary_rows = []
    for mode, rs in by_mode.items():
        n = len(rs)
        acc = sum(r["success"] for r in rs) / max(n, 1)
        lats = [r["latency_ms"] for r in rs]
        steps_ct = [r["steps"] for r in rs]
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
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {summary_fp}")

    # ----- Acceptance -----
    expected_rows = len(tests) * 7  # 6 classic/baselines + RL
    actual_rows = len(joined_rows)
    if actual_rows != expected_rows:
        raise SystemExit(f"❌ Acceptance failed: expected {expected_rows} joined rows, got {actual_rows}")
    print(f"✅ Acceptance OK: {actual_rows} rows (tests × 7 modes)")
    print(f"📝 Traces: {trace_fp}")
