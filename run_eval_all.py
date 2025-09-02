#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment spine (Days 1,2,4,6,8)
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
import time
import random
import argparse
from statistics import median
from typing import List, Dict, Any
from pathlib import Path
import pathlib

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
def load_jsonl(path: pathlib.Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")  # keep empty but present
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------- Paths (domain-aware) ----------------
def tests_path_for_domain(domain: str) -> pathlib.Path:
    """
    Prefer the scaled dataset: tests/<domain>/tests.jsonl
    Fallback to legacy dpp_* paths if missing.
    """
    preferred = ROOT / "tests" / domain / "tests.jsonl"
    if preferred.exists():
        return preferred
    legacy_map = {
        "battery":   ROOT / "tests" / "dpp_rl" / "tests.jsonl",
        "textiles":  ROOT / "tests" / "dpp_textiles" / "tests.jsonl",
        "viessmann": ROOT / "tests" / "dpp_viessmann" / "tests.jsonl",
        "lexmark":   ROOT / "tests" / "dpp_lexmark" / "tests.jsonl",
    }
    return legacy_map.get(domain, preferred)


def seed_path_for_domain(domain: str) -> pathlib.Path:
    """
    Prefer seeds next to the scaled dataset: tests/<domain>/{seed_docs.jsonl,seed_mem.jsonl}
    Then try legacy dpp_* folders; finally return a non-existing preferred path.
    """
    folder = ROOT / "tests" / domain
    for name in ("seed_docs.jsonl", "seed_mem.jsonl"):
        p = folder / name
        if p.exists():
            return p

    legacy_folder = {
        "battery":   ROOT / "tests" / "dpp_rl",
        "textiles":  ROOT / "tests" / "dpp_textiles",
        "viessmann": ROOT / "tests" / "dpp_viessmann",
        "lexmark":   ROOT / "tests" / "dpp_lexmark",
    }.get(domain, ROOT / "tests" / "dpp_rl")

    for name in ("seed_docs.jsonl", "seed_mem.jsonl"):
        p = legacy_folder / name
        if p.exists():
            return p

    return folder / "seed_mem.jsonl"  # non-existing → caller prints skip


# ---------------- Memory seeding (robust to schema) ----------------
def seed_memory(seed_file: str | None = None, domain: str | None = None, session: str = "s1"):
    """
    Accept multiple schemas:
      {"text": ...} or {"memory": ...} or {"content": ...} or {"doc": ...}
    """
    def _extract_text(row: dict) -> str:
        return row.get("text") or row.get("memory") or row.get("content") or row.get("doc") or ""

    # resolve path
    candidates = []
    if seed_file:
        candidates.append(Path(seed_file))
    if domain:
        candidates += [
            Path(f"tests/{domain}/seed_mem.jsonl"),
            Path(f"tests/{domain}/seed_docs.jsonl"),
        ]
    candidates += [
        Path("tests/dpp_rl/seed_mem.jsonl"),
        Path("tests/dpp_rl/seed_docs.jsonl"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if not path:
        print("[seed_memory] no seed file found; skipping.")
        return

    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            row = json.loads(t)
            txt = _extract_text(row)
            if not txt:
                continue
            if hasattr(memory_service, "add_memory"):
                memory_service.add_memory(session, txt)
            elif hasattr(memory_service, "add"):
                memory_service.add(session_id=session, content=txt, metadata={})
            elif hasattr(memory_service, "insert"):
                memory_service.insert(session_id=session, content=txt, metadata={})
            elif hasattr(memory_service, "index_texts"):
                memory_service.index_texts(session_id=session, texts=[txt])
            else:
                raise RuntimeError("memory_service has no add/insert/index_texts method")
            n += 1
    print(f"[seed_memory] Added {n} memories from {path}")


# ---------------- Test loading ----------------
def load_tests(domain: str) -> List[Dict[str, Any]]:
    """
    Canonical schema (enforced):
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


# ---- Day-2 features (robust if feature module missing) ----
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

    # define and export domain
    domain = args.domain
    os.environ["DPP_DOMAIN"] = domain  # visible to services

    set_global_seed(42)

    # Domain-aware seeding (new layout preferred, legacy fallback)
    seed_file = seed_path_for_domain(domain)
    seed_memory(seed_file=seed_file, domain=domain)

    # Build reasoner once; reused by SYM via app.state.reasoner
    app.state.reasoner = build_reasoner(run_owl_rl=True, domain=domain)
    client = TestClient(app)

    rid = time.strftime("%Y%m%d_%H%M%S")
    tests = load_tests(domain)
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
                "session_id": ex.get("session"),  # legacy key for some services
                "query": ex["query"],
                "product": ex.get("product"),
                "mode": mode.value,
            }
            t0 = time.time()
            out = post_solve(client, payload)
            t1 = time.time()

            ans = out.get("answer", "") or ""
            succ = success_contains(ex.get("expected_contains"), ans)
            steps = out.get("steps", []) or []

            feats = extract_features_safe(ex["query"], ex.get("product"), ex["session"])
            sym_step = next((s for s in steps if isinstance(s, dict) and s.get("source") == "SYM"), None)
            cost = episode_cost(steps)

            # Trace row
            trace_row = {
                "id": ex["id"],
                "mode": mode.value,
                "domain": domain,
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
                "sources": out.get("sources", []),
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
                "domain": domain,
                "latency_ms": round((t1 - t0) * 1000.0, 2),
                "steps": len(steps),
                "success": int(succ),
                "answer": ans,
            })

        fp = ART / f"eval_{mode.value}_{rid}.csv"
        write_csv(fp, rows)
        per_mode_paths.append(fp)
        print(f"Wrote {fp}")

    # ----- RL on the same tests -----
    rows = []
    for ex in tests:
        payload = {
            "query": ex["query"],
            "product": ex.get("product"),
            "session": ex.get("session", "default"),
        }
        t0 = time.time()
        out = post_solve_rl(client, payload)
        t1 = time.time()

        ans = out.get("answer", "") or ""
        succ = success_contains(ex.get("expected_contains"), ans)
        steps = out.get("steps", []) or []

        feats = extract_features_safe(ex["query"], ex.get("product"), ex.get("session", "default"))
        action_guess = guess_action_from_steps(steps) or guess_action_from_answer(ans)

        alpha = float(os.getenv("RL_ALPHA", "0.0"))
        cost = episode_cost(steps)
        reward = float(succ) - alpha * (cost / 2.0)  # simple normalization

        sym_step = next((s for s in steps if isinstance(s, dict) and s.get("source") == "SYM"), None)

        trace_row = {
            "id": ex["id"],
            "mode": "RL",
            "domain": domain,
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
            "sources": out.get("sources", []),
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
            "domain": domain,
            "latency_ms": round((t1 - t0) * 1000.0, 2),
            "steps": len(steps),
            "success": int(succ),
            "answer": ans,
        })

    fp_rl = ART / f"eval_RL_{rid}.csv"
    write_csv(fp_rl, rows)
    per_mode_paths.append(fp_rl)
    print(f"Wrote {fp_rl}")

    # ----- Join & summarize -----
    joined_rows: List[Dict[str, Any]] = []
    for p in per_mode_paths:
        with p.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                # normalize types
                row["latency_ms"] = float(row.get("latency_ms", 0.0) or 0.0)
                row["steps"] = int(float(row.get("steps", 0) or 0))
                row["success"] = int(float(row.get("success", 0) or 0))
                joined_rows.append(row)

    joined_fp = ART / f"eval_joined_{rid}.csv"
    write_csv(joined_fp, joined_rows)
    print(f"Wrote {joined_fp}")

    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for r in joined_rows:
        by_mode.setdefault(r["mode"], []).append(r)

    summary_rows = []
    for mode, rs in by_mode.items():
        n = len(rs)
        lats = [float(r["latency_ms"]) for r in rs]
        steps_ct = [int(r["steps"]) for r in rs]
        acc = sum(int(r["success"]) for r in rs) / max(n, 1)
        summary_rows.append({
            "mode": mode,
            "n": n,
            "accuracy": round(acc, 4),
            "avg_latency_ms": round(sum(lats) / max(n, 1), 2) if n else 0.0,
            "median_latency_ms": round(median(lats), 2) if n else 0.0,
            "avg_steps": round(sum(steps_ct) / max(n, 1), 2) if n else 0.0,
        })

    summary_fp = ART / f"eval_summary_{rid}.csv"
    write_csv(summary_fp, summary_rows)
    print(f"Wrote {summary_fp}")

    # ----- Acceptance -----
    expected_rows = len(tests) * 7  # 6 classic/baselines + RL
    actual_rows = len(joined_rows)
    if actual_rows != expected_rows:
        raise SystemExit(f"❌ Acceptance failed: expected {expected_rows} joined rows, got {actual_rows}")
    print(f"✅ Acceptance OK: {actual_rows} rows (tests × 7 modes)")
    print(f"📝 Traces: {trace_fp}")
