#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 9 — Robustness: paraphrase/noise + memory size sweep
- Augment queries: synswap, noise, numshift (+/-)
- Memory scale: duplicate seed docs to reach target sizes
- Modes evaluated: BASE, MEM, RL
- Outputs: artifacts/robustness_{stamp}.csv (+ optional details)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import time
from typing import Any, Dict, List, Tuple

from fastapi.testclient import TestClient
from backend.config.run_modes import parse_modes_csv
from backend.eval.augment import augment
from backend.config.run_modes import RunMode
from backend.services.policy_costs import episode_cost
from backend.services.symbolic_reasoning_service import build_reasoner
from backend.services import memory_service
from backend.main import app

ROOT = pathlib.Path(__file__).parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True, parents=True)

# ---------- helpers reused from run_eval_all ----------
def load_jsonl(path: pathlib.Path) -> List[dict]:
    if not path.exists(): return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def tests_path_for_domain(domain: str) -> pathlib.Path:
    if domain == "textiles":
        return ROOT / "tests" / "dpp_textiles" / "tests.jsonl"
    if domain == "viessmann":
        return ROOT / "tests" / "dpp_viessmann" / "tests.jsonl"
    if domain == "lexmark":
        return ROOT / "tests" / "dpp_lexmark" / "tests.jsonl"
    return ROOT / "tests" / "dpp_rl" / "tests.jsonl"

def seed_path_for_domain(domain: str) -> pathlib.Path:
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
    p = tests_path_for_domain(domain)
    rows = load_jsonl(p)
    for i, ex in enumerate(rows, 1):
        for k in ("id","query","session","type","expected_contains"):
            if k not in ex:
                raise KeyError(f"Missing '{k}' in tests on line {i}")
    return rows

def success_contains(expected: str | None, answer: str | None) -> int:
    if not expected: return 0
    return 1 if str(expected).lower() in (answer or "").lower() else 0

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

# ---------- memory scaling ----------
def reset_and_seed_memory(domain: str) -> int:
    """Reset memory store and seed from domain's seed file. Return count."""
    try:
        memory_service.reset_storage()
    except Exception:
        pass
    seed_file = seed_path_for_domain(domain)
    n = 0
    if seed_file.exists():
        with seed_file.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                row = json.loads(s)
                memory_service.add_memory(row.get("session","default"), row["memory"])
                n += 1
    return n

def grow_memory_to(target: int, domain: str, sessions: List[str]) -> int:
    """
    Duplicate existing memories to reach target entries.
    We round-robin over sessions to keep distribution realistic.
    """
    # snapshot current entries to clone
    base_file = seed_path_for_domain(domain)
    base_docs = []
    if base_file.exists():
        for line in open(base_file, "r", encoding="utf-8"):
            s = line.strip()
            if not s: continue
            base_docs.append(json.loads(s)["memory"])
    if not base_docs:
        # fallback generic fillers
        base_docs = [f"Synthetic memory filler #{i} for domain={domain}" for i in range(20)]

    # count current
    try:
        n_current = len(memory_service._svc().entries)  # type: ignore
    except Exception:
        n_current = 0

    i = 0
    while n_current < target:
        mem = base_docs[i % len(base_docs)]
        sess = sessions[n_current % max(1, len(sessions))]
        memory_service.add_memory(sess, f"{mem} [dup {n_current}]")
        n_current += 1
        i += 1
    return n_current

# ---------- evaluation ----------
def eval_modes(client: TestClient, tests: List[Dict[str, Any]],
               modes: Tuple[RunMode, ...]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], int]:
    """
    Return (accuracy_by_mode, avg_latency_by_mode, avg_cost_by_mode, n)
    """
    acc: Dict[str, float] = {}
    lat: Dict[str, float] = {}
    cost: Dict[str, float] = {}
    n = len(tests)

    for mode in modes:
        correct = 0
        lats: List[float] = []
        costs: List[float] = []

        for ex in tests:
            payload = {"query": ex["query"], "product": ex.get("product"),
                       "session": ex.get("session","s1")}
            t0 = time.time()
            if mode == RunMode.RL:
                out = post_solve_rl(client, payload)
            else:
                payload["mode"] = mode.value
                out = post_solve(client, payload)
            t1 = time.time()

            ans = out.get("answer","")
            correct += success_contains(ex.get("expected_contains"), ans)
            lats.append((t1 - t0) * 1000.0)
            costs.append(episode_cost(out.get("steps", [])))

        acc[mode.value] = round(correct / max(1, n), 4)
        lat[mode.value] = round(sum(lats) / max(1, n), 2)
        cost[mode.value] = round(sum(costs) / max(1, n), 4)

    return acc, lat, cost, n

# ---------- main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=os.environ.get("DPP_DOMAIN","battery"),
                    choices=("battery","textiles","viessmann","lexmark"))
    ap.add_argument("--aug_kinds", default="synswap,noise,num+1",
                    help="comma list of augment kinds: synswap,noise,num+1,num-1")
    ap.add_argument("--aug_intensities", default="1,2",
                    help="comma list of intensities per kind (e.g., 1,2)")
    ap.add_argument("--mem_targets", default="100,1000,5000",
                    help="comma list of memory sizes for scaling")
    ap.add_argument("--modes", default="BASE,MEM,RL",
                    help="comma list of modes (subset of BASE,MEM,RL)")
    args = ap.parse_args()

    os.environ["DPP_DOMAIN"] = args.domain
    client = TestClient(app)
    # build reasoner once (domain-aware)
    app.state.reasoner = build_reasoner(run_owl_rl=True, domain=args.domain)

    # load canonical tests
    tests_canon = load_tests(args.domain)
    sessions = sorted({ex.get("session","s1") for ex in tests_canon})

    rid = time.strftime("%Y%m%d_%H%M%S")
    out_csv = ART / f"robustness_{rid}.csv"
    detail_csv = ART / f"robustness_detail_{rid}.csv"

    # parse lists
    aug_kinds = [k.strip() for k in args.aug_kinds.split(",") if k.strip()]
    aug_ints = [int(x) for x in args.aug_intensities.split(",") if x.strip()]
    mem_targets = [int(x) for x in args.mem_targets.split(",") if x.strip()]
    modes = parse_modes_csv(args.modes)
    if not modes:
        raise SystemExit(f"No valid modes parsed from --modes='{args.modes}'. Try BASE,MEM,RL")

    # open detail file
    with detail_csv.open("w", newline="", encoding="utf-8") as fd:
        wdet = csv.DictWriter(fd, fieldnames=[
            "setting","param","mode","id","query","answer","success","latency_ms","cost"])
        wdet.writeheader()

    rows_out: List[Dict[str, Any]] = []

    # ----- A) Query robustness (augmentations) -----
    for kind in aug_kinds:
        for inten in aug_ints:
            # produce augmented copy of tests
            tests_aug: List[Dict[str, Any]] = []
            for ex in tests_canon:
                q2 = augment(ex["query"], kind=kind, intensity=inten)
                ex2 = dict(ex)
                ex2["id"] = f"{ex['id']}_{kind}{inten}"
                ex2["query"] = q2
                tests_aug.append(ex2)

            # seed memory fresh for this run
            reset_and_seed_memory(args.domain)

            # evaluate
            acc, lat, cost, n = eval_modes(client, tests_aug, modes)

            for m in modes:
                rows_out.append({
                    "setting": "AUGMENT",
                    "param": f"{kind}@{inten}",
                    "mode": m.value,
                    "n": n,
                    "accuracy": acc[m.value],
                    "avg_latency_ms": lat[m.value],
                    "avg_cost": cost[m.value],
                })

            # also dump per-example detail for the first mode (RL preferred)
            prefer = RunMode.RL if RunMode.RL in modes else modes[0]
            for ex in tests_aug:
                payload = {"query": ex["query"], "product": ex.get("product"),
                           "session": ex.get("session","s1")}
                t0 = time.time()
                out = post_solve_rl(client, payload) if prefer == RunMode.RL else post_solve(client, {**payload, "mode": prefer.value})
                t1 = time.time()
                with detail_csv.open("a", newline="", encoding="utf-8") as fd:
                    wdet = csv.DictWriter(fd, fieldnames=["setting","param","mode","id","query","answer","success","latency_ms","cost"])
                    wdet.writerow({
                        "setting":"AUGMENT","param":f"{kind}@{inten}","mode":prefer.value,"id":ex["id"],
                        "query":ex["query"],"answer":out.get("answer",""),
                        "success":success_contains(ex.get("expected_contains"), out.get("answer","")),
                        "latency_ms":round((t1 - t0)*1000.0,2),"cost":round(episode_cost(out.get("steps",[])),4)
                    })

    # ----- B) Memory scaling -----
    for tgt in mem_targets:
        # fresh seed then grow
        reset_and_seed_memory(args.domain)
        grow_memory_to(tgt, args.domain, sessions)

        acc, lat, cost, n = eval_modes(client, tests_canon, modes)
        for m in modes:
            rows_out.append({
                "setting": "MEMSCALE",
                "param": f"mem@{tgt}",
                "mode": m.value,
                "n": n,
                "accuracy": acc[m.value],
                "avg_latency_ms": lat[m.value],
                "avg_cost": cost[m.value],
            })

    # write summary CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["setting","param","mode","n","accuracy","avg_latency_ms","avg_cost"])
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {out_csv}")
    print(f"Wrote {detail_csv}")
