#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 5 — Faithfulness & metrics
- For each test that is CORRECT under a chosen mode (default: MEMSYM),
  * Memory knockout: remove memory path and measure drop (success delta).
  * Rule ablation: disable each rule and measure drop on SYM answer.

Outputs:
  artifacts/faithfulness_{stamp}.csv with columns:
    id, type, used_mem, used_sym, delta_after_knockout,
    delta_rule_R1_BATTERY_SAFETY, delta_rule_R2_WIRELESS_COMPLIANCE, delta_rule_R3_ROHS
Also prints a small summary to stdout.
"""
from __future__ import annotations
import json, time, pathlib, csv
from typing import Dict, Any, List

from fastapi.testclient import TestClient
from backend.main import app
from backend.eval.faithfulness import knockout_memory_then_answer, disable_rule_then_answer

ROOT = pathlib.Path(__file__).parent
ART = ROOT / "artifacts"; ART.mkdir(exist_ok=True, parents=True)
DATA = ROOT / "tests" / "dpp_rl" / "tests.jsonl"

RULE_IDS = ["R1_BATTERY_SAFETY","R2_WIRELESS_COMPLIANCE","R3_ROHS"]

def load_tests() -> List[Dict[str, Any]]:
    with open(DATA, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def success_contains(expected: str | None, answer: str | None) -> int:
    if not expected:
        return 0
    return 1 if str(expected).lower() in (answer or "").lower() else 0

if __name__ == "__main__":
    client = TestClient(app)
    rid = time.strftime("%Y%m%d_%H%M%S")
    tests = load_tests()
    MODE = "MEMSYM"  # evaluate faithfulness on this mode’s correct cases

    # Identify which tests are correct under MODE
    correct_ids = set()
    used_mem_flags = {}
    used_sym_flags = {}
    for ex in tests:
        out = client.post("/solve", json={
            "query": ex["query"], "product": ex.get("product"),
            "session": ex.get("session","s1"), "mode": MODE
        }).json()
        ans = out.get("answer","")
        if success_contains(ex.get("expected_contains"), ans):
            correct_ids.add(ex["id"])
            steps = out.get("steps") or []
            used_mem_flags[ex["id"]] = int(any(s.get("source")=="MEM" for s in steps if isinstance(s, dict)))
            used_sym_flags[ex["id"]] = int(any(s.get("source")=="SYM" for s in steps if isinstance(s, dict)))

    rows: List[Dict[str, Any]] = []
    for ex in tests:
        if ex["id"] not in correct_ids:
            continue

        q, prod, sess = ex["query"], ex.get("product"), ex.get("session","s1")
        base_ans = client.post("/solve", json={
            "query": q, "product": prod, "session": sess, "mode": MODE
        }).json()
        base_succ = success_contains(ex.get("expected_contains"), base_ans.get("answer",""))

        # Memory knockout
        ko = knockout_memory_then_answer(q, prod, sess, MODE)
        ko_succ = success_contains(ex.get("expected_contains"), ko.get("answer",""))
        delta_mem = max(0, base_succ - ko_succ)

        # Rule ablations on SYM path
        deltas_rules = {}
        for rid_one in RULE_IDS:
            abl = disable_rule_then_answer(rid_one, q, prod, sess, "SYM")
            abl_succ = success_contains(ex.get("expected_contains"), abl.get("answer",""))
            deltas_rules[rid_one] = max(0, base_succ - abl_succ)

        row = {
            "id": ex["id"],
            "type": ex.get("type"),
            "used_mem": used_mem_flags.get(ex["id"], 0),
            "used_sym": used_sym_flags.get(ex["id"], 0),
            "delta_after_knockout": delta_mem,
        }
        for rid_one in RULE_IDS:
            row[f"delta_rule_{rid_one}"] = deltas_rules.get(rid_one, 0)
        rows.append(row)

    fp = ART / f"faithfulness_{rid}.csv"
    with fp.open("w", newline="", encoding="utf-8") as f:
        cols = ["id","type","used_mem","used_sym","delta_after_knockout"] + [f"delta_rule_{r}" for r in RULE_IDS]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {fp}")

    # Console summary
    if rows:
        n = len(rows)
        mem_help = sum(r["delta_after_knockout"] > 0 for r in rows) / n
        sym_help = sum(any(r[f"delta_rule_{rid}"] > 0 for rid in RULE_IDS) for r in rows) / n
        print(f"Faithfulness summary on {n} correct cases of {MODE}:")
        print(f"  - Memory causally helped in {mem_help:.1%} of correct cases (knockout drop > 0).")
        print(f"  - Symbolic rules causally helped in {sym_help:.1%} of correct cases (any rule ablation drop > 0).")
    else:
        print("No correct cases under MEMSYM; nothing to evaluate for faithfulness.")
