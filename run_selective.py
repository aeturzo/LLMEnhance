#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 10 — Selective Risk (Uncertainty & Abstention)
- Calls /solve_rl to get confidence (no forced abstain).
- Sweeps thresholds τ in [0.0, 1.0] and computes:
  coverage(τ) = fraction answered (conf >= τ)
  risk(τ)     = errors / answered among those answered
  acc_answered(τ) = correct / answered
- Writes artifacts/selective_{stamp}.csv and a PNG (if matplotlib is available).
"""
from __future__ import annotations

import argparse, csv, json, os, pathlib, time
from typing import Any, Dict, List
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.symbolic_reasoning_service import build_reasoner

ROOT = pathlib.Path(__file__).parent
ART = ROOT / "artifacts"; ART.mkdir(exist_ok=True, parents=True)

def load_jsonl(path: pathlib.Path) -> List[dict]:
    if not path.exists(): return []
    return [json.loads(l) for l in open(path, "r", encoding="utf-8") if l.strip()]

def tests_path_for_domain(domain: str) -> pathlib.Path:
    if domain == "textiles":
        return ROOT / "tests" / "dpp_textiles" / "tests.jsonl"
    if domain == "viessmann":
        return ROOT / "tests" / "dpp_viessmann" / "tests.jsonl"
    if domain == "lexmark":
        return ROOT / "tests" / "dpp_lexmark" / "tests.jsonl"
    return ROOT / "tests" / "dpp_rl" / "tests.jsonl"

def success_contains(expected: str | None, answer: str | None) -> int:
    if not expected: return 0
    return 1 if str(expected).lower() in (answer or "").lower() else 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=os.environ.get("DPP_DOMAIN","battery"),
                    choices=("battery","textiles","viessmann","lexmark"))
    args = ap.parse_args()
    os.environ["DPP_DOMAIN"] = args.domain

    # build reasoner once (domain-aware)
    app.state.reasoner = build_reasoner(run_owl_rl=True, domain=args.domain)
    client = TestClient(app)

    tests = load_jsonl(tests_path_for_domain(args.domain))
    rid = time.strftime("%Y%m%d_%H%M%S")
    out_csv = ART / f"selective_{rid}.csv"

    thresholds = [round(x/100, 2) for x in range(0, 101, 5)]  # 0.00..1.00 step .05
    rows_out: List[Dict[str, Any]] = []

    # turn OFF server-side abstain so we can simulate thresholds here
    os.environ["ABSTAIN_AT"] = "NaN"

    # Collect raw predictions with confidence
    raw: List[Dict[str, Any]] = []
    for ex in tests:
        payload = {"query": ex["query"], "product": ex.get("product"),
                   "session": ex.get("session","s1")}
        out = client.post("/solve_rl", json=payload).json()
        raw.append({
            "id": ex["id"],
            "expected": ex.get("expected_contains"),
            "answer": out.get("answer",""),
            "conf": float(out.get("confidence") or 0.0),
        })

    n = len(raw)
    for tau in thresholds:
        answered = 0; correct = 0; errors = 0
        for r in raw:
            if r["conf"] >= tau:
                answered += 1
                ok = success_contains(r["expected"], r["answer"])
                if ok: correct += 1
                else: errors += 1
        coverage = answered / max(1, n)
        risk = (errors / max(1, answered)) if answered else 0.0
        acc_ans = (correct / max(1, answered)) if answered else 0.0
        rows_out.append({
            "tau": tau,
            "coverage": round(coverage, 4),
            "risk": round(risk, 4),
            "accuracy_answered": round(acc_ans, 4),
            "n": n
        })

    # Write CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tau","coverage","risk","accuracy_answered","n"])
        w.writeheader(); w.writerows(rows_out)
    print(f"Wrote {out_csv}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt  # noqa
        xs = [r["coverage"] for r in rows_out]
        ys = [r["risk"] for r in rows_out]
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Coverage (fraction answered)")
        plt.ylabel("Risk (error rate among answered)")
        plt.title("Selective Risk Curve")
        png = ART / f"selective_{rid}.png"
        plt.savefig(png, bbox_inches="tight", dpi=150)
        print(f"Wrote {png}")
    except Exception as e:
        print(f"[WARN] matplotlib not available or plotting failed: {e}")
