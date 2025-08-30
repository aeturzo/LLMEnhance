#!/usr/bin/env python3
from __future__ import annotations
import os, argparse, json, time, csv, pathlib, random
from typing import List, Dict, Any
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.symbolic_reasoning_service import build_reasoner
from backend.config.run_modes import RunMode

ROOT = pathlib.Path(__file__).parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True, parents=True)

def tests_path_for_domain(domain: str) -> pathlib.Path:
    p = ROOT / "tests" / domain / "tests.jsonl"
    if p.exists(): return p
    legacy = {
        "battery": ROOT / "tests" / "dpp_rl" / "tests.jsonl",
        "lexmark": ROOT / "tests" / "dpp_lexmark" / "tests.jsonl",
        "viessmann": ROOT / "tests" / "dpp_viessmann" / "tests.jsonl",
        "textiles": ROOT / "tests" / "dpp_textiles" / "tests.jsonl",
    }
    return legacy.get(domain, ROOT / "tests" / "dpp_rl" / "tests.jsonl")

def load_jsonl(path: pathlib.Path) -> List[dict]:
    if not path.exists(): return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def post(client: TestClient, payload: dict) -> dict:
    r = client.post("/solve_rl", json=payload)  # selective == abstention logic lives in RL/heuristics
    if r.status_code >= 400:
        raise RuntimeError(f"/solve_rl HTTP {r.status_code}: {r.text}")
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["battery","lexmark","viessmann","textiles"], required=True)
    ap.add_argument("--taus", default="0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95")
    args = ap.parse_args()

    domain = args.domain
    os.environ["DPP_DOMAIN"] = domain
    app.state.reasoner = build_reasoner(run_owl_rl=True, domain=domain)
    client = TestClient(app)

    tests = load_jsonl(tests_path_for_domain(domain))
    if not tests:
        raise SystemExit(f"No tests found for domain={domain}")

    # run once to get per-example confidences (or derive a proxy)
    rows: List[Dict[str, Any]] = []
    for ex in tests:
        out = post(client, {
            "query": ex["query"],
            "product": ex.get("product"),
            "session": ex.get("session","s1"),
        })
        conf = float(out.get("confidence", out.get("prob", out.get("router_conf", 1.0))))  # proxy if missing
        succ = int(str(ex.get("expected_contains","")).lower() in (out.get("answer","")).lower())
        rows.append({"success": succ, "conf": max(0.0, min(1.0, conf))})

    taus = [float(t) for t in args.taus.split(",")]
    rid = time.strftime("%Y%m%d_%H%M%S")
    outp = ART / f"selective_{rid}.csv"
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tau","coverage","risk","accuracy_answered","n","domain"])
        w.writeheader()
        for tau in taus:
            kept = [r for r in rows if r["conf"] >= tau]
            cov = len(kept) / max(1, len(rows))
            if len(kept) == 0:
                w.writerow({"tau":tau,"coverage":0.0,"risk":None,"accuracy_answered":None,"n":len(kept),"domain":domain})
                continue
            acc_ans = sum(r["success"] for r in kept) / len(kept)
            risk = 1.0 - acc_ans
            w.writerow({"tau":tau,"coverage":round(cov,4),"risk":round(risk,4),
                        "accuracy_answered":round(acc_ans,4),"n":len(kept),"domain":domain})
    print(f"Wrote {outp}")

if __name__ == "__main__":
    main()
