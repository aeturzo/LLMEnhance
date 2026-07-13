#!/usr/bin/env python3
"""
End-to-end backend smoke test — exercises imports, carbon service, FastAPI
routes, and (if OPENAI_API_KEY is set) a live one-shot LLM-backed /solve call.

Run on the host machine, NOT the Cowork sandbox (which has no fastapi/openai):

    cd /Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain
    source temp_env.sh        # exports OPENAI_API_KEY
    python scripts/smoke_test_backend.py

Exit code 0 = all checks passed; non-zero = at least one check failed.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def check(name: str, cond: bool, detail: str = "") -> int:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    return 0 if cond else 1


def main() -> int:
    errors = 0
    print(f"\n=== Backend smoke test (Python {sys.version.split()[0]}) ===\n")

    # 1. Carbon service (pure Python — no external deps)
    print("1. Carbon service direct calls")
    from backend.services.carbon_calculation_service import CarbonCalculationService
    svc = CarbonCalculationService()
    cases = [
        ("apple_iphone15_pro_128gb", 66.0, 5.0),
        ("fairphone_4", 43.0, 5.0),
        ("dell_xps14_da14260", 311.0, 5.0),
        ("daikin_altherma_m_hw_260l", 7420.0, 5.0),
        ("generic_bev_pack_60kwh", None, None),
        ("lexmark_mx431adn", None, None),
    ]
    for pid, target, tol in cases:
        sc = {"use_bootstrap_estimates": True} if pid == "lexmark_mx431adn" else None
        r = svc.calculate(pid, sc)
        ok = r.status == "complete"
        detail = f"total={r.total_kg_co2e:.2f}"
        if target is not None and r.total_kg_co2e is not None:
            err_pct = abs(r.total_kg_co2e - target) / target * 100
            detail += f"  vs declared {target}: {err_pct:+.2f}% (tol ±{tol}%)"
            ok = ok and err_pct <= tol
        errors += check(f"carbon.calculate({pid})", ok, detail)

    # 2. FastAPI router smoke
    print("\n2. FastAPI router smoke")
    try:
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)

        r = client.get("/carbon/products")
        errors += check("GET /carbon/products → 200", r.status_code == 200)
        products = r.json() if r.status_code == 200 else []
        expected = {"apple_iphone15_pro_128gb", "fairphone_4", "dell_xps14_da14260",
                    "daikin_altherma_m_hw_260l", "generic_bev_pack_60kwh", "lexmark_mx431adn"}
        missing = expected - set(products)
        errors += check("all 6 products discoverable", not missing,
                        f"missing: {sorted(missing)}" if missing else f"{len(products)} found")

        body = {"product_id": "apple_iphone15_pro_128gb", "include_trace": False,
                "include_ontology_sidecar": False}
        r = client.post("/carbon/calculate", json=body)
        errors += check("POST /carbon/calculate → 200", r.status_code == 200,
                        f"status={r.status_code}")

        # /solve symbolic check
        r = client.post("/solve", json={"mode": "BASE", "query": "ping", "session": "smoke"})
        errors += check("POST /solve (BASE) → 200", r.status_code == 200,
                        f"status={r.status_code}")
    except ImportError as exc:
        errors += check("FastAPI router import", False, f"{exc}")

    # 3. Live OpenAI sanity (optional)
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("LLM_DISABLED"):
        print("\n3. Live OpenAI key sanity (gpt-4o-mini)")
        try:
            from openai import OpenAI
            client = OpenAI()
            t0 = time.time()
            resp = client.chat.completions.create(
                model=os.environ.get("GEN_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": "Reply with the single word PONG."}],
                max_tokens=5, temperature=0,
            )
            dt = time.time() - t0
            answer = (resp.choices[0].message.content or "").strip()
            errors += check(f"OpenAI chat.completions → '{answer}' in {dt:.1f}s",
                            "pong" in answer.lower())
        except ImportError:
            errors += check("openai package", False, "pip install openai")
        except Exception as exc:
            errors += check(f"OpenAI live call", False, f"{type(exc).__name__}: {exc}")
    else:
        print("\n3. Live OpenAI key sanity — SKIPPED (no OPENAI_API_KEY in env)")

    # Summary
    print(f"\n{'='*60}\nSummary: {'ALL PASS' if errors == 0 else f'{errors} FAILURE(S)'}\n")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
