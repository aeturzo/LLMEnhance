# backend/eval/faithfulness.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.symbolic_reasoning_service import disable_rules, enable_all_rules

client = TestClient(app)

def _solve(query: str, product: Optional[str], session: str, mode: str) -> Dict[str, Any]:
    resp = client.post("/solve", json={"query": query, "product": product, "session": session, "mode": mode})
    resp.raise_for_status()
    return resp.json()

def knockout_memory_then_answer(query: str, product: Optional[str], session: str, mode: str = "MEMSYM") -> Dict[str, Any]:
    """
    Simulate a memory knockout by forcing SYM-only answer when mode was MEM or MEMSYM.
    """
    if mode.upper() == "MEM":
        return _solve(query, product, session, "BASE")  # no MEM path
    if mode.upper() == "MEMSYM":
        return _solve(query, product, session, "SYM")
    return _solve(query, product, session, mode)

def disable_rule_then_answer(rule_id: str, query: str, product: Optional[str], session: str, mode: str = "SYM") -> Dict[str, Any]:
    """
    Disable one symbolic rule, answer, then re-enable all rules.
    """
    disable_rules([rule_id])
    try:
        return _solve(query, product, session, mode)
    finally:
        enable_all_rules()
