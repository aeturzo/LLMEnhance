# backend/services/policy_features.py
from __future__ import annotations
from typing import Optional, Dict
from backend.services import memory_service, search_service

# optional import of sym flags; stub to False if not present (wired on Day 3)
try:
    from backend.services.symbolic_reasoning_service import sym_fire_flags  # type: ignore
except Exception:
    def sym_fire_flags(query: str, product: Optional[str]) -> bool:
        return False

def extract_features(query: str, product: Optional[str], session: str) -> Dict[str, float | int]:
    mem = []
    try:
        mem = memory_service.retrieve(session, query, top_k=3)
    except Exception:
        mem = []
    srch = []
    try:
        q = f"{product or ''} {query}".strip()
        srch = search_service.search(q or query, top_k=3)
    except Exception:
        srch = []

    has_num = int(any(ch.isdigit() for ch in query))
    mem_scores = [float(getattr(h, "score", 0.0) or 0.0) for h in mem]
    srch_scores = [float(getattr(h, "score", 0.0) or 0.0) for h in srch]
    feats = {
        "len_query": len(query),
        "has_number": has_num,
        "has_product": int(bool(product)),
        "mem_top": max(mem_scores[:1] or [0.0]),
        "mem_max3": max(mem_scores or [0.0]),
        "search_top": max(srch_scores[:1] or [0.0]),
        "search_max3": max(srch_scores or [0.0]),
        "sym_fired": int(bool(sym_fire_flags(query, product))),
    }
    return feats
