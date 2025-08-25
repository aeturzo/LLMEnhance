# backend/api/solve_rl.py
from __future__ import annotations

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter

from backend.services import memory_service, search_service
from backend.services.policy_costs import COSTS, episode_cost

# Optional: symbolic answerer (graceful if missing)
try:
    from backend.services.symbolic_reasoning_service import answer_symbolic, sym_fire_flags  # type: ignore
except Exception:
    answer_symbolic = None  # type: ignore
    sym_fire_flags = lambda q, p: False  # type: ignore

# Optional: Day-2 features (graceful if missing)
try:
    from backend.services.policy_features import extract_features  # type: ignore
except Exception:
    extract_features = None  # type: ignore

router = APIRouter()


class SolveRLRequest(BaseModel):
    query: Optional[str] = Field(default=None)
    q: Optional[str] = Field(default=None)
    product: Optional[str] = None
    session: Optional[str] = "s1"


def _pick_query(req: SolveRLRequest) -> str:
    return (req.query or req.q or "").strip()


# ---------- Tiny cost-aware router baseline (reacts to RL_ALPHA) -------------
def _safe_features(query: str, product: Optional[str], session: str) -> Dict[str, float | int]:
    if extract_features is None:
        has_num = int(any(ch.isdigit() for ch in query))
        return {
            "len_query": len(query),
            "has_number": has_num,
            "has_product": int(bool(product)),
            "mem_top": 0.0,
            "mem_max3": 0.0,
            "search_top": 0.0,
            "search_max3": 0.0,
            "sym_fired": int(sym_fire_flags(query, product)),
        }
    try:
        return extract_features(query, product, session) or {}
    except Exception:
        return {
            "len_query": len(query),
            "has_number": int(any(ch.isdigit() for ch in query)),
            "has_product": int(bool(product)),
            "mem_top": 0.0,
            "mem_max3": 0.0,
            "search_top": 0.0,
            "search_max3": 0.0,
            "sym_fired": int(sym_fire_flags(query, product)),
        }


def _prob_success_by_action(feats: Dict[str, float | int]) -> Dict[str, float]:
    # Heuristic probabilities from Day-2 features (bounded to [0,1])
    clamp = lambda x: max(0.0, min(1.0, float(x)))
    p_mem = clamp(feats.get("mem_top", 0.0))
    p_search = clamp(feats.get("search_top", 0.0))
    p_sym = 0.85 if int(feats.get("sym_fired", 0)) == 1 else 0.15
    p_base = 0.15

    # Union (independence assumption) for MEMSYM
    p_memsym = 1.0 - (1.0 - p_mem) * (1.0 - p_sym)

    return {
        "BASE": p_base,
        "MEM": p_mem,
        "SYM": p_sym,
        "SEARCH": p_search,
        "MEMSYM": p_memsym,
    }


def _expected_reward(action: str, alpha: float) -> float:
    # Cost normalization roughly to [0,1]; tune if you change COSTS.
    # For composite actions, sum the constituent step costs.
    if action == "MEMSYM":
        raw_cost = COSTS.get("MEM", 0.0) + COSTS.get("SYM", 0.0)
    else:
        raw_cost = COSTS.get(action, 0.0)
    return -alpha * (raw_cost / 2.0)  # success prob added outside


def _choose_action(feats: Dict[str, float | int], alpha: float) -> str:
    ps = _prob_success_by_action(feats)  # estimated success per action
    # Score(action) = p_success(action) + ( - alpha * normalized_cost(action) )
    scores = {a: ps[a] + _expected_reward(a, alpha) for a in ps.keys()}
    # Break ties by lower cost, then by a stable name order
    def act_cost(a: str) -> float:
        return (COSTS.get("MEM", 0) + COSTS.get("SYM", 0)) if a == "MEMSYM" else COSTS.get(a, 0)
    best = sorted(scores.items(), key=lambda kv: (-kv[1], act_cost(kv[0]), kv[0]))[0][0]
    return best


def _compose_answer_for(action: str, query: str, product: Optional[str], session: str) -> Dict[str, Any]:
    """
    Actually run the chosen action and compose a response with a proper steps[] list.
    """
    steps: List[Dict[str, Any]] = []
    parts: List[str] = []
    sources: List[Dict[str, Any]] = []

    # Helpers
    def add_mem():
        hits = []
        try:
            hits = memory_service.retrieve(session_id=session, query=query, top_k=3)
        except Exception:
            pass
        if hits:
            parts.append(f"Memory: {hits[0].content}")
            steps.append({"source": "MEM", "score": getattr(hits[0], "score", None)})
            for h in hits[:2]:
                sources.append({"type": "memory", "score": getattr(h, "score", None), "snippet": h.content})
        return hits

    def add_sym():
        sym = None
        if answer_symbolic is not None:
            try:
                sym = answer_symbolic(query, product, session)
            except Exception:
                sym = None
        if sym:
            parts.append(sym.text)
            steps.append({
                "source": "SYM",
                "sym_trace": {
                    "product": getattr(sym.trace, "product", product),
                    "asserted": getattr(sym.trace, "asserted", []),
                    "inferred": getattr(sym.trace, "inferred", []),
                    "rules_fired": getattr(sym.trace, "rules_fired", []),
                },
                "evidence": sym.evidence,
            })
        return sym

    def add_search():
        hits = []
        search_q = f"{product or ''} {query}".strip() if product else query
        try:
            hits = search_service.search(query_text=search_q, top_k=3)
        except Exception:
            pass
        if hits:
            parts.append(f"Search: {hits[0].text}")
            steps.append({"source": "SEARCH"})
            for h in hits[:2]:
                sources.append({"type": "search", "score": getattr(h, "score", None), "snippet": h.text})
        return hits

    # Execute per action
    if action == "BASE":
        add_search()  # same behavior as BASE in /solve
    elif action == "MEM":
        add_mem()
    elif action == "SYM":
        add_sym()
    elif action == "SEARCH":
        add_search()
    elif action == "MEMSYM":
        # Order: MEM then SYM; if both fail, fall back to SEARCH
        mh = add_mem()
        sh = add_sym()
        if not mh and not sh:
            add_search()
    else:
        # Unknown => safest low-cost fallback
        add_search()

    if not parts:
        parts.append("No result found.")
        steps.append({"source": "BASE"})

    return {
        "answer": " ".join(parts),
        "steps": steps,
        "sources": sources,
    }


@router.post("/solve_rl")
def solve_rl(req: SolveRLRequest):
    text = _pick_query(req)
    session = req.session or "s1"
    product = (req.product or "").strip() or None

    if not text:
        # Do not 422 to avoid breaking batch jobs
        return {
            "answer": "No result found.",
            "steps": [{"source": "BASE"}],
            "sources": [],
            "chosen_action": "BASE",
            "alpha": float(os.getenv("RL_ALPHA", "0.0")),
            "session": session,
            "product": product,
        }

    alpha = float(os.getenv("RL_ALPHA", "0.0"))
    feats = _safe_features(text, product, session)
    action = _choose_action(feats, alpha)
    out = _compose_answer_for(action, text, product, session)

    # Attach bookkeeping for traces
    out.update({
        "chosen_action": action,
        "alpha": alpha,
        "session": session,
        "product": product,
    })
    return out
