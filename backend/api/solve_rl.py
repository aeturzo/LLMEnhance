# backend/api/solve_rl.py
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter
import os, time

from backend.services import memory_service, search_service
from backend.services.policy_features import extract_features  # Day-2 features
from backend.services.policy_confidence import confidence_from_features  # Day-10 confidence
from backend.services.symbolic_reasoning_service import (
    answer_symbolic,   # returns SymAnswer(text, evidence=[(s,p,o)...], fired=bool)
    sym_fire_flags,    # quick boolean: does KG have relevant facts/rules for this product?
)

router = APIRouter()

# -------- Request --------

class SolveRLRequest(BaseModel):
    query: Optional[str] = Field(default=None)
    q: Optional[str] = Field(default=None)
    product: Optional[str] = None
    session: Optional[str] = "s1"
    # Optional: allow evaluator to pass success so we can compute reward server-side
    success: Optional[bool] = None

def _pick_query(req: SolveRLRequest) -> str:
    return (req.query or req.q or "").strip()

# -------- Helpers --------

def _compose_mem_search(query: str, product: Optional[str], session: str,
                        feats: Dict[str, float | int]) -> Dict[str, Any]:
    """
    Adaptive MEM/SEARCH composition using feature thresholds.
    """
    t_mem = float(os.getenv("ADAPTIVE_MEM_T", "0.45"))
    t_sch = float(os.getenv("ADAPTIVE_SEARCH_T", "0.35"))
    mem_top = float(feats.get("mem_top", 0.0) or 0.0)
    sch_top = float(feats.get("search_top", 0.0) or 0.0)

    mem_hits: List[Any] = []
    search_hits: List[Any] = []

    parts: List[str] = []
    steps: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    # MEM (gated by threshold)
    if mem_top >= t_mem:
        try:
            mem_hits = memory_service.retrieve(session_id=session, query=query, top_k=3)
        except Exception:
            mem_hits = []
        if mem_hits:
            parts.append(f"Memory: {mem_hits[0].content}")
            top_score = getattr(mem_hits[0], "score", None)
            steps.append({"source": "MEM", "score": top_score})
            for h in mem_hits[:2]:
                sources.append({"type": "memory", "score": getattr(h, "score", None), "snippet": h.content})

    # SEARCH (gated by threshold, or if no MEM contributed)
    if sch_top >= t_sch or not parts:
        search_q = f"{(product or '').strip()} {query}".strip() if product else query
        try:
            search_hits = search_service.search(query_text=search_q, top_k=3)
        except Exception:
            search_hits = []
        if search_hits:
            parts.append(f"Search: {getattr(search_hits[0], 'text', getattr(search_hits[0], 'content', ''))}")
            top_score = getattr(search_hits[0], "score", None)
            steps.append({"source": "SEARCH", "score": top_score})
            for h in search_hits[:2]:
                sources.append({"type": "search", "score": getattr(h, "score", None), "snippet": getattr(h, "text", getattr(h, "content", ""))})

    if not parts:
        parts.append("No result found.")
        steps.append({"source": "BASE"})

    return {"answer": " ".join(parts), "steps": steps, "sources": sources}


def _compose_with_sym(query: str, product: Optional[str], session: str,
                      feats: Dict[str, float | int]) -> Dict[str, Any]:
    """
    Prefer SYM when KG/rules are relevant; optionally blend in MEM/SEARCH if confident.
    """
    out: Dict[str, Any] = {"answer": "", "steps": [], "sources": []}

    # 1) Symbolic first (authoritative & cheap) if fired
    used_sym = False
    sym_trace: Dict[str, Any] | None = None
    if product and sym_fire_flags(query, product):
        try:
            sym = answer_symbolic(query, product, session)
        except Exception:
            sym = None
        if sym and getattr(sym, "fired", False) and getattr(sym, "text", ""):
            used_sym = True
            out["answer"] = sym.text.strip()
            sym_trace = {
                "rules_fired": ["requiresCompliance", "requiresStep"],  # conservative default labels
                "triples": [f"{s} | {p} | {o}" for (s, p, o) in getattr(sym, "evidence", [])] if getattr(sym, "evidence", None) else [],
            }
            # include a minimal verdict hook if your sym service provides it
            out["steps"].append({"source": "SYM", "sym_trace": sym_trace, "proved": getattr(sym, "proved", None), "refuted": getattr(sym, "refuted", None)})

    # 2) If SYM fired, enrich with MEM/SEARCH if strong
    mem_enrich = float(feats.get("mem_top", 0.0) or 0.0) >= float(os.getenv("ADAPTIVE_MEM_T", "0.45"))
    sch_enrich = float(feats.get("search_top", 0.0) or 0.0) >= float(os.getenv("ADAPTIVE_SEARCH_T", "0.35"))
    if used_sym and (mem_enrich or sch_enrich):
        enr = _compose_mem_search(query, product, session, feats)
        if enr.get("answer") and enr["answer"] != "No result found.":
            out["answer"] = f"{out['answer']} {enr['answer']}".strip()
        out["steps"].extend([s for s in enr.get("steps", []) if s.get("source") in ("MEM", "SEARCH")])
        out["sources"].extend(enr.get("sources", []))
        return out

    # 3) If SYM didn’t fire (or no product), fall back to adaptive MEM/SEARCH
    if not used_sym:
        return _compose_mem_search(query, product, session, feats)

    # 4) SYM-only answer
    return out

# -------- RL endpoint --------

@router.post("/solve_rl")
def solve_rl(req: SolveRLRequest):
    t0 = time.perf_counter()

    text = _pick_query(req)
    session = req.session or "s1"
    product = (req.product or None)

    if not text:
        # empty query: return a well-formed abstain-like record
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        alpha = float(os.getenv("RL_ALPHA", "0.3"))
        cost_norm = 0.0
        return {
            "answer": "No result found.",
            "sources": [],
            "steps": [],
            "policy_used": False,
            "session": session,
            "product": product,
            "confidence": 0.0,
            "abstained": False,
            "latency_ms": latency_ms,
            "tokens_est": 0,
            "alpha": alpha,
            "cost_norm": cost_norm,
            "reward": None,
            "reward_if_success": 1.0 - alpha * cost_norm,
            "reward_if_fail": 0.0 - alpha * cost_norm,
        }

    # ---- Features + confidence (cheap, Day-2 + Day-10) ----
    try:
        feats = extract_features(text, product, session) or {}
    except Exception:
        feats = {}

    conf = confidence_from_features(feats)

    # ---- Optional hard abstain via env threshold (Day-10) ----
    abstain_thresh_s = os.getenv("ABSTAIN_AT", "NaN")
    try:
        abstain_thresh = float(abstain_thresh_s)
    except Exception:
        abstain_thresh = float("nan")
    should_abstain = (abstain_thresh == abstain_thresh) and (conf < abstain_thresh)  # NaN check

    if should_abstain:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        alpha = float(os.getenv("RL_ALPHA", "0.3"))
        cost_norm = 0.0  # abstain doesn’t consume tool steps
        return {
            "answer": "ABSTAIN",
            "sources": [],
            "steps": [{"source": "ABSTAIN"}],
            "policy_used": True,
            "session": session,
            "product": product,
            "confidence": conf,
            "abstained": True,
            "latency_ms": latency_ms,
            "tokens_est": 0,
            "alpha": alpha,
            "cost_norm": cost_norm,
            "reward": None,
            "reward_if_success": 1.0 - alpha * cost_norm,
            "reward_if_fail": 0.0 - alpha * cost_norm,
        }

    # ---- Symbolic-aware composition ----
    out = _compose_with_sym(text, product, session, feats)
    out.setdefault("steps", [])
    out.setdefault("sources", [])
    out.setdefault("answer", "No result found.")

    # ---- Telemetry / cost proxy ----
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    steps_count = len(out["steps"])
    # very rough token estimate (kept simple & deterministic)
    tokens_est = max(0, len(out["answer"].split()))
    # choose one normalized cost (steps-based by default)
    steps_norm = min(1.0, steps_count / float(os.getenv("RL_STEPS_BUDGET", "6")))
    cost_norm = steps_norm  # you can switch to latency/tokens if preferred

    # ---- RL reward hook ----
    alpha = float(os.getenv("RL_ALPHA", "0.3"))
    reward = None
    if req.success is not None:
        # If evaluator supplies success, compute reward here
        success_val = 1.0 if bool(req.success) else 0.0
        reward = success_val - alpha * cost_norm

    # Always include counterfactuals for convenience
    reward_if_success = 1.0 - alpha * cost_norm
    reward_if_fail = 0.0 - alpha * cost_norm

    out.update({
        "policy_used": True,
        "session": session,
        "product": product,
        "confidence": conf,
        "abstained": False,
        "latency_ms": latency_ms,
        "tokens_est": tokens_est,
        "alpha": alpha,
        "cost_norm": cost_norm,
        "reward": reward,
        "reward_if_success": reward_if_success,
        "reward_if_fail": reward_if_fail,
    })
    return out
