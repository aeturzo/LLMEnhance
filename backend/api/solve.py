# backend/api/solve.py
from __future__ import annotations

import os
import math
from typing import Optional, Literal, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Header

from backend.services import memory_service, search_service  # search_service is used as a fallback
from backend.services.policy_router import RouterModel, MODEL_PATH
from backend.services.symbolic_reasoning_service import answer_symbolic, sym_fire_flags

# NEW: hybrid retriever + context-bound answerer
from backend.retrieval.hybrid import HybridRetriever
from backend.api.answerer_ctx import answer_with_context

router = APIRouter()

# -------------------------
# Request model
# -------------------------

class SolveRequest(BaseModel):
    query: Optional[str] = Field(default=None)
    q: Optional[str] = Field(default=None)
    product: Optional[str] = None
    session: Optional[str] = "s1"
    mode: Optional[Literal["BASE", "MEM", "SEARCH", "SYM", "MEMSYM", "ROUTER", "ADAPTIVERAG","RAG_BASE","SYM_ONLY"]] = "BASE"


def _pick_query(req: SolveRequest) -> str:
    return (req.query or req.q or "").strip()

# -------------------------
# Router model loader
# -------------------------

_ROUTER: RouterModel | None = None

def _get_router() -> RouterModel | None:
    global _ROUTER
    if _ROUTER is not None:
        return _ROUTER
    try:
        _ROUTER = RouterModel.load(MODEL_PATH)
    except Exception:
        _ROUTER = None
    return _ROUTER

# -------------------------
# Canonicalize step sources for costing
# -------------------------

_CANON_SOURCES = {"BASE","MEM","SEARCH","SYM","COMPOSE"}
_ALIASES = {"RAG":"SEARCH","ADAPTIVERAG":"SEARCH","ROUTER":"COMPOSE","RL":"COMPOSE"}

def _canon_source(s: str | None) -> str:
    s = (s or "").upper()
    if s in _CANON_SOURCES:
        return s
    return _ALIASES.get(s, "COMPOSE")

# -------------------------
# Feature extraction for routing
# -------------------------

def _safe_features(query: str, product: Optional[str], session: str) -> Dict[str, float | int]:
    """
    Best-effort feature extraction for router and adaptive logic.
    Falls back to simple heuristics if policy_features isn't available.
    """
    try:
        from backend.services.policy_features import extract_features  # type: ignore
        feats = extract_features(query, product, session) or {}
        clean: Dict[str, float | int] = {}
        for k, v in feats.items():
            if isinstance(v, (int, float)):
                clean[k] = v
            elif isinstance(v, bool):
                clean[k] = int(v)
        return clean
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
            "sym_fired": int(sym_fire_flags(query, product)),
        }

# -------------------------
# Confidence computation
# -------------------------

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5 if x == 0 else (0.0 if x < 0 else 1.0)

def _extract_top_score(steps: List[Dict[str, Any]], source_name: str) -> Optional[float]:
    """
    Find the best/first score for the given step source (e.g., 'MEM' or 'SEARCH').
    """
    best = None
    for st in steps:
        if _canon_source(st.get("source")) == source_name.upper():
            sc = st.get("score")
            if sc is None:
                continue
            try:
                scf = float(sc)
            except Exception:
                continue
            if best is None or scf > best:
                best = scf
    return best

def _conf_from_mem(score: Optional[float]) -> Optional[float]:
    if score is None:
        return None
    # Assume memory scores are already 0..1-ish; clamp just in case.
    return max(0.0, min(1.0, float(score)))

def _conf_from_search(score: Optional[float]) -> Optional[float]:
    if score is None:
        return None
    # Cross-encoder scores are roughly logit-ish; squash to 0..1
    return _sigmoid(float(score) / 2.0)

def _conf_from_sym(step: Dict[str, Any]) -> Optional[float]:
    if _canon_source(step.get("source")) != "SYM":
        return None
    proved = step.get("proved")
    refuted = step.get("refuted")
    if proved is True:
        return 1.0
    if refuted is True:
        return 0.0
    # No explicit verdict
    return 0.5

def _attach_confidence(mode: str, out: Dict[str, Any]) -> float:
    """
    Derive a confidence in [0,1] from the evidence we have in steps/sources.
    - MEM: memory top score
    - SEARCH/BASE: search top score
    - SYM: 1.0 proved / 0.0 refuted / 0.5 unknown
    - MEMSYM/ADAPTIVERAG/ROUTER: take a robust max across available signals
    """
    steps: List[Dict[str, Any]] = out.get("steps") or []
    m_top = _extract_top_score(steps, "MEM")
    s_top = _extract_top_score(steps, "SEARCH")

    mem_conf = _conf_from_mem(m_top)
    search_conf = _conf_from_search(s_top)

    sym_conf: Optional[float] = None
    for st in steps:
        if _canon_source(st.get("source")) == "SYM":
            sym_conf = _conf_from_sym(st)
            break

    # Mode-specific selection
    m = (mode or "").upper()
    if m == "MEM":
        return mem_conf if mem_conf is not None else 0.5
    if m in ("SEARCH", "BASE"):
        return search_conf if search_conf is not None else 0.5
    if m == "SYM":
        return sym_conf if sym_conf is not None else 0.5
    if m == "MEMSYM":
        cands = [c for c in (mem_conf, search_conf, sym_conf) if c is not None]
        return max(cands) if cands else 0.5
    if m in ("ROUTER", "ADAPTIVERAG"):
        cands = [c for c in (mem_conf, search_conf, sym_conf) if c is not None]
        return max(cands) if cands else 0.5

    return 0.5

# -------------------------
# Retrieval + Answerer glue
# -------------------------

_RETRIEVER: HybridRetriever | None = None

def _get_retriever() -> HybridRetriever | None:
    """
    Lazy-load the hybrid retriever. If corpus is missing or model errors,
    return None and we'll fall back to search_service.
    """
    global _RETRIEVER
    if _RETRIEVER is not None:
        return _RETRIEVER
    corpus_path = os.getenv("CORPUS_PATH", "backend/corpus/dpp_corpus.jsonl")
    try:
        _RETRIEVER = HybridRetriever(corpus_jsonl=corpus_path)
    except Exception:
        _RETRIEVER = None
    return _RETRIEVER

def _retrieve_context(query: str, product: Optional[str], top_k: int) -> List[Dict[str, Any]]:
    """
    Try hybrid retrieval first; fallback to existing search_service if available.
    Always return a list of dicts with at least: id, title, text, score.
    """
    fused_q = f"{(product or '').strip()} {query}".strip() if product else query
    ret = _get_retriever()
    if ret is not None:
        try:
            hits = ret.search(fused_q)[:top_k]
            # Ensure uniform keys
            out = []
            for h in hits:
                out.append({
                    "id": h.get("id"),
                    "title": h.get("title") or h.get("id"),
                    "text": h.get("text") or "",
                    "score": h.get("score"),
                    "domain": h.get("domain", ""),
                    "source": "retriever"
                })
            return out
        except Exception:
            pass

    # Fallback: search_service (if implemented)
    try:
        raw = search_service.search(query_text=fused_q, top_k=top_k) or []
        out = []
        for r in raw:
            out.append({
                "id": getattr(r, "id", None) or getattr(r, "doc_id", None) or "doc",
                "title": getattr(r, "title", None) or getattr(r, "id", None) or "doc",
                "text": getattr(r, "text", None) or getattr(r, "content", "") or "",
                "score": getattr(r, "score", None),
                "domain": getattr(r, "domain", None) or "",
                "source": "search_service"
            })
        return out
    except Exception:
        return []

def _answer_with_fallback(question: str, passages: List[Dict[str, Any]]) -> str:
    """
    Prefer the LLM answerer with citations; if it's not wired, fall back to
    echoing a high-scoring snippet with a citation.
    """
    try:
        return answer_with_context(question, passages)
    except NotImplementedError:
        pass
    except Exception:
        pass

    if passages:
        top = passages[0]
        snippet = (top.get("text") or "")[:200]
        return f"{snippet} [{top.get('id')}]"
    return "Insufficient context"

def _snippet(s: str, n: int = 220) -> str:
    s = (s or "").strip()
    return (s[:n] + "…") if len(s) > n else s

# -------------------------
# Composition helpers
# -------------------------

def _compose_MEM(query, product, session, steps, sources, parts):
    hits = []
    try:
        hits = memory_service.retrieve(session_id=session, query=query, top_k=3)
    except Exception:
        hits = []
    if hits:
        # main answer material
        parts.append(f"Memory: {hits[0].content}")
        # record score for confidence
        top_score = getattr(hits[0], "score", None)
        steps.append({"source": _canon_source("MEM"), "score": top_score})
        # add sources
        for h in hits[:]:
            sources.append({"type": "memory", "score": getattr(h, "score", None), "snippet": h.content})
    return bool(hits)

def _compose_SEARCH(query, product, session, steps, sources, parts):
    """
    NEW: Use HybridRetriever + context-bound answerer.
    Logs top CE score to steps for confidence; logs per-doc sources.
    """
    top_k = int(os.getenv("RETRIEVE_K", "8"))
    passages = _retrieve_context(query, product, top_k=top_k)

    if passages:
        # Build the final answer with citations
        answer = _answer_with_fallback(query, passages)
        parts.append(answer)

        # Step logging (use top CE score if present)
        top_score = passages[0].get("score", None)
        steps.append({
            "source": _canon_source("SEARCH"),
            "score": top_score,
            "k": len(passages),
            "doc_ids": [p.get("id") for p in passages[:min(5, len(passages))]]
        })

        # Source logging (first few for trace readability)
        for p in passages[:min(5, len(passages))]:
            sources.append({
                "type": "search",
                "id": p.get("id"),
                "score": p.get("score"),
                "title": p.get("title"),
                "snippet": _snippet(p.get("text", "")),
            })
        return True

    # No passages → graceful fallback
    try:
        # Try legacy search_service for a last resort snippet
        hits = search_service.search(query_text=(product + " " + query if product else query), top_k=3) or []
    except Exception:
        hits = []

    if hits:
        parts.append(f"Search: {getattr(hits[0], 'text', getattr(hits[0], 'content', ''))}")
        top_score = getattr(hits[0], "score", None)
        steps.append({"source": _canon_source("SEARCH"), "score": top_score})
        for h in hits[:]:
            sources.append({"type": "search", "score": getattr(h, "score", None),
                            "snippet": getattr(h, "text", getattr(h, "content", ""))})
        return True

    return False

def _compose_SYM(query, product, session, steps, sources, parts):
    sym = None
    try:
        sym = answer_symbolic(query, product, session)
    except Exception:
        sym = None
    if sym:
        parts.append(sym.text)
        steps.append({
            "source": _canon_source("SYM"),
            # expose a minimal verdict for confidence if available
            "proved": getattr(sym, "proved", None),
            "refuted": getattr(sym, "refuted", None),
            "sym_trace": {
                "product": getattr(sym.trace, "product", product),
                "asserted": getattr(sym.trace, "asserted", []),
                "inferred": getattr(sym.trace, "inferred", []),
                "rules_fired": getattr(sym.trace, "rules_fired", []),
            },
            "evidence": getattr(sym, "evidence", None),
        })
    return bool(sym)

def _compose_for_action(action: str, query: str, product: Optional[str], session: str) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    parts: List[str] = []

    a = (action or "").upper()
    if a == "BASE":
        _compose_SEARCH(query, product, session, steps, sources, parts)
    elif a == "MEM":
        _compose_MEM(query, product, session, steps, sources, parts)
    elif a == "SEARCH":
        _compose_SEARCH(query, product, session, steps, sources, parts)
    elif a == "SYM":
        _compose_SYM(query, product, session, steps, sources, parts)
    elif a == "MEMSYM":
        had_mem = _compose_MEM(query, product, session, steps, sources, parts)
        had_sym = _compose_SYM(query, product, session, steps, sources, parts)
        if not had_mem and not had_sym:
            _compose_SEARCH(query, product, session, steps, sources, parts)

    elif a == "RAG_BASE":
    # retrieval only; no MEM or SYM
        _compose_SEARCH(query, product, session, steps, sources, parts)
    elif a == "SYM_ONLY":
         had_sym = _compose_SYM(query, product, session, steps, sources, parts)
         if not had_sym:
             parts.append("Insufficient context")  # abstain-like behavior
    else:
        _compose_SEARCH(query, product, session, steps, sources, parts)

    if not parts:
        parts.append("No result found.")
        steps.append({"source": _canon_source("BASE")})

    return {"answer": " ".join(parts), "steps": steps, "sources": sources}

# -------------------------
# Router helpers
# -------------------------

def _router_predict(model: Any, feats: Dict[str, Any]) -> str:
    """
    Robustly obtain router action without recursion.
    Try route/choose/decide/__call__ before predict; skip predict if it's the same as __call__.
    """
    if model is None:
        return "ADAPTIVERAG"

    # Prefer explicit methods first
    order = ("route", "choose", "decide", "__call__", "predict")
    predict_fn = getattr(model, "predict", None)
    call_fn = getattr(model, "__call__", None)

    for name in order:
        fn = getattr(model, name, None)
        if not callable(fn):
            continue
        # Skip predict if it's exactly the same as __call__ to avoid infinite recursion
        if name == "predict" and call_fn is not None and fn is call_fn:
            continue
        out = fn(feats)
        if isinstance(out, (tuple, list)) and out:
            out = out[0]
        if hasattr(out, "value"):  # Enum
            out = out.value
        if isinstance(out, str):
            return out.upper()

    return "ADAPTIVERAG"

# -------------------------
# Main route
# -------------------------

@router.post("/solve")
def solve(req: SolveRequest, x_run_mode: str | None = Header(default=None)):
    text = _pick_query(req)
    if not text:
        raise HTTPException(status_code=422, detail="Missing query/q")

    mode = (x_run_mode or req.mode or "BASE").upper()
    session = req.session or "s1"
    product = (req.product or "").strip() or None

    # --- ROUTER ---
    if mode == "ROUTER":
        feats = _safe_features(text, product, session)
        model = _get_router()
        action = _router_predict(model, feats)
        if action == "ADAPTIVERAG":
            mode = "ADAPTIVERAG"  # fall through
        else:
            out = _compose_for_action(action, text, product, session)
            # attach confidence based on the underlying chosen action's evidence
            conf = _attach_confidence(action, out)
            out.update({
                "mode": "ROUTER",
                "chosen_action": action,
                "session": session,
                "product": product,
                "confidence": conf,
            })
            return out

    # --- ADAPTIVE-RAG (heuristic) ---
    if mode == "ADAPTIVERAG":
        feats = _safe_features(text, product, session)
        t_mem = float(os.getenv("ADAPTIVE_MEM_T", "0.45"))
        t_search = float(os.getenv("ADAPTIVE_SEARCH_T", "0.55"))
        if float(feats.get("mem_top", 0.0)) >= t_mem:
            action = "MEM"
        elif float(feats.get("search_top", 0.0)) >= t_search:
            action = "SEARCH"
        elif int(feats.get("sym_fired", 0)) == 1:
            action = "SYM"
        else:
            action = "MEMSYM"  # safe combo when signals are weak
        out = _compose_for_action(action, text, product, session)
        conf = _attach_confidence(action, out)
        out.update({
            "mode": "ADAPTIVERAG",
            "chosen_action": action,
            "session": session,
            "product": product,
            "confidence": conf,
        })
        return out

    # --- Classic modes ---
    out = _compose_for_action(mode, text, product, session)
    conf = _attach_confidence(mode, out)
    out.update({"mode": mode, "session": session, "product": product, "confidence": conf})
    return out
