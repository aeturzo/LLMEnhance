# backend/api/solve.py
from __future__ import annotations
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Header

from backend.services import memory_service, search_service

try:
    # New Day-3 API
    from backend.services.symbolic_reasoning_service import answer_symbolic  # type: ignore
except Exception:
    answer_symbolic = None  # type: ignore

router = APIRouter()

class SolveRequest(BaseModel):
    query: Optional[str] = Field(default=None)
    q: Optional[str] = Field(default=None)
    product: Optional[str] = None
    session: Optional[str] = "s1"
    mode: Optional[Literal["BASE", "MEM", "SYM", "MEMSYM"]] = "BASE"

def _pick_query(req: SolveRequest) -> str:
    return (req.query or req.q or "").strip()

@router.post("/solve")
def solve(req: SolveRequest, x_run_mode: str | None = Header(default=None)) -> Dict[str, Any]:
    text = _pick_query(req)
    if not text:
        raise HTTPException(status_code=422, detail="Missing query/q")

    mode = (x_run_mode or req.mode or "BASE").upper()
    session = req.session or "s1"
    product = (req.product or "").strip() or None

    # --- SYMBOLIC (when asked) ---
    sym_ans = None
    if mode in ("SYM", "MEMSYM") and answer_symbolic is not None:
        try:
            sym_ans = answer_symbolic(text, product, session)
        except Exception:
            sym_ans = None

    # --- MEMORY ---
    mem_hits = []
    if mode in ("MEM", "MEMSYM"):
        try:
            mem_hits = memory_service.retrieve(session_id=session, query=text, top_k=3)
        except Exception:
            mem_hits = []

    # --- SEARCH (kept for BASE/SYM/MEMSYM) ---
    search_q = f"{product} {text}".strip() if product else text
    search_hits = []
    if mode in ("BASE", "SYM", "MEMSYM"):
        try:
            search_hits = search_service.search(query_text=search_q, top_k=3)
        except Exception:
            search_hits = []

    parts: List[str] = []
    steps: List[Dict[str, Any]] = []

    if mode == "SYM":
        if sym_ans:
            parts.append(sym_ans.text)
            steps.append({
                "source": "SYM",
                "sym_trace": {
                    "product": getattr(sym_ans.trace, "product", product),
                    "asserted": getattr(sym_ans.trace, "asserted", []),
                    "inferred": getattr(sym_ans.trace, "inferred", []),
                    "rules_fired": getattr(sym_ans.trace, "rules_fired", []),
                },
                "evidence": sym_ans.evidence,
            })
        else:
            parts.append("No result found.")
            steps.append({"source": "SYM"})
        return {
            "mode": mode,
            "answer": "\n".join(parts),
            "steps": steps,
            "session": session,
            "product": product,
        }

    if mode == "MEM":
        if mem_hits:
            parts.append(f"Memory: {mem_hits[0].content}")
            steps.append({"source": "MEM", "score": getattr(mem_hits[0], "score", None)})
        else:
            parts.append("No result found.")
            steps.append({"source": "MEM"})
        return {
            "mode": mode,
            "answer": "\n".join(parts),
            "steps": steps,
            "session": session,
            "product": product,
        }

    if mode == "MEMSYM":
        if mem_hits:
            parts.append(f"Memory: {mem_hits[0].content}")
            steps.append({"source": "MEM", "score": getattr(mem_hits[0], "score", None)})

        if sym_ans:
            parts.append(sym_ans.text)
            steps.append({
                "source": "SYM",
                "sym_trace": {
                    "product": getattr(sym_ans.trace, "product", product),
                    "asserted": getattr(sym_ans.trace, "asserted", []),
                    "inferred": getattr(sym_ans.trace, "inferred", []),
                    "rules_fired": getattr(sym_ans.trace, "rules_fired", []),
                },
                "evidence": sym_ans.evidence,
            })

        if not parts and search_hits:
            parts.append(f"Search: {search_hits[0].text}")
            steps.append({"source": "SEARCH"})

        if not parts:
            parts.append("No result found.")
            steps.append({"source": "MEMSYM"})

        return {
            "mode": mode,
            "answer": "\n".join(parts),
            "steps": steps,
            "session": session,
            "product": product,
        }

    # BASE
    if search_hits:
        parts.append(f"Search: {search_hits[0].text}")
        steps.append({"source": "SEARCH"})
    else:
        parts.append("No result found.")
        steps.append({"source": "BASE"})

    return {
        "mode": mode,
        "answer": "\n".join(parts),
        "steps": steps,
        "session": session,
        "product": product,
    }
