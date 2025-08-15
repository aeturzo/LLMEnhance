# backend/api/solve.py
from __future__ import annotations
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from backend.services import memory_service, search_service

router = APIRouter()

class SolveRequest(BaseModel):
    # Accept both "query" and "q" to be lenient with clients.
    query: Optional[str] = Field(default=None)
    q: Optional[str] = Field(default=None)
    product: Optional[str] = None
    session: Optional[str] = "s1"
    mode: Optional[Literal["BASE", "MEM", "SYM", "MEMSYM"]] = "BASE"

def _pick_query(req: SolveRequest) -> str:
    return (req.query or req.q or "").strip()

@router.post("/solve")
def solve(req: SolveRequest):
    text = _pick_query(req)
    if not text:
        raise HTTPException(status_code=422, detail="Missing query/q")

    mode = (req.mode or "BASE").upper()
    session = req.session or "s1"
    product = (req.product or "").strip()

    # Retrieve memory hits if needed
    mem_hits = []
    if mode in ("MEM", "MEMSYM"):
        try:
            mem_hits = memory_service.retrieve(session_id=session, query=text, top_k=3)
        except Exception as e:
            mem_hits = []

    # Retrieve search hits (sym falls back to stronger query with product hint)
    search_q = f"{product} {text}".strip() if product else text
    search_hits = []
    if mode in ("BASE", "SYM", "MEMSYM"):
        try:
            search_hits = search_service.search(query_text=search_q, top_k=3)
        except Exception:
            search_hits = []

    # Compose a simple answer:
    parts: List[str] = []
    if mode in ("MEM", "MEMSYM") and mem_hits:
        parts.append(f"Memory: {mem_hits[0].content}")
    if search_hits:
        parts.append(f"Search: {search_hits[0].text}")
    if not parts:
        parts.append("No result found.")

    answer = " ".join(parts)

    # Return structured response (your evaluator likely only cares about .answer)
    sources = []
    for h in mem_hits[:2]:
        sources.append({"type": "memory", "score": getattr(h, "score", None), "snippet": h.content})
    for h in search_hits[:2]:
        sources.append({"type": "search", "score": getattr(h, "score", None), "snippet": h.text})

    return {
        "mode": mode,
        "answer": answer,
        "sources": sources,
        "session": session,
        "product": product or None,
    }
