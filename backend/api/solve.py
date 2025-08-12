from __future__ import annotations

import logging
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel

from backend.config.run_modes import RunMode, mode_from_env
from backend.services import memory_service, search_service

# If your reasoner class or builder lives elsewhere, adjust this import.
try:
    from backend.services.symbolic_reasoning_service import SymbolicReasoner
except Exception:  # pragma: no cover
    SymbolicReasoner = None  # type: ignore

router = APIRouter(prefix="/solve", tags=["solve"])
log = logging.getLogger("api.solve")


# ---------- Models ----------
class SolveRequest(BaseModel):
    session_id: str
    query: str
    product: Optional[str] = None  # required only for symbolic checks


class Step(BaseModel):
    kind: str   # "baseline" | "memory" | "symbolic"
    detail: str
    payload: Optional[Any] = None


class SolveResponse(BaseModel):
    mode: RunMode
    steps: List[Step]
    answer: str


# ---------- Dependencies ----------
def get_reasoner(request: Request) -> Optional["SymbolicReasoner"]:
    return getattr(request.app.state, "reasoner", None)


# ---------- Helpers ----------
def _doc_search(query: str, top_k: int = 5):
    """
    Be tolerant of either API shape:
      - search_service.search(query, top_k=...)
      - search_service.search_documents(query)
    """
    if hasattr(search_service, "search"):
        return search_service.search(query, top_k=top_k)
    if hasattr(search_service, "search_documents"):
        return search_service.search_documents(query)
    return []


def _hits_payload(hits: Any) -> Any:
    out = []
    for h in hits or []:
        if hasattr(h, "dict"):
            out.append(h.dict())  # pydantic model
        elif hasattr(h, "__dict__"):
            out.append({k: getattr(h, k) for k in dir(h) if not k.startswith("_")})
        else:
            out.append(str(h))
    return out


# ---------- Route ----------
@router.post(
    "",
    response_model=SolveResponse,
    summary="Hybrid solve (baseline → memory → symbolic)",
)
def solve(
    req: SolveRequest,
    request: Request,
    reasoner: Optional["SymbolicReasoner"] = Depends(get_reasoner),
    x_run_mode: Optional[str] = Header(
        default=None,
        alias="X-Run-Mode",
        description="Override RUN_MODE per request (BASE|MEM|SYM|MEMSYM)",
    ),
):
    # Resolve mode: header (if provided) overrides env/default
    mode = mode_from_env(x_run_mode or None)

    steps: List[Step] = []
    answer = ""

    # 1) baseline — document search (always runs)
    doc_hits = []
    try:
        doc_hits = _doc_search(req.query, top_k=5)
    except Exception as e:
        log.exception("Baseline search failed: %s", e)
    steps.append(Step(kind="baseline", detail=f"{len(doc_hits)} doc hits", payload=_hits_payload(doc_hits)))
    log.info("→ baseline: %d doc hits for %r", len(doc_hits), req.query)

    # 2) memory — only in MEM / MEMSYM
    mem_hits = []
    if mode in (RunMode.MEM, RunMode.MEMSYM):
        try:
            mem_hits = memory_service.retrieve(req.session_id, req.query, top_k=5)
        except Exception as e:
            log.exception("Memory retrieve failed: %s", e)
        steps.append(
            Step(kind="memory", detail=f"{len(mem_hits)} memory hits",
                 payload=[{"content": getattr(h, "content", str(h)), "score": getattr(h, "score", None)} for h in mem_hits])
        )
        log.info("→ memory: %d hits for session=%s", len(mem_hits), req.session_id)
    else:
        steps.append(Step(kind="memory", detail="skipped (mode without memory)"))
        log.info("→ memory: skipped")

    # 3) symbolic — only in SYM / MEMSYM and if product + reasoner available
    sym_payload = None
    if mode in (RunMode.SYM, RunMode.MEMSYM) and reasoner and req.product:
        try:
            requires = [r.split("#")[-1] for r in reasoner.check_compliance_requirements(req.product)]
            missing = [m.split("#")[-1] for m in reasoner.suggest_missing_steps(req.product)]
            sym_payload = {"requires": requires, "missing": missing}
            steps.append(Step(kind="symbolic", detail="compliance check", payload=sym_payload))
            log.info("→ symbolic: %s", sym_payload)
        except Exception as e:
            log.exception("Symbolic step failed: %s", e)
            steps.append(Step(kind="symbolic", detail="error", payload={"error": str(e)}))
    else:
        steps.append(Step(kind="symbolic", detail="skipped (no product or reasoner/mode)"))
        log.info("→ symbolic: skipped")

    # 4) aggregator — prefer memory > docs; append symbolic advice when present
    if mem_hits:
        answer = getattr(mem_hits[0], "content", str(mem_hits[0]))
    elif doc_hits:
        top = doc_hits[0]
        snippet = getattr(top, "snippet", None)
        name = getattr(top, "document_name", None) or getattr(top, "name", None)
        answer = snippet or name or "Found a relevant document."
    elif isinstance(sym_payload, dict):
        reqs = ", ".join(sym_payload.get("requires", []))
        miss = ", ".join(sym_payload.get("missing", [])) or "none"
        answer = f"Symbolic analysis for {req.product or 'the product'} → requires: {reqs}; missing: {miss}"
    else:
        answer = "No relevant information found."

    if (mem_hits or doc_hits) and isinstance(sym_payload, dict):
        reqs = ", ".join(sym_payload.get("requires", []))
        miss = ", ".join(sym_payload.get("missing", [])) or "none"
        answer = f"{answer} | requires: {reqs}; missing: {miss}"

    return SolveResponse(mode=mode, steps=steps, answer=answer)
