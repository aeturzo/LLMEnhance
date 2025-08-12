from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Any

from backend.services import memory_service

router = APIRouter(prefix="/memory", tags=["memory"])

class PutReq(BaseModel):
    session_id: str
    content: str
    metadata: Optional[dict[str, Any]] = None  # ignored by add_memory

@router.post("/put", summary="Store a memory item for a session")
def put_memory(req: PutReq):
    try:
        # Your MemoryService uses add_memory(session_id, content)
        if hasattr(memory_service, "add_memory"):
            memory_service.add_memory(req.session_id, req.content)
            return {"ok": True}
        raise HTTPException(
            status_code=500,
            detail="MemoryService is missing add_memory(session_id, content).",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
