from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services import search_service
from backend.services.ingestion_service import Document

router = APIRouter(prefix="/ingest", tags=["ingest"])

class PutDoc(BaseModel):
    document_name: str
    text: str

@router.post("/put", summary="Index a simple text document for baseline search")
def put_doc(req: PutDoc):
    try:
        # EXACTLY what your service wants: a Document instance as sole arg
        doc = Document(name=req.document_name, content=req.text)
        ok = search_service.add_document(doc)
        return {"ok": bool(ok)}
    except Exception as e:
        # Bubble up the precise error so we can see it if anything else breaks
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
