from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from backend.services import search_service

router = APIRouter(prefix="/docs", tags=["docs"])

class PutDoc(BaseModel):
    document_name: str
    text: str
    doc_id: Optional[str] = None

@router.post("/put", summary="Index a simple text document for baseline search")
def put_doc(req: PutDoc):
    added = False
    # Try common ingestion method names
    if hasattr(search_service, "add_document"):
        search_service.add_document(req.document_name, req.text, doc_id=req.doc_id)
        added = True
    elif hasattr(search_service, "index_document"):
        search_service.index_document(req.document_name, req.text, doc_id=req.doc_id)
        added = True
    elif hasattr(search_service, "upsert"):
        search_service.upsert([{"id": req.doc_id or req.document_name, "text": req.text, "name": req.document_name}])
        added = True

    if not added:
        raise HTTPException(status_code=500, detail="No add/index/upsert method found on search_service")

    return {"ok": True}
