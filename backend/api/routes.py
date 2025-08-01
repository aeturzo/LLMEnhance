from fastapi import APIRouter, File, UploadFile, HTTPException
from backend.services import ingestion_service
from backend.services import search_service
from backend.models.schemas import DocumentResponse, SearchResponse, QueryRequest

router = APIRouter()

@router.post("/upload", response_model=DocumentResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a DPP file and process it."""
    try:
        doc = ingestion_service.process_file(file)
        search_service.add_document(doc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"filename": doc.name, "status": "ingested"}

@router.post("/index", response_model=dict)
async def build_index():
    """Build or rebuild the search index from ingested documents."""
    success = search_service.build_index()
    if not success:
        raise HTTPException(status_code=500, detail="Index building failed")
    return {"status": "index_built", "document_count": search_service.doc_count()}

@router.post("/search", response_model=SearchResponse)
async def semantic_search(query: QueryRequest):
    """Perform a semantic search over indexed documents."""
    results = search_service.search(query_text=query.query, top_k=5)
    return {"query": query.query, "results": results}
