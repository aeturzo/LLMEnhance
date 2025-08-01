from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str

class SearchResult(BaseModel):
    document_name: str
    snippet: str = ""
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

class DocumentResponse(BaseModel):
    filename: str
    status: str
