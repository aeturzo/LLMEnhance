import faiss
import numpy as np
from typing import List
from backend.services.ingestion_service import Document
from backend.services.embedding_service import EmbeddingService
from backend.models.schemas import SearchResult

class SemanticSearchService:
    """Manage FAISS index and perform semantic search."""
    def __init__(self, embedding_service: EmbeddingService, vector_dim: int = 768):
        self.embedding_service = embedding_service
        self.vector_dim = vector_dim
        self.index = faiss.IndexFlatIP(vector_dim)
        self.documents: List[Document] = []

    def add_document(self, doc: Document) -> bool:
        self.documents.append(doc)
        return True

    def build_index(self) -> bool:
        if not self.documents:
            return False
        self.index.reset()
        vectors = []
        for doc in self.documents:
            vec = self.embedding_service.generate_embedding(doc.content)
            vectors.append(vec)
        try:
            matrix = np.stack(vectors).astype('float32')
        except Exception:
            return False
        self.index.add(matrix)
        return True

    def search(self, query_text: str, top_k: int = 5) -> List[SearchResult]:
        if self.index.ntotal == 0:
            return []
        query_vec = self.embedding_service.generate_embedding(query_text)
        query_vec = query_vec.reshape(1, -1)
        distances, indices = self.index.search(query_vec, top_k)
        results: List[SearchResult] = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            score = float(distances[0][rank])
            snippet = (doc.content[:100] + "...") if doc.content else ""
            results.append(SearchResult(document_name=doc.name, snippet=snippet, score=score))
        return results

    def doc_count(self) -> int:
        return len(self.documents)
