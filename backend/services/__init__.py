from .embedding_service import EmbeddingService
from .search_service import SemanticSearchService
from .memory_service import MemoryService
from backend.config.config import settings

# Instantiate shared services with configured embedding backend
embedding_service = EmbeddingService(
    vector_dim=int(settings.EMBEDDING_VECTOR_DIM),   # hint; backend may override internally
    model_name=settings.EMBEDDING_MODEL_NAME,
)

REAL_DIM = embedding_service.vector_dim  # <-- actual backend dimension

search_service = SemanticSearchService(
    embedding_service=embedding_service,
    vector_dim=REAL_DIM,                 # make FAISS match embeddings
)
memory_service = MemoryService(
    embedding_service=embedding_service,
    vector_dim=REAL_DIM,                 # make FAISS match embeddings
)