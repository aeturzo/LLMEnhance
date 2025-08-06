from .embedding_service import EmbeddingService
from .search_service import SemanticSearchService
from .memory_service import MemoryService



embedding_service = EmbeddingService()
search_service = SemanticSearchService(embedding_service=embedding_service)
memory_service   = MemoryService(embedding_service=embedding_service)  