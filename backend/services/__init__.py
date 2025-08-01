from .embedding_service import EmbeddingService
from .search_service import SemanticSearchService

embedding_service = EmbeddingService()
search_service = SemanticSearchService(embedding_service=embedding_service)
