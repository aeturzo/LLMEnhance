from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

log = logging.getLogger("services.embedding")


class EmbeddingService:
    """
    Pluggable embedding backend with **single** and **batch** APIs.

    model_name:
      - "minilm" / "all-MiniLM-L6-v2" → sentence-transformers (local)
      - "gpt4o-mini" / "openai"       → OpenAI embeddings (requires OPENAI_API_KEY)
      - anything else                 → deterministic random fallback (dev only)

    Public methods expected by the rest of the codebase:
      - generate_embedding(text: str)          -> np.ndarray (dim,)
      - generate_embeddings(texts: List[str])  -> np.ndarray (n, dim)

    Aliases provided for compatibility:
      - embed(text)         -> generate_embedding(text)
      - embed_many(texts)   -> generate_embeddings(texts)
      - encode(texts)       -> generate_embeddings(texts)
    """

    def __init__(self, vector_dim: int = 768, model_name: str = "gpt4o-mini"):
        self.vector_dim = int(vector_dim)
        self.model_name = (model_name or "").lower()
        self._backend = None
        self._backend_type: Optional[str] = None  # "st" | "openai" | None

        # Initialize chosen backend if available
        if "mini" in self.model_name or self.model_name.startswith("all-"):
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                name = "all-MiniLM-L6-v2"
                self._backend = SentenceTransformer(name)
                self._backend_type = "st"
                self.vector_dim = int(self._backend.get_sentence_embedding_dimension())
                log.info("Embedding backend: SentenceTransformer(%s) [%d dims]", name, self.vector_dim)
            except Exception as e:  # pragma: no cover
                log.warning("MiniLM backend unavailable (%s); falling back to random.", e)
                self._backend = None
                self._backend_type = None

        elif "gpt4o" in self.model_name or "openai" in self.model_name:
            try:
                from openai import OpenAI  # type: ignore
                self._backend = OpenAI()
                self._backend_type = "openai"
                log.info("Embedding backend: OpenAI (model=%s)", self.model_name)
                # vector_dim will be inferred on first call if needed
            except Exception as e:  # pragma: no cover
                log.warning("OpenAI backend unavailable (%s); falling back to random.", e)
                self._backend = None
                self._backend_type = None

    # -------- Single text --------
    def generate_embedding(self, text: str) -> np.ndarray:
        """Return a single embedding vector (dim,)."""
        if self._backend_type == "st":
            vec = self._backend.encode([text], normalize_embeddings=True)[0]  # type: ignore[attr-defined]
            return np.asarray(vec, dtype=np.float32)

        if self._backend_type == "openai":
            model = (
                (self.model_name if "embedding" in self.model_name else None)
                or os.getenv("OPENAI_EMBEDDING_MODEL")
                or "text-embedding-3-small"
            )
            out = self._backend.embeddings.create(model=model, input=text)  # type: ignore[attr-defined]
            emb = np.asarray(out.data[0].embedding, dtype=np.float32)
            # Update vector_dim if this backend differs from default
            if self.vector_dim != emb.shape[0]:
                self.vector_dim = int(emb.shape[0])
            return emb

        # Deterministic random fallback (dev only, no network)
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.random(self.vector_dim, dtype=np.float32)

    # -------- Batch texts --------
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Return a batch of embeddings with shape (n, dim)."""
        if not texts:
            return np.zeros((0, self.vector_dim), dtype=np.float32)

        if self._backend_type == "st":
            vecs = self._backend.encode(texts, normalize_embeddings=True)  # type: ignore[attr-defined]
            arr = np.asarray(vecs, dtype=np.float32)
            # Ensure 2D shape (n, dim)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr

        if self._backend_type == "openai":
            model = (
                (self.model_name if "embedding" in self.model_name else None)
                or os.getenv("OPENAI_EMBEDDING_MODEL")
                or "text-embedding-3-small"
            )
            out = self._backend.embeddings.create(model=model, input=texts)  # type: ignore[attr-defined]
            data = sorted(out.data, key=lambda d: d.index)  # ensure order matches inputs
            arr = np.asarray([d.embedding for d in data], dtype=np.float32)
            if self.vector_dim != arr.shape[1]:
                self.vector_dim = int(arr.shape[1])
            return arr

        # Fallback: deterministic random per text, stacked
        mats = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            mats.append(rng.random(self.vector_dim, dtype=np.float32))
        return np.vstack(mats)

    # -------- Aliases (compatibility) --------
    def embed(self, text: str) -> np.ndarray:
        return self.generate_embedding(text)

    def embed_many(self, texts: List[str]) -> np.ndarray:
        return self.generate_embeddings(texts)

    def encode(self, texts: List[str]) -> np.ndarray:
        # Some callers may expect an 'encode' that returns batch embeddings
        return self.generate_embeddings(texts)
