# backend/services/embedding_service.py
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger("services.embedding")


def _l2_normalize_2d(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize each row of a 2D array. Returns float32.
    """
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    out = arr / norms
    return out.astype(np.float32, copy=False)


class EmbeddingService:
    """
    Central, pluggable embedding backend used by memory/search.

    Backends supported (auto-selected from model_name/env):
      - SentenceTransformers ("st"): MiniLM etc.  (local)
      - OpenAI ("openai"): text-embedding-*        (requires OPENAI_API_KEY)
      - HF Transformers ("hf"): plain transformers (local, OSS, mean-pool)

    Public API (used by the codebase):
      - generate_embedding(text: str)          -> np.ndarray (dim,)
      - generate_embeddings(texts: List[str])  -> np.ndarray (n, dim)
      - embed(text) / embed_many(texts) / encode(texts)  (aliases)

    Choosing the backend:
      * MiniLM testing (default):
          EMBED_MODEL_NAME unset  -> "sentence-transformers/all-MiniLM-L6-v2" (st, dim=384)
      * OpenAI (maps "gpt4o-mini" to embeddings):
          EMBED_MODEL_NAME="gpt4o-mini" or "openai" or any "text-embedding-*" (openai)
      * OSS/HF without sentence-transformers:
          EMBED_MODEL_NAME="hf:intfloat/e5-base-v2" (hf)  # example (dim=768)
          EMBED_MODEL_NAME="hf:sentence-transformers/all-MiniLM-L6-v2" (hf, dim=384)
    """

    def __init__(self, model_name: Optional[str] = None, vector_dim: Optional[int] = None):
        # Resolve desired model name
        env_name = os.getenv("EMBED_MODEL_NAME") or os.getenv("LLM_EMBED_MODEL")
        self.model_name_raw = (model_name or env_name or "sentence-transformers/all-MiniLM-L6-v2").strip()
        self.model_name_lc = self.model_name_raw.lower()

        self._backend_type: Optional[str] = None  # "st" | "openai" | "hf" | None
        self._backend = None
        self.vector_dim: int = int(vector_dim or 0)

        # Select backend
        if self._is_openai(self.model_name_lc):
            self._init_openai()
        elif self._is_hf(self.model_name_lc):
            self._init_hf()
        elif self._is_st(self.model_name_lc):
            self._init_st()
        else:
            # Heuristics: prefer ST for "all-" or "minilm", else HF if prefixed, else OpenAI if "gpt4o"
            if ("mini" in self.model_name_lc) or self.model_name_lc.startswith("all-") or "sentence-transformers/" in self.model_name_lc:
                self._init_st()
            elif "gpt4o" in self.model_name_lc or "openai" in self.model_name_lc or "text-embedding" in self.model_name_lc:
                self._init_openai()
            else:
                # Fallback to ST MiniLM for local dev
                self.model_name_raw = "sentence-transformers/all-MiniLM-L6-v2"
                self.model_name_lc = self.model_name_raw.lower()
                self._init_st()

        if not self._backend_type:
            log.warning("No embedding backend available; using deterministic random vectors.")
            if not self.vector_dim:
                self.vector_dim = 384  # safe default

    # ----------------------- Backend selection helpers -----------------------

    @staticmethod
    def _is_st(name: str) -> bool:
        return (
            name.startswith("sentence-transformers/") or
            name.startswith("all-") or
            "minilm" in name or
            name == "st"
        )

    @staticmethod
    def _is_openai(name: str) -> bool:
        return (
            name == "openai" or
            "gpt4o" in name or
            name.startswith("text-embedding-")
        )

    @staticmethod
    def _is_hf(name: str) -> bool:
        return name.startswith("hf:")

    # ----------------------------- ST backend -------------------------------

    def _init_st(self) -> None:
        """
        SentenceTransformers backend (local). Default to MiniLM if a generic "minilm" or "all-" was given.
        """
        try:
            # Import inside to avoid heavy import at module load time.
            from sentence_transformers import SentenceTransformer  # type: ignore

            name = self.model_name_raw
            if name in {"minilm", "all-minilm-l6-v2"} or name == "st":
                name = "sentence-transformers/all-MiniLM-L6-v2"

            self._backend = SentenceTransformer(name)
            self._backend_type = "st"
            self.vector_dim = int(self._backend.get_sentence_embedding_dimension())  # type: ignore[attr-defined]
            log.info("Embedding backend: ST (%s) [%d dims]", name, self.vector_dim)
        except Exception as e:
            log.warning("ST backend unavailable (%s); falling back to random.", e)
            self._backend = None
            self._backend_type = None

    # ----------------------------- OpenAI backend ---------------------------

    def _init_openai(self) -> None:
        """
        OpenAI embeddings. We map gpt4o-mini → text-embedding-3-small by default.
        Requires OPENAI_API_KEY in environment.
        """
        try:
            from openai import OpenAI  # type: ignore
            self._backend = OpenAI()
            self._backend_type = "openai"
            log.info("Embedding backend: OpenAI (requested=%s)", self.model_name_raw)
            # vector_dim inferred on first call unless OPENAI_EMBEDDING_MODEL is set explicitly
        except Exception as e:
            log.warning("OpenAI backend unavailable (%s); falling back to random.", e)
            self._backend = None
            self._backend_type = None

    def _resolve_openai_model(self) -> str:
        """
        Choose an OpenAI embedding model. If user set a text-embedding-* model, respect it.
        Otherwise:
          - "gpt4o-mini" / "gpt-4o-mini" -> "text-embedding-3-small"
          - default -> env OPENAI_EMBEDDING_MODEL or "text-embedding-3-small"
        """
        # User explicitly provided embeddings model
        if self.model_name_lc.startswith("text-embedding-"):
            return self.model_name_raw

        # Map gpt4o names to embeddings
        if "gpt4o" in self.model_name_lc or "gpt-4o" in self.model_name_lc:
            return "text-embedding-3-small"

        return (
            os.getenv("OPENAI_EMBEDDING_MODEL")
            or "text-embedding-3-small"
        )

    # ----------------------------- HF (Transformers) backend ----------------

    def _init_hf(self) -> None:
        """
        HF transformers backend (no sentence-transformers, mean-pooled last hidden state).
        Example name: hf:intfloat/e5-base-v2  OR  hf:sentence-transformers/all-MiniLM-L6-v2
        """
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            _, model_id = self.model_name_raw.split(":", 1)
            self._hf_tok = AutoTokenizer.from_pretrained(model_id)
            self._hf_model = AutoModel.from_pretrained(model_id)
            self._hf_model.eval()
            self._backend = (self._hf_tok, self._hf_model)
            self._backend_type = "hf"
            self.vector_dim = int(self._hf_model.config.hidden_size)  # type: ignore[attr-defined]
            log.info("Embedding backend: HF (%s) [%d dims]", model_id, self.vector_dim)
        except Exception as e:
            log.warning("HF backend unavailable (%s); falling back to random.", e)
            self._backend = None
            self._backend_type = None

    # ----------------------------- Public API --------------------------------

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Return a single embedding vector (dim,). Always float32 and L2-normalized.
        """
        if self._backend_type == "st":
            vec = self._backend.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]  # type: ignore[attr-defined]
            return vec.astype(np.float32, copy=False)

        if self._backend_type == "openai":
            model = self._resolve_openai_model()
            out = self._backend.embeddings.create(model=model, input=text)  # type: ignore[attr-defined]
            emb = np.asarray(out.data[0].embedding, dtype=np.float32)
            if not self.vector_dim:
                self.vector_dim = int(emb.shape[0])
            return _l2_normalize_2d(emb).reshape(-1)

        if self._backend_type == "hf":
            tok, mdl = self._backend  # type: ignore[assignment]
            import torch  # local import
            with torch.no_grad():
                enc = tok([text], padding=True, truncation=True, max_length=256, return_tensors="pt")
                out = mdl(**enc).last_hidden_state  # [1, T, H]
                mask = enc["attention_mask"].unsqueeze(-1)  # [1, T, 1]
                embs = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
                vec = embs[0].cpu().numpy()
            return _l2_normalize_2d(vec).reshape(-1)

        # Deterministic random fallback (dev only)
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        if not self.vector_dim:
            self.vector_dim = 384
        return _l2_normalize_2d(rng.random(self.vector_dim)).reshape(-1)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Return (n, dim). Always float32 and L2-normalized.
        """
        if not texts:
            if not self.vector_dim:
                self.vector_dim = 384
            return np.zeros((0, self.vector_dim), dtype=np.float32)

        if self._backend_type == "st":
            arr = self._backend.encode(texts, convert_to_numpy=True, normalize_embeddings=True)  # type: ignore[attr-defined]
            arr = np.asarray(arr, dtype=np.float32)
            return arr if arr.ndim == 2 else arr.reshape(1, -1)

        if self._backend_type == "openai":
            model = self._resolve_openai_model()
            out = self._backend.embeddings.create(model=model, input=texts)  # type: ignore[attr-defined]
            data = sorted(out.data, key=lambda d: d.index)
            arr = np.asarray([d.embedding for d in data], dtype=np.float32)
            if not self.vector_dim:
                self.vector_dim = int(arr.shape[1])
            return _l2_normalize_2d(arr)

        if self._backend_type == "hf":
            tok, mdl = self._backend  # type: ignore[assignment]
            import torch  # local import
            batches: List[np.ndarray] = []
            bs = int(os.getenv("EMBED_BATCH_SIZE", "32"))
            with torch.no_grad():
                for i in range(0, len(texts), bs):
                    chunk = texts[i:i+bs]
                    enc = tok(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt")
                    out = mdl(**enc).last_hidden_state  # [B, T, H]
                    mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
                    embs = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
                    batches.append(embs.cpu().numpy())
            arr = np.vstack(batches)
            return _l2_normalize_2d(arr)

        # Fallback: deterministic random per text
        if not self.vector_dim:
            self.vector_dim = 384
        mats = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            mats.append(rng.random(self.vector_dim))
        return _l2_normalize_2d(np.vstack(mats))

    # ----------------------------- Aliases -----------------------------------

    def embed(self, text: str) -> np.ndarray:
        return self.generate_embedding(text)

    def embed_many(self, texts: List[str]) -> np.ndarray:
        return self.generate_embeddings(texts)

    def encode(self, texts: List[str]) -> np.ndarray:
        # Some callers expect an 'encode' that returns batch embeddings
        return self.generate_embeddings(texts)


# Singleton accessor used by memory/search services
@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingService:
    """
    Returns a cached EmbeddingService configured by EMBED_MODEL_NAME (if set).
    """
    name = os.getenv("EMBED_MODEL_NAME") or os.getenv("LLM_EMBED_MODEL") or None
    svc = EmbeddingService(model_name=name)
    log.info("get_embedder(): backend=%s, model=%s, dim=%s",
             svc._backend_type, svc.model_name_raw, svc.vector_dim or "auto")
    return svc
