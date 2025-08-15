#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backend/services/search_service.py  (FAISS-free, pure NumPy)

Lightweight vector search over a document corpus with cosine similarity
(dot product on L2-normalized embeddings). Mirrors MemoryService’s style and
exposes both a class API and module-level wrapper functions for back-compat.

Public API
----------
Class:
    - SearchService.add_document(text: str, meta: dict | None = None) -> SearchDoc
    - SearchService.search(query_text: str, top_k: int = 5) -> list[SearchDoc]
    - SearchService.reset_storage() -> None
    - SearchService.reindex() -> None

Module-level wrappers (what your RL code calls):
    - add_document(text: str, meta: dict | None = None) -> SearchDoc
    - search(query_text: str, top_k: int = 5) -> list[SearchDoc]
    - reset_storage() -> None
    - reindex() -> None
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np

from backend.services.embedding_service import get_embedder, EmbeddingService

# ---------------------------------------------------------------------------
# Persistence paths (absolute, resolved from repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEARCH_META_PATH = os.path.join(REPO_ROOT, "search.meta.jsonl")  # JSONL rows: {"text", "meta", "timestamp"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_f32c(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype="float32"), dtype="float32")

def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    mat = _as_f32c(mat)
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return _as_f32c(mat / n)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class SearchDoc:
    text: str
    meta: Dict[str, Any]
    embedding: np.ndarray
    timestamp: datetime
    score: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "meta": self.meta,
            "timestamp": self.timestamp.isoformat(),
        }

# ---------------------------------------------------------------------------
# Service (pure NumPy)
# ---------------------------------------------------------------------------
class SearchService:
    def __init__(self, embedding_service: Optional[EmbeddingService] = None, vector_dim: Optional[int] = None):
        self.embedding_service: EmbeddingService = embedding_service or get_embedder()
        backend_dim = int(getattr(self.embedding_service, "vector_dim", 0) or 0)
        self.vector_dim = int(vector_dim or backend_dim or 384)

        self.docs: List[SearchDoc] = []
        self._mat: np.ndarray = np.zeros((0, self.vector_dim), dtype="float32")  # (N, D) L2-normalized

        self._load_meta_and_rebuild()

    # ---------- Persistence ----------
    def _load_meta_and_rebuild(self) -> None:
        if os.path.exists(SEARCH_META_PATH):
            try:
                with open(SEARCH_META_PATH, "r", encoding="utf-8") as f:
                    self.docs = []
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        d = json.loads(s)
                        self.docs.append(
                            SearchDoc(
                                text=d["text"],
                                meta=d.get("meta") or {},
                                embedding=np.zeros(self.vector_dim, dtype="float32"),  # placeholder
                                timestamp=datetime.fromisoformat(d["timestamp"]),
                            )
                        )
            except Exception as exc:
                print(f"[SearchService] Failed to load search meta ({exc}); starting with empty corpus.")
                self.docs = []

        if self.docs:
            self._reindex_from_texts()
        else:
            self._mat = np.zeros((0, self.vector_dim), dtype="float32")

    def _persist(self) -> None:
        with open(SEARCH_META_PATH, "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d.as_dict()) + "\n")

    def _reindex_from_texts(self) -> None:
        if not self.docs:
            self._mat = np.zeros((0, self.vector_dim), dtype="float32")
            self._persist()
            return

        texts = [d.text for d in self.docs]
        if hasattr(self.embedding_service, "generate_embeddings"):
            mat = np.asarray(self.embedding_service.generate_embeddings(texts), dtype="float32")
        else:
            mat = np.vstack([self._embed(t) for t in texts]).astype("float32")

        mat = _l2_normalize_rows(mat)
        for doc, v in zip(self.docs, mat):
            doc.embedding = v
        self._mat = mat
        self._persist()

    # ---------- Public API ----------
    def add_document(self, text: str, meta: Optional[Dict[str, Any]] = None) -> SearchDoc:
        vec = self._embed(text)
        vec = _as_f32c(vec).reshape(1, -1)
        self._mat = np.vstack([self._mat, vec])
        doc = SearchDoc(text=text, meta=meta or {}, embedding=vec[0], timestamp=datetime.utcnow())
        self.docs.append(doc)
        self._persist()
        return doc

    def search(self, query_text: str, top_k: int = 5) -> List[SearchDoc]:
        if self._mat.shape[0] == 0:
            return []

        qv = self._embed(query_text).reshape(1, -1)  # (1, D)
        sims = (self._mat @ qv.T).ravel()

        k = int(min(top_k, sims.shape[0]))
        idxs = np.argpartition(-sims, k - 1)[:k]
        idxs = idxs[np.argsort(-sims[idxs])]

        out: List[SearchDoc] = []
        for rank, idx in enumerate(idxs):
            d = self.docs[int(idx)]
            d.score = float(sims[int(idx)])
            out.append(d)
        return out

    def reset_storage(self) -> None:
        try:
            if os.path.exists(SEARCH_META_PATH):
                os.remove(SEARCH_META_PATH)
        except Exception as exc:
            print(f"[SearchService] Failed to remove {SEARCH_META_PATH}: {exc}")
        self.docs = []
        self._mat = np.zeros((0, self.vector_dim), dtype="float32")
        self._persist()

    def reindex(self) -> None:
        self._reindex_from_texts()

    # ---------- Helpers ----------
    def _embed(self, text: str) -> np.ndarray:
        if hasattr(self.embedding_service, "generate_embedding"):
            vec = np.asarray(self.embedding_service.generate_embedding(text), dtype="float32")
        else:
            vec = np.asarray(self.embedding_service.embed(text), dtype="float32")  # type: ignore[attr-defined]

        if vec.shape[0] != self.vector_dim:
            self.vector_dim = int(vec.shape[0])
            self._reindex_from_texts()

        nrm = float(np.linalg.norm(vec)) + 1e-12
        return _as_f32c(vec / nrm)


# ---------------------------------------------------------------------------
# Module-level singleton + wrappers (back-compat)
# ---------------------------------------------------------------------------
_DEFAULT_SVC: Optional[SearchService] = None

def _svc() -> SearchService:
    global _DEFAULT_SVC
    if _DEFAULT_SVC is None:
        _DEFAULT_SVC = SearchService()
    return _DEFAULT_SVC

def add_document(text: str, meta: Optional[Dict[str, Any]] = None) -> SearchDoc:
    return _svc().add_document(text, meta)

def search(query_text: str, top_k: int = 5) -> List[SearchDoc]:
    return _svc().search(query_text, top_k)

def reset_storage() -> None:
    _svc().reset_storage()

def reindex() -> None:
    _svc().reindex()
