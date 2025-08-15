#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backend/services/memory_service.py  (FAISS-free, pure NumPy)

Long-term episodic memory with on-disk persistence of JSONL metadata and a
vector matrix kept in RAM. Cosine similarity via dot-product of L2-normalized
embeddings.

Public API:
- Class:
    - MemoryService.add_memory(session_id, content) -> MemoryEntry
    - MemoryService.retrieve(session_id, query, top_k=5) -> List[MemoryEntry]
    - MemoryService.flush_session(session_id) -> int
    - MemoryService.reset_storage() -> None
    - MemoryService.reindex() -> None
- Module-level wrappers (for backward compatibility with existing code):
    - add_memory(session_id, content)
    - retrieve(session_id, query, top_k=5)
    - flush_session(session_id)
    - reset_storage()
    - reindex()
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Optional

import numpy as np

from backend.services.embedding_service import EmbeddingService, get_embedder

# ---------------------------------------------------------------------------
# Persistence paths (absolute, resolved from repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
META_PATH = os.path.join(REPO_ROOT, "memory.meta.jsonl")  # JSONL metadata only


def _as_f32c(a: np.ndarray) -> np.ndarray:
    """Return a float32, C-contiguous ndarray view/copy."""
    return np.ascontiguousarray(np.asarray(a, dtype="float32"), dtype="float32")


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """L2-normalize rows of (N, D)."""
    mat = _as_f32c(mat)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return _as_f32c(mat / norms)


class MemoryEntry:
    """Container for a single memory record."""
    def __init__(
        self,
        session_id: str,
        content: str,
        embedding: np.ndarray,
        timestamp: Optional[datetime] = None,
    ):
        self.session_id = session_id
        self.content = content
        self.embedding = _as_f32c(embedding)  # L2-normalized
        self.timestamp: datetime = timestamp or datetime.utcnow()
        self.score: Optional[float] = None  # filled by retrieve()

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        score_str = f"{self.score:.3f}" if self.score is not None else "–"
        return f"<MemoryEntry sess={self.session_id!r} ts={self.timestamp.isoformat()} score={score_str}>"


# ---------------------------------------------------------------------------
# Service (pure NumPy index)
# ---------------------------------------------------------------------------
class MemoryService:
    """Vector-based long-term memory with NumPy cosine search."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_dim: int | None = None,
    ):
        # Embeddings
        self.embedding_service: EmbeddingService = embedding_service or get_embedder()

        # Prefer backend's declared dim; fall back to provided hint or 384
        backend_dim = int(getattr(self.embedding_service, "vector_dim", 0) or 0)
        self.vector_dim = int(vector_dim or backend_dim or 384)

        # Storage
        self.entries: List[MemoryEntry] = []
        self._mat: np.ndarray = np.zeros((0, self.vector_dim), dtype="float32")  # (N, D) L2-normalized

        # Load metadata (if any) and rebuild vectors
        self._load_meta_and_rebuild()

    # ----------------------------- Persistence -----------------------------
    def _load_meta_and_rebuild(self) -> None:
        """Load JSONL metadata and rebuild the vector matrix from content."""
        if os.path.exists(META_PATH):
            try:
                with open(META_PATH, "r", encoding="utf-8") as f:
                    self.entries = []
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        data = json.loads(s)
                        self.entries.append(
                            MemoryEntry(
                                session_id=data["session_id"],
                                content=data["content"],
                                embedding=np.zeros(self.vector_dim, dtype="float32"),  # placeholder
                                timestamp=datetime.fromisoformat(data["timestamp"]),
                            )
                        )
            except Exception as exc:
                print(f"[MemoryService] Failed to load metadata ({exc}); starting with empty entries.")
                self.entries = []

        # Rebuild vectors if we have entries
        if self.entries:
            self._rebuild_index_from_entries()
        else:
            self._mat = np.zeros((0, self.vector_dim), dtype="float32")

    def _persist(self) -> None:
        """Persist metadata JSONL."""
        with open(META_PATH, "w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(e.as_dict()) + "\n")

    def _rebuild_index_from_entries(self) -> None:
        """Re-embed all entries and rebuild the NumPy matrix."""
        if not self.entries:
            self._mat = np.zeros((0, self.vector_dim), dtype="float32")
            self._persist()
            return

        texts = [e.content for e in self.entries]
        if hasattr(self.embedding_service, "generate_embeddings"):
            mat = np.asarray(self.embedding_service.generate_embeddings(texts), dtype="float32")
        else:
            mat = np.vstack([self._embed(t) for t in texts]).astype("float32")

        mat = _l2_normalize_rows(mat)

        # Update entries and in-memory matrix
        for e, v in zip(self.entries, mat):
            e.embedding = v
        self._mat = mat
        self._persist()

    # ------------------------------- Public API ----------------------------
    def add_memory(self, session_id: str, content: str) -> MemoryEntry:
        """Store a memory row and append its vector, then persist."""
        vec = self._embed(content)                      # (D,)
        vec = _as_f32c(vec).reshape(1, -1)              # (1, D)
        self._mat = np.vstack([self._mat, vec])         # append row

        entry = MemoryEntry(session_id, content, vec[0])
        self.entries.append(entry)
        self._persist()
        return entry

    def retrieve(self, session_id: str, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Return top-k memories for this session ordered by cosine similarity."""
        if self._mat.shape[0] == 0:
            return []

        qv = self._embed(query).reshape(1, -1)          # (1, D), L2-normalized
        sims = (self._mat @ qv.T).ravel()               # cosine via dot product

        # Top-k over all, then filter by session
        k = int(min(top_k, sims.shape[0]))
        idxs = np.argpartition(-sims, k - 1)[:k]
        idxs = idxs[np.argsort(-sims[idxs])]

        hits: List[MemoryEntry] = []
        for rank, idx in enumerate(idxs):
            e = self.entries[int(idx)]
            if e.session_id == session_id:
                e.score = float(sims[int(idx)])
                hits.append(e)
        return hits

    def flush_session(self, session_id: str) -> int:
        """Remove all memories for a session and rebuild matrix."""
        kept = [e for e in self.entries if e.session_id != session_id]
        removed = len(self.entries) - len(kept)
        if removed:
            self.entries = kept
            self._rebuild_index_from_entries()
        return removed

    # ------------------------------- Maintenance helpers -------------------
    def reset_storage(self) -> None:
        """Delete meta file on disk and reset in-memory state."""
        try:
            if os.path.exists(META_PATH):
                os.remove(META_PATH)
        except Exception as exc:
            print(f"[MemoryService] Failed to remove {META_PATH}: {exc}")
        self.entries = []
        self._mat = np.zeros((0, self.vector_dim), dtype="float32")
        self._persist()

    def reindex(self) -> None:
        """Force a fresh re-embedding & rebuild from current entries."""
        self._rebuild_index_from_entries()

    # ------------------------------- Helpers -------------------------------
    def _embed(self, text: str) -> np.ndarray:
        """Single-text embedding normalized to unit length (float32)."""
        if hasattr(self.embedding_service, "generate_embedding"):
            vec = np.asarray(self.embedding_service.generate_embedding(text), dtype="float32")
        else:
            vec = np.asarray(self.embedding_service.embed(text), dtype="float32")  # type: ignore[attr-defined]

        # If backend dimension differs (e.g., first OpenAI call), realign and rebuild
        if vec.shape[0] != self.vector_dim:
            self.vector_dim = int(vec.shape[0])
            # Resize matrix & rebuild from text to match the new dim
            self._rebuild_index_from_entries()

        # L2-normalize and return
        nrm = float(np.linalg.norm(vec)) + 1e-12
        return _as_f32c(vec / nrm)


# ---------------------------------------------------------------------------
# Module-level singleton + wrapper functions (back-compat)
# ---------------------------------------------------------------------------
_DEFAULT_SVC: Optional[MemoryService] = None

def _svc() -> MemoryService:
    global _DEFAULT_SVC
    if _DEFAULT_SVC is None:
        _DEFAULT_SVC = MemoryService()
    return _DEFAULT_SVC

# Old call style: backend.services.memory_service.add_memory(...)
def add_memory(session_id: str, content: str) -> MemoryEntry:
    return _svc().add_memory(session_id, content)

def retrieve(session_id: str, query: str, top_k: int = 5) -> List[MemoryEntry]:
    return _svc().retrieve(session_id, query, top_k)

def flush_session(session_id: str) -> int:
    return _svc().flush_session(session_id)

def reset_storage() -> None:
    _svc().reset_storage()

def reindex() -> None:
    _svc().reindex()
