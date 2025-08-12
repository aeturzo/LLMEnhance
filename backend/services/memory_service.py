#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backend/services/memory_service.py

Long-term episodic memory service for the hybrid LLM architecture with on-disk
persistence of BOTH the FAISS vector index **and** the accompanying metadata
(`MemoryEntry` rows).

- Always initializes `self.entries`
- Loads FAISS + JSONL meta on startup
- Auto-rebuilds FAISS from entries if dim/count mismatch
- Uses the embedding backend's REAL dimension
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Optional

import faiss
import numpy as np

from backend.services.embedding_service import EmbeddingService

# ---------------------------------------------------------------------------
# Persistence paths (relative to repo root)
# ---------------------------------------------------------------------------
INDEX_PATH = "memory.index"          # FAISS binary
META_PATH = "memory.meta.jsonl"      # 1 JSON object per MemoryEntry


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
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
        self.embedding = embedding  # ℓ2-normalised vector (float32)
        self.timestamp: datetime = timestamp or datetime.utcnow()
        self.score: Optional[float] = None  # filled by .retrieve()

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
# Service
# ---------------------------------------------------------------------------
class MemoryService:
    """Vector-based long-term memory with FAISS similarity search."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_dim: int | None = None,
    ):
        # Embeddings
        self.embedding_service = embedding_service
        self.vector_dim = int(vector_dim or getattr(embedding_service, "vector_dim", 768))

        # Storage
        self.entries: List[MemoryEntry] = []
        self.index = faiss.IndexFlatIP(self.vector_dim)

        # Load persisted state (and heal if needed)
        self._load_index_and_meta()

    # ----------------------------- Persistence -----------------------------
    def _load_index_and_meta(self) -> None:
        """Load FAISS index + JSONL metadata if present; heal dim/count mismatches."""
        # Load FAISS index (if any)
        if os.path.exists(INDEX_PATH):
            try:
                self.index = faiss.read_index(INDEX_PATH)
            except Exception as exc:
                print(f"[MemoryService] Corrupt index ({exc}); recreating empty index.")
                self.index = faiss.IndexFlatIP(self.vector_dim)

        # Dim healing: if loaded index dimension != backend dim, rebuild empty
        idx_dim = getattr(self.index, "d", self.vector_dim)
        if idx_dim != self.vector_dim:
            print(f"[MemoryService] Index dim {idx_dim} != backend dim {self.vector_dim}; rebuilding empty index.")
            self.index = faiss.IndexFlatIP(self.vector_dim)

        # Load metadata (entries)
        if os.path.exists(META_PATH):
            try:
                with open(META_PATH, "r", encoding="utf-8") as f:
                    self.entries = []
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        self.entries.append(
                            MemoryEntry(
                                session_id=data["session_id"],
                                content=data["content"],
                                embedding=np.zeros(self.vector_dim, dtype="float32"),  # placeholder; will be rebuilt
                                timestamp=datetime.fromisoformat(data["timestamp"]),
                            )
                        )
            except Exception as exc:
                print(f"[MemoryService] Failed to load metadata ({exc}); starting with empty entries.")
                self.entries = []

        # If counts mismatch or embeddings are placeholders, rebuild index from entries
        if self.entries:
            needs_rebuild = (
                self.index.ntotal != len(self.entries)
                or any(e.embedding is None or not np.any(e.embedding) for e in self.entries)
            )
            if needs_rebuild:
                self._rebuild_index_from_entries()

    def _persist(self) -> None:
        """Persist both FAISS index and metadata (idempotent)."""
        # Write FAISS index (even if 0 vectors, to keep things consistent)
        faiss.write_index(self.index, INDEX_PATH)
        # Write metadata JSONL
        with open(META_PATH, "w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(e.as_dict()) + "\n")

    def _rebuild_index_from_entries(self) -> None:
        """Re-embed all entries and rebuild FAISS (used on load/heal)."""
        self.index = faiss.IndexFlatIP(self.vector_dim)
        if not self.entries:
            self._persist()
            return

        texts = [e.content for e in self.entries]
        # Prefer batch embedding if available
        if hasattr(self.embedding_service, "generate_embeddings"):
            mat = np.asarray(self.embedding_service.generate_embeddings(texts), dtype="float32")
        else:
            # Fallback: single-by-single
            mat = np.vstack([self._embed(t) for t in texts]).astype("float32")

        # Normalize to unit length for IP ~ cosine
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        mat = (mat / norms).astype("float32")

        # Update in-memory entries with fresh embeddings
        for e, v in zip(self.entries, mat):
            e.embedding = v

        self.index.add(mat)
        self._persist()

    # ------------------------------- Public API ----------------------------
    def add_memory(self, session_id: str, content: str) -> MemoryEntry:
        """Store a memory row and append its vector to FAISS, then persist."""
        # Ensure entries list exists (defensive)
        if not hasattr(self, "entries"):
            self.entries = []

        emb = self._embed(content)
        self.index.add(emb.reshape(1, -1))
        entry = MemoryEntry(session_id, content, emb)
        self.entries.append(entry)
        self._persist()
        return entry

    def retrieve(self, session_id: str, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Return top-k memories for this session ordered by similarity."""
        if self.index.ntotal == 0:
            return []

        qv = self._embed(query)
        D, I = self.index.search(qv.reshape(1, -1), top_k)
        hits: List[MemoryEntry] = []
        for rank, idx in enumerate(I[0]):
            if idx == -1 or idx >= len(self.entries):
                continue
            entry = self.entries[idx]
            if entry.session_id == session_id:
                entry.score = float(D[0][rank])
                hits.append(entry)
        return hits

    def flush_session(self, session_id: str) -> int:
        """Remove all memories for a session and rebuild FAISS."""
        if not hasattr(self, "entries"):
            self.entries = []

        kept = [e for e in self.entries if e.session_id != session_id]
        removed = len(self.entries) - len(kept)
        if removed:
            self.entries = kept
            if kept:
                # Rebuild FAISS from kept entries
                self._rebuild_index_from_entries()
            else:
                # No entries left: reset and persist clean state
                self.index = faiss.IndexFlatIP(self.vector_dim)
                self._persist()
        return removed

    # ------------------------------- Helpers -------------------------------
    def _embed(self, text: str) -> np.ndarray:
        """Single-text embedding normalized to unit length (float32)."""
        # Call the embedding backend (single or batch API)
        if hasattr(self.embedding_service, "generate_embedding"):
            vec = np.asarray(self.embedding_service.generate_embedding(text), dtype="float32")
        else:
            # Extremely unlikely given our EmbeddingService, but keep a fallback
            vec = np.asarray(self.embedding_service.embed(text), dtype="float32")  # type: ignore[attr-defined]
        # Normalize for inner-product similarity
        norm = float(np.linalg.norm(vec)) + 1e-12
        return (vec / norm).astype("float32")
