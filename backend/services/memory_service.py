#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backend/services/memory_service.py
Long‑term episodic memory service for the hybrid LLM architecture with on‑disk
persistence of BOTH the FAISS vector index **and** the accompanying metadata
(`MemoryEntry` rows).

This version fixes the earlier bug where only `memory.index` was written, so
`self.entries` became empty after a restart. We now write a compact JSONL file
alongside the FAISS binary and reload it at startup.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Optional

import faiss
import numpy as np

# Local import
from backend.services.embedding_service import EmbeddingService

# ---------------------------------------------------------------------------
# Constants for persistence
# ---------------------------------------------------------------------------
INDEX_PATH = "memory.index"          # FAISS binary
META_PATH = "memory.meta.jsonl"      # 1 JSON object per MemoryEntry


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
        self.embedding = embedding  # ℓ2‑normalised vector
        self.timestamp: datetime = timestamp or datetime.utcnow()
        self.score: Optional[float] = None  # filled by .retrieve()

    # ------------------------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"<MemoryEntry sess={self.session_id!r} ts={self.timestamp.isoformat()} "
            f"score={self.score:.3f if self.score is not None else '–'}>"
        )


class MemoryService:
    """Vector‑based long‑term memory with FAISS similarity search."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_dim: int = 768,
    ):
        self.embedding_service = embedding_service
        self.vector_dim = vector_dim

        self.index = faiss.IndexFlatIP(vector_dim)
        self.entries: List[MemoryEntry] = []
        self._load_index_and_meta()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_index_and_meta(self) -> None:
        """Load FAISS index + JSONL metadata if present."""
        if os.path.exists(INDEX_PATH):
            try:
                self.index = faiss.read_index(INDEX_PATH)
            except Exception as exc:  # corrupt file
                print(f"[MemoryService] Corrupt index ({exc}); recreating.")
                self.index = faiss.IndexFlatIP(self.vector_dim)

        if os.path.exists(META_PATH):
            try:
                with open(META_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        self.entries.append(
                            MemoryEntry(
                                data["session_id"],
                                data["content"],
                                np.zeros(self.vector_dim, dtype="float32"),  # placeholder; not needed for look‑up
                                datetime.fromisoformat(data["timestamp"]),
                            )
                        )
            except Exception as exc:
                print(f"[MemoryService] Failed to load metadata ({exc}); entries cleared.")
                self.entries = []

    def _persist(self) -> None:
        """Persist both FAISS index and metadata."""
        if self.index.ntotal > 0:
            faiss.write_index(self.index, INDEX_PATH)
            with open(META_PATH, "w", encoding="utf-8") as f:
                for e in self.entries:
                    f.write(json.dumps(e.as_dict()) + "\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_memory(self, session_id: str, content: str) -> MemoryEntry:
        emb = self._embed(content)
        self.index.add(emb.reshape(1, -1))
        entry = MemoryEntry(session_id, content, emb)
        self.entries.append(entry)
        self._persist()
        return entry

    def retrieve(self, session_id: str, query: str, top_k: int = 5) -> List[MemoryEntry]:
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
        kept = [e for e in self.entries if e.session_id != session_id]
        removed = len(self.entries) - len(kept)
        if removed:
            self.entries = kept
            self.index.reset()
            if kept:
                mat = np.stack([e.embedding for e in kept]).astype("float32")
                self.index.add(mat)
            self._persist()
        return removed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedding_service.generate_embedding(text)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.astype("float32")