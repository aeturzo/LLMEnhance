# -*- coding: utf-8 -*-
"""
HybridRetriever: robust BM25/TF-IDF (+ optional dense) retriever.

Design goals (per project docs):
- Return ranked passages WITH provenance: IDs/scores and basic metadata.  # LLMEnhance.pdf §4 (Retrieval) 
- Be lenient about corpus schema: accept {id|pid|passage_id}, {text|content|body}.  # your KeyError 'id'
- Provide a simple .search(query) that old code can call with just one arg.
- Optionally combine dense and sparse scores if sentence-transformers is available.

Outputs (list[dict]):
{
  "id": "<pid>",            # stable passage id; also mirrored as 'pid' for compatibility
  "pid": "<pid>",
  "doc_id": "<doc id or ''>",
  "domain": "<domain or ''>",
  "title": "<title or ''>",
  "text": "<passage text>",
  "rank": 1,                # 1-based
  "score": 12.345,          # combined score
  "source": "corpus"
}
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# --- Optional deps: fall back gracefully if missing
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SK = True
except Exception:
    _HAS_SK = False
    TfidfVectorizer = None  # type: ignore

try:
    # sentence-transformers is optional; we won't import heavy models unless asked
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _HAS_ST = True
except Exception:
    _HAS_ST = False
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore


@dataclass
class Passage:
    pid: str
    text: str
    doc_id: str = ""
    domain: str = ""
    title: str = ""


class HybridRetriever:
    def __init__(self, corpus_path: str | Path,
                 use_dense: bool = True,
                 dense_model_name: str = "intfloat/e5-base-v2",
                 topk_default: int = 5) -> None:
        self.corpus_path = str(corpus_path)
        self.topk_default = topk_default
        self.use_dense = bool(use_dense and _HAS_ST)

        # 1) load corpus robustly (accept id/pid/passsage_id; text/content/body)
        self.passages: List[Passage] = self._load_corpus(self.corpus_path)

        # 2) build sparse index
        self._build_sparse()

        # 3) build dense index (optional)
        self._dense_model_name = dense_model_name
        self._dense_model = None
        self._dense_mat = None
        if self.use_dense and len(self.passages) > 0:
            try:
                self._dense_model = SentenceTransformer(self._dense_model_name)
                texts = [p.text for p in self.passages]
                self._dense_mat = self._dense_model.encode(texts, normalize_embeddings=True)
            except Exception:
                # fail soft: disable dense if anything goes wrong
                self.use_dense = False
                self._dense_model = None
                self._dense_mat = None

    # ---------------------------------------------------------------------
    # Internal: corpus loading / indexing
    # ---------------------------------------------------------------------
    def _load_corpus(self, path: str) -> List[Passage]:
        out: List[Passage] = []
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"corpus not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)

                # robust field mapping
                pid = o.get("id") or o.get("pid") or o.get("passage_id")
                text = o.get("text") or o.get("content") or o.get("body") or ""
                if not pid or not text:
                    # skip unusable rows
                    continue

                doc_id = o.get("doc_id") or o.get("docId") or ""
                domain = o.get("domain") or ""
                title = o.get("title") or o.get("name") or ""

                out.append(Passage(pid=pid, text=text, doc_id=doc_id, domain=domain, title=title))

        if not out:
            raise ValueError(f"no usable passages in {p} (expected id/pid + text fields)")
        return out

    def _build_sparse(self) -> None:
        texts = [p.text for p in self.passages]
        if _HAS_SK:
            # reasonable defaults for short passages
            self._tfidf = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=200000
            )
            self._tfidf_mat = self._tfidf.fit_transform(texts)
        else:
            # lightweight fallback: bag-of-words into dicts
            self._tfidf = None
            self._tfidf_mat = None
            self._bow_vocab = {}
            for t in texts:
                for w in t.lower().split():
                    self._bow_vocab[w] = self._bow_vocab.get(w, 0) + 1

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def search(self, query: str, *args, **kwargs) -> List[Dict[str, Any]]:
        """
        Old signature: search(query) -> list[dict]
        New options (backward compatible):
          - positional k as 2nd arg, or kwargs: k / n / top_k
        """
        k = self._parse_k(args, kwargs)

        # 1) sparse scores
        sp_scores = self._score_sparse(query)

        # 2) dense scores (optional)
        if self.use_dense and self._dense_model is not None and self._dense_mat is not None:
            qv = self._dense_model.encode([query], normalize_embeddings=True)[0]
            dn = (self._dense_mat @ qv).astype(float)  # cosine since normalized
            # blend: simple z-normalized sum
            sp = _zscore(sp_scores)
            dd = _zscore(dn)
            scores = 0.5 * sp + 0.5 * dd
        else:
            scores = sp_scores

        # 3) top-k
        ranked = _argtopk(scores, k)

        # 4) format rows with provenance
        out: List[Dict[str, Any]] = []
        for rank, idx in enumerate(ranked, 1):
            psg = self.passages[idx]
            out.append({
                "id": psg.pid,
                "pid": psg.pid,               # alias to help any code expecting 'pid'
                "doc_id": psg.doc_id,
                "domain": psg.domain,
                "title": psg.title,
                "text": psg.text,
                "rank": rank,
                "score": float(scores[idx]),
                "source": "corpus"
            })
        return out

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _parse_k(self, args, kwargs) -> int:
        # allow r.search(q), r.search(q, 5), r.search(q, k=5), r.search(q, n=5), r.search(q, top_k=5)
        k = None
        if len(args) >= 1:
            try:
                k = int(args[0])
            except Exception:
                k = None
        if k is None:
            for key in ("k", "n", "top_k", "topk"):
                if key in kwargs:
                    try:
                        k = int(kwargs[key])
                        break
                    except Exception:
                        pass
        return k if (isinstance(k, int) and k > 0) else self.topk_default

    def _score_sparse(self, query: str):
        if _HAS_SK and self._tfidf is not None:
            qv = self._tfidf.transform([query])
            # cosine similarity into 1D vector
            sims = cosine_similarity(qv, self._tfidf_mat)[0]
            return sims
        else:
            # crude fallback: token overlap
            qset = set(query.lower().split())
            scores = []
            for p in self.passages:
                overlap = len(qset.intersection(p.text.lower().split()))
                scores.append(float(overlap))
            return scores


# --- small utilities ---------------------------------------------------------

def _argtopk(arr, k: int) -> List[int]:
    # works for list or numpy
    if np is not None and hasattr(arr, "shape"):
        idx = np.argsort(arr)[-k:][::-1]
        return idx.tolist()
    else:
        return sorted(range(len(arr)), key=lambda i: arr[i], reverse=True)[:k]


def _zscore(x):
    # works for list or numpy array
    if np is not None and hasattr(x, "mean"):
        mu = float(x.mean())
        sd = float(x.std()) or 1.0
        return (x - mu) / sd
    else:
        x = list(x)
        mu = sum(x) / max(1, len(x))
        var = sum((v - mu) ** 2 for v in x) / max(1, len(x))
        sd = math.sqrt(var) or 1.0
        return [(v - mu) / sd for v in x]
