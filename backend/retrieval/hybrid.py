from pathlib import Path
from typing import List, Dict, Any
import json, os
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

class HybridRetriever:
    """
    Hybrid sparse+dense retrieval with cross-encoder reranking.
    Caches embeddings to artifacts/cache to avoid recompute.
    """
    def __init__(self, corpus_jsonl: str,
                 dense_model: str = "intfloat/e5-large-v2",
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 k_bm25: int = 50, k_dense: int = 50, k_final: int = 10,
                 cache_dir: str = "artifacts/cache"):
        self.k_bm25, self.k_dense, self.k_final = k_bm25, k_dense, k_final
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_path = Path(corpus_jsonl)
        self.docs, self.texts = self._load_corpus(self.corpus_path)

        # BM25
        self.tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)

        # Dense encoder
        self.embedder = SentenceTransformer(dense_model)
        self.doc_emb = self._load_or_build_embeddings(self.texts, dense_model)

        # Cross-encoder
        self.reranker = CrossEncoder(cross_encoder_model)

    def _load_corpus(self, p: Path):
        docs, texts = [], []
        with p.open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                d = {
                    "id": o["id"],
                    "title": o.get("title",""),
                    "text": o.get("text",""),
                    "domain": o.get("domain",""),
                    "source": o.get("source",""),
                }
                docs.append(d); texts.append(d["text"] or "")
        return docs, texts

    def _emb_cache_paths(self, dense_model: str):
        stem = self.corpus_path.stem + "_" + dense_model.replace("/", "_")
        return (self.cache_dir / f"{stem}.npy", self.cache_dir / f"{stem}.meta")

    def _load_or_build_embeddings(self, texts, dense_model: str):
        nppath, metapath = self._emb_cache_paths(dense_model)
        if nppath.exists() and metapath.exists():
            with open(metapath, "r") as m:
                m_info = json.load(m)
            if m_info.get("n_docs") == len(texts):
                return np.load(nppath)
        # build
        with torch.no_grad():
            emb = self.embedder.encode(texts, normalize_embeddings=True,
                                       batch_size=128, convert_to_numpy=True)
        np.save(nppath, emb)
        with open(metapath, "w") as m:
            json.dump({"n_docs": len(texts)}, m)
        return emb

    def search(self, query: str) -> List[Dict[str, Any]]:
        # Sparse
        bm_scores = self.bm25.get_scores(query.split())
        bm_idx = np.argsort(-bm_scores)[:self.k_bm25]

        # Dense
        q_emb = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        dn_scores = (self.doc_emb @ q_emb)
        dn_idx = np.argsort(-dn_scores)[:self.k_dense]

        # Union and rerank
        cand = sorted(set(bm_idx.tolist() + dn_idx.tolist()))
        pairs = [[query, self.texts[i]] for i in cand]
        ce = self.reranker.predict(pairs)
        order = np.argsort(-ce)[:self.k_final]

        out = []
        for rank, j in enumerate(order):
            i = cand[j]
            out.append({
                "rank": rank+1,
                "score": float(ce[j]),
                **self.docs[i]
            })
        return out
