#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick sweep of HybridRetriever hyperparameters to pick sensible defaults.
This ONLY instantiates/warms the retriever to verify the combo is valid.
Use the results to set defaults (or env vars) and then re-run eval normally.
"""

from __future__ import annotations
import argparse, itertools, csv
from pathlib import Path

def main(corpus: str, out_csv: str):
    try:
        # Import inside main so the module import doesn't fail if deps missing.
        from backend.retrieval.hybrid import HybridRetriever  # type: ignore
    except Exception as e:
        raise SystemExit(f"Could not import HybridRetriever: {e}")

    grid = {
        "k_bm25": [30, 50, 80],
        "k_dense": [16, 32, 64],
        "k_final": [6, 8, 10],
    }

    rows = []
    for k_bm25, k_dense, k_final in itertools.product(grid["k_bm25"], grid["k_dense"], grid["k_final"]):
        rec = {"k_bm25": k_bm25, "k_dense": k_dense, "k_final": k_final}
        try:
            _ = HybridRetriever(corpus, k_bm25=k_bm25, k_dense=k_dense, k_final=k_final)
            rec["ok"] = 1
            rec["note"] = ""
        except TypeError:
            # If your HybridRetriever doesn’t accept kwargs, try positional fallback
            try:
                _ = HybridRetriever(corpus)
                rec["ok"] = 1
                rec["note"] = "kwargs not supported; using class defaults"
            except Exception as e:
                rec["ok"] = 0
                rec["note"] = f"init failed: {e}"
        except Exception as e:
            rec["ok"] = 0
            rec["note"] = f"init failed: {e}"
        rows.append(rec)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["k_bm25","k_dense","k_final","ok","note"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="backend/corpus/dpp_corpus.jsonl")
    ap.add_argument("--out", default="tables/retriever_sweep.csv")
    a = ap.parse_args()
    main(a.corpus, a.out)
