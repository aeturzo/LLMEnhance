"""
A/B benchmark with full debug trace.
doc-only  vs  doc+memory
"""
import json, time, statistics as stats, pathlib
from types import SimpleNamespace
from textwrap import shorten

from backend.services import search_service, memory_service

ROOT = pathlib.Path(__file__).parent / "full_eval"
DOCS, MEMS, QUERIES = (ROOT / fn for fn in (
    "doc_corpus.jsonl", "memory_pairs.jsonl", "queries.jsonl"))

SIM_THRESHOLD = 0.20        # treat memory hit as valid only above this

# ---------- helpers --------------------------------------------------------
def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]

def ingest_docs():
    search_service.documents.clear()
    for row in load(DOCS):
        search_service.add_document(
            SimpleNamespace(name=row["doc_id"], content=row["text"]))
    search_service.build_index()

def add_mem():
    for row in load(MEMS):
        memory_service.add_memory(row["session"], row["memory"])

def best(snippet_obj):
    if not snippet_obj:
        return "–––"
    hit = snippet_obj[0]
    text = hit.content if hasattr(hit, "content") else hit.snippet
    return shorten(text, 60, placeholder="…")

# core router --------------------------------------------------------------
def pipeline(query, sid, use_memory):
    """Returns chosen_store ('mem'|'doc'|'none'), answer_text, mem_score, doc_score."""
    mem_hits = memory_service.retrieve(sid, query, top_k=1) if use_memory else []
    doc_hits = search_service.search(query_text=query, top_k=1)

    mem_score = mem_hits[0].score if mem_hits else 0.0
    doc_score = doc_hits[0].score if doc_hits else 0.0

    if mem_score >= SIM_THRESHOLD:
        return "mem", best(mem_hits), mem_score, doc_score
    if doc_hits:
        return "doc", best(doc_hits), mem_score, doc_score
    return "none", "–––", mem_score, doc_score

# ------------------ the test ----------------------------------------------
def test_ab_memory_verbose(capfd):
    ingest_docs()
    add_mem()
    qs = load(QUERIES)

    for label, flag in (("MEM OFF", False), ("MEM ON ", True)):
        print(f"\n=== {label} ===")
        hits, lats = 0, []
        for row in qs:
            t0 = time.time()
            store, ans, mscore, dscore = pipeline(row["query"], row["session"], flag)
            lats.append((time.time() - t0)*1e3)
            if store != "none":
                hits += 1
            print(f"Q: {row['query']}\n"
                  f"   mem={mscore:.2f}  doc={dscore:.2f}   "
                  f"⟶ {store.upper():3}   {ans}")
        print(f"{label} | Recall {hits}/{len(qs)}  "
              f"p50 {stats.median(lats):.2f} ms")

    assert hits >= len(qs)/2, "Memory did not improve recall"  # coarse check
