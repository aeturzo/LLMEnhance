import json, time, statistics as stats, pathlib
from textwrap import shorten
from types import SimpleNamespace

from backend.services import memory_service, search_service

ROOT = pathlib.Path(__file__).parent
DOCS    = ROOT / "full_eval" / "doc_corpus.jsonl"
MEMS    = ROOT / "full_eval" / "memory_pairs.jsonl"
QUERIES = ROOT / "full_eval" / "queries.jsonl"

SIM_THRESHOLD = 0.50   # ignore mem hits below this cosine score

# ---------- helpers ------------------------------------------------------
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)

def ingest_doc_corpus():
    """Wrap each row in an object with '.content' attr expected by search_service."""
    for row in load_jsonl(DOCS):
        doc = SimpleNamespace(                   # minimal stub
            name=row["doc_id"],
            content=row["text"],
        )
        search_service.add_document(doc)
    search_service.build_index()

def populate_memory():
    for row in load_jsonl(MEMS):
        memory_service.add_memory(row["session"], row["memory"])

# --- helper ---------------------------------------------------------------
def best_snippet(hit_list):
    if not hit_list:
        return "–––"
    hit = hit_list[0]
    # SearchResult from search_service  ➜  has .snippet   (attribute)
    # MemoryEntry from memory_service   ➜  has .content
    if hasattr(hit, "snippet"):
        txt = hit.snippet
    else:                               # MemoryEntry
        txt = hit.content
    return shorten(txt, 45, placeholder="…")

# ---------- test ---------------------------------------------------------
def test_full_pipeline(capfd):
    ingest_doc_corpus()
    populate_memory()

    latencies, rights = [], []
    for q in load_jsonl(QUERIES):
        sid, query, expected = q["session"], q["query"], q["expected"]

        t0 = time.time()
        mem_hits = memory_service.retrieve(sid, query, top_k=1)
        doc_hits = search_service.search(query_text=query, top_k=1)

        print(f"   mem={mem_hits[0].score if mem_hits else 0:.2f} "f"doc={doc_hits[0].score if doc_hits else 0:.2f}")
        latency = (time.time() - t0) * 1e3

        # pick winner
        if mem_hits and mem_hits[0].score >= SIM_THRESHOLD:
            src, answer = "mem", best_snippet(mem_hits)
        elif doc_hits:
            src, answer = "doc", best_snippet(doc_hits)
        else:
            src, answer = "none", "–––"

        latencies.append(latency)
        rights.append(src == expected)

        print(f"[{src.upper():4}] {shorten(query, 33):33} "
              f"→ {answer:<45} ({latency:5.2f} ms)")

    print("— summary —")
    print(f"Accuracy {sum(rights)}/{len(rights)} | "
          f"p50 {stats.median(latencies):.2f} ms  "
          f"p95 {stats.quantiles(latencies, n=100)[94]:.2f} ms")

    assert all(rights), "Some queries hit the wrong store"
