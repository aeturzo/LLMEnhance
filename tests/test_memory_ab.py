import json, time, statistics as stats, pathlib
from backend.services import memory_service, search_service

DATA = pathlib.Path(__file__).parent / "memory_eval" / "dpp_samples.jsonl"

def load_pairs():
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):   # ← add this
                continue
            yield json.loads(line)


def run_case(pair, use_memory: bool):
    sid, text, query, expected = (
        pair["session"], pair["content"], pair["query"], pair["relevant"]
    )
    memory_service.flush_session(sid)
    memory_service.add_memory(sid, text)

    t0 = time.time()
    if use_memory:
        got = bool(memory_service.retrieve(sid, query, top_k=1))
    else:
        got = bool(search_service.search(query_text=query, top_k=1))

    return (time.time() - t0) * 1e3, got == expected   # latency-ms, correct?

def eval_pipeline(use_memory: bool):
    lats, rights = zip(*(run_case(p, use_memory) for p in load_pairs()))
    print(f"{'MEM ON ' if use_memory else 'MEM OFF'} | "
          f"Acc {sum(rights)}/{len(rights)} | "
          f"p50 {stats.median(lats):.2f} ms")
    return sum(rights)

def test_memory_beats_baseline():
    baseline = eval_pipeline(False)
    with_mem = eval_pipeline(True)
    assert with_mem >= baseline, "Memory should not be worse than baseline"

