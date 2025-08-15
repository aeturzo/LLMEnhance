import os, csv, json, uuid, pathlib, time
from fastapi.testclient import TestClient
from backend.config.run_modes import RunMode
from backend.services.symbolic_reasoning_service import build_reasoner
from backend.services import memory_service
from backend.main import app

ROOT = pathlib.Path(__file__).parent
DATA_COMBINED = ROOT / "tests" / "combined_eval" / "combined.jsonl"  # your existing mixed set
DATA_RL = ROOT / "tests" / "dpp_rl" / "episodes.jsonl"
DATA_MEMSEED = ROOT / "tests" / "dpp_rl" / "seed_mem.jsonl"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

def seed_memory():
    if DATA_MEMSEED.exists():
        with open(DATA_MEMSEED, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    memory_service.add_memory(row["session"], row["memory"])

def try_load_jsonl(path: pathlib.Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def success_contains(ex: dict, answer: str) -> str:
    gold = ex.get("gold") or {}
    s = gold.get("contains")
    if s is None: return ""
    return "1" if s.lower() in (answer or "").lower() else "0"

if __name__ == "__main__":
    app.state.reasoner = build_reasoner(run_owl_rl=True)
    seed_memory()
    client = TestClient(app)

    rid = uuid.uuid4().hex[:8]

    # 1) Classic modes
    combined = try_load_jsonl(DATA_COMBINED)
    for mode in (RunMode.BASE, RunMode.MEM, RunMode.SYM, RunMode.MEMSYM):
        rows = []
        for ex in combined:
            t0 = time.time()
            resp = client.post("/solve", json=ex, headers={"X-Run-Mode": mode.value})
            t1 = time.time()
            out = resp.json()
            rows.append({
                "id": ex.get("id","-"),
                "mode": mode.value,
                "latency_ms": round((t1-t0)*1000, 2),
                "steps": len(out.get("steps", [])),
                "answer": out.get("answer",""),
                "success_contains": success_contains(ex, out.get("answer",""))
            })
        fp = ART / f"eval_{mode.value}_{rid}.csv"
        with fp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        print(f"Wrote {fp}")

    # 2) RL on DPP episodes
    episodes = try_load_jsonl(DATA_RL)
    rows = []
    for ex in episodes:
        t0 = time.time()
        payload = {"query": ex["query"], "product": ex.get("product"), "session": ex.get("session","default")}
        resp = client.post("/solve_rl", json=payload)
        t1 = time.time()
        out = resp.json()
        rows.append({
            "id": ex["id"],
            "mode": "RL",
            "latency_ms": round((t1-t0)*1000,2),
            "steps": len(out.get("steps", [])),
            "answer": out.get("answer",""),
            "success_contains": success_contains(ex, out.get("answer",""))
        })
    fp = ART / f"eval_RL_{rid}.csv"
    with fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"Wrote {fp}")
