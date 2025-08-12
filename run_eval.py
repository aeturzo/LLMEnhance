import os, csv, json, uuid, pathlib, time
from backend.config.run_modes import RunMode
from backend.services.symbolic_reasoning_service import build_reasoner
from backend.services import memory_service, search_service
from fastapi.testclient import TestClient
from backend.main import app

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "tests" / "combined_eval" / "combined.jsonl"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

def run(mode: RunMode):
    os.environ["RUN_MODE"] = mode.value
    app.state.run_mode = mode
    app.state.reasoner = build_reasoner(run_owl_rl=(mode in (RunMode.SYM, RunMode.MEMSYM)))
    client = TestClient(app)
    rid = f"{mode.value}-{uuid.uuid4().hex[:6]}"
    rows = []
    for line in DATA.read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        t0 = time.time()
        resp = client.post("/api/solve", json={"session_id": ex["session"], "query": ex["query"], "product": ex.get("product")})
        t1 = time.time()
        out = resp.json()
        rows.append({
            "id": ex["id"], "mode": mode.value, "latency_ms": round((t1-t0)*1000,2),
            "answer": out["answer"], "steps": len(out["steps"])
        })
    fp = ART / f"{rid}.csv"
    with fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"Wrote {fp}")
    return fp

if __name__ == "__main__":
    print("Seeding combined eval...")
    import tests.combined_eval.seed_combined  # runs on import
    for m in (RunMode.BASE, RunMode.MEM, RunMode.SYM, RunMode.MEMSYM):
        run(m)
