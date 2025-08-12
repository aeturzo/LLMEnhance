import json, pathlib
from backend.services import memory_service, search_service, ingestion_service

ROOT = pathlib.Path(__file__).parent
for p in (ROOT.parent / "full_eval").glob("doc_*.jsonl"):
    for line in p.read_text(encoding="utf-8").splitlines():
        ingestion_service.process_text(json.loads(line)["text"])

for line in (ROOT / "mem.jsonl").read_text(encoding="utf-8").splitlines():
    obj = json.loads(line)
    memory_service.store(obj["session"], obj["memory"])
print("Seeded docs + memory.")