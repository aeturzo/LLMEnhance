import json, sys, pathlib, glob

ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
OUT   = ROOT / "backend" / "corpus" / "dpp_corpus.jsonl"

def flatten_seed_docs():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w", encoding="utf-8") as out:
        for domain in ("battery","lexmark","viessmann"):
            # prefer seed_docs.jsonl, fallback to docs.jsonl
            for fp in [TESTS/domain/"seed_docs.jsonl", TESTS/domain/"docs.jsonl"]:
                if not fp.exists(): continue
                for line in fp.read_text(encoding="utf-8").splitlines():
                    if not line.strip(): continue
                    doc = json.loads(line)
                    doc_id = doc.get("doc_id") or doc.get("id") or f"{domain}_doc_{n}"
                    title  = doc.get("title","")
                    if "chunks" in doc:                      # doc-with-chunks -> many passages
                        for ch in doc["chunks"]:
                            pid = ch.get("pid"); txt = (ch.get("text") or "").strip()
                            if not pid or not txt: continue
                            out.write(json.dumps({"pid": pid, "text": txt, "doc_id": doc_id, "title": title}, ensure_ascii=False) + "\n")
                            n += 1
                    elif "pid" in doc and "text" in doc:     # already-flat passage
                        out.write(json.dumps({"pid": doc["pid"], "text": doc["text"], "doc_id": doc_id, "title": title}, ensure_ascii=False) + "\n")
                        n += 1
                    elif doc.get("text","").strip():         # fallback: dump doc-level text
                        pid = f"{doc_id}#p1"
                        out.write(json.dumps({"pid": pid, "text": doc["text"].strip(), "doc_id": doc_id, "title": title}, ensure_ascii=False) + "\n")
                        n += 1
    print(f"Wrote {OUT} with {n} passages")

if __name__ == "__main__":
    flatten_seed_docs()
