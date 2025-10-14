#!/usr/bin/env python3
import json, os, pathlib, hashlib

def ensure_dir(p):
    pathlib.Path(os.path.dirname(p)).mkdir(parents=True, exist_ok=True)

def write_jsonl(path, rows):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def rid(domain, idx):  # reproducible id
    return hashlib.sha1(f"{domain}-{idx}".encode()).hexdigest()[:10]

def qa(domain, idx, typ, q, a):
    rid = hashlib.sha1(f"{domain}-{idx}".encode()).hexdigest()[:10]
    return {
        "id": rid,
        "type": typ,
        "domain": domain,
        "question": q,
        "query": q,        # <-- add this line
        "answer": a,
    }

def main():
    # ----- EVAL FILES (these are linted) -----
    write_jsonl("tests/battery/tests.jsonl", [
        qa("battery", 1, "logic",  "Is recycled content ≥ 50% considered compliant?", "yes"),
        qa("battery", 2, "recall", "What is the canonical name for 'EPR'?", "Extended Producer Responsibility"),
        qa("battery", 3, "open",   "Quote the clause on collection targets (cite).", "Manufacturers must meet collection targets of ... [doc:eu-batt-2023]"),
    ])
    write_jsonl("tests/lexmark/tests.jsonl", [
        qa("lexmark", 1, "recall", "Which toner fits MS821?", "54G0H00"),
        qa("lexmark", 2, "logic",  "If paper jam is cleared and door closed, can printing resume?", "yes"),
    ])
    write_jsonl("tests/viessmann/tests.jsonl", [
        qa("viessmann", 1, "recall", "What does COP stand for?", "coefficient of performance"),
        qa("viessmann", 2, "open",   "Quote the definition of COP (cite).", "The ratio of heat output to electrical input. [doc:cop-def]"),
    ])
    # Optional extra suites if present in your repo:
    write_jsonl("tests/combined_eval/combined.jsonl", [
        qa("battery",  1, "logic",  "Does a cell with 60% recycled content comply?", "yes"),
        qa("lexmark",  2, "recall", "MS821 compatible toner?", "54G0H00"),
    ])
    write_jsonl("tests/dpp_textiles/tests.jsonl", [
        qa("textiles", 1, "recall", "What is the canonical name for 'DoC'?", "Declaration of Conformity"),
        qa("textiles", 2, "logic",  "If fiber content label is missing, is EU labeling compliant?", "no"),
    ])
    write_jsonl("tests/dpp_rl/tests.jsonl", [
        qa("battery",  1, "open",   "Quote clause about safety datasheets (cite).", "Provide a Material Safety Data Sheet ... [doc:msds]"),
    ])

    # ----- NON-EVAL FILES (ignored by linter) -----
    # Keep your seeds as-is or (re)write minimal valid JSON if empty:
    write_jsonl("tests/battery/seed_docs.jsonl", [
        {"id":"doc:eu-batt-2023","domain":"battery","type":"doc","question":"clause","answer":"Manufacturers must meet collection targets of ..."},
    ])
    write_jsonl("tests/battery/seed_mem.jsonl", [
        {"domain":"battery","type":"profile","question":"User printer model?","answer":"MS821"},
    ])
    write_jsonl("tests/lexmark/seed_docs.jsonl", [
        {"id":"doc:ms821-manual","domain":"lexmark","type":"doc","question":"clear jam","answer":"Open tray, remove jam, close tray."},
    ])
    write_jsonl("tests/lexmark/seed_mem.jsonl", [
        {"domain":"lexmark","type":"inventory","question":"Default toner model?","answer":"54G0H00"},
    ])
    write_jsonl("tests/viessmann/seed_docs.jsonl", [
        {"id":"doc:cop-def","domain":"viessmann","type":"doc","question":"cop def","answer":"The ratio of heat output to electrical input."},
    ])
    # If these exist in your tree, seed minimally:
    write_jsonl("tests/full_eval/doc_corpus.jsonl", [
        {"id":"doc:corpus-1","domain":"battery","type":"doc","question":"k","answer":"v"},
    ])
    write_jsonl("tests/full_eval/memory_pairs.jsonl", [
        {"id":"mp-1","domain":"battery","type":"pair","question":"support email","answer":"support@example.com"},
    ])
    write_jsonl("tests/full_eval/queries.jsonl", [
        {"id":"q-1","domain":"battery","type":"query","question":"What is support email?","answer":"support@example.com"},
    ])
    write_jsonl("tests/memory_eval/dpp_samples.jsonl", [
        {"id":"mem-1","domain":"battery","type":"memory","question":"Store: plan","answer":"Pro"},
    ])
    write_jsonl("tests/dpp_rl/episodes.jsonl", [
        {"id":"ep-1","domain":"battery","type":"episode","question":"state","answer":"action"},
    ])
    write_jsonl("tests/dpp_rl/seed_mem.jsonl", [
        {"id":"sm-1","domain":"battery","type":"memory","question":"foo","answer":"bar"},
    ])
    write_jsonl("tests/combined_eval/mem.jsonl", [
        {"id":"m-1","domain":"battery","type":"mem","question":"k","answer":"v"},
    ])
    write_jsonl("tests/dpp_textiles/seed_docs.jsonl", [
        {"id":"doc:text-1","domain":"textiles","type":"doc","question":"k","answer":"v"},
    ])

if __name__ == "__main__":
    main()
