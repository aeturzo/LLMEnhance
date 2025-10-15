# scripts/gen_synth.py
# Generate synthetic compliance QA + docs + memory for one domain.
import argparse, json, random
from pathlib import Path

CHEM = ["Li-ion","Li-poly","NiMH","NiCd"]
STD  = {"Li-ion":"EN_62133_2","Li-poly":"EN_62133_2","NiMH":"IEC_61951","NiCd":"IEC_61951"}
LIM  = {"Pb":100, "Cd":50, "Hg":5}

def pid(doc_id, n): return f"{doc_id}#p{n}"

def wjsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for r in rows: w.write(json.dumps(r, ensure_ascii=False)+"\n")

def gen(domain, n_dev=300, n_test=300, target_docs=220, min_mem=80, seed=73):
    rng = random.Random(seed)
    dev, test, mem, docs = [], [], [], []
    products = [f"{domain[:1].upper()}-{i:04d}" for i in range(1, target_docs+1)]

    for k, prod in enumerate(products, 1):
        chem = rng.choice(CHEM); std = STD[chem]
        doc_id = f"doc_{domain}_{prod}"
        date   = f"2024-06-{rng.randint(10,28)}"
        metal  = rng.choice(list(LIM)); lim = LIM[metal]
        ok, bad = rng.randint(0,lim), rng.randint(lim+1, lim+50)

        chunks = [
          {"pid": pid(doc_id,1), "text": f"{prod} chemistry: {chem}. Report date {date}."},
          {"pid": pid(doc_id,2), "text": f"{std} tests passed for {prod}."},
          {"pid": pid(doc_id,3), "text": f"{metal} content measured {ok} ppm (< {lim})."},
          {"pid": pid(doc_id,4), "text": f"{metal} content measured {bad} ppm (> {lim})."},
        ]
        docs.append({"doc_id": doc_id, "title": f"Test Report for {prod}", "text": "", "chunks": chunks,
                     "meta":{"product": prod, "date": date, "domain": domain}})

        split = "dev" if k % 2 == 0 else "test"
        out = dev if split=="dev" else test

        out.append({
          "id": f"{domain}.Q{k*10+1:08d}","domain":domain,"type":"compliance",
          "question": f"Is {prod} compliant with {std}?",
          "gold_answer":"yes","gold_rationale":"evidence-backed",
          "gold_evidence":[pid(doc_id,2)],"product":prod,"aliases":[prod.replace('-','')],
          "symbols_expected":[f"requiresCompliance({prod},{std})"] if chem in ("Li-ion","Li-poly") else [],
          "memory_refs":[]
        })
        out.append({
          "id": f"{domain}.Q{k*10+2:08d}","domain":domain,"type":"compliance",
          "question": f"Does {prod} violate {metal} threshold limits?",
          "gold_answer":"yes","gold_rationale":"evidence-backed",
          "gold_evidence":[pid(doc_id,4)],"product":prod,"aliases":[prod.replace('-','')],
          "symbols_expected":[f"exceedsLimit({prod},{metal})"],"memory_refs":[]
        })
        out.append({
          "id": f"{domain}.Q{k*10+3:08d}","domain":domain,"type":"compliance",
          "question": f"Is there sufficient information to confirm {prod} compliance with {std}?",
          "gold_answer":"insufficient","gold_rationale":"not enough evidence",
          "gold_evidence":[pid(doc_id,1)],"product":prod,"aliases":[prod.replace('-','')],
          "symbols_expected":[f"requiresCompliance({prod},{std})"] if chem in ("Li-ion","Li-poly") else [],
          "memory_refs":[]
        })
        if rng.random() < 0.30:
            key = f"mem_{prod}_std"
            mem.append({"key":key,"value":f"{prod} passed {std} on {date}.",
                        "scope":"session","tags":[domain,"compliance",prod]})
            out.append({
              "id": f"{domain}.Q{k*10+4:08d}","domain":domain,"type":"compliance",
              "question": f"(Memory) Did {prod} pass {std} per the most recent report?",
              "gold_answer":"yes","gold_rationale":"memory-backed",
              "gold_evidence":[pid(doc_id,1)],"product":prod,"aliases":[prod.replace('-','')],
              "symbols_expected":[f"requiresCompliance({prod},{std})"],"memory_refs":[key]
            })

    # pad/trim to exact sizes
    if len(dev) < n_dev: dev += dev[:(n_dev-len(dev))]
    if len(test) < n_test: test += test[:(n_test-len(test))]
    dev, test = dev[:n_dev], test[:n_test]

    # ensure min memory
    i = 0
    while len(mem) < min_mem:
        base = mem[i % max(1,len(mem))]; i += 1
        mem.append({**base, "key": base["key"] + f"_extra{i}"})

    outdir = Path(f"tests/{domain}")
    wjsonl(outdir/"dev.jsonl", dev)
    wjsonl(outdir/"test.jsonl", test)
    wjsonl(outdir/"seed_mem.jsonl", mem)
    wjsonl(outdir/"seed_docs.jsonl", docs)
    print(f"[OK] {domain}: dev={len(dev)} test={len(test)} mem={len(mem)} docs={len(docs)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["battery","lexmark","viessmann"])
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--docs", type=int, default=220)
    ap.add_argument("--mem", type=int, default=80)
    ap.add_argument("--seed", type=int, default=73)
    args = ap.parse_args()
    gen(args.domain, args.n_dev, args.n_test, args.docs, args.mem, args.seed)
