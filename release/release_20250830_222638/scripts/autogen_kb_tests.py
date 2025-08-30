#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autogenerate tests.jsonl from domain ontology:

- recall (literals): (s, p, "literal") in asserted graph
- recall (objects):  (s, p, o) where o is URIRef (object property)
- logic: inferred types s rdf:type C (present in closure but NOT asserted)

Writes/updates: tests/<domain>/tests.jsonl
De-dups by (query, expected_contains).
"""
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, Set

from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL
from owlrl import DeductiveClosure, OWLRL_Semantics

ROOT = Path(__file__).resolve().parents[1]
ONT = ROOT / "backend" / "ontologies"

def ttl_for(domain: str) -> Path:
    cand = ONT / f"{domain}_ontology.ttl"
    return cand if cand.exists() else (ONT / "dpp_ontology.ttl")

def localname(u: URIRef) -> str:
    s = str(u)
    if "#" in s: s = s.split("#", 1)[1]
    elif "/" in s: s = s.rsplit("/", 1)[1]
    return s

def label_of(g: Graph, u: URIRef) -> str:
    lab = next((str(o) for o in g.objects(u, RDFS.label)), None)
    return lab or localname(u)

def sanitize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def load_graph(path: Path) -> tuple[Graph, Graph]:
    g0 = Graph()
    g0.parse(path.as_posix(), format="turtle")
    g = Graph()
    for t in g0.triples((None, None, None)):
        g.add(t)
    DeductiveClosure(OWLRL_Semantics).expand(g)
    return g0, g

def literal_facts(g: Graph) -> Iterable[tuple[URIRef, URIRef, Literal]]:
    bad = {RDFS.label, RDF.type, OWL.sameAs}
    for s, p, o in g.triples((None, None, None)):
        if isinstance(o, Literal) and p not in bad:
            yield s, p, o

def object_facts(g: Graph) -> Iterable[tuple[URIRef, URIRef, URIRef]]:
    bad = {RDFS.label, RDF.type, OWL.sameAs}
    for s, p, o in g.triples((None, None, None)):
        if isinstance(o, URIRef) and p not in bad:
            yield s, p, o

def inferred_types(g0: Graph, g: Graph) -> Iterable[tuple[URIRef, URIRef]]:
    base_types: Set[tuple[URIRef, URIRef]] = set((s, o) for s, _, o in g0.triples((None, RDF.type, None)))
    for s, _, o in g.triples((None, RDF.type, None)):
        if (s, o) not in base_types:
            yield s, o

def make_q_literal(slab: str, plab: str, lit: str) -> tuple[str, str]:
    q = f"What is the {plab} of {slab}?"
    return q, lit

def make_q_object(slab: str, plab: str, olab: str) -> tuple[str, str]:
    # Use same language so eval substring rule still works
    q = f"What is the {plab} of {slab}?"
    return q, olab

def make_q_logic(slab: str, clab: str) -> tuple[str, str]:
    q = f"Is {slab} a {clab}?"
    return q, clab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["battery","lexmark","viessmann"])
    ap.add_argument("--n_recall", type=int, default=500)   # TOTAL recall (literal+object)
    ap.add_argument("--n_logic", type=int, default=400)
    ap.add_argument("--cap_per_subject", type=int, default=6, help="Max recall Qs per subject to avoid explosion")
    args = ap.parse_args()

    out_dir = ROOT / "tests" / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tests.jsonl"

    # existing for dedup
    seen = set()
    existing = []
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            j = json.loads(line)
            existing.append(j)
            seen.add((j.get("query",""), j.get("expected_contains","")))

    g0, g = load_graph(ttl_for(args.domain))

    # --- recall: literals ---
    rec_lit = []
    subj_counts: dict[str, int] = {}
    for s, p, o in literal_facts(g0):
        slab, plab, lit = label_of(g, s), label_of(g, p), str(o)
        q, exp = sanitize(f"What is the {plab} of {slab}?"), sanitize(lit)
        key = (q, exp)
        if key in seen: continue
        cnt = subj_counts.get(slab, 0)
        if cnt >= args.cap_per_subject: continue
        rec_lit.append({"type":"recall","query":q,"expected_contains":exp,"session":"s1"})
        subj_counts[slab] = cnt + 1
        seen.add(key)

    # --- recall: objects (single-hop object properties) ---
    rec_obj = []
    for s, p, o in object_facts(g0):
        slab, plab, olab = label_of(g, s), label_of(g, p), label_of(g, o)
        # skip overly-generic object labels
        if not olab or len(olab) < 2: continue
        q, exp = sanitize(f"What is the {plab} of {slab}?"), sanitize(olab)
        key = (q, exp)
        if key in seen: continue
        cnt = subj_counts.get(slab, 0)
        if cnt >= args.cap_per_subject: continue
        rec_obj.append({"type":"recall","query":q,"expected_contains":exp,"session":"s1"})
        subj_counts[slab] = cnt + 1
        seen.add(key)

    # sample to reach desired recall target
    random.shuffle(rec_lit); random.shuffle(rec_obj)
    recall_pool = rec_lit + rec_obj
    random.shuffle(recall_pool)
    recall_take = recall_pool[:args.n_recall]

    # --- logic from inferred types ---
    log_items = []
    for s, cls in inferred_types(g0, g):
        slab, clab = label_of(g, s), label_of(g, cls)
        q, exp = sanitize(f"Is {slab} a {clab}?"), sanitize(clab)
        key = (q, exp)
        if key in seen: continue
        log_items.append({"type":"logic","query":q,"expected_contains":exp,"session":"s1"})
        seen.add(key)
    random.shuffle(log_items)
    log_take = log_items[:args.n_logic]

    rows = existing + [
        {**ex, "id": ex.get("id") or f"{args.domain}.kb.{ex['type']}.{i:05d}"}
        for i, ex in enumerate(recall_take + log_take, start=len(existing)+1)
    ]

    with out_path.open("w", encoding="utf-8") as f:
        for j in rows:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    print(json.dumps({
        "domain": args.domain,
        "ttl": ttl_for(args.domain).as_posix(),
        "added_recall_total": len(recall_take),
        "added_logic": len(log_take),
        "total": len(rows),
        "out": out_path.as_posix()
    }, indent=2))

if __name__ == "__main__":
    main()
