#!/usr/bin/env python3
"""
Generate JSONL tests and memory seeds from a TTL ontology + YAML config.

NEW in this version:
- Safe ASK substitution (no str.format on SPARQL braces)
- Recall answers resolve rdfs:label for resource objects
- --variants_* flags: expand each underlying fact into many paraphrased Qs
  so you can hit target sizes even if the KG is small.

Outputs:
  tests/<domain>/tests.jsonl
  tests/<domain>/seed_mem.jsonl
"""

import argparse, json, random, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from rdflib import Graph, URIRef, RDF, RDFS, Literal

# ------------ utils ------------
def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_graph(ttl_path: Path) -> Graph:
    g = Graph()
    g.parse(ttl_path.as_posix(), format="turtle")
    return g

def localname(uri: URIRef) -> str:
    s = str(uri)
    if "#" in s: return s.split("#")[-1]
    if "/" in s: return s.rstrip("/").split("/")[-1]
    return s

def first_literal(g: Graph, s: URIRef, preds: List[URIRef]) -> Optional[str]:
    for p in preds:
        for o in g.objects(s, p):
            if isinstance(o, Literal):
                return str(o)
            else:
                try:
                    for lab in g.objects(o, RDFS.label):
                        return str(lab)
                except Exception:
                    pass
    return None

def choose_name(g: Graph, s: URIRef, name_preds: List[URIRef]) -> str:
    n = first_literal(g, s, name_preds)
    return n or localname(s)

# light paraphrase/noise machinery
REPS = [
    (r"\bWhat is\b", ["What's", "What is the", "State the", "Kindly state the"]),
    (r"\bAccording to\b", ["Per", "As per", "Based on", "According to"]),
    (r"\bDoes\b", ["Is it true that", "Do we have", "Do we know if", "Does"]),
    (r"\bIs\b", ["Is it the case that", "Would you say", "Is"]),
]
TRAILS = ["", "", "", " (per DPP)", " (datasheet v2)", " (records)"]

def paraphrase_once(q: str, rnd: random.Random) -> str:
    out = q
    for pat, choices in REPS:
        out = re.sub(pat, rnd.choice(choices), out, flags=re.IGNORECASE)
    return out

def add_noise(q: str, rnd: random.Random) -> str:
    toks = q.split()
    if len(toks) > 3 and rnd.random() < 0.5:
        i = rnd.randrange(len(toks))
        toks[i] = toks[i].rstrip(".,?") + rnd.choice(["", "", "s", "e"])
    return " ".join(toks) + rnd.choice(TRAILS)

def variantize(q: str, n: int, rnd: random.Random) -> List[str]:
    out = []
    for _ in range(n):
        qq = q
        if rnd.random() < 0.9:
            qq = paraphrase_once(qq, rnd)
        if rnd.random() < 0.9:
            qq = add_noise(qq, rnd)
        out.append(qq)
    return out

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ------------ generation ------------
def select_products(g: Graph, class_iri: Optional[str]) -> List[URIRef]:
    if class_iri:
        cls = URIRef(class_iri)
        xs = [s for s in g.subjects(RDF.type, cls)]
        if xs:
            return xs
    # fallback: anything with an rdfs:label and at least one outgoing predicate
    subs = set()
    for s, _, _ in g.triples((None, RDFS.label, None)):
        subs.add(s)
    return list(subs)

def recall_items(g: Graph, prod: URIRef, fields: Dict[str, str]) -> List[Tuple[str,str]]:
    """
    For each predicate listed in YAML recall_fields, fetch the first value.
    If the object is a resource, use its rdfs:label if available, else localname.
    """
    out = []
    for slot, pred_iri in fields.items():
        p = URIRef(pred_iri)
        vals = []
        for o in g.objects(prod, p):
            if isinstance(o, Literal):
                vals.append(str(o))
            else:
                lab = first_literal(g, o, [RDFS.label])
                vals.append(lab or localname(o))
        if vals:
            out.append((slot, vals[0]))
    return out

def ask_boolean(g: Graph, ask_sparql: str, product_uri: URIRef, prefixes: Dict[str,str]) -> bool:
    # Only substitute {product}; do NOT interpret other braces from SPARQL.
    q = ask_sparql.replace("{product}", str(product_uri))
    res = g.query(q, initNs={k: URIRef(v) for k, v in prefixes.items()})
    return bool(getattr(res, "askAnswer", res))

def make_recall_q(name: str, slot: str) -> str:
    patterns = [
        f"What is the {slot} of {name}?",
        f"For {name}, what is the {slot}?",
        f"{name} — state the {slot}.",
        f"Please provide the {slot} for {name}.",
    ]
    return random.choice(patterns)

def make_open_q(name: str, slot: str, tmpl: str) -> str:
    return tmpl.replace("{name}", name).replace("{slot}", slot)

def mk_id(prefix: str, i: int) -> str:
    return f"{prefix}-{i:06d}"

def gen_for_domain(cfg_path: Path, out_dir: Path,
                   n_recall: int, n_logic: int, n_open: int,
                   v_recall: int, v_logic: int, v_open: int) -> Tuple[int,int,int]:
    cfg = load_yaml(cfg_path)
    g = load_graph(Path(cfg["ttl_path"]))

    # config parts
    prefixes = cfg.get("prefixes", {})
    class_iri = cfg.get("product_selector", {}).get("class_iri")
    name_preds = [URIRef(p) for p in cfg.get("name_predicates", [str(RDFS.label)])]
    fields: Dict[str,str] = cfg.get("recall_fields", {})
    logic_checks = cfg.get("logic_checks", [])
    open_templates = cfg.get("open_templates", ["According to documentation, what is the {slot} of {name}?"])
    parap_rate = float(cfg.get("paraphrase_rate", 0.2))
    noise_rate = float(cfg.get("noise_rate", 0.1))

    prods = select_products(g, class_iri)
    if not prods:
        raise SystemExit(f"No products found for class: {class_iri}")

    ensure_dir(out_dir)
    tests_fp = out_dir / "tests.jsonl"
    seeds_fp = out_dir / "seed_mem.jsonl"
    tests_fp.unlink(missing_ok=True)
    seeds_fp.unlink(missing_ok=True)

    rnd = random.Random(42)
    tid = 0

    # memory seeds: write once per product per field
    with seeds_fp.open("w", encoding="utf-8") as wseed:
        for p in prods:
            name = choose_name(g, p, name_preds)
            for slot, pred_iri in fields.items():
                piri = URIRef(pred_iri)
                for val in g.objects(p, piri):
                    text_val = str(val) if isinstance(val, Literal) else (first_literal(g, val, [RDFS.label]) or localname(val))
                    text = f"{name} {slot} is {text_val}."
                    rec = {"id": mk_id("mem", tid), "text": text, "tags": [slot, localname(p)]}
                    wseed.write(json.dumps(rec) + "\n")
                    tid += 1

    # pools
    recall_pool: List[Tuple[URIRef,str,str]] = []  # (prod, slot, answer)
    for p in prods:
        items = recall_items(g, p, fields)
        for slot, val in items:
            recall_pool.append((p, slot, val))

    logic_pool: List[Tuple[URIRef,dict,bool]] = [] # (prod, check_cfg, bool)
    for p in prods:
        for chk in logic_checks:
            ok = ask_boolean(g, chk["positive_ask"], p, prefixes)
            logic_pool.append((p, chk, ok))

    rnd.shuffle(recall_pool)
    rnd.shuffle(logic_pool)

    def maybe_paraphrase(q: str) -> str:
        return paraphrase_once(q, rnd) if rnd.random() < parap_rate else q

    def maybe_noise(q: str) -> str:
        return add_noise(q, rnd) if rnd.random() < noise_rate else q

    nrec = nlogic = nopen = 0
    with tests_fp.open("w", encoding="utf-8") as w:
        # RECALL (variant expansion)
        rid = 0
        for p, slot, ans in recall_pool:
            if nrec >= n_recall: break
            name = choose_name(g, p, name_preds)
            base_q = make_recall_q(name, slot)
            variants = variantize(base_q, max(1, v_recall), rnd)
            for q in variants:
                if nrec >= n_recall: break
                q = maybe_paraphrase(maybe_noise(q))
                rec = {
                    "id": mk_id("rec", rid),
                    "type": "recall",
                    "session": "s1",
                    "product": localname(p),
                    "query": q,
                    "expected_contains": str(ans),
                    "ontology_refs": [slot],
                }
                w.write(json.dumps(rec) + "\n")
                nrec += 1; rid += 1

        # LOGIC (variant expansion)
        lid = 0
        for p, chk, ok in logic_pool:
            if nlogic >= n_logic: break
            name = choose_name(g, p, name_preds)
            base_q = chk["question"].format(name=name)
            variants = variantize(base_q, max(1, v_logic), rnd)
            for q in variants:
                if nlogic >= n_logic: break
                q = maybe_paraphrase(maybe_noise(q))
                rec = {
                    "id": mk_id("log", lid),
                    "type": "logic",
                    "session": "s1",
                    "product": localname(p),
                    "query": q,
                    "expected_contains": chk["yes_answer"] if ok else chk["no_answer"],
                    "ontology_refs": [chk["name"]],
                }
                w.write(json.dumps(rec) + "\n")
                nlogic += 1; lid += 1

        # OPEN (variant expansion from recall slots)
        oid = 0
        k = 0
        while nopen < n_open and k < len(recall_pool):
            p, slot, ans = recall_pool[k]; k += 1
            name = choose_name(g, p, name_preds)
            tmpl = rnd.choice(open_templates)
            base_q = tmpl.replace("{name}", name).replace("{slot}", slot)
            variants = variantize(base_q, max(1, v_open), rnd)
            for q in variants:
                if nopen >= n_open: break
                q = maybe_paraphrase(maybe_noise(q))
                rec = {
                    "id": mk_id("opn", oid),
                    "type": "open",
                    "session": "s1",
                    "product": localname(p),
                    "query": q,
                    "expected_contains": str(ans),
                    "ontology_refs": [slot],
                }
                w.write(json.dumps(rec) + "\n")
                nopen += 1; oid += 1

    return nrec, nlogic, nopen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["battery","lexmark","viessmann","veissmann"], required=True)
    ap.add_argument("--n_recall", type=int, default=150)
    ap.add_argument("--n_logic", type=int, default=120)
    ap.add_argument("--n_open", type=int, default=90)
    ap.add_argument("--variants_recall", type=int, default=1)
    ap.add_argument("--variants_logic", type=int, default=1)
    ap.add_argument("--variants_open", type=int, default=1)
    args = ap.parse_args()

    # map alt spelling to canonical folder name
    dom = "viessmann" if args.domain == "veissmann" else args.domain

    cfg_path = Path(f"backend/config/domains/{dom}.yml")
    out_dir = Path(f"tests/{dom}")
    ensure_dir(out_dir)
    nrec, nlog, nopen = gen_for_domain(
        cfg_path, out_dir,
        args.n_recall, args.n_logic, args.n_open,
        args.variants_recall, args.variants_logic, args.variants_open
    )
    print(json.dumps({"domain": dom, "recall": nrec, "logic": nlog, "open": nopen}, indent=2))

if __name__ == "__main__":
    main()
