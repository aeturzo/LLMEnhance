#!/usr/bin/env python3
import argparse, json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# -------------------- Helpers --------------------

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def load_jsonl(p: Path):
    if not p.exists(): return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try: out.append(json.loads(ln))
                except: pass
    return out

def append_jsonl(p: Path, rows):
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def pick_text(obj):
    return obj.get("text","") if isinstance(obj, dict) else ""

def make_id_factory(existing_ids, prefix):
    n = 0
    def next_id():
        nonlocal n
        while True:
            cand = f"{prefix}-{n:06d}"
            n += 1
            if cand not in existing_ids:
                existing_ids.add(cand)
                return cand
    return next_id

# -------------------- Extractors --------------------

# Accept :, =, en dash (–), em dash (—), and hyphen (-) as separators
SEP = r"[:=\-–—]"
KV_ANY = re.compile(rf"(?m)^\s*([A-Za-z][\w\s\-/()%]+?)\s*{SEP}\s*([^\n]+?)\s*$")

# Two-space aligned “Label  Value”
KV_TWOSPACE = re.compile(r"(?m)^\s*([A-Za-z][\w\s\-/()%]+?)\s{2,}([^\n]+?)\s*$")

# Bullet-style lines we’ll attempt to split on the first separator
BULLET = re.compile(r"(?m)^\s*(?:[•\-\*]\s+)([^\n]+?)\s*$")

NUMERIC = re.compile(
    r"\b\d+(?:[\.,]\d+)?\s?(?:kWh|Wh|W|kW|V|mAh|Ah|A|°C|C|kg|g|mm|cm|m|%|ppm|bar|psi|years?|months?|days?)\b",
    flags=re.IGNORECASE
)

# Heading “— Model Name …” pattern
HEADING_MODEL = re.compile(r"—\s*([^(\n]{3,80})")

def extract_title_like(text: str):
    first = text.splitlines()[0] if text else ""
    m = HEADING_MODEL.search(first)
    if not m: return None
    val = m.group(1).split("(")[0].strip()
    if not (3 <= len(val) <= 80): return None
    return ("model name", val)

def kv_pairs_rich(text: str):
    """Yield (label, value) from multiple formatting styles."""
    seen = set()
    # 1) Classic separators
    for m in KV_ANY.finditer(text):
        label, value = m.group(1).strip(), m.group(2).strip()
        k = (label.lower(), value)
        if label and value and k not in seen:
            seen.add(k); yield label, value
    # 2) Two-space alignment
    for m in KV_TWOSPACE.finditer(text):
        label, value = m.group(1).strip(), m.group(2).strip()
        k = (label.lower(), value)
        if label and value and k not in seen:
            seen.add(k); yield label, value
    # 3) Bullet lines: split on first separator if present
    for m in BULLET.finditer(text):
        line = m.group(1).strip()
        m2 = re.search(SEP, line)
        if m2:
            label = line[:m2.start()].strip()
            value = line[m2.end():].strip()
            if label and value:
                k = (label.lower(), value)
                if k not in seen:
                    seen.add(k); yield label, value

# -------------------- Generators (schema-compliant) --------------------

def mk_open_item(next_id, source_id, label, span):
    q = f"According to {source_id}, what is the {label}?"
    return {
        "id": next_id(),
        "type": "open",
        "session": "s_docs",
        "product": source_id,
        "query": q,
        "expected_contains": span,
        "meta": {"source_id": source_id, "label": label, "from": "docs"},
    }

def mk_recall_item(next_id, source_id, label, value):
    q = f"According to {source_id}, what is the {label}?"
    return {
        "id": next_id(),
        "type": "recall",
        "session": "s_docs",
        "product": source_id,
        "query": q,
        "expected_contains": value,
        "meta": {"source_id": source_id, "label": label, "from": "docs"},
    }

def mk_recall_numeric(next_id, source_id, numval, ctx):
    snip = re.sub(r"\s+", " ", ctx).strip()
    if len(snip) > 120: snip = snip[:120] + "..."
    q = f"In {source_id}, what is the value reported near: \"{snip}\"?"
    return {
        "id": next_id(),
        "type": "recall",
        "session": "s_docs",
        "product": source_id,
        "query": q,
        "expected_contains": numval,
        "meta": {"source_id": source_id, "from": "numeric"},
    }

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--n_open", type=int, default=800)
    ap.add_argument("--n_recall", type=int, default=250)
    args = ap.parse_args()

    dom = args.domain
    tests_path = ROOT / "tests" / dom / "tests.jsonl"
    seeds_path = ROOT / "tests" / dom / "seed_docs.jsonl"

    if not seeds_path.exists():
        print(json.dumps({"domain": dom, "error": f"missing {seeds_path}"}))
        return 0

    existing = load_jsonl(tests_path)
    existing_ids = {str(x.get("id")) for x in existing if x.get("id")}
    next_open_id = make_id_factory(existing_ids, "docopen")
    next_rec_id  = make_id_factory(existing_ids, "docrec")

    seen_queries = {norm(x.get("query") or "") for x in existing}
    seen_queries.discard("")

    seeds = load_jsonl(seeds_path)

    doc_ids = set()
    open_added, recall_added = [], []

    # pass 0: heading/title → open
    for idx, s in enumerate(seeds):
        if len(open_added) >= args.n_open: break
        text = pick_text(s)
        if not text: continue
        source_id = f"{dom}_seed_{idx:04d}"
        doc_ids.add(source_id)
        tkv = extract_title_like(text)
        if tkv:
            label, span = tkv
            item = mk_open_item(next_open_id, source_id, label, span)
            k = norm(item["query"])
            if k and k not in seen_queries:
                open_added.append(item); seen_queries.add(k)

    # pass 1: KV pairs → recall if numeric-with-unit else open
    for idx, s in enumerate(seeds):
        if len(open_added) >= args.n_open and len(recall_added) >= args.n_recall: break
        text = pick_text(s)
        if not text: continue
        source_id = f"{dom}_seed_{idx:04d}"
        doc_ids.add(source_id)

        for label, value in kv_pairs_rich(text):
            if len(recall_added) < args.n_recall and NUMERIC.search(value):
                item = mk_recall_item(next_rec_id, source_id, label, value)
                k = norm(item["query"])
                if k and k not in seen_queries:
                    recall_added.append(item); seen_queries.add(k)
            elif len(open_added) < args.n_open:
                # allow digits in open answers (e.g., model names)
                item = mk_open_item(next_open_id, source_id, label, value)
                k = norm(item["query"])
                if k and k not in seen_queries:
                    open_added.append(item); seen_queries.add(k)
        if len(open_added) >= args.n_open and len(recall_added) >= args.n_recall:
            break

    # pass 2: numeric fallbacks
    if len(recall_added) < args.n_recall:
        for idx, s in enumerate(seeds):
            if len(recall_added) >= args.n_recall: break
            text = pick_text(s)
            if not text: continue
            source_id = f"{dom}_seed_{idx:04d}"
            doc_ids.add(source_id)
            for m in NUMERIC.finditer(text):
                numval = m.group(0).strip()
                ctx = text[max(0, m.start()-40): m.end()+40]
                item = mk_recall_numeric(next_rec_id, source_id, numval, ctx)
                k = norm(item["query"])
                if k and k not in seen_queries:
                    recall_added.append(item); seen_queries.add(k)
                if len(recall_added) >= args.n_recall: break

    new_rows = open_added + recall_added
    if new_rows:
        append_jsonl(tests_path, new_rows)

    print(json.dumps({
        "domain": dom,
        "doc_sources": len(doc_ids),
        "added_doc_recall": len(recall_added),
        "added_open": len(open_added),
        "total": len(existing) + len(new_rows),
        "out": str(tests_path.resolve())
    }, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
