#!/usr/bin/env python3
"""Run ADAPTIVERAG on the corrected release-clean benchmark.

This evaluator is intentionally separate from run_eval_all.py because the clean
release benchmark lives in artifacts/release_benchmark_clean_docs.json rather
than tests/{domain}/tests.jsonl. It writes a baseline-compatible CSV so it can
be used directly by scripts/aggregate_baselines.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The clean evaluator does not need paid embeddings. temp_env.sh may set the
# embedding backend to OpenAI for other experiments; force a local/default model
# here unless explicitly overridden.
if os.environ.get("ADAPTIVERAG_CLEAN_ALLOW_OPENAI_EMBED") != "1":
    os.environ["EMBED_MODEL_NAME"] = os.environ.get(
        "ADAPTIVERAG_CLEAN_EMBED_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

from backend.main import app  # noqa: E402
from backend.retrieval.hybrid import HybridRetriever  # noqa: E402
from backend.services.symbolic_reasoning_service import build_reasoner  # noqa: E402
from scripts.run_baselines import (  # noqa: E402
    DEFAULT_CLEAN_RELEASE_JSON,
    DEFAULT_DOCS_ROOT,
    EVAL_CSV_FIELDS,
    load_clean_release_benchmark,
    load_ontology_lines,
    load_seed_doc_chunks,
    load_seed_mem_facts,
    score_answer,
)

DOMAINS = ("battery", "lexmark", "viessmann")
BAD_API_TERMS = ("RateLimitError", "APITimeoutError", "APIConnectionError", "APIStatusError", "InternalServerError")


def compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def ontology_header(lines: List[str], max_chars: int = 900) -> str:
    out: List[str] = []
    used = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("@prefix") or stripped.startswith("#"):
            continue
        keep = (
            "rdfs:Class" in stripped
            or "owl:Class" in stripped
            or "rdfs:subClassOf" in stripped
            or "rdf:Property" in stripped
            or "rdfs:domain" in stripped
            or "rdfs:range" in stripped
            or re.search(r"\b(?:ex|ext):[A-Za-z][A-Za-z0-9_.-]+\s+a\s+(?:ex|ext):", stripped)
        )
        if not keep:
            continue
        if used + len(line) + 1 > max_chars:
            break
        out.append(line.rstrip())
        used += len(line) + 1
    return "\n".join(out)


def ontology_windows(domain: str, window: int = 42, overlap: int = 10) -> List[Dict[str, str]]:
    lines = load_ontology_lines(domain)
    if not lines:
        return []
    header = ontology_header(lines)
    rows: List[Dict[str, str]] = []

    note = (
        "Ontology note: XML Schema datatypes such as integer, nonPositiveInteger, "
        "unsignedLong, dateTime, and dateTimeStamp are Datatype. In OWL/RDF, "
        "named ontology resources can be treated as Thing when class membership "
        "is queried."
    )
    rows.append({
        "id": f"clean_{domain}_ontology_schema",
        "doc_id": f"clean_{domain}_ontology",
        "domain": domain,
        "title": f"{domain} ontology schema",
        "text": f"Domain: {domain}\n{note}\n\nOntology schema/context:\n{header}",
    })

    label_lines: Dict[str, str] = {}
    resource_labels: Dict[str, str] = {}
    label_values: Dict[str, str] = {}
    subclasses: Dict[str, List[str]] = defaultdict(list)
    for line in lines:
        stripped = line.strip()
        m = re.match(r"^(\S+)\s+rdfs:label\s+\"([^\"]+)\"", stripped)
        if not m:
            sm = re.match(r"^(\S+).*?\brdfs:subClassOf\s+(\S+)", stripped)
            if sm:
                subclasses[sm.group(1)].append(sm.group(2).rstrip(" .;"))
            continue
        label_lines.setdefault(m.group(1), line.rstrip())
        resource_labels.setdefault(m.group(1), m.group(2))
        label_values.setdefault(m.group(2).lower(), m.group(1))

    def qname_label(ref: str) -> str:
        if ref in resource_labels:
            return resource_labels[ref]
        name = ref.split(":", 1)[-1].strip(" .;")
        return re.sub(r"(?<!^)([A-Z])", r" \1", name).replace("_", " ").strip()

    def class_ancestors(cls: str) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        stack = [cls]
        while stack:
            cur = stack.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            out.append(cur)
            stack.extend(subclasses.get(cur, []))
        return out

    seen_labels: set[str] = set()
    for i, line in enumerate(lines):
        m = re.match(r"^(\S+)\s+rdfs:label\s+\"([^\"]+)\"", line.strip())
        if not m:
            continue
        subject, label = m.group(1), m.group(2)
        label_key = label.lower()
        # Keep first occurrence only. The release ontology contains repeated
        # synthetic labels; clean gold was generated from the first occurrence.
        if label_key in seen_labels:
            continue
        seen_labels.add(label_key)

        start = i
        for j in range(i, max(-1, i - 10), -1):
            stripped = lines[j].strip()
            if stripped.startswith(subject + " a ") or stripped.startswith(subject + " rdf:type"):
                start = j
                break
        end = min(len(lines), start + window)
        prefix_match = re.match(r"^(.*_)\d+$", subject)
        if prefix_match:
            prefix = re.escape(prefix_match.group(1))
            for j in range(start + 1, min(len(lines), start + window * 2)):
                stripped = lines[j].strip()
                if re.match(rf"^{prefix}\d+\s+a\s+", stripped) and not stripped.startswith(subject + " "):
                    end = j
                    break
        chunk_lines = [ln.rstrip() for ln in lines[start:end] if ln.strip()]
        type_classes = []
        for ln in chunk_lines:
            tm = re.match(rf"^{re.escape(subject)}\s+a\s+(\S+)", ln.strip())
            if tm:
                type_classes.extend(x.rstrip(" .;") for x in tm.group(1).split(","))
        refs = set(re.findall(r"\b(?:ext|ex):[A-Za-z0-9_.\-]+", "\n".join(chunk_lines)))
        for ref in sorted(refs):
            label_line = label_lines.get(ref)
            if label_line and label_line not in chunk_lines:
                chunk_lines.append(label_line)
        inferred_labels: List[str] = []
        for cls in type_classes:
            for anc in class_ancestors(cls):
                inferred_labels.append(qname_label(anc))
        inferred_labels.append("Thing")
        inferred_labels = list(dict.fromkeys(x for x in inferred_labels if x))
        if inferred_labels:
            chunk_lines.insert(0, f'Inferred class labels for "{label}": ' + ", ".join(inferred_labels) + ".")

        body = "\n".join(chunk_lines)
        text = (
            f"Domain: {domain}\n"
            f"Ontology facts:\n{body}\n\n"
            f"{note}\n\n"
            f"Ontology schema/context:\n{header}"
        ).strip()
        rows.append({
            "id": f"clean_{domain}_ontology_{len(rows):06d}",
            "doc_id": f"clean_{domain}_ontology",
            "domain": domain,
            "title": f"{domain} ontology entity {label}",
            "text": text,
        })
    return rows


def build_clean_corpus(docs_root: Path, out: Path, force: bool = False) -> Path:
    if out.exists() and not force:
        return out
    rows: List[Dict[str, Any]] = []
    for domain in DOMAINS:
        for doc in load_seed_doc_chunks(docs_root, domain):
            text = (
                f"Document ID: {doc['id']}\n"
                f"Domain: {domain}\n"
                f"Source: {doc.get('source') or doc['id']}\n\n"
                f"{doc['text']}"
            )
            rows.append({
                "id": doc["id"],
                "doc_id": doc["id"],
                "domain": domain,
                "title": doc.get("source") or doc["id"],
                "text": text,
            })
        for fact in load_seed_mem_facts(docs_root, domain):
            text = f"Domain: {domain}\nMemory fact ID: {fact['id']}\n\n{fact['text']}"
            rows.append({
                "id": f"{domain}_mem_{fact['id']}",
                "doc_id": f"{domain}_seed_mem",
                "domain": domain,
                "title": f"{domain} memory fact",
                "text": text,
            })
        rows.extend(ontology_windows(domain))
    n = write_jsonl(out, rows)
    print(f"[adaptiverag-clean] wrote clean corpus: {out} rows={n}")
    return out


def load_filter_ids(path: Path | None) -> set[Tuple[str, str | None]] | None:
    if not path:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["rows"] if isinstance(data, dict) and "rows" in data else data
    keep: set[Tuple[str, str | None]] = set()
    for entry in items:
        if isinstance(entry, str):
            if "|" in entry:
                qid, dom = entry.split("|", 1)
                keep.add((qid, dom))
            else:
                keep.add((entry, None))
        elif isinstance(entry, dict):
            keep.add((entry["id"], entry.get("domain")))
    return keep


def apply_filters(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    keep_ids = load_filter_ids(args.filter_ids)
    prefixes = None
    if args.filter_prefix:
        prefixes = {p.strip() for p in args.filter_prefix.split(",") if p.strip()}

    out: List[Dict[str, Any]] = []
    for row in rows:
        if prefixes is not None:
            pref = row["id"].split("-")[0] if "-" in row["id"] else row["id"].split(".")[0]
            if pref not in prefixes:
                continue
        if keep_ids is not None:
            if (row["id"], row["domain"]) not in keep_ids and (row["id"], None) not in keep_ids:
                continue
        out.append(row)

    out.sort(key=lambda r: (r["id"], r["domain"]))

    if args.sample_mixed:
        out = mixed_sample(out, args.sample_mixed)
    if args.limit:
        out = out[:args.limit]
    return out


def mixed_sample(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["domain"], row.get("type", "open"))].append(row)
    keys = sorted(buckets)
    selected: List[Dict[str, Any]] = []
    idx = 0
    while len(selected) < n and keys:
        key = keys[idx % len(keys)]
        bucket = buckets[key]
        if bucket:
            selected.append(bucket.pop(0))
        keys = [k for k in keys if buckets[k]]
        idx += 1
    selected.sort(key=lambda r: (r["id"], r["domain"]))
    return selected


def existing_done(path: Path) -> set[Tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        return {(r["id"], r["domain"]) for r in csv.DictReader(fh)}


def configure_runtime(corpus_path: Path, llm_disabled: bool) -> TestClient:
    os.environ["CORPUS_PATH"] = str(corpus_path.resolve())
    os.environ.setdefault("RETRIEVE_K", "8")
    # Keep context tight. This prevents records questions from being polluted
    # by later product-document passages after the relevant memory/ontology rows.
    os.environ["MAX_PASSAGES"] = os.environ.get("ADAPTIVERAG_CLEAN_MAX_PASSAGES", "4")
    os.environ.setdefault("MAX_CTX_CHARS", "12000")
    os.environ.setdefault("OPENAI_RESPONSES_DISABLED", "1")
    # The release-clean rows often lack product IDs for KB questions. Keep the
    # adaptive policy from taking SEARCH solely because the fallback feature
    # value is 0.0; MEMSYM still falls back to retrieval when symbolic reasoning
    # cannot fire, while true logic rows can use the KG path first.
    os.environ["ADAPTIVE_MEM_T"] = "999.0"
    os.environ["ADAPTIVE_SEARCH_T"] = "0.0"
    os.environ["DISABLE_CARBON_ROUTING"] = "1"
    if llm_disabled:
        os.environ["LLM_DISABLED"] = "1"
    elif os.environ.get("LLM_DISABLED") == "1":
        del os.environ["LLM_DISABLED"]

    import backend.api.solve as solve_mod

    solve_mod._RETRIEVER = HybridRetriever(corpus_path=corpus_path, use_dense=False, topk_default=8)
    app.state.reasoners = {domain: build_reasoner(run_owl_rl=True, domain=domain) for domain in DOMAINS}
    app.state.reasoner = app.state.reasoners["battery"]
    return TestClient(app)


def audit_retrieval(rows: List[Dict[str, Any]], retriever: HybridRetriever, top_k: int = 8) -> Dict[str, Any]:
    checked = 0
    gold_in_context = 0
    misses: List[Dict[str, str]] = []
    by_type: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        expected = row.get("expected_contains") or ""
        if not expected:
            continue
        query = f"{row.get('product') or row['domain']} {row['query']}".strip()
        hits = retriever.search(query, k=top_k)
        context = "\n".join(h.get("text") or "" for h in hits)
        ok = compact_text(expected) in compact_text(context)
        checked += 1
        gold_in_context += int(ok)
        by_type[row.get("type", "open")][0] += int(ok)
        by_type[row.get("type", "open")][1] += 1
        if not ok and len(misses) < 10:
            misses.append({
                "id": row["id"],
                "domain": row["domain"],
                "type": row.get("type", ""),
                "expected_contains": expected,
                "top_ids": ",".join(str(h.get("id")) for h in hits[:3]),
            })
    return {
        "checked": checked,
        "gold_in_context": gold_in_context,
        "rate": gold_in_context / checked if checked else 0.0,
        "by_type": {k: {"ok": v[0], "n": v[1], "rate": v[0] / v[1] if v[1] else 0.0} for k, v in sorted(by_type.items())},
        "misses": misses,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-json", type=Path, default=DEFAULT_CLEAN_RELEASE_JSON)
    ap.add_argument("--docs-root", type=Path, default=DEFAULT_DOCS_ROOT)
    ap.add_argument("--corpus-out", type=Path, default=REPO_ROOT / "artifacts" / "adaptiverag_clean" / "corpus_release_clean.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-mixed", type=int, default=None, help="balanced sample across domain/type before --limit")
    ap.add_argument("--filter-prefix", default=None)
    ap.add_argument("--filter-ids", type=Path, default=None)
    ap.add_argument("--force-corpus", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--llm-disabled", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    rows = load_clean_release_benchmark(args.benchmark_json)
    keys = {(r["id"], r["domain"]) for r in rows}
    missing_gold = [r for r in rows if not r.get("expected_contains")]
    if len(keys) != len(rows):
        raise SystemExit(f"duplicate (id, domain) keys in clean benchmark: rows={len(rows)} unique={len(keys)}")
    if missing_gold:
        raise SystemExit(f"clean benchmark has rows without gold: {len(missing_gold)}")

    filtered = apply_filters(rows, args)
    print(f"[adaptiverag-clean] benchmark rows={len(rows)} filtered={len(filtered)} out={args.out}")
    corpus_path = build_clean_corpus(args.docs_root, args.corpus_out, force=args.force_corpus)
    client = configure_runtime(corpus_path, llm_disabled=args.llm_disabled)

    import backend.api.solve as solve_mod

    if args.audit_only:
        audit = audit_retrieval(filtered, solve_mod._RETRIEVER, top_k=int(os.environ.get("RETRIEVE_K", "8")))
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0 if audit["rate"] >= 0.85 else 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_done(args.out)
    if done:
        print(f"[adaptiverag-clean] resuming: {len(done)} rows already present")
    write_header = not args.out.exists()
    ok_total = 0
    n_total = 0
    api_error_rows = 0

    with args.out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVAL_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for idx, row in enumerate(filtered, start=1):
            key = (row["id"], row["domain"])
            if key in done:
                continue
            os.environ["DPP_DOMAIN"] = row["domain"]
            payload = {
                "mode": "ADAPTIVERAG",
                "query": row["query"],
                "product": row.get("product") or row["domain"],
                "session": row.get("session") or "s1",
                "session_id": row.get("session") or "s1",
            }
            t0 = time.time()
            try:
                resp = client.post("/solve", json=payload)
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
                out = resp.json()
                answer = out.get("answer", "") or ""
                steps = out.get("steps", []) or []
                confidence = out.get("confidence", "")
            except Exception as exc:
                answer = f"{type(exc).__name__}: {exc}"
                steps = []
                confidence = ""
            latency_ms = int((time.time() - t0) * 1000)
            success = max(0, score_answer(answer, row["expected_contains"], row.get("type", ""), row["query"]))
            ok_total += success
            n_total += 1
            if any(term in answer for term in BAD_API_TERMS):
                api_error_rows += 1
            writer.writerow({
                "id": row["id"],
                "mode": "ADAPTIVERAG",
                "type": row.get("type", ""),
                "domain": row["domain"],
                "query": row["query"],
                "product": row.get("product", ""),
                "session": row.get("session", "s1"),
                "success": success,
                "steps": json.dumps(steps)[:5000],
                "correct": success,
                "latency_ms": latency_ms,
                "confidence": confidence,
                "confidence_raw": confidence,
                "confidence_cal": confidence,
                "cost_retrieval_calls": 1,
                "cost_rule_checks": 0,
                "cost_tokens_in": 0,
                "cost_tokens_out": 0,
                "n_steps": len(steps),
                "answer": answer[:1000],
                "expected_contains": row["expected_contains"],
                "cost_usd_running": "0.0000",
            })
            fh.flush()
            if idx % 25 == 0:
                print(f"[ADAPTIVERAG] {idx}/{len(filtered)} acc_so_far={ok_total / max(n_total, 1):.4f} last_success={success}")
            if args.sleep:
                time.sleep(args.sleep)

    with args.out.open(encoding="utf-8") as fh:
        out_rows = list(csv.DictReader(fh))
    scored = [r for r in out_rows if r.get("expected_contains")]
    acc = sum(int(r["success"]) for r in scored) / len(scored) if scored else 0.0
    api_error_rows = sum(1 for r in out_rows if any(term in (r.get("answer") or "") for term in BAD_API_TERMS))
    print(f"[ADAPTIVERAG] DONE n={len(scored)} accuracy={acc:.4f} api_error_rows={api_error_rows} out={args.out.resolve()}")
    return 0 if api_error_rows == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
