#!/usr/bin/env python3
"""Run same-row architecture smoke comparisons for AUTO_COMPOSE vs baselines."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if os.environ.get("ADAPTIVERAG_CLEAN_ALLOW_OPENAI_EMBED") != "1":
    os.environ["EMBED_MODEL_NAME"] = os.environ.get(
        "ADAPTIVERAG_CLEAN_EMBED_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

from openai import OpenAI  # type: ignore  # noqa: E402

from backend.services import memory_service  # noqa: E402
from scripts.run_adaptiverag_clean import (  # noqa: E402
    DEFAULT_DOCS_ROOT,
    build_clean_corpus,
    compact_text,
    configure_runtime,
)
from scripts.run_baselines import ontology_context, query_terms  # noqa: E402
from scripts.run_baselines import load_seed_doc_chunks, load_seed_mem_facts, ranked_items  # noqa: E402


MODES = ("AUTO_COMPOSE", "GPT4O_LONGCTX", "LINC", "LOGIC_LM")

PRODUCTS = {
    "battery": {
        "ProductA": {
            "components": ["WirelessModule1", "Board1", "Battery1"],
            "standards": ["EN 62133-2", "Battery Safety Standard", "Wireless Compliance Standard", "RoHS"],
            "steps": ["BatteryTestStep", "WirelessTestStep", "Battery Test Step", "Wireless Test Step"],
        },
        "ProductB": {
            "components": ["Board2"],
            "standards": ["RoHS"],
            "steps": [],
        },
        "ProductC": {
            "components": ["WirelessModule2"],
            "standards": ["Wireless Compliance Standard"],
            "steps": [],
        },
    },
    "lexmark": {
        "PrinterL1": {
            "components": ["Wlan1", "Board1", "Head1", "Toner1"],
            "standards": ["Wireless Compliance", "WEEE Marking", "EMC Compliance", "IEC 62368-1 Safety", "RoHS"],
            "steps": ["Wireless Test", "Print Quality Test", "Label Check"],
        },
        "PrinterL2": {
            "components": ["Toner2", "Board2", "Head2"],
            "standards": ["WEEE Marking", "EMC Compliance", "IEC 62368-1 Safety"],
            "steps": ["Print Quality Test", "Label Check"],
        },
    },
    "viessmann": {
        "ProductV1": {
            "components": ["Radio1", "Comp1", "Board1"],
            "standards": ["EU F-Gas Regulation", "Electrical Safety Check", "RoHS", "Wireless Compliance"],
            "steps": ["Leak Check", "Pressure Test", "Electrical Safety Test", "Wireless Test"],
        },
        "ProductV2": {
            "components": ["Comp2", "Board2"],
            "standards": ["Electrical Safety Check"],
            "steps": ["Leak Check", "Pressure Test", "Electrical Safety Test"],
        },
    },
}

PACKAGING = [
    "recycled cardboard",
    "an aluminum pouch",
    "a double-walled carton",
    "a returnable crate",
    "molded pulp packaging",
    "a reusable transit box",
    "paper-based cushioning",
    "a low-plastic shipper",
]

SUPPLIERS = [
    "GreenCells",
    "NordicPack",
    "PrintPack",
    "HeatPack",
    "EcoWrap",
    "CircularBox",
    "DPP Logistics",
    "MaterialLoop",
]

TEMPLATES = [
    {
        "subtype": "memory_symbolic_compliance_packaging",
        "query": "For {product}, combine my preferred packaging with one compliance standard required by the product record.",
        "groups": lambda facts, memory: [[memory["packaging"]], facts["standards"]],
    },
    {
        "subtype": "memory_symbolic_compliance_supplier",
        "query": "For {product}, answer with my preferred supplier and one compliance standard supported by the product record.",
        "groups": lambda facts, memory: [[memory["supplier"]], facts["standards"]],
    },
    {
        "subtype": "memory_symbolic_component_supplier",
        "query": "What is my preferred supplier for {product}, and what component is documented for {product}?",
        "groups": lambda facts, memory: [[memory["supplier"]], facts["components"]],
    },
    {
        "subtype": "memory_symbolic_component_packaging",
        "query": "For {product}, give my preferred packaging and one documented component.",
        "groups": lambda facts, memory: [[memory["packaging"]], facts["components"]],
    },
    {
        "subtype": "memory_symbolic_step_packaging",
        "query": "For {product}, give my preferred packaging and one required test step.",
        "groups": lambda facts, memory: [[memory["packaging"]], facts["steps"]],
        "requires_steps": True,
    },
]


def build_rows(n: int) -> List[Dict[str, Any]]:
    products: List[tuple[str, str, Dict[str, List[str]]]] = []
    for domain, items in PRODUCTS.items():
        for product, facts in items.items():
            products.append((domain, product, facts))

    rows: List[Dict[str, Any]] = []
    i = 0
    while len(rows) < n:
        domain, product, facts = products[i % len(products)]
        template = TEMPLATES[(i // len(products)) % len(TEMPLATES)]
        i += 1
        if template.get("requires_steps") and not facts["steps"]:
            continue
        memory = {
            "packaging": PACKAGING[len(rows) % len(PACKAGING)],
            "supplier": SUPPLIERS[(len(rows) * 3) % len(SUPPLIERS)],
        }
        row = {
            "id": f"arch-smoke100-{len(rows) + 1:03d}",
            "domain": domain,
            "type": "compose",
            "subtype": template["subtype"],
            "product": product,
            "session": f"arch_smoke100_s{len(rows) + 1:03d}",
            "memory": (
                f"For {product}, the preferred packaging is {memory['packaging']} "
                f"and the preferred supplier is {memory['supplier']}."
            ),
            "query": template["query"].format(product=product),
            "expected_groups": template["groups"](facts, memory),
        }
        rows.append(row)
    return rows


def infer_product(row: Dict[str, Any]) -> str:
    query = str(row.get("query") or "")
    m = re.match(r"^\s*is\s+(.+?)\s+an?\s+.+?\??\s*$", query, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.match(r"^\s*what is the\s+.+?\s+of\s+(.+?)\??\s*$", query, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    for pattern in (
        r"\bProductV\d+\b",
        r"\bPrinterL\d+\b",
        r"\bProduct[A-Za-z0-9_-]+\b",
        r"\bLiCell-\d+\b",
        r"\b(?:battery|lexmark|viessmann)_seed_\d+\b",
    ):
        m = re.search(pattern, query)
        if m:
            return m.group(0)
    return ""


def load_benchmark_rows(path: Path, limit: int | None = None, sample_mixed: int | None = None) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    out: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        r["product"] = (r.get("product") or r.get("solver_product") or infer_product(r)).strip()
        r.setdefault("session", "s1")
        if not isinstance(r.get("expected_groups"), list) or not r["expected_groups"]:
            expected = str(r.get("expected_contains") or "")
            r["expected_groups"] = [[part.strip()] for part in expected.split("||") if part.strip()]
        out.append(r)
    out.sort(key=lambda r: (r["id"], r["domain"]))
    if sample_mixed:
        buckets: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in out:
            buckets[(row["domain"], row.get("subtype") or row.get("type", ""))].append(row)
        keys = sorted(buckets)
        selected: List[Dict[str, Any]] = []
        idx = 0
        while len(selected) < sample_mixed and keys:
            key = keys[idx % len(keys)]
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop(0))
            keys = [k for k in keys if buckets[k]]
            idx += 1
        out = sorted(selected, key=lambda r: (r["id"], r["domain"]))
    if limit:
        out = out[:limit]
    return out


def score_answer(answer: str, groups: List[List[str]]) -> int:
    ca = compact_text(answer)
    return int(all(any(compact_text(value) in ca for value in group) for group in groups))


def existing_done(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(encoding="utf-8") as fh:
        return {
            (r.get("mode", ""), r.get("id", ""), r.get("domain", ""))
            for r in csv.DictReader(fh)
            if r.get("mode") and r.get("id") and r.get("domain")
        }


def as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def estimated_cost_usd(tokens_in: int, tokens_out: int, in_per_1m: float, out_per_1m: float) -> float:
    return (tokens_in / 1_000_000.0) * in_per_1m + (tokens_out / 1_000_000.0) * out_per_1m


def transient_llm_failure(trace: Dict[str, Any], answer: str) -> bool:
    path = str(trace.get("path") or "")
    reason = str(trace.get("reason") or "")
    haystack = f"{path} {reason} {answer}"
    markers = (
        "llm_error",
        "snippet_fallback",
        "Connection error",
        "APITimeout",
        "APIConnection",
        "APIStatus",
        "RateLimit",
        "InternalServer",
        "timeout",
    )
    return any(marker.lower() in haystack.lower() for marker in markers)


def solve_auto_with_retries(test_client: Any, payload: Dict[str, Any], retries: int, sleep_s: float) -> Dict[str, Any]:
    last: Dict[str, Any] | None = None
    for attempt in range(1, max(1, retries) + 1):
        resp = test_client.post("/solve_auto", json=payload)
        data = resp.json()
        trace = data.get("answer_trace") or {}
        answer = data.get("answer", "") or ""
        if not transient_llm_failure(trace, answer):
            return data
        last = data
        if attempt < retries:
            wait = max(1.0, sleep_s) * attempt
            print(
                f"[arch-smoke] transient AUTO_COMPOSE LLM failure; retry {attempt}/{retries} after {wait:.1f}s",
                flush=True,
            )
            time.sleep(wait)
    trace = (last or {}).get("answer_trace") or {}
    raise SystemExit(
        "AUTO_COMPOSE LLM failure persisted after retries: "
        f"path={trace.get('path')} reason={trace.get('reason')}"
    )


def prior_token_cost(path: Path, in_per_1m: float, out_per_1m: float) -> float:
    if not path.exists() or path.stat().st_size == 0:
        return 0.0
    total = 0.0
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            total += estimated_cost_usd(
                as_int(row.get("tokens_in")),
                as_int(row.get("tokens_out")),
                in_per_1m,
                out_per_1m,
            )
    return total


def context_for(row: Dict[str, Any], doc_cache: Dict[str, List[Dict[str, str]]], mem_cache: Dict[str, List[Dict[str, str]]]) -> str:
    test = {
        "id": row["id"],
        "domain": row["domain"],
        "type": "compose",
        "query": row["query"],
        "product": row["product"],
        "ontology_refs": [row["product"], "component", "compliance", "standard"],
    }
    if isinstance(row.get("gold_evidence"), list):
        test["ontology_refs"].extend(str(ev.get("value", "")) for ev in row["gold_evidence"] if isinstance(ev, dict))
    terms = query_terms(test)
    terms.extend([row["product"], "component", "compliance", "standard", "requiresStep", "requiresCompliance"])
    sections: List[str] = []

    memory_seed = str(row.get("memory_seed") or row.get("memory") or "").strip()
    if memory_seed:
        sections.append(f"## Session memory\n{memory_seed}")
    if "memory" in (row.get("required_sources") or []) and not memory_seed:
        domain = row["domain"]
        if domain not in mem_cache:
            mem_cache[domain] = load_seed_mem_facts(DEFAULT_DOCS_ROOT, domain)
        mem_hits = ranked_items(mem_cache[domain], terms, product=row["product"])[:10]
        if mem_hits:
            sections.append("## Relevant memory facts\n" + "\n".join(f"[{m['id']}] {m['text']}" for m in mem_hits))

    needs_document = "document" in (row.get("required_sources") or []) or bool(
        re.search(r"\b(?:battery|lexmark|viessmann)_seed_\d{4}\b", f"{row.get('query','')} {row.get('product','')}")
    )
    if needs_document:
        domain = row["domain"]
        if domain not in doc_cache:
            doc_cache[domain] = load_seed_doc_chunks(DEFAULT_DOCS_ROOT, domain)
        wanted_ids = set(re.findall(r"\b(?:battery|lexmark|viessmann)_seed_\d{4}\b", f"{row.get('query','')} {row.get('product','')}"))
        doc_hits: List[Dict[str, str]] = []
        for doc in doc_cache[domain]:
            if doc["id"] in wanted_ids:
                doc_hits.append(doc)
        seen = {doc["id"] for doc in doc_hits}
        for doc in ranked_items(doc_cache[domain], terms, product=row["product"]):
            if doc["id"] not in seen:
                doc_hits.append(doc)
                seen.add(doc["id"])
            if len(doc_hits) >= 8:
                break
        if doc_hits:
            sections.append(
                "## Relevant document chunks\n"
                + "\n\n".join(f"[{d['id']} source={d['source']}]\n{d['text']}" for d in doc_hits[:8])
            )

    if "symbolic" in (row.get("required_sources") or []) or row.get("type") in {"logic", "recall"}:
        symbolic_values = [
            str(ev.get("value", "")).strip()
            for ev in (row.get("gold_evidence") or [])
            if isinstance(ev, dict) and ev.get("source") == "symbolic" and str(ev.get("value", "")).strip()
        ]
        if symbolic_values:
            sections.append(
                "## Symbolic KG evidence\n"
                + "\n".join(f"- {row['product']}: {value}" for value in symbolic_values)
            )
        ontology = ontology_context(row["domain"], terms, max_chars=7000)
        if ontology:
            sections.append(f"## Raw ontology snippets\n{ontology}")

    return "\n\n".join(sections)


def call_baseline(
    client: OpenAI,
    model: str,
    mode: str,
    row: Dict[str, Any],
    doc_cache: Dict[str, List[Dict[str, str]]],
    mem_cache: Dict[str, List[Dict[str, str]]],
) -> tuple[str, int, int]:
    ctx = context_for(row, doc_cache, mem_cache)
    if mode == "GPT4O_LONGCTX":
        system = (
            "You are a product-passport reasoning assistant. Answer using ONLY "
            "the provided context. Return a concise answer containing all requested "
            "values. If insufficient, say INSUFFICIENT EVIDENCE."
        )
        user = f"Domain: {row['domain']}\nContext:\n{ctx}\n\nQuestion: {row['query']}"
    elif mode == "LINC":
        system = (
            "You are a neuro-symbolic reasoning agent in the style of LINC. "
            "Extract premises from the supplied context, express the question "
            "as a goal, reason step by step, and return one final answer containing "
            "all requested values. Use only the supplied context."
        )
        user = (
            f"Domain: {row['domain']}\nContext:\n{ctx}\n\nQuestion: {row['query']}\n\n"
            "Output format:\nPremises: ...\nGoal: ...\nReasoning: ...\nFinal: <one line>"
        )
    else:
        system = (
            "You are Logic-LM. Classify the question, extract relevant premises "
            "from the provided context, reason over them, and return a single final "
            "answer containing all requested values. Use only context."
        )
        user = (
            f"Domain: {row['domain']}\nContext:\n{ctx}\n\nQuestion: {row['query']}\n\n"
            "Output format:\nKind: RULE|LOOKUP|COMPOSE\nReasoning: ...\nAnswer: <one line>"
        )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=220,
    )
    answer = (resp.choices[0].message.content or "").strip()
    if mode == "LOGIC_LM":
        m = re.search(r"Answer\s*:\s*(.+)", answer, flags=re.IGNORECASE | re.DOTALL)
        if m:
            answer = m.group(1).strip()
    elif mode == "LINC":
        m = re.search(r"Final\s*:\s*(.+)", answer, flags=re.IGNORECASE | re.DOTALL)
        if m:
            answer = m.group(1).strip()
    return answer, int(resp.usage.prompt_tokens), int(resp.usage.completion_tokens)


def call_baseline_with_retries(
    client: OpenAI,
    model: str,
    mode: str,
    row: Dict[str, Any],
    doc_cache: Dict[str, List[Dict[str, str]]],
    mem_cache: Dict[str, List[Dict[str, str]]],
    retries: int,
    sleep_s: float,
) -> tuple[str, int, int]:
    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return call_baseline(client, model, mode, row, doc_cache, mem_cache)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                wait = max(1.0, sleep_s) * attempt
                print(
                    f"[arch-smoke] transient {mode} LLM failure; retry {attempt}/{retries} after {wait:.1f}s: {exc}",
                    flush=True,
                )
                time.sleep(wait)
    raise SystemExit(f"{mode} LLM failure persisted after retries: {last_exc}")


def summarize(rows: List[Dict[str, Any]], out_path: Path, summary_path: Path) -> None:
    by_mode: Dict[str, List[int]] = defaultdict(list)
    by_mode_subtype: Dict[tuple[str, str], List[int]] = defaultdict(list)
    with out_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            s = int(r["success"])
            by_mode[r["mode"]].append(s)
            by_mode_subtype[(r["mode"], r["subtype"])].append(s)

    lines = ["# Architecture Smoke Comparison", "", f"Rows: {len(rows)}", "", "## Overall", "", "| Mode | Correct | Accuracy |", "|---|---:|---:|"]
    for mode in MODES:
        vals = by_mode.get(mode, [])
        lines.append(f"| `{mode}` | {sum(vals)} / {len(vals)} | {sum(vals) / len(vals) if vals else 0:.4f} |")
    lines.extend(["", "## By Subtype", "", "| Mode | Subtype | Correct | Accuracy |", "|---|---|---:|---:|"])
    for (mode, subtype), vals in sorted(by_mode_subtype.items()):
        lines.append(f"| `{mode}` | {subtype} | {sum(vals)} / {len(vals)} | {sum(vals) / len(vals) if vals else 0:.4f} |")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None, help="Optional prepared benchmark JSON. If omitted, generate synthetic smoke rows.")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-mixed", type=int, default=None)
    ap.add_argument("--model", default=os.environ.get("GEN_MODEL", "gpt-4o-mini"))
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "auto_compose_v1" / "eval_arch_smoke100_all3_comparison.csv")
    ap.add_argument("--summary", type=Path, default=REPO_ROOT / "artifacts" / "auto_compose_v1" / "eval_arch_smoke100_all3_summary.md")
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--row-retries", type=int, default=4)
    ap.add_argument("--fresh", action="store_true", help="Overwrite any existing output instead of resuming it.")
    ap.add_argument("--budget-usd", type=float, default=None, help="Best-effort cap based on token usage returned by the API.")
    ap.add_argument("--input-usd-per-1m", type=float, default=0.15)
    ap.add_argument("--output-usd-per-1m", type=float, default=0.60)
    args = ap.parse_args()

    try:
        harness_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        harness_commit = "unknown"

    rows = load_benchmark_rows(args.data, limit=args.limit, sample_mixed=args.sample_mixed) if args.data else build_rows(args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["AUTO_COMPOSE_DISABLE_PRE_LLM_DIRECT_FALLBACK"] = "1"
    os.environ.pop("AUTO_COMPOSE_DISABLE_DIRECT_FALLBACK", None)
    os.environ["AUTO_COMPOSE_FAIL_ON_LLM_ERROR"] = "1"
    if args.fresh and args.out.exists():
        args.out.unlink()

    corpus = build_clean_corpus(
        DEFAULT_DOCS_ROOT,
        REPO_ROOT / "artifacts" / "auto_compose_v1" / "corpus_auto_compose.jsonl",
        force=False,
    )
    test_client = configure_runtime(corpus, llm_disabled=False)
    openai_client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        timeout=env_float("OPENAI_TIMEOUT", 60.0),
        max_retries=as_int(os.environ.get("OPENAI_MAX_RETRIES")),
    )
    doc_cache: Dict[str, List[Dict[str, str]]] = {}
    mem_cache: Dict[str, List[Dict[str, str]]] = {}

    fields = [
        "id", "mode", "domain", "subtype", "product", "session", "query",
        "expected_groups", "success", "llm_used", "answer", "tokens_in",
        "tokens_out", "answer_trace", "model", "temperature",
        "max_output_tokens", "harness_commit",
    ]
    done = existing_done(args.out)
    running_cost = prior_token_cost(args.out, args.input_usd_per_1m, args.output_usd_per_1m)
    write_header = not args.out.exists() or args.out.stat().st_size == 0
    print(
        f"[arch-smoke] rows={len(rows)} out={args.out} "
        f"resume_done={len(done)} prior_token_cost=${running_cost:.4f}",
        flush=True,
    )

    with args.out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            writer.writeheader()

        for row in rows:
            memory_seed = str(row.get("memory_seed") or row.get("memory") or "").strip()
            if memory_seed:
                memory_service.flush_session(row["session"])
                memory_service.add_memory(row["session"], memory_seed)

        try:
            for idx, row in enumerate(rows, start=1):
                key = ("AUTO_COMPOSE", row["id"], row["domain"])
                if key in done:
                    continue
                data = solve_auto_with_retries(test_client, {
                    "query": row["query"],
                    "product": row["product"],
                    "domain": row["domain"],
                    "session": row["session"],
                    "top_k_search": 4,
                    "top_k_memory": 3,
                }, retries=args.row_retries, sleep_s=args.sleep)
                answer = data.get("answer", "") or ""
                trace = data.get("answer_trace") or {}
                tokens_in = as_int(trace.get("prompt_tokens"))
                tokens_out = as_int(trace.get("completion_tokens"))
                running_cost += estimated_cost_usd(tokens_in, tokens_out, args.input_usd_per_1m, args.output_usd_per_1m)
                writer.writerow({
                    "id": row["id"],
                    "mode": "AUTO_COMPOSE",
                    "domain": row["domain"],
                    "subtype": row["subtype"],
                    "product": row["product"],
                    "session": row["session"],
                    "query": row["query"],
                    "expected_groups": json.dumps(row["expected_groups"], ensure_ascii=False),
                    "success": score_answer(answer, row["expected_groups"]),
                    "llm_used": int(bool(trace.get("llm_used"))),
                    "answer": answer,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "answer_trace": json.dumps(trace, ensure_ascii=False),
                    "model": str(trace.get("model") or trace.get("configured_model") or args.model),
                    "temperature": "API-path dependent; see release/baseline_configs.json",
                    "max_output_tokens": 256,
                    "harness_commit": harness_commit,
                })
                fh.flush()
                if idx % 20 == 0:
                    print(f"[arch-smoke] AUTO_COMPOSE {idx}/{len(rows)} est_cost=${running_cost:.4f}", flush=True)
                if args.budget_usd is not None and running_cost > args.budget_usd:
                    raise SystemExit(f"budget exceeded after AUTO_COMPOSE {idx}: ${running_cost:.4f} > ${args.budget_usd:.4f}")
                if args.sleep:
                    time.sleep(args.sleep)

            for mode in ("GPT4O_LONGCTX", "LINC", "LOGIC_LM"):
                for idx, row in enumerate(rows, start=1):
                    key = (mode, row["id"], row["domain"])
                    if key in done:
                        continue
                    answer, tokens_in, tokens_out = call_baseline_with_retries(
                        openai_client,
                        args.model,
                        mode,
                        row,
                        doc_cache,
                        mem_cache,
                        retries=args.row_retries,
                        sleep_s=args.sleep,
                    )
                    running_cost += estimated_cost_usd(tokens_in, tokens_out, args.input_usd_per_1m, args.output_usd_per_1m)
                    writer.writerow({
                        "id": row["id"],
                        "mode": mode,
                        "domain": row["domain"],
                        "subtype": row["subtype"],
                        "product": row["product"],
                        "session": row["session"],
                        "query": row["query"],
                        "expected_groups": json.dumps(row["expected_groups"], ensure_ascii=False),
                        "success": score_answer(answer, row["expected_groups"]),
                        "llm_used": 1,
                        "answer": answer,
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "answer_trace": "",
                        "model": args.model,
                        "temperature": "0.0",
                        "max_output_tokens": 220,
                        "harness_commit": harness_commit,
                    })
                    fh.flush()
                    if idx % 20 == 0:
                        print(f"[arch-smoke] {mode} {idx}/{len(rows)} est_cost=${running_cost:.4f}", flush=True)
                    if args.budget_usd is not None and running_cost > args.budget_usd:
                        raise SystemExit(f"budget exceeded after {mode} {idx}: ${running_cost:.4f} > ${args.budget_usd:.4f}")
                    if args.sleep:
                        time.sleep(args.sleep)
        finally:
            for row in rows:
                memory_service.flush_session(row["session"])

    summarize(rows, args.out, args.summary)
    print(args.summary.read_text(encoding="utf-8"))
    print(f"[arch-smoke] out={args.out.resolve()}")
    print(f"[arch-smoke] summary={args.summary.resolve()}")
    print(f"[arch-smoke] estimated_token_cost=${running_cost:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
