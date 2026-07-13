#!/usr/bin/env python3
"""
External LLM baseline harness — runs three reviewer-defensible baselines against
the synthetic DPP-like benchmark. Outputs CSV in the same schema as
artifacts/eval_<MODE>_*.csv so it plugs into the existing aggregation pipeline.

Modes
-----
  GPT4O_LONGCTX  long-context GPT-4o[-mini] with the per-domain seed corpus
  LINC           LINC-style premise/goal/reasoning chain (Olausson et al. 2023)
  LOGIC_LM       Logic-LM-style classify-then-dispatch (Pan et al. 2023)
  STUB           offline placeholder for plumbing-only tests

Test sets (pick exactly one of these inputs)
------------------------------------------
  --benchmark release       release/release_20250902_215155 JSONLs (8,093 rows,
                            every row has gold). Default.
  --benchmark paper         exact 3,429 rows from the paper's pooled CSV
                            (artifacts/paper_split.json). Gold is attached only
                            when the release query is byte-identical; otherwise
                            the row is written unscored to avoid false matches.
  --benchmark paper-matched release-benchmark rows whose (id, domain) keys also
                            occur in the paper split (1,345 rows, all with gold).
                            This keeps question/gold/context consistent while
                            preserving paper-overlap keys for diagnostics.

Filters
-------
  --filter-prefix docopen,docrec,log
  --filter-ids /path/to/ids.json     (JSON list of "id" or "id|domain" strings,
                                      or list of {id, domain} dicts, or a
                                      {"rows":[...]} object)
  --limit 25
  --sort                              (default on — deterministic ordering)

Cost controls
-------------
  --budget-usd 5.00                   abort if running cost exceeds this
  --price-in 0.15                     USD per 1M input tokens (gpt-4o-mini default)
  --price-out 0.60                    USD per 1M output tokens
  --print-prompt                      print the exact prompt for the first
                                      filtered query and exit (NO API calls)

Resume
------
Output CSV is appended-to. If it already exists, completed (id, domain) pairs
are skipped. Kill the process and re-run; it picks up where it left off.

Examples
--------
  # 25-query smoke test (~$0.05)
  python scripts/run_baselines.py --mode LOGIC_LM \
      --benchmark release --limit 25 \
      --out artifacts/baseline_smoke.csv

  # Prompt audit (no API call):
  python scripts/run_baselines.py --mode GPT4O_LONGCTX \
      --benchmark release --filter-prefix docopen --limit 1 \
      --print-prompt

  # Full release run (the harness prints an estimate before making API calls):
  for m in GPT4O_LONGCTX LINC LOGIC_LM; do
      python scripts/run_baselines.py --mode $m \
          --benchmark release \
          --out artifacts/eval_${m}_full.csv \
          --budget-usd 10.00
  done
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PAPER_SPLIT_JSON = REPO_ROOT / "artifacts" / "paper_split.json"
DEFAULT_PAPER_MATCHED_JSON = REPO_ROOT / "artifacts" / "paper_split_with_gold.json"
DEFAULT_RELEASE_JSON = REPO_ROOT / "artifacts" / "release_benchmark.json"
DEFAULT_CLEAN_RELEASE_JSON = REPO_ROOT / "artifacts" / "release_benchmark_clean_docs.json"
DEFAULT_DOCS_ROOT = REPO_ROOT / "release" / "release_20250902_215155" / "tests"

EVAL_CSV_FIELDS = [
    "id",
    "mode",
    "type",
    "domain",
    "query",
    "product",
    "session",
    "success",
    "steps",
    "correct",
    "latency_ms",
    "confidence",
    "confidence_raw",
    "confidence_cal",
    "cost_retrieval_calls",
    "cost_rule_checks",
    "cost_tokens_in",
    "cost_tokens_out",
    "n_steps",
    "answer",
    "expected_contains",
    "cost_usd_running",
]


# ---------------------------------------------------------------------------
# Benchmark loaders. Each returns a list of dicts containing:
#   id, domain, type, query, expected_contains, product, session
# ---------------------------------------------------------------------------
def load_release_benchmark(path: Path = DEFAULT_RELEASE_JSON) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    out: List[Dict[str, Any]] = []
    for r in data["rows"]:
        out.append({
            "id": r["id"],
            "domain": r["domain"],
            "type": r.get("type", "open"),
            "query": r.get("query") or r.get("question") or "",
            "expected_contains": r.get("expected_contains", ""),
            "product": r.get("product", ""),
            "session": r.get("session", "s1"),
            "meta": r.get("meta", {}),
            "ontology_refs": r.get("ontology_refs", []),
        })
    return out


def load_clean_release_benchmark(path: Path = DEFAULT_CLEAN_RELEASE_JSON) -> List[Dict[str, Any]]:
    return load_release_benchmark(path)


def load_paper_split(path: Path = DEFAULT_PAPER_SPLIT_JSON) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    # The paper split doesn't carry expected_contains. Attach gold only when the
    # release row has the same query text; matching by id/domain alone is unsafe
    # because archived paper queries and release queries are not byte-identical.
    release_rows = {(r["id"], r["domain"]): r for r in load_release_benchmark()}
    out: List[Dict[str, Any]] = []
    for r in data["rows"]:
        rel = release_rows.get((r["id"], r["domain"]))
        gold = rel["expected_contains"] if rel and rel.get("query") == r.get("query") else ""
        out.append({
            "id": r["id"],
            "domain": r["domain"],
            "type": r["type"],
            "query": r["query"],
            "expected_contains": gold,
            "product": r.get("product", ""),
            "session": r.get("session", "s1"),
            "meta": r.get("meta", {}),
            "ontology_refs": r.get("ontology_refs", []),
        })
    return out


def load_paper_matched(path: Path = DEFAULT_PAPER_MATCHED_JSON) -> List[Dict[str, Any]]:
    del path  # Kept for backward-compatible signature; see benchmark docstring.
    paper_keys = {
        (r["id"], r["domain"])
        for r in json.loads(DEFAULT_PAPER_SPLIT_JSON.read_text())["rows"]
    }
    return [
        r for r in load_release_benchmark()
        if (r["id"], r["domain"]) in paper_keys
    ]


BENCHMARK_LOADERS = {
    "release": load_release_benchmark,
    "release-clean": load_clean_release_benchmark,
    "paper": load_paper_split,
    "paper-matched": load_paper_matched,
}


def load_seed_docs(docs_root: Path, domain: str, max_chars: int = 12_000) -> str:
    path = docs_root / domain / "seed_docs.jsonl"
    if not path.exists():
        return ""
    chunks: List[str] = []
    used = 0
    with path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = rec.get("text") or rec.get("content") or rec.get("title") or json.dumps(rec)
            if not text:
                continue
            if used + len(text) > max_chars:
                remaining = max_chars - used
                if remaining > 200:
                    chunks.append(text[:remaining])
                    used += remaining
                break
            chunks.append(text)
            used += len(text)
    return "\n\n---\n\n".join(chunks)


def load_seed_doc_chunks(docs_root: Path, domain: str) -> List[Dict[str, str]]:
    """Load seed document chunks with stable IDs matching generated doc tests."""
    path = docs_root / domain / "seed_docs.jsonl"
    if not path.exists():
        return []
    out: List[Dict[str, str]] = []
    with path.open() as fh:
        for idx, line in enumerate(fh):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = rec.get("text") or rec.get("content") or ""
            if not text and isinstance(rec.get("chunks"), list):
                parts = []
                for ch in rec["chunks"]:
                    if isinstance(ch, dict):
                        parts.append(ch.get("text", ""))
                    else:
                        parts.append(str(ch))
                text = "\n".join(p for p in parts if p)
            if not text:
                continue
            sid = f"{domain}_seed_{idx:04d}"
            out.append({
                "id": sid,
                "source": str(rec.get("source") or rec.get("doc_id") or sid),
                "text": text,
            })
    return out


def load_seed_mem_facts(docs_root: Path, domain: str) -> List[Dict[str, str]]:
    path = docs_root / domain / "seed_mem.jsonl"
    if not path.exists():
        return []
    out: List[Dict[str, str]] = []
    with path.open() as fh:
        for idx, line in enumerate(fh):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = rec.get("text") or rec.get("content") or ""
            if text:
                out.append({"id": str(rec.get("id") or f"mem-{idx:06d}"), "text": text})
    return out


def ontology_paths_for(domain: str) -> List[Path]:
    release_onto = REPO_ROOT / "release" / "release_20250902_215155" / "backend" / "ontologies"
    # Match scripts/autogen_kb_tests.py: battery falls back to dpp_ontology.ttl,
    # while lexmark/viessmann use their domain ontology. Do not mix in
    # *_augment.ttl here; those files contain overlapping synthetic entities and
    # can contradict the gold answers generated from the base ontology.
    names = {
        "battery": ["dpp_ontology.ttl"],
        "lexmark": ["lexmark_ontology.ttl"],
        "viessmann": ["viessmann_ontology.ttl"],
    }.get(domain, ["dpp_ontology.ttl"])
    paths: List[Path] = []
    for name in names:
        p = release_onto / name
        if p.exists() and p not in paths:
            paths.append(p)
    return paths


def load_ontology_lines(domain: str) -> List[str]:
    lines: List[str] = []
    for path in ontology_paths_for(domain):
        try:
            lines.extend(path.read_text(encoding="utf-8", errors="ignore").splitlines())
        except Exception:
            continue
    return lines


STOPWORDS = {
    "according", "about", "after", "before", "based", "context", "does", "have",
    "into", "mentioned", "near", "please", "provide", "record", "records",
    "reported", "specification", "state", "that", "the", "this", "true", "value",
    "what", "when", "where", "which", "with", "would", "from", "list", "give",
    "doc", "kb", "logic", "open", "recall", "product",
}


def query_terms(test: Dict[str, Any]) -> List[str]:
    fields = [
        test.get("id", ""),
        test.get("query", ""),
        test.get("product", ""),
        " ".join(test.get("ontology_refs") or []),
    ]
    meta = test.get("meta") or {}
    if isinstance(meta, dict):
        fields.extend(str(v) for v in meta.values() if isinstance(v, (str, int, float)))
    raw = " ".join(str(x) for x in fields if x)
    terms = []
    for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-]{1,}", raw):
        t = term.strip().lower()
        if len(t) >= 3 and t not in STOPWORDS:
            terms.append(t)
    # Preserve quoted snippets as high-value retrieval cues.
    for quoted in re.findall(r'"([^"]{8,160})"', str(test.get("query", ""))):
        for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-]{2,}", quoted):
            terms.append(term.lower())
    return list(dict.fromkeys(terms))


def ranked_items(items: List[Dict[str, str]], terms: List[str], product: str = "") -> List[Dict[str, str]]:
    scored: List[Tuple[int, int, Dict[str, str]]] = []
    product_l = product.lower()
    for idx, item in enumerate(items):
        hay = f"{item.get('id', '')} {item.get('source', '')} {item.get('text', '')}".lower()
        score = 0
        if product_l and product_l in hay:
            score += 200
        for term in terms:
            if term in hay:
                score += 5 if len(term) >= 6 else 2
        if score:
            scored.append((score, -idx, item))
    scored.sort(reverse=True)
    return [x[2] for x in scored]


def ontology_context(domain: str, terms: List[str], max_chars: int = 4_000) -> str:
    lines = load_ontology_lines(domain)
    if not lines:
        return ""
    # Terms with digits or hyphens are usually entity identifiers
    # (e.g. LiCell-064, ProductC) — far more discriminative than generic words
    # like "voltage". Process specific terms FIRST so identifier matches reach
    # the model even when common terms swamp the budget.
    def specificity(t: str) -> int:
        # Higher = more specific. Penalise terms that match thousands of lines.
        s = 0
        if any(c.isdigit() for c in t): s += 20
        if "-" in t or "_" in t: s += 5
        if "." in t: s += 5
        s += min(10, len(t))   # longer terms tend to be more specific
        if t in {"voltage","capacity","battery","product","model","name","passport","standard","compliance"}:
            s -= 30
        return s
    sorted_terms = sorted({t for t in terms if len(t) >= 3}, key=specificity, reverse=True)

    def first_entity_block(term: str) -> List[str]:
        """Return the first contiguous ontology block for an entity label term."""
        if not (any(c.isdigit() for c in term) or "-" in term or "_" in term):
            return []
        for i, line in enumerate(lines):
            low = line.lower()
            if term not in low or "rdfs:label" not in low:
                continue
            subj_match = re.match(r"^(\S+)\s+rdfs:label\b", line.strip())
            if not subj_match:
                continue
            subject = subj_match.group(1)
            start = i
            for j in range(i, max(-1, i - 10), -1):
                stripped = lines[j].strip()
                if stripped.startswith(subject + " a ") or stripped.startswith(subject + " rdf:type"):
                    start = j
                    break
            end = min(len(lines), start + 40)
            prefix_match = re.match(r"^(.*_)\d+$", subject)
            if prefix_match:
                prefix = re.escape(prefix_match.group(1))
                for j in range(start + 1, min(len(lines), start + 80)):
                    stripped = lines[j].strip()
                    if re.match(rf"^{prefix}\d+\s+a\s+", stripped) and not stripped.startswith(subject + " "):
                        end = j
                        break
            block = lines[start:end]
            refs = set(re.findall(r"\b(?:ext|ex):[A-Za-z0-9_.\-]+", "\n".join(block)))
            for ref in sorted(refs):
                if any(existing.strip().startswith(ref + " rdfs:label") for existing in block):
                    continue
                for label_line in lines:
                    if label_line.strip().startswith(ref + " rdfs:label") and label_line not in block:
                        block.append(label_line)
                        break
            return block
        return []

    entity_blocks: List[str] = []
    entity_seen: set = set()
    for term in sorted_terms:
        block = first_entity_block(term)
        if not block:
            continue
        for line in block:
            if line not in entity_seen:
                entity_seen.add(line)
                entity_blocks.append(line)
        # The most specific entity is enough for single-entity lookup questions.
        break

    # Priority 1: term-matched windows (with context), in order of term
    # specificity. Highest-specificity term gets first shot at the budget.
    term_blocks: List[str] = []
    seen_idx: set = set()
    if not entity_blocks:
        for term in sorted_terms:
            for i, line in enumerate(lines):
                if term in line.lower():
                    lo = max(0, i - 2)
                    hi = min(len(lines), i + 8)
                    for j in range(lo, hi):
                        if j in seen_idx:
                            continue
                        seen_idx.add(j)
                        term_blocks.append(lines[j])

    entity_prefixes = {
        ref.split(":", 1)[0]
        for ref in re.findall(r"\b(?:ext|ex):[A-Za-z0-9_.\-]+", "\n".join(entity_blocks))
    }

    # Priority 2: a compact class/property declaration header so the model can
    # generalise (e.g. LithiumIonBattery -> Product). Capped at ~1/4 of budget.
    header_budget = max(200, max_chars // 4)
    header_lines: List[str] = []
    header_used = 0
    header_candidates: List[str] = []
    if entity_prefixes:
        for line in lines:
            stripped = line.strip()
            if not any(stripped.startswith(prefix + ":") for prefix in entity_prefixes):
                continue
            if re.match(r"^(?:ext|ex):[A-Za-z]+_\d+\b", stripped):
                continue
            if (
                "owl:Class" in stripped
                or "rdfs:subClassOf" in stripped
                or "owl:ObjectProperty" in stripped
                or "rdfs:domain" in stripped
                or "rdfs:range" in stripped
                or " rdfs:label " in stripped
            ):
                header_candidates.append(line)
    header_candidates.extend(lines[:140])
    for line in header_candidates:
        stripped = line.strip()
        if not stripped or stripped.startswith("@prefix") or stripped.startswith("#"):
            continue
        # Keep only class/property declarations, not facts (which we want term-
        # matched lookups for).
        if " a " not in line and "rdf:type" not in line and "rdfs:subClassOf" not in line and "rdfs:domain" not in line and "rdfs:range" not in line and " rdfs:label " not in line:
            continue
        header_lines.append(line)
        header_used += len(line) + 1
        if header_used > header_budget:
            break

    out: List[str] = []
    used = 0
    seen: set = set()
    # Entity block first, if available.
    for line in entity_blocks:
        line = line.rstrip()
        if not line or line in seen:
            continue
        seen.add(line)
        if used + len(line) + 1 > max_chars:
            break
        out.append(line)
        used += len(line) + 1
    # Term matches first.
    for line in term_blocks:
        line = line.rstrip()
        if not line or line in seen:
            continue
        seen.add(line)
        if used + len(line) + 1 > max_chars:
            break
        out.append(line)
        used += len(line) + 1
    # Then header declarations to fill remaining space.
    for line in header_lines:
        line = line.rstrip()
        if not line or line in seen:
            continue
        seen.add(line)
        if used + len(line) + 1 > max_chars:
            break
        out.append(line)
        used += len(line) + 1
    out.append("Known XML Schema datatypes such as integer, nonPositiveInteger, unsignedLong, dateTime, and dateTimeStamp are Datatype.")
    return "\n".join(out)


def build_retrieved_context(
    test: Dict[str, Any],
    docs_root: Path,
    doc_cache: Dict[str, List[Dict[str, str]]],
    mem_cache: Dict[str, List[Dict[str, str]]],
    max_chars: int,
) -> str:
    domain = test["domain"]
    terms = query_terms(test)
    product = str(test.get("product") or "")
    if domain not in doc_cache:
        doc_cache[domain] = load_seed_doc_chunks(docs_root, domain)
    if domain not in mem_cache:
        mem_cache[domain] = load_seed_mem_facts(docs_root, domain)

    sections: List[Tuple[str, str]] = []
    row_id = str(test.get("id") or "")
    is_kb_row = ".kb." in row_id

    mem_hits = ranked_items(mem_cache[domain], terms, product=product)[:10]
    if mem_hits:
        sections.append(("Relevant memory facts", "\n".join(f"[{m['id']}] {m['text']}" for m in mem_hits)))

    type_membership_query = bool(re.match(r"^\s*is\s+.+?\s+(?:a|an)\s+.+?\?\s*$", str(test.get("query") or ""), flags=re.IGNORECASE))
    if is_kb_row or (test.get("type") == "logic" and type_membership_query):
        onto = ontology_context(domain, terms, max_chars=min(4_500, max_chars))
        if onto:
            sections.append(("Relevant ontology snippets", onto))

    if test.get("type") != "logic" and not is_kb_row:
        doc_hits = ranked_items(doc_cache[domain], terms, product=product)
        if not doc_hits and doc_cache[domain]:
            doc_hits = doc_cache[domain][:8]
        doc_blocks = []
        for d in doc_hits[:12]:
            doc_blocks.append(f"[{d['id']} source={d['source']}]\n{d['text']}")
        if doc_blocks:
            sections.append(("Relevant document chunks", "\n\n".join(doc_blocks)))

    out: List[str] = []
    used = 0
    for title, body in sections:
        block = f"## {title}\n{body}".strip()
        if used + len(block) + 2 > max_chars:
            remaining = max_chars - used - len(title) - 8
            if remaining > 500:
                block = f"## {title}\n{body[:remaining]}"
            else:
                continue
        out.append(block)
        used += len(block) + 2
        if used >= max_chars:
            break
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing (source temp_env.sh first)")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("pip install openai  (host machine; sandbox proxy blocked)") from exc
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("‑", "-").replace("—", "-").replace("–", "-")
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text))


def logic_target_from_query(query: str) -> str:
    m = re.match(r"^\s*is\s+.+?\s+(?:a|an)\s+(.+?)\?\s*$", query or "", flags=re.IGNORECASE)
    return normalize_text(m.group(1)) if m else ""


def first_answer_token(answer: str) -> str:
    a = normalize_text(answer)
    m = re.search(r"(?:answer|final)\s*:\s*([a-z]+)", a)
    if m:
        return m.group(1)
    m = re.match(r"[^a-z0-9]*([a-z]+)", a)
    return m.group(1) if m else ""


def answer_polarity(answer: str) -> Optional[bool]:
    a = normalize_text(answer)
    token = first_answer_token(a)
    if token in {"yes", "true", "correct", "supported", "entailed"}:
        return True
    if token in {"no", "false", "incorrect", "insufficient", "unknown"}:
        return False
    if "insufficient" in a or "not enough evidence" in a:
        return False
    return None


def contains_expected(answer: str, expected_contains: str) -> bool:
    a = normalize_text(answer)
    e = normalize_text(expected_contains)
    if e in a:
        return True
    # Accept harmless spacing/punctuation variants, e.g. "12V" vs "12 V".
    e_compact = compact_text(e)
    return bool(e_compact and e_compact in compact_text(a))


def score_answer(answer: str, expected_contains: str, qtype: str, query: str = "") -> int:
    """Returns 1 / 0, or -1 if no gold available (will be excluded from accuracy)."""
    if not expected_contains:
        return -1
    e = normalize_text(expected_contains)
    if qtype == "logic":
        polarity = answer_polarity(answer)
        if e in {"yes", "true"}:
            return 1 if polarity is True else 0
        if e in {"no", "false"}:
            return 1 if polarity is False else 0
        target = logic_target_from_query(query)
        if target and compact_text(target) == compact_text(e):
            return 1 if polarity is True or contains_expected(answer, expected_contains) else 0
    return 1 if contains_expected(answer, expected_contains) else 0


# ---------------------------------------------------------------------------
# Baseline prompts and runners
# ---------------------------------------------------------------------------
def _is_retryable_openai_error(exc: Exception) -> bool:
    name = type(exc).__name__
    return name in {
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "InternalServerError",
        "RateLimitError",
    }


def _chat(client, model: str, system: str, user: str, max_tokens: int = 200):
    max_attempts = 8
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=60,
            )
            break
        except Exception as exc:
            if not _is_retryable_openai_error(exc) or attempt == max_attempts - 1:
                raise
            delay = min(45.0, 2.0 * (attempt + 1))
            print(f"[run_baselines] retryable OpenAI error: {type(exc).__name__}; retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
    answer = (resp.choices[0].message.content or "").strip()
    usage = {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}
    return answer, usage


def prompt_gpt4o_longctx(question: str, qtype: str, domain: str, seed_context: str) -> Tuple[str, str]:
    system = (
        "You are a product-passport reasoning assistant. Answer using ONLY the "
        "provided context. If the context does not contain enough evidence, "
        "respond with 'INSUFFICIENT EVIDENCE'. For yes/no logic questions, "
        "respond with a single word 'yes' or 'no'. For factual recall/open "
        "questions, respond with a short phrase that quotes the relevant value."
    )
    user = (
        f"Domain: {domain}\nQuestion type: {qtype}\n\n"
        f"Context:\n{seed_context}\n\n"
        f"Question: {question}"
    )
    return system, user


def prompt_linc(question: str, qtype: str, domain: str, seed_context: str) -> Tuple[str, str]:
    system = (
        "You are a neuro-symbolic reasoning agent in the style of LINC "
        "(Olausson et al., 2023). Solve the question by: "
        "(1) extracting premises as first-order-logic statements; "
        "(2) writing the question as a goal predicate; "
        "(3) reasoning step by step; "
        "(4) emitting a single final answer. For yes/no questions answer "
        "'yes' or 'no'. For factual questions answer with the exact phrase. "
        "If the premises are insufficient answer 'INSUFFICIENT'."
    )
    user = (
        f"Domain: {domain}\nQuestion type: {qtype}\n\n"
        f"Context (treat as premises):\n{seed_context}\n\n"
        f"Question: {question}\n\n"
        f"Output format:\nPremises: ...\nGoal: ...\nReasoning: ...\nFinal: <answer>"
    )
    return system, user


def prompt_logic_lm(question: str, qtype: str, domain: str, seed_context: str) -> Tuple[str, str]:
    system = (
        "You are Logic-LM (Pan et al., 2023). Step 1: classify the question "
        "as RULE (yes/no compliance), LOOKUP (factual recall), or OPEN. "
        "Step 2: solve it using only the provided context. For RULE answer "
        "'yes' or 'no'. For LOOKUP/OPEN return the exact factual phrase. If "
        "the context is insufficient, return 'INSUFFICIENT'."
    )
    user = (
        f"Domain: {domain}\nDeclared question type: {qtype}\n\n"
        f"Context:\n{seed_context}\n\n"
        f"Question: {question}\n\n"
        f"Output format:\nKind: RULE|LOOKUP|OPEN\nAnswer: <one line>"
    )
    return system, user


def call_gpt4o_longctx(client, model, question, qtype, domain, seed_context):
    system, user = prompt_gpt4o_longctx(question, qtype, domain, seed_context)
    return _chat(client, model, system, user, max_tokens=200)


def call_linc(client, model, question, qtype, domain, seed_context):
    system, user = prompt_linc(question, qtype, domain, seed_context)
    answer, usage = _chat(client, model, system, user, max_tokens=300)
    m = re.search(r"Final\s*:\s*(.+)", answer, flags=re.IGNORECASE)
    return (m.group(1).strip() if m else answer), usage


def call_logic_lm(client, model, question, qtype, domain, seed_context):
    system, user = prompt_logic_lm(question, qtype, domain, seed_context)
    answer, usage = _chat(client, model, system, user, max_tokens=200)
    m = re.search(r"Answer\s*:\s*(.+)", answer, flags=re.IGNORECASE)
    return (m.group(1).strip() if m else answer), usage


def call_stub(client, model, question, qtype, domain, seed_context):
    return ("yes" if qtype == "logic" else "PLACEHOLDER"), {"in": 0, "out": 0}


MODE_DISPATCH = {
    "GPT4O_LONGCTX": call_gpt4o_longctx,
    "LINC": call_linc,
    "LOGIC_LM": call_logic_lm,
    "STUB": call_stub,
}

PROMPT_BUILDERS = {
    "GPT4O_LONGCTX": prompt_gpt4o_longctx,
    "LINC": prompt_linc,
    "LOGIC_LM": prompt_logic_lm,
}

MODE_MAX_OUTPUT = {
    "GPT4O_LONGCTX": 200,
    "LINC": 300,
    "LOGIC_LM": 200,
    "STUB": 0,
}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def apply_filters(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    keep_prefixes: Optional[set] = None
    if args.filter_prefix:
        keep_prefixes = {p.strip() for p in args.filter_prefix.split(",") if p.strip()}
    keep_ids: Optional[set] = None
    if args.filter_ids:
        data = json.loads(Path(args.filter_ids).read_text())
        keep_ids = set()
        items = data["rows"] if isinstance(data, dict) and "rows" in data else data
        for entry in items:
            if isinstance(entry, str):
                if "|" in entry:
                    qid, dom = entry.split("|", 1)
                    keep_ids.add((qid, dom))
                else:
                    keep_ids.add((entry, None))
            elif isinstance(entry, dict):
                keep_ids.add((entry["id"], entry.get("domain")))
    filtered: List[Dict[str, Any]] = []
    for r in rows:
        if keep_prefixes is not None:
            pref = r["id"].split("-")[0] if "-" in r["id"] else r["id"].split(".")[0]
            if pref not in keep_prefixes:
                continue
        if keep_ids is not None:
            if (r["id"], r["domain"]) not in keep_ids and (r["id"], None) not in keep_ids:
                continue
        filtered.append(r)
    if args.sort:
        filtered.sort(key=lambda r: (r["id"], r["domain"]))
    if args.limit:
        filtered = filtered[: args.limit]
    return filtered


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------
def usd_cost(tokens_in: int, tokens_out: int, price_in: float, price_out: float) -> float:
    return tokens_in * price_in / 1_000_000 + tokens_out * price_out / 1_000_000


def format_usd(amount: float) -> str:
    return f"${amount:.4f}" if amount < 0.10 else f"${amount:.2f}"


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text or "") / 4))


def estimate_prompt_usage(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    doc_cache: Dict[str, List[Dict[str, str]]],
    mem_cache: Dict[str, List[Dict[str, str]]],
) -> Tuple[int, int, float]:
    if args.mode == "STUB":
        return 0, 0, 0.0
    builder = PROMPT_BUILDERS[args.mode]
    max_out = MODE_MAX_OUTPUT.get(args.mode, 200)
    tokens_in = 0
    tokens_out = 0
    for test in rows:
        context = build_retrieved_context(test, args.docs, doc_cache, mem_cache, args.max_ctx_chars)
        system, user = builder(test["query"], test["type"], test["domain"], context)
        tokens_in += approx_tokens(system) + approx_tokens(user) + 16
        tokens_out += max_out
    return tokens_in, tokens_out, usd_cost(tokens_in, tokens_out, args.price_in, args.price_out)


def already_done_and_usage(out_path: Path) -> Tuple[set, int, int]:
    if not out_path.exists():
        return set(), 0, 0
    done = set()
    tokens_in = 0
    tokens_out = 0
    with out_path.open() as fh:
        for row in csv.DictReader(fh):
            done.add((row["id"], row["domain"]))
            try:
                tokens_in += int(row.get("cost_tokens_in") or 0)
                tokens_out += int(row.get("cost_tokens_out") or 0)
            except ValueError:
                pass
    return done, tokens_in, tokens_out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=list(MODE_DISPATCH.keys()))
    ap.add_argument("--benchmark", default="release", choices=list(BENCHMARK_LOADERS.keys()))
    ap.add_argument("--docs", type=Path, default=DEFAULT_DOCS_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--filter-prefix", default=None)
    ap.add_argument("--filter-ids", type=Path, default=None)
    ap.add_argument("--sort", action="store_true", default=True)
    ap.add_argument("--no-sort", dest="sort", action="store_false")
    ap.add_argument("--model", default=os.environ.get("GEN_MODEL", "gpt-4o-mini"))
    ap.add_argument("--max-ctx-chars", type=int, default=12_000)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--budget-usd", type=float, default=10.0)
    ap.add_argument("--price-in", type=float, default=0.15)
    ap.add_argument("--price-out", type=float, default=0.60)
    ap.add_argument("--print-prompt", action="store_true")
    args = ap.parse_args()

    rows = BENCHMARK_LOADERS[args.benchmark]()
    print(f"[run_baselines] benchmark={args.benchmark}  raw rows={len(rows)}")
    filtered = apply_filters(rows, args)
    print(f"[run_baselines] after filters: {len(filtered)} rows")

    doc_cache: Dict[str, List[Dict[str, str]]] = {}
    mem_cache: Dict[str, List[Dict[str, str]]] = {}

    def context_for(test: Dict[str, Any]) -> str:
        return build_retrieved_context(test, args.docs, doc_cache, mem_cache, args.max_ctx_chars)

    if args.print_prompt:
        if not filtered:
            print("No rows after filtering.")
            return 1
        if args.mode == "STUB":
            print("STUB mode has no prompt.")
            return 0
        r = filtered[0]
        seed = context_for(r)
        builder = PROMPT_BUILDERS[args.mode]
        system, user = builder(r["query"], r["type"], r["domain"], seed)
        print(f"=== PROMPT AUDIT: mode={args.mode} id={r['id']} domain={r['domain']} ===\n")
        print("--- system ---")
        print(system)
        print("\n--- user ---")
        print(user)
        print("\n--- expected_contains ---")
        print(r.get("expected_contains", "(none)"))
        return 0

    done, running_in, running_out = already_done_and_usage(args.out)
    prior_cost = usd_cost(running_in, running_out, args.price_in, args.price_out)
    remaining = [r for r in filtered if (r["id"], r["domain"]) not in done]
    est_in, est_out, est_cost = estimate_prompt_usage(remaining, args, doc_cache, mem_cache)
    avg_in = int(est_in / len(remaining)) if remaining else 0
    avg_out = int(est_out / len(remaining)) if remaining else 0
    print(
        f"[run_baselines] estimated remaining cost: {format_usd(est_cost)} "
        f"({len(remaining)} calls; avg approx {avg_in} in / {avg_out} max out tokens)"
    )
    if prior_cost:
        print(f"[run_baselines] existing output cost from resumable CSV: {format_usd(prior_cost)}")
    if prior_cost + est_cost > args.budget_usd:
        print(f"\nERROR: estimated total cost {format_usd(prior_cost + est_cost)} > --budget-usd {format_usd(args.budget_usd)}.")
        print("Use --limit N, --filter-prefix, --filter-ids, or raise --budget-usd.")
        return 2

    client = None if args.mode == "STUB" else get_openai_client()
    runner = MODE_DISPATCH[args.mode]
    if done:
        print(f"[run_baselines] resuming — {len(done)} already done")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    write_header = not args.out.exists()
    running_cost = prior_cost

    with args.out.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVAL_CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for idx, test in enumerate(filtered, start=1):
            key = (test["id"], test["domain"])
            if key in done:
                continue
            seed = context_for(test)
            t0 = time.time()
            try:
                answer, usage = runner(client, args.model, test["query"], test["type"], test["domain"], seed)
                err = None
            except Exception as exc:
                answer = ""
                usage = {"in": 0, "out": 0}
                err = f"{type(exc).__name__}: {exc}"
            latency_ms = int((time.time() - t0) * 1000)
            running_in += usage.get("in", 0)
            running_out += usage.get("out", 0)
            running_cost = usd_cost(running_in, running_out, args.price_in, args.price_out)
            score = score_answer(answer, test["expected_contains"], test["type"], test["query"])
            success = max(0, score)
            row = {
                "id": test["id"],
                "mode": args.mode,
                "type": test["type"],
                "domain": test["domain"],
                "query": test["query"],
                "product": test.get("product", ""),
                "session": test.get("session", "s1"),
                "success": success,
                "steps": json.dumps([{"source": args.mode, "text": answer[:500]}]),
                "correct": success,
                "latency_ms": latency_ms,
                "confidence": 0.85 if success else 0.45,
                "confidence_raw": 0.85 if success else 0.45,
                "confidence_cal": 0.85 if success else 0.45,
                "cost_retrieval_calls": 1 if seed else 0,
                "cost_rule_checks": 0,
                "cost_tokens_in": usage.get("in", 0),
                "cost_tokens_out": usage.get("out", 0),
                "n_steps": 1,
                "answer": (answer if not err else err)[:500],
                "expected_contains": test["expected_contains"],
                "cost_usd_running": f"{running_cost:.4f}",
            }
            writer.writerow(row)
            fh.flush()
            if idx % 25 == 0:
                print(f"[{args.mode}] {idx}/{len(filtered)}  cost=${running_cost:.2f}  last_success={success}")
            if running_cost > args.budget_usd:
                print(f"\nERROR: running cost ${running_cost:.2f} exceeded budget ${args.budget_usd:.2f}. Aborting.")
                return 3
            if args.sleep:
                time.sleep(args.sleep)

    print(f"\n[{args.mode}] DONE. tokens in={running_in:,}  out={running_out:,}  total cost=${running_cost:.4f}")
    with args.out.open() as fh:
        rows_out = list(csv.DictReader(fh))
    scored = [r for r in rows_out if r.get("expected_contains")]
    if scored:
        acc = sum(int(r["success"]) for r in scored) / len(scored)
        print(f"[{args.mode}] scored rows (have gold): n={len(scored)}  accuracy={acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
