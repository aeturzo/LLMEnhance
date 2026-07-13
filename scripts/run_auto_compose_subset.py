#!/usr/bin/env python3
"""Run strict AUTO_COMPOSE on a prepared JSON benchmark subset."""
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
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if os.environ.get("ADAPTIVERAG_CLEAN_ALLOW_OPENAI_EMBED") != "1":
    os.environ["EMBED_MODEL_NAME"] = os.environ.get(
        "ADAPTIVERAG_CLEAN_EMBED_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

from scripts.run_adaptiverag_clean import (  # noqa: E402
    BAD_API_TERMS,
    DEFAULT_DOCS_ROOT,
    build_clean_corpus,
    configure_runtime,
    compact_text,
    mixed_sample,
)
from scripts.run_baselines import score_answer  # noqa: E402
from backend.services import memory_service  # noqa: E402


CSV_FIELDS = [
    "id",
    "mode",
    "type",
    "subtype",
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
    "answer_trace",
    "llm_used",
    "provider",
    "model",
    "api",
    "included_source_types",
    "included_passage_ids",
    "fallback_path",
    "composed_context_chars",
]


def load_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    out: List[Dict[str, Any]] = []
    for row in rows:
        expected = row.get("expected_contains", "")
        if isinstance(expected, list):
            expected_s = " || ".join(str(x) for x in expected)
        else:
            expected_s = str(expected)
        r = dict(row)
        r["expected_contains"] = expected_s
        r.setdefault("session", "s1")
        r["product"] = (r.get("product") or r.get("solver_product") or infer_product(r)).strip()
        out.append(r)
    return out


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


def existing_done(path: Path) -> set[Tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        return {(r["id"], r["domain"]) for r in csv.DictReader(fh)}


def score_row(answer: str, row: Dict[str, Any]) -> int:
    expected_groups = row.get("expected_groups")
    if isinstance(expected_groups, list) and expected_groups:
        return int(all(
            any(compact_text(str(value)) in compact_text(answer) for value in group)
            for group in expected_groups
        ))
    expected_all = row.get("expected_all")
    if isinstance(expected_all, list) and expected_all:
        return int(all(compact_text(str(v)) in compact_text(answer) for v in expected_all))
    expected = row.get("expected_contains", "")
    if "||" in str(expected):
        parts = [p.strip() for p in str(expected).split("||") if p.strip()]
        return int(all(compact_text(p) in compact_text(answer) for p in parts))
    return max(0, score_answer(answer, str(expected), row.get("type", ""), row.get("query", "")))


def context_chars_from_steps(steps: List[Dict[str, Any]]) -> int:
    total = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        snippet = step.get("snippet")
        if isinstance(snippet, str):
            total += len(snippet)
    return total


def trace_int(trace: Dict[str, Any], key: str) -> int | str:
    try:
        value = trace.get(key)
        if value in ("", None):
            return ""
        return int(value)
    except Exception:
        return ""


def included_passage_ids(steps: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for step in steps:
        if isinstance(step, dict) and step.get("source") == "COMPOSE":
            ids = step.get("included_passage_ids") or []
            out.extend(str(x) for x in ids if x)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--docs-root", type=Path, default=DEFAULT_DOCS_ROOT)
    ap.add_argument("--corpus-out", type=Path, default=REPO_ROOT / "artifacts" / "auto_compose_v1" / "corpus_auto_compose.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-mixed", type=int, default=None)
    ap.add_argument("--strict-llm-final", action="store_true")
    ap.add_argument("--llm-disabled", action="store_true")
    ap.add_argument("--force-corpus", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    rows = load_rows(args.data)
    keys = {(r["id"], r["domain"]) for r in rows}
    if len(keys) != len(rows):
        raise SystemExit(f"duplicate (id, domain) keys: rows={len(rows)} unique={len(keys)}")
    missing = [
        r for r in rows
        if any(not str(r.get(k, "")).strip() for k in ("id", "domain", "type", "query", "expected_contains", "product", "session"))
    ]
    if missing:
        raise SystemExit(f"rows missing required fields: {len(missing)}")

    rows.sort(key=lambda r: (r["id"], r["domain"]))
    if args.sample_mixed:
        rows = mixed_sample(rows, args.sample_mixed)
    if args.limit:
        rows = rows[:args.limit]

    print(f"[auto-compose] data rows={len(keys)} filtered={len(rows)} out={args.out}")
    corpus_path = build_clean_corpus(args.docs_root, args.corpus_out, force=args.force_corpus)
    if args.strict_llm_final:
        os.environ["AUTO_COMPOSE_DISABLE_DIRECT_FALLBACK"] = "1"
    else:
        os.environ.pop("AUTO_COMPOSE_DISABLE_DIRECT_FALLBACK", None)
    client = configure_runtime(corpus_path, llm_disabled=args.llm_disabled)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_done(args.out)
    write_header = not args.out.exists()
    ok_total = 0
    n_total = 0
    llm_used_total = 0
    api_error_rows = 0
    by_type: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    seeded_sessions: set[str] = set()

    try:
        with args.out.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            for idx, row in enumerate(rows, start=1):
                key = (row["id"], row["domain"])
                if key in done:
                    continue
                payload = {
                    "query": row["query"],
                    "product": row.get("product") or row.get("solver_product") or "",
                    "domain": row["domain"],
                    "session": row.get("session") or "s1",
                    "top_k_search": 4,
                    "top_k_memory": 3,
                }
                memory_seed = str(row.get("memory_seed") or "").strip()
                if memory_seed:
                    memory_service.flush_session(payload["session"])
                    memory_service.add_memory(payload["session"], memory_seed)
                    seeded_sessions.add(payload["session"])
                t0 = time.time()
                try:
                    resp = client.post("/solve_auto", json=payload)
                    if resp.status_code >= 400:
                        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
                    out = resp.json()
                    answer = out.get("answer", "") or ""
                    steps = out.get("steps", []) or []
                    sources = out.get("sources", []) or []
                    answer_trace = out.get("answer_trace") or {}
                    confidence = out.get("confidence", "")
                except Exception as exc:
                    answer = f"{type(exc).__name__}: {exc}"
                    steps = []
                    sources = []
                    answer_trace = {}
                    confidence = ""
                latency_ms = int((time.time() - t0) * 1000)
                success = score_row(answer, row)
                ok_total += success
                n_total += 1
                by_type[row.get("type", "")][0] += success
                by_type[row.get("type", "")][1] += 1
                llm_used = bool(answer_trace.get("llm_used"))
                llm_used_total += int(llm_used)
                if any(term in answer for term in BAD_API_TERMS):
                    api_error_rows += 1
                src_types = sorted({str(src.get("type")) for src in sources if isinstance(src, dict) and src.get("type")})
                pids = included_passage_ids(steps)
                writer.writerow({
                    "id": row["id"],
                    "mode": "AUTO_COMPOSE",
                    "type": row.get("type", ""),
                    "subtype": row.get("subtype", ""),
                    "domain": row["domain"],
                    "query": row["query"],
                    "product": payload["product"],
                    "session": payload["session"],
                    "success": success,
                    "steps": json.dumps(steps, ensure_ascii=False)[:5000],
                    "correct": success,
                    "latency_ms": latency_ms,
                    "confidence": confidence,
                    "confidence_raw": confidence,
                    "confidence_cal": confidence,
                    "cost_retrieval_calls": 1,
                    "cost_rule_checks": "",
                    "cost_tokens_in": trace_int(answer_trace, "prompt_tokens"),
                    "cost_tokens_out": trace_int(answer_trace, "completion_tokens"),
                    "n_steps": len(steps),
                    "answer": answer[:1200],
                    "expected_contains": row.get("expected_contains", ""),
                    "cost_usd_running": "",
                    "answer_trace": json.dumps(answer_trace, ensure_ascii=False)[:2000],
                    "llm_used": int(llm_used),
                    "provider": answer_trace.get("provider"),
                    "model": answer_trace.get("model"),
                    "api": answer_trace.get("api"),
                    "included_source_types": ",".join(src_types),
                    "included_passage_ids": ",".join(pids),
                    "fallback_path": answer_trace.get("path", ""),
                    "composed_context_chars": context_chars_from_steps(steps),
                })
                fh.flush()
                if idx % 10 == 0:
                    print(
                        f"[auto-compose] {idx}/{len(rows)} "
                        f"acc_so_far={ok_total / max(n_total, 1):.4f} "
                        f"llm_used={llm_used_total}/{n_total} last_success={success}"
                    )
                if args.sleep:
                    time.sleep(args.sleep)
    finally:
        for session in seeded_sessions:
            memory_service.flush_session(session)

    print(f"[auto-compose] DONE n={n_total} accuracy={ok_total / max(n_total, 1):.4f} llm_used={llm_used_total}/{n_total} api_error_rows={api_error_rows}")
    print(json.dumps({k: {"ok": v[0], "n": v[1], "acc": v[0] / v[1] if v[1] else 0.0} for k, v in sorted(by_type.items())}, indent=2))
    return 0 if api_error_rows == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
