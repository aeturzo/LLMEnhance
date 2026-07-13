#!/usr/bin/env python3
"""Build a clean release benchmark with regenerated document QA rows.

The archived release benchmark contains valid KB/fact rows, but some document
QA rows no longer match the current seed_docs.jsonl order/content. This script
does not mutate the release bundle. It writes a new artifact that:

- keeps non-document rows from artifacts/release_benchmark.json;
- drops old docopen/docrec/docfix/*.doc.* rows;
- regenerates document open/recall rows from the current seed_docs.jsonl files;
- uses unique cleandocopen/cleandocrec IDs to avoid accidental joins against
  archived paper rows with different question text.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_JSON = REPO_ROOT / "artifacts" / "release_benchmark.json"
DEFAULT_DOCS_ROOT = REPO_ROOT / "release" / "release_20250902_215155" / "tests"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "release_benchmark_clean_docs.json"
DOMAINS = ("battery", "lexmark", "viessmann")

NUMERIC_RE = re.compile(
    r"\b\d+(?:[\.,]\d+)?\s?(?:kWh|Wh|W|kW|V|mAh|Ah|A|\u00b0C|C|kg|g|mm|cm|m|%|ppm|bar|psi|years?|months?|days?|pages?|s|dpi|MB|GHz)\b",
    flags=re.IGNORECASE,
)
WS_RE = re.compile(r"\s+")


def compact(text: str) -> str:
    return WS_RE.sub(" ", (text or "").strip())


def norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def canonical_expected(value: str) -> str:
    value = compact(value).strip(" .;")
    # Keep a concise core answer when the source adds a trailing qualifier such
    # as "(sample)" or "(very compact)". The core value remains a source
    # substring and is a better target for substring scoring.
    stripped = re.sub(r"\s+\([^)]{1,60}\)$", "", value).strip(" .;")
    return stripped if stripped else value


def is_old_doc_row(row: Dict[str, Any]) -> bool:
    rid = str(row.get("id") or "")
    return (
        rid.startswith("docopen-")
        or rid.startswith("docrec-")
        or rid.startswith("docfix-")
        or ".doc." in rid
    )


def load_seed_docs(docs_root: Path, domain: str) -> List[Dict[str, str]]:
    path = docs_root / domain / "seed_docs.jsonl"
    out: List[Dict[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            rec = json.loads(line)
            text = rec.get("text") or rec.get("content") or ""
            if not text and isinstance(rec.get("chunks"), list):
                parts = []
                for chunk in rec["chunks"]:
                    parts.append(chunk.get("text", "") if isinstance(chunk, dict) else str(chunk))
                text = "\n".join(p for p in parts if p)
            if not text:
                continue
            out.append({
                "source_id": f"{domain}_seed_{idx:04d}",
                "source": str(rec.get("source") or rec.get("doc_id") or f"{domain}_seed_{idx:04d}"),
                "text": text,
            })
    return out


def split_inline_kv(line: str) -> Iterable[Tuple[str, str]]:
    line = compact(line.lstrip(" -*\u2022\t"))
    if ":" not in line:
        return []
    label_re = re.compile(r"([A-Za-z][A-Za-z0-9 /()\u00d7.+\-&]{1,60}):\s*")
    matches = [
        m for m in label_re.finditer(line)
        if m.group(1).strip().lower() not in {"http", "https"}
    ]
    if not matches:
        return []
    pairs = []
    for idx, match in enumerate(matches):
        label = compact(match.group(1))
        value_start = match.end()
        value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
        value = compact(line[value_start:value_end])
        if not label or not value:
            continue
        if len(label) > 60 or len(value) > 180:
            continue
        if "|" in value:
            continue
        if not re.search(r"[A-Za-z0-9]", value):
            continue
        if label.startswith(("http", "https")):
            continue
        if set(label) <= {"-", "=", "_"}:
            continue
        if value.lower() in {"(enter)", "(verify)", "(enter if applicable)"}:
            continue
        pairs.append((label, value))
    return pairs


def iter_kv_pairs(text: str) -> Iterable[Tuple[str, str]]:
    seen = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or set(line) <= {"-", "=", "_"}:
            continue
        for label, value in split_inline_kv(line):
            key = (norm_key(label), norm_key(value))
            if key in seen:
                continue
            seen.add(key)
            yield label, value


def make_row(
    rid: str,
    qtype: str,
    domain: str,
    source_id: str,
    query: str,
    expected: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": rid,
        "domain": domain,
        "type": qtype,
        "query": query,
        "expected_contains": expected,
        "product": source_id,
        "session": "s_docs_clean",
        "meta": meta,
        "ontology_refs": [],
    }


def generate_doc_rows(
    docs_root: Path,
    original_doc_counts: Counter,
    per_source_open_cap: int,
    per_source_recall_cap: int,
    include_numeric_fallback: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"domains": {}}

    for domain in DOMAINS:
        docs = load_seed_docs(docs_root, domain)
        candidates: Dict[str, List[Dict[str, Any]]] = {"open": [], "recall": []}
        seen = set()

        for doc in docs:
            source_id = doc["source_id"]
            open_count = 0
            recall_count = 0
            for label, value in iter_kv_pairs(doc["text"]):
                expected = canonical_expected(value)
                qtype = "recall" if NUMERIC_RE.search(expected) else "open"
                if qtype == "open" and open_count >= per_source_open_cap:
                    continue
                if qtype == "recall" and recall_count >= per_source_recall_cap:
                    continue
                query = f"According to {source_id}, what is the {label}?"
                key = (qtype, domain, norm_key(query), norm_key(expected))
                if key in seen:
                    continue
                seen.add(key)
                if qtype == "open":
                    open_count += 1
                else:
                    recall_count += 1
                candidates[qtype].append(make_row(
                    rid="",
                    qtype=qtype,
                    domain=domain,
                    source_id=source_id,
                    query=query,
                    expected=expected,
                    meta={"source_id": source_id, "source": doc["source"], "label": label, "from": "clean_kv"},
                ))

            # Numeric fallback for rows where the label is not explicit.
            if include_numeric_fallback and recall_count < per_source_recall_cap:
                for match in NUMERIC_RE.finditer(doc["text"]):
                    if recall_count >= per_source_recall_cap:
                        break
                    expected = compact(match.group(0))
                    snippet = compact(doc["text"][max(0, match.start() - 45): match.end() + 45])
                    if len(snippet) > 130:
                        snippet = snippet[:130] + "..."
                    query = f"In {source_id}, what is the value reported near: \"{snippet}\"?"
                    key = ("recall", domain, norm_key(query), norm_key(expected))
                    if key in seen:
                        continue
                    seen.add(key)
                    recall_count += 1
                    candidates["recall"].append(make_row(
                        rid="",
                        qtype="recall",
                        domain=domain,
                        source_id=source_id,
                        query=query,
                        expected=expected,
                        meta={"source_id": source_id, "source": doc["source"], "from": "clean_numeric"},
                    ))

        domain_report: Dict[str, Any] = {
            "seed_docs": len(docs),
            "candidate_open": len(candidates["open"]),
            "candidate_recall": len(candidates["recall"]),
            "targets": {},
            "selected": {},
        }

        for qtype, prefix in (("open", "cleandocopen"), ("recall", "cleandocrec")):
            target = int(original_doc_counts.get((domain, qtype), 0))
            selected = candidates[qtype][:target]
            domain_report["targets"][qtype] = target
            domain_report["selected"][qtype] = len(selected)
            for idx, row in enumerate(selected):
                row["id"] = f"{prefix}-{idx:06d}"
                rows.append(row)
        report["domains"][domain] = domain_report

    return rows, report


def validate_doc_rows(rows: List[Dict[str, Any]], docs_root: Path) -> Dict[str, Any]:
    docs_by_domain = {domain: {d["source_id"]: d["text"] for d in load_seed_docs(docs_root, domain)} for domain in DOMAINS}
    total = 0
    ok = 0
    misses = []
    for row in rows:
        text = docs_by_domain[row["domain"]].get(row["product"], "")
        total += 1
        expected_norm = re.sub(r"[^a-z0-9]+", "", row["expected_contains"].lower())
        text_norm = re.sub(r"[^a-z0-9]+", "", text.lower())
        if expected_norm and expected_norm in text_norm:
            ok += 1
        elif len(misses) < 10:
            misses.append({
                "id": row["id"],
                "domain": row["domain"],
                "product": row["product"],
                "expected_contains": row["expected_contains"],
            })
    return {"doc_rows": total, "gold_in_target_doc": ok, "misses": misses}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-json", type=Path, default=DEFAULT_RELEASE_JSON)
    ap.add_argument("--docs-root", type=Path, default=DEFAULT_DOCS_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-source-open-cap", type=int, default=10)
    ap.add_argument("--per-source-recall-cap", type=int, default=25)
    ap.add_argument("--include-numeric-fallback", action="store_true")
    args = ap.parse_args()

    release_data = json.loads(args.release_json.read_text(encoding="utf-8"))
    original_rows = release_data["rows"]
    original_doc_counts = Counter(
        (row["domain"], row.get("type", "open"))
        for row in original_rows
        if is_old_doc_row(row)
    )

    kept_rows = [row for row in original_rows if not is_old_doc_row(row)]
    clean_doc_rows, generation_report = generate_doc_rows(
        args.docs_root,
        original_doc_counts,
        args.per_source_open_cap,
        args.per_source_recall_cap,
        args.include_numeric_fallback,
    )
    rows = kept_rows + clean_doc_rows
    rows.sort(key=lambda row: (row["id"], row["domain"]))
    validation = validate_doc_rows(clean_doc_rows, args.docs_root)

    out = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_release_json": str(args.release_json.resolve()),
            "docs_root": str(args.docs_root.resolve()),
            "dropped_old_doc_rows": len(original_rows) - len(kept_rows),
            "kept_non_doc_rows": len(kept_rows),
            "generated_doc_rows": len(clean_doc_rows),
            "total_rows": len(rows),
            "original_doc_counts": {f"{domain}:{qtype}": count for (domain, qtype), count in sorted(original_doc_counts.items())},
            "generation": generation_report,
            "validation": validation,
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["meta"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
