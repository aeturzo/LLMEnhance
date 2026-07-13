#!/usr/bin/env python3
"""
Build an objectively-verified subset of the release-clean benchmark.

The release-clean benchmark (artifacts/release_benchmark_clean_docs.json, 6,915
rows) contains a minority of rows that NO system can answer reliably because
the benchmark itself is internally inconsistent:

  * contradictory duplicated ontology facts (e.g. vol_064 labelled both
    "12 V" and "3.7 V" in the same ontology file);
  * non-atomic / placeholder document gold answers;
  * records-style logic rows with no explicit supporting evidence.

This script applies OBJECTIVE exclusion rules — none of which look at any
model's predictions — and emits a verified subset plus an auditable report.

Inputs
------
  artifacts/release_benchmark_clean_docs.json
  release/release_20250902_215155/backend/ontologies/*.ttl
  release/release_20250902_215155/tests/<domain>/seed_docs.jsonl
  release/release_20250902_215155/tests/<domain>/seed_mem.jsonl

Outputs
-------
  artifacts/release_clean_verified_v1.json
  artifacts/release_clean_verified_v1_ids.json
  artifacts/release_clean_verified_v1_exclusions.csv
  artifacts/release_clean_verified_v1_report.md
  artifacts/release_clean_verified_v1_suggested_repairs.csv   (diagnostic only)

The verified subset KEEPS original ids, questions and gold answers unchanged,
so the already-completed external baseline CSVs can simply be filtered to the
verified id set (see scripts/filter_eval_to_verified_subset.py). No baseline
rerun is needed; only ADAPTIVERAG must be rerun on the verified ids.

Exclusion reason codes
----------------------
  duplicate_key                  (id, domain) appears more than once
  missing_gold                   expected_contains empty
  unsupported_schema             row missing required fields
  doc_placeholder_gold           gold is a template placeholder, e.g. "(enter …)"
  doc_gold_non_atomic            gold span longer than MAX_DOC_GOLD_CHARS
  doc_label_malformed            generated document label is not an atomic field
  doc_label_ambiguous            target document contains multiple values for label
  doc_gold_not_in_target_doc     doc gold not found in its target seed document
  records_missing_evidence       records-style row with no seed-memory evidence
  records_logic_not_verified     records-style yes/no row needs derived semantics
  kb_conflicting_entity_relation KB recall: entity+relation resolves to >1 value
  kb_gold_not_uniquely_supported KB recall: gold not the unique ontology value
  logic_class_not_derivable      KB logic: class membership not derivable
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_clean_release_benchmark import iter_kv_pairs, norm_key  # noqa: E402

DEFAULT_BENCHMARK = REPO_ROOT / "artifacts" / "release_benchmark_clean_docs.json"
RELEASE_ROOT = REPO_ROOT / "release" / "release_20250902_215155"
ONTOLOGY_ROOT = RELEASE_ROOT / "backend" / "ontologies"
DOCS_ROOT = RELEASE_ROOT / "tests"

MAX_DOC_GOLD_CHARS = 80

ONTOLOGY_FILES = {
    "battery": ["battery_augment.ttl", "dpp_ontology.ttl"],
    "lexmark": ["lexmark_augment.ttl", "lexmark_ontology.ttl"],
    "viessmann": ["viessmann_augment.ttl", "viessmann_ontology.ttl"],
}

# Generic OWL/RDF/XSD classes a KB-logic membership question can target without
# needing explicit per-entity assertions.
GENERIC_LOGIC_CLASSES = {
    "thing", "class", "datatype", "annotationproperty", "objectproperty",
    "datatypeproperty", "property", "resource", "namedindividual",
}
XSD_DATATYPES = {
    "integer", "string", "boolean", "decimal", "float", "double", "datetime",
    "datetimestamp", "date", "time", "nonnegativeinteger", "nonpositiveinteger",
    "positiveinteger", "negativeinteger", "unsignedlong", "unsignedint",
    "unsignedshort", "unsignedbyte", "long", "int", "short", "byte", "anyuri",
}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("‑", "-").replace("–", "-").replace("—", "-")
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize(text))


# ---------------------------------------------------------------------------
# Ontology parsing — one triple per line in these .ttl files
# ---------------------------------------------------------------------------
class Ontology:
    """A flat triple store parsed from the release .ttl files for one domain."""

    def __init__(self) -> None:
        # node -> set of label strings
        self.labels: Dict[str, Set[str]] = defaultdict(set)
        # node -> set of class nodes (rdf:type / 'a')
        self.types: Dict[str, Set[str]] = defaultdict(set)
        # class -> set of parent classes
        self.subclass: Dict[str, Set[str]] = defaultdict(set)
        # (subject, property) -> set of object nodes
        self.rel: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        # property node -> set of label strings
        self.prop_labels: Dict[str, Set[str]] = defaultdict(set)
        # node -> set of declared kinds (owl:Class, owl:ObjectProperty, …)
        self.declared: Dict[str, Set[str]] = defaultdict(set)

    def load(self, paths: List[Path]) -> None:
        for path in paths:
            if not path.exists():
                continue
            for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("@"):
                    continue
                line = line.rstrip(".").strip()
                if not line:
                    continue
                # Turtle: a statement is "SUBJ pred obj ; pred obj ; pred obj"
                parts = [p.strip() for p in self._split_top(line, ";") if p.strip()]
                if not parts:
                    continue
                first = parts[0].split(None, 2)
                if len(first) < 3:
                    continue
                subj = first[0]
                po_pairs = [(first[1], first[2])]
                for extra in parts[1:]:
                    bits = extra.split(None, 1)
                    if len(bits) == 2:
                        po_pairs.append((bits[0], bits[1]))
                for pred, obj in po_pairs:
                    obj = obj.strip()
                    if pred in ("a", "rdf:type"):
                        for cls in self._split_top(obj, ","):
                            cls = cls.strip()
                            if cls.startswith("owl:") or cls.startswith("rdfs:"):
                                self.declared[subj].add(cls)
                            self.types[subj].add(cls)
                    elif pred == "rdfs:subClassOf":
                        for cls in self._split_top(obj, ","):
                            self.subclass[subj].add(cls.strip())
                    elif pred == "rdfs:label":
                        lbl = self._literal(obj)
                        if lbl is not None:
                            self.labels[subj].add(lbl)
                    else:
                        lit = self._literal(obj)
                        if lit is not None:
                            self.rel[(subj, pred)].add(f'"{lit}"')
                        else:
                            for tok in self._split_top(obj, ","):
                                self.rel[(subj, pred)].add(tok.strip())

        # property labels: a node declared an *Property (or named ext:has*…)
        # that carries an rdfs:label is usable for relation lookups.
        for node, kinds in self.declared.items():
            if any("Property" in k for k in kinds):
                for lbl in self.labels.get(node, ()):
                    self.prop_labels[node].add(lbl)
        for node in list(self.labels):
            local = node.split(":")[-1]
            if local.startswith("has") or local in ("madeBy", "compatibleWith", "usesMaterial", "hasComponent"):
                for lbl in self.labels.get(node, ()):
                    self.prop_labels[node].add(lbl)

    @staticmethod
    def _split_top(text: str, sep: str) -> List[str]:
        """Split on `sep` but not inside quoted literals."""
        out: List[str] = []
        buf: List[str] = []
        in_quote = False
        for ch in text:
            if ch == '"':
                in_quote = not in_quote
            if ch == sep and not in_quote:
                out.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        out.append("".join(buf))
        return out

    @staticmethod
    def _literal(obj: str) -> Optional[str]:
        m = re.match(r'^"([^"]*)"', obj)
        return m.group(1) if m else None

    # -- queries -----------------------------------------------------------
    def nodes_for_label(self, label: str) -> List[str]:
        target = compact(label)
        return [n for n, lbls in self.labels.items()
                if any(compact(l) == target for l in lbls)]

    def property_for_label(self, label: str) -> List[str]:
        target = compact(label)
        out = [p for p, lbls in self.prop_labels.items()
               if any(compact(l) == target for l in lbls)]
        # also accept the bare local name, e.g. "hasComponent"
        if not out:
            for node in self.labels:
                if compact(node.split(":")[-1]) == target and "Property" in " ".join(self.declared.get(node, ())):
                    out.append(node)
        return sorted(set(out))

    def superclasses(self, cls: str, _seen: Optional[Set[str]] = None) -> Set[str]:
        if _seen is None:
            _seen = set()
        if cls in _seen:
            return set()
        _seen.add(cls)
        out = {cls}
        for parent in self.subclass.get(cls, ()):
            out |= self.superclasses(parent, _seen)
        return out

    def type_closure(self, node: str) -> Set[str]:
        out: Set[str] = set()
        for cls in self.types.get(node, ()):
            out |= self.superclasses(cls)
        return out


_ONTOLOGY_CACHE: Dict[str, Ontology] = {}


def ontology_for(domain: str) -> Ontology:
    if domain not in _ONTOLOGY_CACHE:
        onto = Ontology()
        onto.load([ONTOLOGY_ROOT / fn for fn in ONTOLOGY_FILES.get(domain, [])])
        _ONTOLOGY_CACHE[domain] = onto
    return _ONTOLOGY_CACHE[domain]


# ---------------------------------------------------------------------------
# Seed docs / memory
# ---------------------------------------------------------------------------
_SEED_DOC_CACHE: Dict[str, Dict[str, str]] = {}
_SEED_MEM_CACHE: Dict[str, List[str]] = {}


def seed_docs(domain: str) -> Dict[str, str]:
    if domain not in _SEED_DOC_CACHE:
        mapping: Dict[str, str] = {}
        path = DOCS_ROOT / domain / "seed_docs.jsonl"
        if path.exists():
            for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                text = rec.get("text") or rec.get("content") or ""
                mapping[f"{domain}_seed_{idx:04d}"] = text
                if rec.get("source"):
                    mapping[str(rec["source"])] = text
        _SEED_DOC_CACHE[domain] = mapping
    return _SEED_DOC_CACHE[domain]


def seed_mem(domain: str) -> List[str]:
    if domain not in _SEED_MEM_CACHE:
        facts: List[str] = []
        path = DOCS_ROOT / domain / "seed_mem.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("text"):
                    facts.append(rec["text"])
        _SEED_MEM_CACHE[domain] = facts
    return _SEED_MEM_CACHE[domain]


# ---------------------------------------------------------------------------
# Verification per row
# ---------------------------------------------------------------------------
PLACEHOLDER_RE = re.compile(r"\(enter|tbd|todo|xxx|placeholder|to be confirmed", re.IGNORECASE)
NON_ATOMIC_DOC_GOLD_RE = re.compile(
    r"(\*\*|\.\s+(?:it|this|these|values?|manufacturer|official)\b)",
    re.IGNORECASE,
)
MALFORMED_DOC_LABEL_RE = re.compile(r"(^[^A-Za-z0-9]+|[^A-Za-z0-9%)]$|[()])")
KB_RECALL_RE = re.compile(r"what is the (.+?) of (.+?)\??$", re.IGNORECASE)
KB_LOGIC_RE = re.compile(r"is (.+?) an? (.+?)\??$", re.IGNORECASE)


def verify_doc_row(row: Dict[str, Any]) -> Optional[str]:
    """cleandocopen / cleandocrec rows — gold must be atomic and in target doc."""
    gold = (row.get("expected_contains") or "").strip()
    if PLACEHOLDER_RE.search(gold):
        return "doc_placeholder_gold"
    if len(gold) > MAX_DOC_GOLD_CHARS or NON_ATOMIC_DOC_GOLD_RE.search(gold):
        return "doc_gold_non_atomic"
    meta = row.get("meta") or {}
    source_id = meta.get("source_id") or row.get("product") or ""
    docs = seed_docs(row["domain"])
    text = docs.get(source_id, "")
    if not text:
        return "doc_gold_not_in_target_doc"
    label = str(meta.get("label") or "").strip()
    if label:
        if MALFORMED_DOC_LABEL_RE.search(label):
            return "doc_label_malformed"
        values = [
            value for field, value in iter_kv_pairs(text)
            if norm_key(field) == norm_key(label)
        ]
        distinct = {compact(value) for value in values if value}
        if len(distinct) > 1:
            return "doc_label_ambiguous"
    if compact(gold) not in compact(text):
        return "doc_gold_not_in_target_doc"
    return None


def verify_records_row(row: Dict[str, Any]) -> Optional[str]:
    """opn / rec / log records-style rows — need explicit seed-memory evidence."""
    gold = (row.get("expected_contains") or "").strip()
    product = str(row.get("product") or "")
    facts = seed_mem(row["domain"])
    prod_facts = [f for f in facts if product and product.lower() in f.lower()]
    if not prod_facts:
        return "records_missing_evidence"
    qtype = row.get("type")
    if qtype in ("open", "recall"):
        # gold must be supported by at least one product memory fact
        if any(compact(gold) in compact(f) for f in prod_facts):
            return None
        return "records_missing_evidence"
    # The log-* rows are generated yes/no questions whose truth depends on
    # derived ontology semantics (component type, material use, compliance
    # closure). The seed-memory facts alone are not a complete truth source, so
    # these rows are not part of the objective verified subset.
    if compact(gold) in ("yes", "no"):
        return "records_logic_not_verified"
    return "records_missing_evidence"


def verify_kb_recall(row: Dict[str, Any]) -> Optional[str]:
    onto = ontology_for(row["domain"])
    m = KB_RECALL_RE.search(normalize(row["query"]))
    if not m:
        return "unsupported_schema"
    rel_label, ent_label = m.group(1).strip(), m.group(2).strip()
    gold = (row.get("expected_contains") or "").strip()

    # meta-questions like "what is the domain of hasComponent?" — keep only if
    # uniquely answerable; otherwise drop as unsupported schema.
    ent_nodes = onto.nodes_for_label(ent_label)
    if not ent_nodes:
        return "kb_gold_not_uniquely_supported"

    props = onto.property_for_label(rel_label)
    if not props:
        return "kb_gold_not_uniquely_supported"

    # collect all candidate value labels
    value_labels: Set[str] = set()
    for ent in ent_nodes:
        for prop in props:
            for target in onto.rel.get((ent, prop), ()):
                if target.startswith('"'):
                    value_labels.add(target.strip('"'))
                else:
                    for lbl in onto.labels.get(target, ()):
                        value_labels.add(lbl)
    distinct = {compact(v) for v in value_labels if v}
    if not distinct:
        return "kb_gold_not_uniquely_supported"
    if len(distinct) > 1:
        return "kb_conflicting_entity_relation"
    if compact(gold) not in distinct:
        return "kb_gold_not_uniquely_supported"
    return None


def verify_kb_logic(row: Dict[str, Any]) -> Optional[str]:
    onto = ontology_for(row["domain"])
    gold = (row.get("expected_contains") or "").strip()
    gold_c = compact(gold)

    # Generic universal / meta classes — always derivable.
    if gold_c in {compact(x) for x in GENERIC_LOGIC_CLASSES}:
        return None
    if gold_c in {compact(x) for x in XSD_DATATYPES}:
        return None

    m = KB_LOGIC_RE.search(normalize(row["query"]))
    if not m:
        return "unsupported_schema"
    ent_label = m.group(1).strip()
    ent_nodes = onto.nodes_for_label(ent_label)
    if not ent_nodes:
        # entities that are themselves values (e.g. "2200 mAh") — class
        # membership of a value node; keep only the generic cases handled above.
        return "logic_class_not_derivable"
    # is gold class reachable in the type closure of any node bearing this label?
    for ent in ent_nodes:
        closure = onto.type_closure(ent)
        closure_labels = set()
        for cls in closure:
            closure_labels.add(compact(cls.split(":")[-1]))
            for lbl in onto.labels.get(cls, ()):
                closure_labels.add(compact(lbl))
        if gold_c in closure_labels:
            return None
    return "logic_class_not_derivable"


def verify_row(row: Dict[str, Any]) -> Optional[str]:
    """Returns an exclusion reason code, or None if the row passes."""
    required = ("id", "domain", "type", "query", "expected_contains")
    if any(k not in row for k in required):
        return "unsupported_schema"
    if not (row.get("expected_contains") or "").strip():
        return "missing_gold"

    rid = row["id"]
    if rid.startswith("cleandoc"):
        return verify_doc_row(row)
    if rid.startswith(("opn-", "rec-", "log-")):
        return verify_records_row(row)
    if ".kb.recall" in rid:
        return verify_kb_recall(row)
    if ".kb.logic" in rid:
        return verify_kb_logic(row)
    # Unknown row family — keep conservatively only if gold is atomic.
    if len(row["expected_contains"]) > MAX_DOC_GOLD_CHARS:
        return "doc_gold_non_atomic"
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts")
    ap.add_argument("--tag", default="release_clean_verified_v1")
    args = ap.parse_args()

    data = json.loads(args.benchmark.read_text())
    rows = data["rows"] if isinstance(data, dict) else data
    print(f"[verify] loaded {len(rows)} rows from {args.benchmark}")

    seen_keys: Set[Tuple[str, str]] = set()
    kept: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, str]] = []
    reason_counts: Dict[str, int] = defaultdict(int)

    for row in rows:
        key = (row.get("id", ""), row.get("domain", ""))
        if key in seen_keys:
            reason = "duplicate_key"
        else:
            seen_keys.add(key)
            reason = verify_row(row)
        if reason:
            reason_counts[reason] += 1
            exclusions.append({
                "id": row.get("id", ""),
                "domain": row.get("domain", ""),
                "type": row.get("type", ""),
                "reason": reason,
                "expected_contains": (row.get("expected_contains") or "")[:120],
                "query": (row.get("query") or "")[:160],
            })
        else:
            kept.append(row)

    # ---- Outputs ----------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    verified_json = args.out_dir / f"{args.tag}.json"
    verified_ids = args.out_dir / f"{args.tag}_ids.json"
    exclusions_csv = args.out_dir / f"{args.tag}_exclusions.csv"
    report_md = args.out_dir / f"{args.tag}_report.md"
    repairs_csv = args.out_dir / f"{args.tag}_suggested_repairs.csv"

    verified_json.write_text(json.dumps({
        "meta": {
            "source_benchmark": str(args.benchmark),
            "n_input": len(rows),
            "n_verified": len(kept),
            "n_excluded": len(exclusions),
            "max_doc_gold_chars": MAX_DOC_GOLD_CHARS,
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        },
        "rows": kept,
    }, indent=1))

    verified_ids.write_text(json.dumps(
        [f"{r['id']}|{r['domain']}" for r in kept], indent=1))

    with exclusions_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "domain", "type", "reason",
                                           "expected_contains", "query"])
        w.writeheader()
        w.writerows(exclusions)

    # Diagnostic-only suggested repairs for non-atomic doc gold (NOT applied).
    with repairs_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "domain", "reason", "current_gold", "suggested_atomic_gold"])
        for ex in exclusions:
            if ex["reason"] == "doc_gold_non_atomic":
                cur = ex["expected_contains"]
                # heuristic: first sentence / first 80 chars up to a period
                suggestion = re.split(r"(?<=[.;])\s", cur)[0][:MAX_DOC_GOLD_CHARS]
                w.writerow([ex["id"], ex["domain"], ex["reason"], cur, suggestion])

    # ---- Report -----------------------------------------------------------
    by_type = defaultdict(int)
    by_domain = defaultdict(int)
    for r in kept:
        by_type[r.get("type", "?")] += 1
        by_domain[r.get("domain", "?")] += 1

    lines = [
        f"# Release-clean verified subset — {args.tag}",
        "",
        f"- Input rows: **{len(rows)}**",
        f"- Verified (kept): **{len(kept)}**",
        f"- Excluded: **{len(exclusions)}**  ({len(exclusions)/len(rows)*100:.1f}%)",
        "",
        "## Exclusions by reason",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {reason} | {count} |")
    lines += [
        "",
        "## Verified subset composition",
        "",
        "| Dimension | Breakdown |",
        "|---|---|",
        f"| By type | {', '.join(f'{k}: {v}' for k, v in sorted(by_type.items()))} |",
        f"| By domain | {', '.join(f'{k}: {v}' for k, v in sorted(by_domain.items()))} |",
        "",
        "## Acceptance gate",
        "",
        f"- Verified subset size: **{len(kept)}** "
        f"({'OK — >=5000' if len(kept) >= 5000 else 'OK — >=4000' if len(kept) >= 4000 else 'TOO SMALL — <4000, do not use' if len(kept) < 3000 else 'MARGINAL — 3000-4000'})",
        "",
        "All exclusion rules are objective and independent of any model's "
        "predictions (see scripts/build_release_clean_verified_subset.py "
        "docstring). External baseline CSVs can be filtered to "
        f"`{verified_ids.name}` with no rerun; only ADAPTIVERAG must be rerun.",
    ]
    report_md.write_text("\n".join(lines))

    # ---- Console ----------------------------------------------------------
    print(f"[verify] kept {len(kept)}  excluded {len(exclusions)}")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"         {reason:32s} {count}")
    print(f"[verify] wrote:")
    for p in (verified_json, verified_ids, exclusions_csv, report_md, repairs_csv):
        print(f"         {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
