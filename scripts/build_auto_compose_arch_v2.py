#!/usr/bin/env python3
"""
Build the architecture-focused benchmark `auto_compose_arch_3000_v2`.

This benchmark tests the intended LLMEnhance composition architecture:
interpret -> gather evidence (session memory + documents + symbolic KG) ->
compose -> answer. A long-context single-pass LLM can match the architecture
on atomic one-fact rows, so 1,500 of the 3,000 rows are *multi-source* rows
that genuinely require composing evidence from two or three distinct sources.

Composition (3,000 rows)
------------------------
  1,500 atomic verified rows         (from release_clean_verified_v1.json)
  1,500 multi-source rows
        375 memory_symbolic_compliance
        375 memory_symbolic_component
        300 memory_symbolic_step
        225 memory_document
        225 memory_document_symbolic

Scoring
-------
Every row carries `expected_groups` — a list of groups. A row is correct iff
the answer contains at least one accepted value from EVERY group. This avoids
false failures when "one compliance standard" admits several valid answers.

Grounding rules (objective; no invented facts)
----------------------------------------------
  * symbolic facts come from the Codex-verified PRODUCTS table (components /
    standards / steps that exist in the current ontologies + records);
  * document facts come from verified seed-document rows in
    release_clean_verified_v1.json;
  * memory facts are session-scoped preferences seeded per row in
    `memory_seed` — they exist nowhere else, so a multi-source row cannot be
    solved from documents or the KG alone.

Outputs
-------
  artifacts/auto_compose_v2/auto_compose_arch_3000_v2.json
  artifacts/auto_compose_v2/auto_compose_arch_3000_v2_ids.json
  artifacts/auto_compose_v2/auto_compose_arch_3000_v2_report.md
  artifacts/auto_compose_v2/auto_compose_arch_3000_v2_validation.csv

Usage
-----
  python scripts/build_auto_compose_arch_v2.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_JSON = REPO_ROOT / "artifacts" / "release_clean_verified_v1.json"
OUT_DIR = REPO_ROOT / "artifacts" / "auto_compose_v2"
SEED = 20260521

# ---------------------------------------------------------------------------
# Codex-verified product facts (components / standards / steps that exist in
# the current ontologies + records — do NOT add anything not listed here).
# ---------------------------------------------------------------------------
PRODUCTS: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "battery": {
        "ProductA": {
            "components": ["WirelessModule1", "Board1", "Battery1"],
            "standards": ["EN 62133-2", "Battery Safety Standard",
                          "Wireless Compliance Standard", "RoHS"],
            "steps": ["BatteryTestStep", "WirelessTestStep"],
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
            "standards": ["Wireless Compliance", "WEEE Marking", "EMC Compliance",
                          "IEC 62368-1 Safety", "RoHS"],
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
            "standards": ["EU F-Gas Regulation", "Electrical Safety Check",
                          "RoHS", "Wireless Compliance"],
            "steps": ["Leak Check", "Pressure Test", "Electrical Safety Test",
                      "Wireless Test"],
        },
        "ProductV2": {
            "components": ["Comp2", "Board2"],
            "standards": ["Electrical Safety Check"],
            "steps": ["Leak Check", "Pressure Test", "Electrical Safety Test"],
        },
    },
}

# Memory-only preference vocabulary (these exist ONLY in seeded session memory).
PACKAGING = [
    "recycled cardboard", "aluminum pouch", "double-walled carton",
    "returnable crate", "molded pulp packaging", "reusable transit box",
    "paper-based cushioning", "low-plastic shipper",
]
SUPPLIERS = [
    "GreenCells", "NordicPack", "PrintPack", "HeatPack",
    "EcoWrap", "CircularBox", "DPP Logistics", "MaterialLoop",
]

# Paraphrase templates. {p}=product, {dp}=memory descriptor (packaging/supplier),
# {sp}=symbolic descriptor, {doc}=seed-doc id, {label}=doc label.
TEMPLATES: Dict[str, List[str]] = {
    "memory_symbolic_compliance": [
        "For {p}, combine my preferred {dp} with one compliance standard required by the product record.",
        "Give my preferred {dp} for {p} and one compliance standard {p} must meet.",
        "Using my saved preference, state my preferred {dp} for {p} together with one standard the {p} record requires.",
    ],
    "memory_symbolic_component": [
        "For {p}, give my preferred {dp} and one component documented for {p}.",
        "State my preferred {dp} for {p} and one component the {p} record lists.",
        "Combine my preferred {dp} for {p} with one documented {p} component.",
    ],
    "memory_symbolic_step": [
        "For {p}, give my preferred {dp} and one required test step for {p}.",
        "State my preferred {dp} for {p} and one test step the {p} record requires.",
        "Combine my preferred {dp} for {p} with one required {p} verification step.",
    ],
    "memory_document": [
        "For {doc}, state my preferred {dp} and the {label} recorded in its product passport.",
        "Give my preferred {dp} for {doc} together with the {label} from the {doc} passport.",
        "Combine my preferred {dp} for {doc} with the {label} documented in {doc}.",
    ],
    "memory_document_symbolic": [
        "Using my saved preference and both product records: give my preferred {dp} for {p}, one compliance standard {p} must meet, and the {label} recorded in {doc}.",
        "Combine my preferred {dp} for {p}, one documented component of {p}, and the {label} stated in the {doc} passport.",
        "State my preferred {dp} for {p}, one required test step for {p}, and the {label} from the {doc} record.",
    ],
}

ATOMIC_SOURCE = {  # id-family -> evidence source for atomic rows
    "cleandoc": "document",
    "kb": "symbolic",
    "records": "memory",
}


def compact(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("‑", "-").replace("–", "-").replace("—", "-")
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9]+", "", text)


def infer_product_from_query(query: str) -> str:
    for pattern in (
        r"\bProductV\d+\b",
        r"\bPrinterL\d+\b",
        r"\bProduct[A-Za-z0-9_-]+\b",
        r"\bLiCell-\d+\b",
        r"\b(?:battery|lexmark|viessmann)_seed_\d+\b",
    ):
        m = re.search(pattern, query or "")
        if m:
            return m.group(0)
    m = re.match(r"^\s*is\s+(.+?)\s+an?\s+.+?\??\s*$", query or "", flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.match(r"^\s*what is the\s+.+?\s+of\s+(.+?)\??\s*$", query or "", flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def display_doc_label(label: str) -> str:
    label = re.sub(r"^\s*kg\s+", "", label or "", flags=re.IGNORECASE)
    # "Standard" appears under the warranty section in the source passports.
    # Use the section-aware label in generated questions so models do not
    # confuse it with compliance standards or nearby packaging text.
    if compact(label) == "standard":
        return "Warranty Standard"
    return label


def sanitize_query(query: str) -> str:
    query = query or ""
    query = re.sub(r"(what is the)\s+kg\s+", r"\1 ", query, flags=re.IGNORECASE)
    return query


def atomic_source(rid: str) -> str:
    if rid.startswith("cleandoc"):
        return "document"
    if ".kb." in rid:
        return "symbolic"
    if rid.startswith(("log-", "rec-", "opn-")):
        return "memory"
    return "document"


def atomic_expected_groups(row: Dict[str, Any], source: str, gold: str) -> List[List[str]]:
    """Return accepted answer groups for atomic rows.

    The release ``opn``/``rec`` rows sometimes ask for "a component" or "a
    standard" while the product record contains several valid values. Accept
    any grounded product value for those underspecified prompts instead of
    treating one arbitrary release gold as the only correct answer.
    """
    if source != "memory":
        return [[gold]]
    dom = row.get("domain", "")
    product = row.get("product", "") or infer_product_from_query(row.get("query", ""))
    facts = PRODUCTS.get(dom, {}).get(product)
    if not facts:
        return [[gold]]
    q = (row.get("query") or "").lower()
    if "component" in q:
        return [list(facts["components"])]
    if "standard" in q or "conforms to" in q or "compliance" in q:
        return [list(facts["standards"])]
    if "step" in q or "test" in q:
        return [list(facts["steps"] or [gold])]
    return [[gold]]


# ---------------------------------------------------------------------------
# Atomic rows
# ---------------------------------------------------------------------------
def build_atomic_rows(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    verified = json.loads(VERIFIED_JSON.read_text())["rows"]
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in verified:
        by_domain[r["domain"]].append(r)
    domains = sorted(by_domain)
    per_domain = n // len(domains)

    picked: List[Dict[str, Any]] = []
    for di, dom in enumerate(domains):
        pool = by_domain[dom][:]
        rng.shuffle(pool)
        # last domain absorbs the rounding remainder
        take = per_domain if di < len(domains) - 1 else n - len(picked)
        # spread across types within the domain
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in pool:
            by_type[r["type"]].append(r)
        types = sorted(by_type)
        chosen: List[Dict[str, Any]] = []
        idx = 0
        while len(chosen) < take and any(by_type[t] for t in types):
            t = types[idx % len(types)]
            if by_type[t]:
                chosen.append(by_type[t].pop())
            idx += 1
        picked.extend(chosen[:take])

    rows: List[Dict[str, Any]] = []
    for r in picked:
        src = atomic_source(r["id"])
        gold = r["expected_contains"]
        groups = atomic_expected_groups(r, src, gold)
        rows.append({
            "domain": r["domain"],
            "type": r.get("type", "open"),
            "subtype": "atomic_verified",
            "query": sanitize_query(r["query"]),
            "product": r.get("product", "") or infer_product_from_query(r["query"]),
            "memory_seed": "",
            "expected_groups": groups,
            "required_sources": [src],
            "gold_evidence": [{"source": src, "value": gold}],
            "expected_contains": gold,
            "origin_id": r["id"],
        })
    return rows


# ---------------------------------------------------------------------------
# Multi-source rows
# ---------------------------------------------------------------------------
def load_doc_facts() -> Dict[str, List[Tuple[str, str, str]]]:
    verified = json.loads(VERIFIED_JSON.read_text())["rows"]
    facts: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for r in verified:
        if not r["id"].startswith("cleandoc"):
            continue
        meta = r.get("meta") or {}
        sid = meta.get("source_id") or r.get("product", "")
        label = meta.get("label", "fact")
        val = r["expected_contains"]
        if sid and val:
            facts[r["domain"]].append((sid, display_doc_label(label), val))
    return facts


def product_list() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for dom, items in PRODUCTS.items():
        for prod in items:
            out.append((dom, prod))
    return out


def domain_balanced_products(step_only: bool = False) -> List[Tuple[str, str]]:
    """Round-robin over DOMAINS first (so domain counts stay balanced), and
    within each domain rotate its products."""
    domains = sorted(PRODUCTS)
    per_domain: Dict[str, List[str]] = {}
    for dom in domains:
        prods = [p for p in PRODUCTS[dom]
                 if (not step_only) or PRODUCTS[dom][p]["steps"]]
        per_domain[dom] = prods
    # interleave: dom0[p], dom1[p], dom2[p], dom0[p+1], ...
    order: List[Tuple[str, str]] = []
    rounds = max(len(v) for v in per_domain.values())
    for rnd in range(rounds):
        for dom in domains:
            prods = per_domain[dom]
            if prods:
                order.append((dom, prods[rnd % len(prods)]))
    return order


def build_multisource_rows(rng: random.Random) -> List[Dict[str, Any]]:
    doc_facts = load_doc_facts()
    products = domain_balanced_products(step_only=False)
    step_products = domain_balanced_products(step_only=True)
    rows: List[Dict[str, Any]] = []

    spec = [
        ("memory_symbolic_compliance", 375, "standards"),
        ("memory_symbolic_component", 375, "components"),
        ("memory_symbolic_step", 300, "steps"),
        ("memory_document", 225, None),
        ("memory_document_symbolic", 225, None),
    ]

    mem_counter = 0  # drives packaging/supplier rotation for diversity

    for subtype, count, fact_key in spec:
        pool = step_products if subtype == "memory_symbolic_step" else products
        for i in range(count):
            dom, prod = pool[i % len(pool)]
            facts = PRODUCTS[dom][prod]
            template = TEMPLATES[subtype][i % len(TEMPLATES[subtype])]

            # alternate packaging / supplier as the memory descriptor
            use_supplier = (i % 2 == 1)
            packaging = PACKAGING[mem_counter % len(PACKAGING)]
            supplier = SUPPLIERS[(mem_counter * 3) % len(SUPPLIERS)]
            mem_counter += 1
            if use_supplier:
                mem_desc, mem_value = "supplier", supplier
            else:
                mem_desc, mem_value = "packaging", packaging

            memory_seed = (
                f"For {prod}, the preferred packaging is {packaging} "
                f"and the preferred supplier is {supplier}."
            )

            row: Dict[str, Any] = {
                "domain": dom,
                "type": "compose",
                "subtype": subtype,
                "product": prod,
                "memory_seed": memory_seed,
            }

            if subtype in ("memory_symbolic_compliance", "memory_symbolic_component",
                           "memory_symbolic_step"):
                sym_values = facts[fact_key]
                row["query"] = template.format(p=prod, dp=mem_desc)
                row["expected_groups"] = [[mem_value], list(sym_values)]
                row["required_sources"] = ["memory", "symbolic"]
                row["gold_evidence"] = [
                    {"source": "memory", "value": mem_value},
                    {"source": "symbolic", "value": sym_values[0]},
                ]
                row["expected_contains"] = f"{mem_value} || {sym_values[0]}"

            elif subtype == "memory_document":
                dfacts = doc_facts[dom]
                sid, label, val = dfacts[(i * 7 + 3) % len(dfacts)]
                # the memory_seed must reference the doc product
                row["product"] = sid
                row["memory_seed"] = (
                    f"For {sid}, the preferred packaging is {packaging} "
                    f"and the preferred supplier is {supplier}."
                )
                row["query"] = template.format(doc=sid, dp=mem_desc, label=label)
                row["expected_groups"] = [[mem_value], [val]]
                row["required_sources"] = ["memory", "document"]
                row["gold_evidence"] = [
                    {"source": "memory", "value": mem_value},
                    {"source": "document", "value": val},
                ]
                row["expected_contains"] = f"{mem_value} || {val}"

            else:  # memory_document_symbolic
                dfacts = doc_facts[dom]
                sid, label, val = dfacts[(i * 11 + 5) % len(dfacts)]
                # rotate which symbolic fact family this row uses
                sym_key = ["standards", "components", "steps"][i % 3]
                sym_values = facts[sym_key] or facts["standards"] or facts["components"]
                row["query"] = template.format(p=prod, dp=mem_desc, doc=sid, label=label)
                row["expected_groups"] = [[mem_value], list(sym_values), [val]]
                row["required_sources"] = ["memory", "symbolic", "document"]
                row["gold_evidence"] = [
                    {"source": "memory", "value": mem_value},
                    {"source": "symbolic", "value": sym_values[0]},
                    {"source": "document", "value": val},
                ]
                row["expected_contains"] = f"{mem_value} || {sym_values[0]} || {val}"
                row["doc_ref"] = sid

            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
REQUIRED_KEYS = ("id", "domain", "type", "subtype", "query", "product",
                 "session", "expected_groups", "required_sources")


def validate(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], List[str]]:
    """Returns (validation_records, fatal_errors)."""
    records: List[Dict[str, str]] = []
    fatal: List[str] = []
    seen_ids: set = set()

    for r in rows:
        rid = r["id"]
        problems: List[str] = []

        for k in REQUIRED_KEYS:
            if k not in r or r[k] in (None, "", []):
                if not (k == "product" and r["subtype"] == "atomic_verified"):
                    problems.append(f"missing:{k}")
        if rid in seen_ids:
            problems.append("duplicate_id")
        seen_ids.add(rid)

        groups = r.get("expected_groups") or []
        if not all(isinstance(g, list) and g for g in groups):
            problems.append("empty_or_malformed_group")

        if r["subtype"] != "atomic_verified":
            # multi-source rows must have >= 2 groups from >= 2 sources
            if len(groups) < 2:
                problems.append("multisource_lt2_groups")
            srcs = {e["source"] for e in r.get("gold_evidence", [])}
            if len(srcs) < 2:
                problems.append("multisource_lt2_sources")
            # every gold value must be grounded in its declared source
            for ev in r.get("gold_evidence", []):
                src, val = ev["source"], ev["value"]
                if src == "memory":
                    if compact(val) not in compact(r["memory_seed"]):
                        problems.append(f"memory_value_not_in_seed:{val}")
                elif src == "symbolic":
                    dom = r["domain"]
                    prod = r["product"] if r["product"] in PRODUCTS.get(dom, {}) else None
                    ok = False
                    if prod:
                        facts = PRODUCTS[dom][prod]
                        allv = facts["components"] + facts["standards"] + facts["steps"]
                        ok = any(compact(val) == compact(x) for x in allv)
                    if not ok:
                        problems.append(f"symbolic_value_ungrounded:{val}")
                # document values are validated against the verified doc-fact
                # set at build time (they come straight from it), so no
                # re-check needed here.
            # single-source-solvable guard: the memory group value must not
            # appear in any non-memory group
            mem_vals = set()
            for ev in r.get("gold_evidence", []):
                if ev["source"] == "memory":
                    mem_vals.add(compact(ev["value"]))
            for gi, g in enumerate(groups):
                # group 0 is the memory group by construction
                if gi == 0:
                    continue
                if any(compact(v) in mem_vals for v in g):
                    problems.append("memory_value_leaks_into_other_group")

        records.append({
            "id": rid,
            "domain": r["domain"],
            "subtype": r["subtype"],
            "n_groups": str(len(groups)),
            "required_sources": "|".join(r.get("required_sources", [])),
            "status": "OK" if not problems else "FAIL",
            "problems": ";".join(problems),
        })
        if problems:
            fatal.append(f"{rid}: {problems}")
    return records, fatal


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--atomic", type=int, default=1500)
    args = ap.parse_args()
    rng = random.Random(SEED)

    atomic = build_atomic_rows(args.atomic, rng)
    multi = build_multisource_rows(rng)
    all_rows = atomic + multi

    # assign ids + sessions
    for i, r in enumerate(all_rows, start=1):
        r["id"] = f"arch-v2-{i:06d}"
        r["session"] = f"arch_v2_s{i:06d}"

    records, fatal = validate(all_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bench_json = args.out_dir / "auto_compose_arch_3000_v2.json"
    ids_json = args.out_dir / "auto_compose_arch_3000_v2_ids.json"
    report_md = args.out_dir / "auto_compose_arch_3000_v2_report.md"
    valid_csv = args.out_dir / "auto_compose_arch_3000_v2_validation.csv"

    sub_counts = Counter(r["subtype"] for r in all_rows)
    dom_counts = Counter(r["domain"] for r in all_rows)
    n_atomic = sum(1 for r in all_rows if r["subtype"] == "atomic_verified")
    n_multi = len(all_rows) - n_atomic
    n_fail = sum(1 for rec in records if rec["status"] == "FAIL")

    bench_json.write_text(json.dumps({
        "meta": {
            "name": "auto_compose_arch_3000_v2",
            "seed": SEED,
            "total_rows": len(all_rows),
            "atomic_rows": n_atomic,
            "multisource_rows": n_multi,
            "subtype_counts": dict(sorted(sub_counts.items())),
            "domain_counts": dict(sorted(dom_counts.items())),
            "validation_failures": n_fail,
            "scoring": "expected_groups — answer must contain >=1 accepted value per group",
        },
        "rows": all_rows,
    }, indent=1, ensure_ascii=False))

    ids_json.write_text(json.dumps(
        [f"{r['id']}|{r['domain']}" for r in all_rows], indent=1))

    with valid_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "domain", "subtype", "n_groups",
                                           "required_sources", "status", "problems"])
        w.writeheader()
        w.writerows(records)

    # ---- report ----------------------------------------------------------
    by_sub_dom = Counter((r["subtype"], r["domain"]) for r in all_rows)
    lines = [
        "# auto_compose_arch_3000_v2 — build report",
        "",
        f"- Total rows: **{len(all_rows)}**",
        f"- Atomic verified rows: **{n_atomic}**",
        f"- Multi-source rows: **{n_multi}**",
        f"- Validation failures: **{n_fail}** "
        f"({'PASS — all rows valid' if n_fail == 0 else 'FAIL — see validation CSV'})",
        "",
        "## Row counts by subtype",
        "",
        "| Subtype | Count |",
        "|---|---:|",
    ]
    for sub, c in sorted(sub_counts.items()):
        lines.append(f"| {sub} | {c} |")
    lines += ["", "## Row counts by domain", "", "| Domain | Count |", "|---|---:|"]
    for dom, c in sorted(dom_counts.items()):
        lines.append(f"| {dom} | {c} |")
    lines += ["", "## Subtype × domain", "", "| Subtype | battery | lexmark | viessmann |",
              "|---|---:|---:|---:|"]
    for sub in sorted(sub_counts):
        lines.append(f"| {sub} | {by_sub_dom.get((sub,'battery'),0)} | "
                      f"{by_sub_dom.get((sub,'lexmark'),0)} | "
                      f"{by_sub_dom.get((sub,'viessmann'),0)} |")
    lines += ["", "## Validation checks", "",
              f"- required keys present: {'PASS' if n_fail==0 else 'see CSV'}",
              f"- multi-source rows have >=2 groups from >=2 sources: "
              f"{'PASS' if n_fail==0 else 'see CSV'}",
              f"- every gold value grounded in its declared source: "
              f"{'PASS' if n_fail==0 else 'see CSV'}",
              f"- no memory value leaking into a non-memory group: "
              f"{'PASS' if n_fail==0 else 'see CSV'}",
              f"- unique ids: {'PASS' if len({r['id'] for r in all_rows})==len(all_rows) else 'FAIL'}",
              ""]
    # 5 examples per subtype
    lines += ["## Examples (5 rows per subtype)", ""]
    for sub in sorted(sub_counts):
        ex = [r for r in all_rows if r["subtype"] == sub][:5]
        lines.append(f"### {sub}")
        lines.append("")
        for r in ex:
            lines.append(f"- `{r['id']}` [{r['domain']}] {r['query']}")
            lines.append(f"  - expected_groups: {json.dumps(r['expected_groups'], ensure_ascii=False)}")
            if r["memory_seed"]:
                lines.append(f"  - memory_seed: {r['memory_seed']}")
        lines.append("")
    report_md.write_text("\n".join(lines))

    print(f"[v2] total={len(all_rows)}  atomic={n_atomic}  multi={n_multi}  failures={n_fail}")
    for sub, c in sorted(sub_counts.items()):
        print(f"     {sub:32s} {c}")
    print(f"[v2] domains: {dict(sorted(dom_counts.items()))}")
    print(f"[v2] wrote: {bench_json}")
    print(f"[v2] wrote: {ids_json}")
    print(f"[v2] wrote: {report_md}")
    print(f"[v2] wrote: {valid_csv}")
    if fatal:
        print(f"[v2] VALIDATION FAILURES ({len(fatal)}):")
        for f in fatal[:15]:
            print(f"     {f}")
        return 1
    print("[v2] VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
