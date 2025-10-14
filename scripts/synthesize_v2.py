#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic QA for each domain:
- templated recall/open
- light paraphrasing
- contrastive negatives
Outputs: tests/<domain>/tests_synth.jsonl
"""

from __future__ import annotations
import json, random, hashlib
from pathlib import Path
from typing import Dict, List

# Seed templates per domain (extend freely)
TEMPLATES: Dict[str, List[Dict]] = {
    "battery": [
        {"type": "recall", "q": "What is the official term for '{alias}'?", "a_from": "alias_target"},
        {"type": "open",   "q": "Quote one sentence that supports '{claim}'.", "a_from": "support"},
        {"type": "logic",  "q": "Is the statement true: {claim}?", "a_from": "yesno"},
    ],
    "lexmark": [
        {"type": "recall", "q": "Which toner model is compatible with {device}?", "a_from": "device_toner"},
    ],
}

# Domain facts used by templates
FACTS = {
    "battery": {
        "alias_map": {
            "EPR": "Extended Producer Responsibility",
            "DoC": "Declaration of Conformity",
            "COP": "coefficient of performance",
        },
        "supports": [
            ("Producers must finance take-back schemes.", "take-back schemes are mandated"),
            ("DoC is required for EU market access.", "Declaration of Conformity is required"),
        ],
        "yesno": [
            ("A cell with 60% recycled content meets the target.", "yes"),
            ("COP refers to energy efficiency ratio.", "yes"),
        ],
    },
    "lexmark": {
        "device_toner": [
            ("MS821", "54G0H00"),
            ("MX722", "62D5H00"),
        ]
    },
}

PARAPHRASES = [
    ("official term", "canonical name"),
    ("Which", "What"),
    ("Is the statement true", "Does the following hold"),
    ("supports", "evidences"),
    ("required", "mandatory"),
]

def _hash_id(domain: str, payload: str) -> str:
    return hashlib.sha1(f"{domain}|{payload}".encode()).hexdigest()[:10]

def _emit(line: dict, fh):
    fh.write(json.dumps(line, ensure_ascii=False) + "\n")

def synth_for_domain(domain: str, n_per_template: int = 30) -> List[dict]:
    out: List[dict] = []
    facts = FACTS.get(domain, {})
    for tpl in TEMPLATES.get(domain, []):
        for _ in range(n_per_template):
            typ = tpl["type"]
            q = tpl["q"]
            # Fill slots
            if tpl["a_from"] == "alias_target":
                alias, target = random.choice(list((facts.get("alias_map") or {"EPR":"Extended Producer Responsibility"}).items()))
                qf = q.replace("{alias}", alias)
                a = target
            elif tpl["a_from"] == "support":
                sent, gloss = random.choice(facts.get("supports") or [("DoC is required", "DoC required")])
                qf = q.replace("{claim}", gloss)
                a = sent
            elif tpl["a_from"] == "yesno":
                claim, yn = random.choice(facts.get("yesno") or [("EPR is mandatory", "yes")])
                qf = q.replace("{claim}", claim)
                a = yn
            elif tpl["a_from"] == "device_toner":
                device, toner = random.choice(facts.get("device_toner") or [("MS821","54G0H00")])
                qf = q.replace("{device}", device)
                a = toner
            else:
                qf, a = q, "unknown"

            # light paraphrase
            for s, t in random.sample(PARAPHRASES, k=random.randint(0, min(3,len(PARAPHRASES)))):
                qf = qf.replace(s, t)

            rid = _hash_id(domain, f"{typ}|{qf}|{a}")
            out.append({
                "id": rid,
                "type": typ,
                "domain": domain,
                "question": qf,
                "query": qf,
                "answer": a,
                "expected_contains": a,
                "meta": {"synthetic": True}
            })

            # contrastive negative for recall/open: same question with wrong answer
            if typ in ("recall","open"):
                wrong = a if isinstance(a, str) else str(a)
                # cheap corruption: shuffle or add prefix
                wrong_cf = f"NOT {wrong}"
                rid2 = _hash_id(domain, f"neg|{typ}|{qf}|{wrong_cf}")
                out.append({
                    "id": rid2,
                    "type": typ,
                    "domain": domain,
                    "question": qf,
                    "query": qf,
                    "answer": wrong,  # gold stays true answer
                    "expected_contains": a,  # success only if correct span is found
                    "meta": {"synthetic": True, "contrastive": True}
                })
    return out

def main():
    root = Path("tests")
    for dom in TEMPLATES.keys():
        if not (root / dom / "tests.jsonl").exists():
            continue
        rows = synth_for_domain(dom, n_per_template=15)
        outp = root / dom / "tests_synth.jsonl"
        with outp.open("w", encoding="utf-8") as f:
            for r in rows:
                _emit(r, f)
        print(f"Wrote {outp} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
