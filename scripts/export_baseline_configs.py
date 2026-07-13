#!/usr/bin/env python3
"""Export evidence-backed baseline configurations for the COMPASS release."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.api.answerer_ctx import SYSTEM as COMPASS_SYSTEM  # noqa: E402
from scripts import run_baselines as baselines  # noqa: E402


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def sha256(relative: str) -> str:
    return hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()


def rendered_prompt(mode: str) -> dict[str, str]:
    system, user = baselines.PROMPT_BUILDERS[mode](
        "{question}", "{question_type}", "{domain}", "{selected_context}"
    )
    return {"system": system, "user_template": user}


def verified_config(mode: str) -> dict[str, Any]:
    return {
        "benchmark": "verified_release_6270",
        "historical_model_alias": "gpt-4o-mini",
        "historical_exact_model_snapshot": None,
        "historical_snapshot_status": (
            "UNKNOWN: archived CSVs contain token counts but no model field; "
            "the harness default and run documentation identify only the mutable alias"
        ),
        "proposed_frozen_snapshot_for_rerun": "gpt-4o-mini-2024-07-18",
        "temperature": 0.0,
        "seed": None,
        "seed_status": "not set by the harness",
        "max_context_chars": 12000,
        "max_output_tokens": baselines.MODE_MAX_OUTPUT[mode],
        "retry_policy": {
            "max_attempts": 8,
            "retryable_errors": sorted(
                [
                    "APIConnectionError",
                    "APITimeoutError",
                    "APIStatusError",
                    "InternalServerError",
                    "RateLimitError",
                ]
            ),
            "backoff_seconds": "min(45, 2 * attempt_number)",
            "request_timeout_seconds": 60,
        },
        "evidence_selection": {
            "method": "deterministic lexical/product retrieval from seed documents, memory facts, and ontology snippets",
            "uses_gold_for_selection": False,
            "oracle_context": False,
            "limits": {
                "memory_hits": 10,
                "document_chunks": 12,
                "ontology_chars": 4500,
                "total_chars": 12000,
            },
            "correction": (
                "The paper must not call this oracle context: gold annotations are not "
                "used to select evidence and the supplied context is retrieval-ranked and capped."
            ),
        },
        "prompt": rendered_prompt(mode),
        "answer_processing": {
            "GPT4O_LONGCTX": "raw stripped response",
            "LINC": "extract text after the final 'Final:' marker when present",
            "LOGIC_LM": "extract text after the final 'Answer:' marker when present",
        }[mode],
        "validator": {
            "source": "scripts/run_baselines.py:normalize_text/score_answer",
            "rules": "type-aware deterministic polarity or normalized substring/compact match; no LLM judge",
        },
        "source": "scripts/run_baselines.py",
        "source_sha256": sha256("scripts/run_baselines.py"),
    }


def compositional_config(mode: str) -> dict[str, Any]:
    if mode == "GPT4O_LONGCTX":
        system = (
            "You are a product-passport reasoning assistant. Answer using ONLY "
            "the provided context. Return a concise answer containing all requested "
            "values. If insufficient, say INSUFFICIENT EVIDENCE."
        )
        output_format = "raw stripped response"
    elif mode == "LINC":
        system = (
            "You are a neuro-symbolic reasoning agent in the style of LINC. "
            "Extract premises from the supplied context, express the question "
            "as a goal, reason step by step, and return one final answer containing "
            "all requested values. Use only the supplied context."
        )
        output_format = "extract text after the final 'Final:' marker when present"
    else:
        system = (
            "You are Logic-LM. Classify the question, extract relevant premises "
            "from the provided context, reason over them, and return a single final "
            "answer containing all requested values. Use only context."
        )
        output_format = "extract text after the final 'Answer:' marker when present"
    return {
        "benchmark": "compositional_3000",
        "historical_run_status": (
            "not run; added for the Phase B frozen comparison"
            if mode == "LINC"
            else "archived run exists"
        ),
        "historical_model_alias": None if mode == "LINC" else "gpt-4o-mini",
        "historical_exact_model_snapshot": None,
        "proposed_frozen_snapshot_for_rerun": "gpt-4o-mini-2024-07-18",
        "temperature": 0.0,
        "seed": None,
        "max_context_chars": None,
        "max_context_status": "no explicit total character cap in the compositional context builder",
        "max_output_tokens": 220,
        "retry_policy": {"outer_row_attempts": 4, "linear_wait_multiplier_seconds": 1.0},
        "evidence_selection": {
            "method": "benchmark-declared document, memory, and symbolic sections",
            "uses_gold_for_selection": True,
            "oracle_context": True,
            "qualification": (
                "the constructed benchmark supplies declared source-specific evidence, including "
                "symbolic values from gold_evidence; this applies only to the compositional study"
            ),
        },
        "prompt": {
            "system": system,
            "user_template": "Domain: {domain}\nContext:\n{context}\n\nQuestion: {question}",
        },
        "answer_processing": output_format,
        "validator": {
            "source": "scripts/run_arch_smoke_comparison.py:score_answer",
            "rules": "every expected value group must match after compact normalization; no LLM judge",
        },
        "source": "scripts/run_arch_smoke_comparison.py",
        "source_sha256": sha256("scripts/run_arch_smoke_comparison.py"),
    }


def build() -> dict[str, Any]:
    systems: list[dict[str, Any]] = []
    labels = {
        "GPT4O_LONGCTX": (
            "GPT-4o-mini + retrieved context",
            "prompted retrieval-context baseline; not an oracle long-context baseline",
        ),
        "LINC": (
            "LINC-style prompted baseline",
            "prompted approximation; it does not execute the published LINC theorem-prover pipeline",
        ),
        "LOGIC_LM": (
            "Logic-LM-style prompted baseline",
            "prompted approximation; it does not execute the full published Logic-LM pipeline",
        ),
    }
    for mode in ("GPT4O_LONGCTX", "LINC", "LOGIC_LM"):
        label, fidelity = labels[mode]
        systems.append(
            {
                "id": mode,
                "paper_label_recommended": label,
                "implementation_fidelity": fidelity,
                "evaluations": [verified_config(mode), compositional_config(mode)],
            }
        )

    systems.append(
        {
            "id": "COMPASS",
            "historical_mode_id": "ADAPTIVERAG (verified release), AUTO_COMPOSE (compositional benchmark)",
            "paper_label_recommended": "COMPASS",
            "implementation_fidelity": "project system",
            "historical_model_alias": "gpt-4o-mini",
            "historical_exact_model_snapshot": None,
            "historical_snapshot_status": (
                "UNKNOWN for the verified-release CSV; compositional traces record the mutable gpt-4o-mini alias"
            ),
            "proposed_frozen_snapshot_for_rerun": "gpt-4o-mini-2024-07-18",
            "system_prompt": COMPASS_SYSTEM,
            "user_template": "Question: {question}\n\nContext:\n{formatted_context}\n\nAnswer:",
            "temperature": None,
            "temperature_status": (
                "API-path dependent in current code: Responses API and modern OpenAI chat path do not set it; "
                "OpenRouter/legacy fallbacks set 0.2"
            ),
            "seed": None,
            "max_context_chars": 12000,
            "max_passages": 6,
            "max_chars_per_passage": 2000,
            "max_output_tokens": 256,
            "retry_policy": {
                "sdk_max_retries_default": 0,
                "architecture_harness_outer_row_retries": 4,
            },
            "answer_processing": (
                "citation enforcement plus deterministic field, memory, and symbolic fallbacks; "
                "final answer provenance is recorded in answer_trace"
            ),
            "validators": {
                "verified_release": "same stored success/expected_contains schema aggregated for every system",
                "compositional": "all expected value groups must appear after compact normalization",
                "no_llm_judge": True,
            },
            "sources": [
                {
                    "path": "backend/api/answerer_ctx.py",
                    "sha256": sha256("backend/api/answerer_ctx.py"),
                },
                {
                    "path": "backend/api/solve_auto.py",
                    "sha256": sha256("backend/api/solve_auto.py"),
                },
                {
                    "path": "scripts/run_arch_smoke_comparison.py",
                    "sha256": sha256("scripts/run_arch_smoke_comparison.py"),
                },
            ],
        }
    )
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "status": (
            "Historical configuration audit. Proposed snapshot fields are plans for Phase B, "
            "not claims about completed historical runs."
        ),
        "systems": systems,
        "known_gaps": [
            "Exact historical model snapshots cannot be recovered from the verified-release CSVs.",
            "The compositional LINC-style path is newly added and has no historical result; it must be run in Phase B.",
            "The verified and compositional harnesses use different prompt templates.",
            "The paper's oracle-context label is not supported by the implemented evidence selector.",
        ],
    }


def render_markdown(config: dict[str, Any]) -> str:
    lines = [
        "# Baseline configurations",
        "",
        "This file renders `baseline_configs.json`. Unknown historical values are left unknown; proposed frozen values have not yet been run.",
        "",
        "| System ID | Recommended paper label | Historical snapshot | Context | Fidelity |",
        "|---|---|---|---|---|",
    ]
    for system in config["systems"]:
        if system["id"] == "COMPASS":
            snapshot = system["historical_exact_model_snapshot"] or "unknown (alias only)"
            context = f"{system['max_passages']} passages / {system['max_context_chars']} chars"
        else:
            ev = system["evaluations"][0]
            snapshot = ev["historical_exact_model_snapshot"] or "unknown (alias only)"
            context = f"retrieved, {ev['max_context_chars']} chars; not oracle"
        lines.append(
            f"| `{system['id']}` | {system['paper_label_recommended']} | {snapshot} | {context} | {system['implementation_fidelity']} |"
        )
    lines.extend(["", "## Required paper corrections", ""])
    lines.extend(
        [
            "- Replace “oracle context” with “retrieval-selected context” unless a genuinely gold-selected oracle is implemented and rerun.",
            "- Refer to LINC and Logic-LM as prompted, style-based approximations; the harness does not run their published solver pipelines.",
            "- Do not claim exact snapshots were exported for historical runs. The exact snapshot is unknown and the archived CSVs record no model field.",
            "- Phase B must use the explicit `gpt-4o-mini-2024-07-18` snapshot and record it per row/manifest.",
        ]
    )
    lines.extend(["", "## Known gaps", ""])
    lines.extend(f"- {gap}" for gap in config["known_gaps"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=REPO_ROOT / "release" / "baseline_configs.json")
    parser.add_argument("--markdown", type=Path, default=REPO_ROOT / "release" / "BASELINES.md")
    args = parser.parse_args()
    config = build()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(config), encoding="utf-8")
    print(args.json.relative_to(REPO_ROOT))
    print(args.markdown.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
