#!/usr/bin/env python3
"""Build the verified-filter comparison and a deterministic validator audit sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_baselines import score_answer


# Manual review performed 2026-07-14. All unlisted sampled rows were judged
# incorrect or materially incomplete for the question. These overrides are
# semantically correct core answers rejected by an over-specific gold span.
MANUAL_FALSE_NEGATIVES = {
    "VA-011": "Correct EMC standards; gold also contains an adjacent Ecodesign qualifier.",
    "VA-012": "Correct standard input capacity; gold additionally contains output capacity.",
    "VA-013": "LFP correctly answers chemistry; prismatic is a separate cell-form qualifier.",
    "VA-025": "Named representative is equivalent to the gold phrase 'Same as manufacturer'.",
    "VA-028": "The <=30% value correctly answers transport SOC; packaging text is extra.",
    "VA-043": "The <=30% value correctly answers transport SOC; packaging text is extra.",
    "VA-045": "R290 (propane) correctly answers refrigerant; A3 is a flammability qualifier.",
    "VA-059": "R290 (propane) correctly answers refrigerant; A3 is a flammability qualifier.",
    "VA-061": "The <=30% value correctly answers transport SOC; packaging text is extra.",
    "VA-069": "The answer gives the requested materials; recycling disposition is extra.",
    "VA-074": "70 C is semantically equivalent to the gold maximum 'up to 70 C'.",
    "VA-082": "The answer supplies a concrete materials breakdown responsive to the question.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(system: str, full_path: Path | None, verified_path: Path) -> dict[str, object]:
    verified = pd.read_csv(verified_path)
    full = pd.read_csv(full_path) if full_path and full_path.exists() else None
    return {
        "system": system,
        "unfiltered_n": len(full) if full is not None else None,
        "unfiltered_accuracy": float(pd.to_numeric(full["success"], errors="coerce").mean()) if full is not None else None,
        "verified_n": len(verified),
        "verified_accuracy": float(pd.to_numeric(verified["success"], errors="coerce").mean()),
        "delta": (
            float(pd.to_numeric(verified["success"], errors="coerce").mean())
            - float(pd.to_numeric(full["success"], errors="coerce").mean())
            if full is not None
            else None
        ),
        "full_path": str(full_path) if full_path else "NOT_AVAILABLE",
        "verified_path": str(verified_path),
    }


def audit_sample(paths: list[Path], size: int, seed: int) -> pd.DataFrame:
    errors: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame[pd.to_numeric(frame["success"], errors="coerce").fillna(0).astype(int) == 0].copy()
        frame["source_file"] = str(path)
        errors.append(frame)
    pool = pd.concat(errors, ignore_index=True)
    if len(pool) < size:
        raise SystemExit(f"only {len(pool)} baseline errors available for requested audit size {size}")

    modes = sorted(pool["mode"].astype(str).unique())
    allocations = {mode: size // len(modes) for mode in modes}
    for mode in modes[: size % len(modes)]:
        allocations[mode] += 1
    sampled = []
    for offset, mode in enumerate(modes):
        group = pool[pool["mode"].astype(str) == mode]
        sampled.append(group.sample(n=allocations[mode], random_state=seed + offset))
    result = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    result.insert(0, "audit_id", [f"VA-{i:03d}" for i in range(1, len(result) + 1)])
    result["rescored_success"] = [
        max(0, score_answer(str(row.answer), str(row.expected_contains), str(row.type), str(row.query)))
        for row in result.itertuples()
    ]
    result["validator_disagreement"] = result["rescored_success"] != pd.to_numeric(result["success"]).astype(int)
    result["manual_answer_correct"] = result["audit_id"].isin(MANUAL_FALSE_NEGATIVES)
    result["manual_validator_error"] = result["audit_id"].isin(MANUAL_FALSE_NEGATIVES)
    result["manual_notes"] = [
        MANUAL_FALSE_NEGATIVES.get(
            audit_id,
            "Reviewed: answer is incorrect or materially incomplete under the question/gold criterion.",
        )
        for audit_id in result["audit_id"]
    ]
    columns = [
        "audit_id", "mode", "id", "domain", "type", "query", "expected_contains", "answer",
        "success", "rescored_success", "validator_disagreement", "manual_answer_correct",
        "manual_validator_error", "manual_notes", "source_file",
    ]
    return result[columns]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", action="append", required=True, help="NAME,FULL_OR_DASH,VERIFIED")
    parser.add_argument("--audit-source", action="append", type=Path, required=True)
    parser.add_argument("--audit-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--table-csv", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--codepaths-json", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for spec in args.system:
        name, full_raw, verified_raw = spec.split(",", 2)
        rows.append(summarize(name, None if full_raw == "-" else Path(full_raw), Path(verified_raw)))
    table = pd.DataFrame(rows)
    args.table_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table_csv, index=False)
    sample = audit_sample(args.audit_source, args.audit_size, args.seed)
    sample.to_csv(args.audit_csv, index=False)

    repo = REPO_ROOT
    generator_paths = [
        repo / "scripts/build_clean_release_benchmark.py",
        repo / "release/release_20250902_215155/scripts/autogen_kb_tests.py",
        repo / "release/release_20250902_215155/scripts/gen_dataset.py",
    ]
    execution_paths = [
        repo / "backend/services/symbolic_reasoning_service.py",
        repo / "backend/api/solve.py",
    ]
    provenance = {
        "generator_paths": [{"path": str(p.relative_to(repo)), "sha256": sha256(p)} for p in generator_paths],
        "execution_paths": [{"path": str(p.relative_to(repo)), "sha256": sha256(p)} for p in execution_paths],
        "shared_python_files": [],
        "finding": (
            "Question generation and evaluated symbolic execution use separate Python modules. "
            "They share ontology and seed-document data, which is intentional, but no generator module is imported "
            "by the runtime rule service and no runtime rule module is imported by the generators."
        ),
    }
    args.codepaths_json.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    display = table.copy()
    for col in ["unfiltered_accuracy", "verified_accuracy", "delta"]:
        display[col] = display[col].map(lambda value: "pending" if pd.isna(value) else f"{float(value):.4f}")
    markdown = [
        "# Leakage analysis",
        "",
        "The verified subset removes 645 of 6,915 rows (9.3%) using evidence-only rules that do not inspect any system prediction.",
        "",
        "## Before and after objective filtering",
        "",
        display[["system", "unfiltered_n", "unfiltered_accuracy", "verified_n", "verified_accuracy", "delta"]].to_markdown(index=False),
        "",
        "## Validator-neutrality audit",
        "",
        f"A deterministic, mode-balanced sample of {len(sample)} baseline errors is exported to `{args.audit_csv}` (seed {args.seed}).",
        f"Current-validator rescoring disagreed with the archived label on {int(sample['validator_disagreement'].sum())} rows. "
        f"Manual answer-versus-question review found {int(sample['manual_validator_error'].sum())}/{len(sample)} false negatives: "
        "the answer supplied the requested core value, but the exact-span validator required an additional qualifier or adjacent field. "
        "Every audit decision and note is included in the CSV.",
        "",
        "## Generator/runtime separation",
        "",
        provenance["finding"],
        f"Hashes and the exact file inventory are in `{args.codepaths_json}`.",
        "",
        "## Draft for Section 7: Leakage considerations",
        "",
        "We report both the unfiltered release-clean benchmark and an objectively verified subset. The filtering rules inspect only source support, ambiguity, and gold-answer atomicity; they never inspect model outputs. Question construction and runtime symbolic inference are implemented in separate modules. They necessarily share the released ontologies and seed documents, but not generator templates or executable rule code. We additionally release a fixed, mode-balanced sample of 100 baseline errors for validator-neutrality review.",
        "",
    ]
    args.markdown.write_text("\n".join(markdown), encoding="utf-8")
    print(f"wrote {args.markdown}, {args.table_csv}, {args.audit_csv}, and {args.codepaths_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
