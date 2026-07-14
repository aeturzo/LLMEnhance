#!/usr/bin/env python3
"""Build a non-destructive SHA-256 manifest for COMPASS paper evidence.

The manifest records the exact source, benchmark, and result files that must be
protected before a frozen rerun. It never modifies an input and refuses to
replace an existing output unless --force is supplied.
"""

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

GROUPS = {
    "benchmarks": [
        "artifacts/paper_split.json",
        "artifacts/paper_split_with_gold.json",
        "artifacts/release_clean_verified_v1.json",
        "artifacts/release_clean_verified_v1_ids.json",
        "artifacts/auto_compose_v2/auto_compose_arch_3000_v2.json",
    ],
    "current_results": [
        "artifacts/eval_joined_pooled_20260202_192420.csv",
        "artifacts/verified_v1/eval_ADAPTIVERAG_verified_v1_full.csv",
        "artifacts/verified_v1/eval_GPT4O_LONGCTX_verified_v1.csv",
        "artifacts/verified_v1/eval_LINC_verified_v1.csv",
        "artifacts/verified_v1/eval_LOGIC_LM_verified_v1.csv",
        "artifacts/auto_compose_v2/final_arch_3000_all3.csv",
        "experiments/openfoodfacts_20260506_165710/system_eval_20260506_174429_rescored/off_system_eval_gpt4o_mini.csv",
        "artifacts/phase_bc_20260714_frozen/verified/COMPASS.csv",
        "artifacts/phase_bc_20260714_frozen/verified/GPT4O_LONGCTX.csv",
        "artifacts/phase_bc_20260714_frozen/verified/LINC.csv",
        "artifacts/phase_bc_20260714_frozen/verified/LOGIC_LM.csv",
        "artifacts/phase_bc_20260714_frozen/compositional/all_systems.csv",
    ],
    "ontology_and_rules": [
        "backend/ontologies/dpp_ontology.ttl",
        "backend/ontologies/carbon_ontology.ttl",
        "backend/services/symbolic_reasoning_service.py",
        "backend/services/carbon_ontology_service.py",
        "backend/config/domains/battery.yml",
        "backend/config/domains/lexmark.yml",
        "backend/config/domains/viessmann.yml",
    ],
    "prompts_and_routing": [
        "backend/api/answerer_ctx.py",
        "backend/api/solve.py",
        "backend/api/solve_auto.py",
        "backend/services/policy_router.py",
        "scripts/run_baselines.py",
        "scripts/run_arch_smoke_comparison.py",
        "scripts/run_external_baselines_batch.py",
    ],
    "validators_and_exporters": [
        "scripts/filter_eval_to_verified_subset.py",
        "scripts/aggregate_verified_v1.py",
        "scripts/aggregate_baselines.py",
        "scripts/export_tables.py",
        "scripts/run_paper_pipeline.py",
        "backend/eval/mcnemar.py",
        "backend/eval/stats_polish.py",
        "scripts/analyze_leakage.py",
        "scripts/audit_pooled_rerun_inputs.py",
        "scripts/calibration_diagnostics.py",
        "scripts/plot_exact_risk_coverage.py",
        "scripts/package_frozen_results.py",
    ],
    "calibration": [
        "artifacts/calibration_fit.json",
        "artifacts/calibration_meta.json",
        "artifacts/selective/selective_calibrated.csv",
        "artifacts/paper_handoff_20260603/off_calibration_bins.csv",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def file_record(relative: str) -> dict[str, Any]:
    path = REPO_ROOT / relative
    if not path.is_file():
        return {"path": relative, "status": "missing"}
    stat = path.stat()
    return {
        "path": relative,
        "status": "present",
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
        "git_tracked": bool(git("ls-files", "--error-unmatch", relative)) if _is_tracked(relative) else False,
    }


def _is_tracked(relative: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if output.exists() and not args.force:
        parser.error(f"output already exists: {output}; use --force to replace it")

    status = git("status", "--porcelain=v1", "--untracked-files=no")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "root": str(REPO_ROOT),
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "tracked_worktree_clean": not bool(status),
            "tracked_status": status.splitlines(),
        },
        "groups": {
            group: [file_record(relative) for relative in relatives]
            for group, relatives in GROUPS.items()
        },
    }
    records = [record for records in manifest["groups"].values() for record in records]
    manifest["summary"] = {
        "files_declared": len(records),
        "files_present": sum(record["status"] == "present" for record in records),
        "files_missing": sum(record["status"] == "missing" for record in records),
        "present_bytes": sum(record.get("size_bytes", 0) for record in records),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output.relative_to(REPO_ROOT)}")
    print(json.dumps(manifest["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
