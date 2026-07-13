#!/usr/bin/env python3
"""Preflight the COMPASS frozen rerun without invoking any model API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {
    "artifacts/paper_split.json": 3429,
    "artifacts/release_clean_verified_v1.json": 6270,
    "artifacts/auto_compose_v2/auto_compose_arch_3000_v2.json": 3000,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def benchmark_count(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return len(data)
    return len(data.get("rows", []))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=REPO_ROOT / "release/paper_freeze_preflight_20260713/EXISTING_ARTIFACT_INVENTORY.json",
    )
    parser.add_argument(
        "--parity-report",
        type=Path,
        default=REPO_ROOT / "release/paper_freeze_preflight_20260713/BASELINE_PARITY.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper_results_20260713_frozen",
    )
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--require-api-key", action="store_true")
    args = parser.parse_args()

    blockers: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    counts = {}
    for relative, expected in EXPECTED_COUNTS.items():
        path = REPO_ROOT / relative
        actual = benchmark_count(path) if path.is_file() else None
        counts[relative] = {"expected": expected, "actual": actual}
        if actual != expected:
            blockers.append(f"{relative}: expected {expected} rows, found {actual}")
    checks["benchmark_counts"] = counts

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    protected_groups = ("benchmarks", "current_results", "calibration")
    hash_mismatches = []
    for group in protected_groups:
        for record in inventory["groups"][group]:
            path = REPO_ROOT / record["path"]
            current = sha256(path) if path.is_file() else None
            if current != record.get("sha256"):
                hash_mismatches.append(record["path"])
    checks["protected_hash_mismatches"] = hash_mismatches
    if hash_mismatches:
        blockers.append(f"protected inputs/results changed: {hash_mismatches}")

    parity = json.loads(args.parity_report.read_text(encoding="utf-8"))
    checks["baseline_parity"] = parity["status"]
    if not str(parity["status"]).startswith("PASS"):
        blockers.append(f"baseline parity status is {parity['status']}")
    if "LIMITATION" in str(parity["status"]):
        warnings.append("20 historical LINC rows cannot be rescored because archived answers were truncated")

    router = json.loads((REPO_ROOT / "artifacts/policy_router.json").read_text(encoding="utf-8"))
    router_source = str(router.get("source_eval") or "")
    checks["router_source"] = router_source
    if "/dev/" not in router_source.replace("\\", "/"):
        blockers.append(f"router was not selected from a dev artifact: {router_source}")

    calibrator = json.loads((REPO_ROOT / "artifacts/calibration_fit.json").read_text(encoding="utf-8"))
    calibrator_source = str(calibrator.get("joined") or "")
    checks["calibrator_source"] = calibrator_source
    if "/dev/" not in calibrator_source.replace("\\", "/"):
        blockers.append(f"calibrator was not fitted from a dev artifact: {calibrator_source}")

    output_exists = args.output_dir.exists()
    checks["output_dir"] = str(args.output_dir)
    checks["output_dir_exists"] = output_exists
    if output_exists:
        blockers.append(f"refusing to reuse existing output directory: {args.output_dir}")

    clean = not bool(git("status", "--porcelain=v1", "--untracked-files=all"))
    checks["git_commit"] = git("rev-parse", "HEAD")
    checks["git_clean"] = clean
    if not clean:
        message = "working tree is not clean; Phase A must be committed before tagging/running"
        (blockers if args.require_clean else warnings).append(message)

    api_key_available = bool(os.environ.get("OPENAI_API_KEY"))
    checks["openai_api_key_available"] = api_key_available
    if not api_key_available:
        message = "OPENAI_API_KEY is not set in this shell"
        (blockers if args.require_api_key else warnings).append(message)

    checks["frozen_model_snapshot"] = args.model
    if args.model != "gpt-4o-mini-2024-07-18":
        blockers.append("Phase B model must be the explicit gpt-4o-mini-2024-07-18 snapshot")

    arch_source = (REPO_ROOT / "scripts/run_arch_smoke_comparison.py").read_text(encoding="utf-8")
    checks["compositional_linc_path_present"] = '"LINC"' in arch_source
    if not checks["compositional_linc_path_present"]:
        blockers.append("compositional harness has no LINC path")

    free_bytes = shutil.disk_usage(REPO_ROOT).free
    checks["disk_free_bytes"] = free_bytes
    if free_bytes < 5 * 1024**3:
        blockers.append("less than 5 GiB free for isolated outputs")

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not blockers else "BLOCKED",
        "paid_calls_made": False,
        "checks": checks,
        "warnings": warnings,
        "blockers": blockers,
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
