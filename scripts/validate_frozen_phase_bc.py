#!/usr/bin/env python3
"""Validate frozen Phase B result parity and compute paired McNemar tests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scipy.stats import binomtest


VERIFIED_MODES = ("COMPASS", "GPT4O_LONGCTX", "LINC", "LOGIC_LM")
COMPOSITIONAL_MODES = ("AUTO_COMPOSE", "GPT4O_LONGCTX", "LINC", "LOGIC_LM")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_score(row: dict[str, str]) -> int:
    raw = row.get("success") or row.get("correct")
    value = int(float(raw or 0))
    if value not in (0, 1):
        raise ValueError(f"non-binary score for {row.get('id')}: {raw!r}")
    return value


def keyed(rows: list[dict[str, str]]) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["id"], row.get("domain", ""))
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = binary_score(row)
    return result


def paired_record(benchmark: str, reference: str, comparison: str,
                  a: dict[tuple[str, str], int], b: dict[tuple[str, str], int]) -> dict[str, object]:
    if set(a) != set(b):
        raise ValueError(f"key mismatch: {benchmark} {reference} vs {comparison}")
    reference_only = sum(a[k] == 1 and b[k] == 0 for k in a)
    comparison_only = sum(a[k] == 0 and b[k] == 1 for k in a)
    discordant = reference_only + comparison_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(reference_only, discordant, 0.5, alternative="two-sided").pvalue
    )
    return {
        "benchmark": benchmark,
        "reference": reference,
        "comparison": comparison,
        "n": len(a),
        "reference_correct": sum(a.values()),
        "comparison_correct": sum(b.values()),
        "reference_accuracy": sum(a.values()) / len(a),
        "comparison_accuracy": sum(b.values()) / len(b),
        "reference_only": reference_only,
        "comparison_only": comparison_only,
        "discordant": discordant,
        "mcnemar_exact_two_sided_p": p_value,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    issues: list[str] = []
    verified: dict[str, list[dict[str, str]]] = {}
    for mode in VERIFIED_MODES:
        rows = load_rows(args.frozen_results / "verified" / f"{mode}.csv")
        verified[mode] = rows
        if len(rows) != 6270:
            issues.append(f"verified {mode}: expected 6270 rows, got {len(rows)}")

    compositional_rows = load_rows(args.frozen_results / "compositional" / "all_systems.csv")
    compositional = {
        mode: [row for row in compositional_rows if row.get("mode") == mode]
        for mode in COMPOSITIONAL_MODES
    }
    for mode, rows in compositional.items():
        if len(rows) != 3000:
            issues.append(f"compositional {mode}: expected 3000 rows, got {len(rows)}")

    try:
        verified_keyed = {mode: keyed(rows) for mode, rows in verified.items()}
        compositional_keyed = {mode: keyed(rows) for mode, rows in compositional.items()}
    except ValueError as exc:
        issues.append(str(exc))
        verified_keyed = {}
        compositional_keyed = {}

    expected_model = "gpt-4o-mini-2024-07-18"
    for mode in ("GPT4O_LONGCTX", "LINC", "LOGIC_LM"):
        models = {row.get("model", "") for row in verified[mode]}
        if models != {expected_model}:
            issues.append(f"verified {mode}: resolved models {sorted(models)}")
    for mode, rows in compositional.items():
        models = {row.get("model", "") for row in rows}
        if models != {expected_model}:
            issues.append(f"compositional {mode}: resolved models {sorted(models)}")

    paired: list[dict[str, object]] = []
    if verified_keyed and compositional_keyed:
        for mode in VERIFIED_MODES[1:]:
            paired.append(paired_record(
                "verified_release_6270", "COMPASS", mode,
                verified_keyed["COMPASS"], verified_keyed[mode],
            ))
        for mode in COMPOSITIONAL_MODES[1:]:
            paired.append(paired_record(
                "compositional_3000", "AUTO_COMPOSE", mode,
                compositional_keyed["AUTO_COMPOSE"], compositional_keyed[mode],
            ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "paired_mcnemar.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]) if paired else ["benchmark"])
        writer.writeheader()
        writer.writerows(paired)

    report = {
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "verified_rows_by_mode": {mode: len(rows) for mode, rows in verified.items()},
        "compositional_rows_by_mode": {mode: len(rows) for mode, rows in compositional.items()},
        "resolved_model_required": expected_model,
        "paired_tests": paired,
    }
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Frozen Phase B validation",
        "",
        f"Status: **{report['status'].upper()}**.",
        "",
        f"All evaluated LLM rows must resolve to `{expected_model}`.",
        "",
        "| Benchmark | Reference | Comparison | n | Reference only | Comparison only | Exact p |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['benchmark']} | {row['reference']} | {row['comparison']} | {row['n']} | "
            f"{row['reference_only']} | {row['comparison_only']} | {row['mcnemar_exact_two_sided_p']:.3g} |"
        )
    if issues:
        lines.extend(["", "## Issues", "", *[f"- {issue}" for issue in issues]])
    (args.output_dir / "VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    package_root = args.output_dir.parent
    manifest_path = package_root / "PACKAGE_MANIFEST.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    package_files = sorted(
        path for path in package_root.rglob("*")
        if path.is_file() and path != manifest_path
    )
    package_manifest = {
        "schema_version": 2,
        "freeze_tag": previous.get("freeze_tag", "unknown"),
        "validation_status": report["status"],
        "files": [
            {
                "path": str(path.relative_to(package_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in package_files
        ],
    }
    manifest_path.write_text(json.dumps(package_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
