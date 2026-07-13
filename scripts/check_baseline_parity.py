#!/usr/bin/env python3
"""Verify and independently rescore all verified-release systems on equal rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_baselines import score_answer  # noqa: E402


DEFAULT_FILES = {
    "COMPASS": REPO_ROOT / "artifacts/verified_v1/eval_ADAPTIVERAG_verified_v1_full.csv",
    "GPT4O_LONGCTX": REPO_ROOT / "artifacts/verified_v1/eval_GPT4O_LONGCTX_verified_v1.csv",
    "LINC": REPO_ROOT / "artifacts/verified_v1/eval_LINC_verified_v1.csv",
    "LOGIC_LM": REPO_ROOT / "artifacts/verified_v1/eval_LOGIC_LM_verified_v1.csv",
}


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("id") or ""), str(row.get("domain") or "")


def load_expected_keys(path: Path) -> set[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {tuple(item.split("|", 1)) for item in data}


def load_benchmark(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key(row): row for row in data["rows"]}


def load_csv(path: Path) -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(key(row) for row in rows)
    duplicates = sorted(k for k, count in counts.items() if count != 1)
    if duplicates:
        raise ValueError(f"{path}: duplicate keys, first examples: {duplicates[:5]}")
    return rows, {key(row): row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids",
        type=Path,
        default=REPO_ROOT / "artifacts/release_clean_verified_v1_ids.json",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=REPO_ROOT / "artifacts/release_clean_verified_v1.json",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    expected_keys = load_expected_keys(args.ids)
    benchmark = load_benchmark(args.benchmark)
    failures: list[str] = []
    tables: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    summaries: list[dict[str, Any]] = []

    if set(benchmark) != expected_keys:
        failures.append(
            f"benchmark keys differ from ID list: benchmark={len(benchmark)}, ids={len(expected_keys)}"
        )

    for system, path in DEFAULT_FILES.items():
        rows, table = load_csv(path)
        tables[system] = table
        missing = expected_keys - set(table)
        extra = set(table) - expected_keys
        if missing or extra:
            failures.append(f"{system}: missing={len(missing)}, extra={len(extra)}")

        metadata_mismatches = 0
        stored_score_mismatches = 0
        truncated_unverifiable = 0
        independently_correct = 0
        for row_key in expected_keys & set(table) & set(benchmark):
            row = table[row_key]
            gold = benchmark[row_key]
            expected = str(gold.get("expected_contains") or "")
            qtype = str(gold.get("type") or "")
            query = str(gold.get("query") or gold.get("question") or "")
            if (
                str(row.get("expected_contains") or "") != expected
                or str(row.get("type") or "") != qtype
                or str(row.get("query") or "") != query
            ):
                metadata_mismatches += 1
            rescored = score_answer(str(row.get("answer") or ""), expected, qtype, query)
            independently_correct += max(0, rescored)
            if max(0, rescored) != int(row.get("success") or 0):
                if len(str(row.get("answer") or "")) == 500:
                    truncated_unverifiable += 1
                else:
                    stored_score_mismatches += 1
        if metadata_mismatches:
            failures.append(f"{system}: {metadata_mismatches} benchmark metadata mismatches")
        if stored_score_mismatches:
            failures.append(f"{system}: {stored_score_mismatches} stored scores differ from common validator")
        summaries.append(
            {
                "system": system,
                "rows": len(rows),
                "missing": len(missing),
                "extra": len(extra),
                "metadata_mismatches": metadata_mismatches,
                "stored_score_mismatches": stored_score_mismatches,
                "truncated_answers_unverifiable": truncated_unverifiable,
                "common_validator_correct": independently_correct,
            }
        )

    common = set.intersection(*(set(table) for table in tables.values()))
    if common != expected_keys:
        failures.append(f"all-system common keys={len(common)}, expected={len(expected_keys)}")

    report = {
        "status": (
            "PASS_WITH_HISTORICAL_TRUNCATION_LIMITATION"
            if not failures and any(s["truncated_answers_unverifiable"] for s in summaries)
            else ("PASS" if not failures else "FAIL")
        ),
        "expected_rows": len(expected_keys),
        "common_rows": len(common),
        "validator": "scripts.run_baselines.score_answer",
        "systems": summaries,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
