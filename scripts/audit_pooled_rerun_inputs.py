#!/usr/bin/env python3
"""Audit whether the historical 3,429-row pooled suite can be rescored."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["rows"] if isinstance(value, dict) and "rows" in value else value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--gold-subset", type=Path, required=True)
    parser.add_argument("--release-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split = rows(args.split)
    gold_subset = rows(args.gold_subset)
    release = rows(args.release_benchmark)
    release_by_key = {(r["id"], r["domain"], r.get("query")): r for r in release}
    exact_release_matches = sum(
        bool(release_by_key.get((r["id"], r["domain"], r.get("query")), {}).get("expected_contains"))
        for r in split
    )
    split_keys = {(r["id"], r["domain"], r.get("query")) for r in split}
    subset_keys = {(r["id"], r["domain"], r.get("query")) for r in gold_subset}
    report = {
        "schema_version": 1,
        "status": "BLOCKED_MISSING_GOLD" if len(gold_subset) != len(split) else "READY",
        "split": {"path": str(args.split), "rows": len(split), "sha256": sha256(args.split)},
        "gold_subset": {
            "path": str(args.gold_subset),
            "rows": len(gold_subset),
            "keys_with_expected_contains": sum(bool(r.get("expected_contains")) for r in gold_subset),
            "keys_present_in_split": len(split_keys & subset_keys),
            "sha256": sha256(args.gold_subset),
        },
        "exact_query_matches_with_release_gold": exact_release_matches,
        "missing_gold_rows": len(split) - len(gold_subset),
        "safe_to_rescore_all_3429": len(gold_subset) == len(split) and subset_keys == split_keys,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["safe_to_rescore_all_3429"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
