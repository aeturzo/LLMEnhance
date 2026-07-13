#!/usr/bin/env python3
"""
Filter already-completed external-baseline CSVs to the verified subset.

The three external baselines (GPT4O_LONGCTX, LINC, LOGIC_LM) were already run
against the full 6,915-row release-clean benchmark. This script restricts each
baseline CSV to the verified `(id, domain)` keys produced by
scripts/build_release_clean_verified_subset.py — NO model rerun is needed.

Inputs
------
  artifacts/release_clean_verified_v1_ids.json
  artifacts/eval_GPT4O_LONGCTX_clean_full.csv
  artifacts/eval_LINC_clean_full.csv
  artifacts/eval_LOGIC_LM_clean_full.csv

Outputs
-------
  artifacts/verified_v1/eval_GPT4O_LONGCTX_verified_v1.csv
  artifacts/verified_v1/eval_LINC_verified_v1.csv
  artifacts/verified_v1/eval_LOGIC_LM_verified_v1.csv
  artifacts/verified_v1/filter_report.md

Validation (fails loud if violated)
-----------------------------------
  * every verified (id, domain) must be present in each baseline CSV
  * row counts must match across all filtered files
  * no API-error rows in the filtered output

Usage
-----
  python scripts/filter_eval_to_verified_subset.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDS = REPO_ROOT / "artifacts" / "release_clean_verified_v1_ids.json"
DEFAULT_BASELINES = [
    REPO_ROOT / "artifacts" / "eval_GPT4O_LONGCTX_clean_full.csv",
    REPO_ROOT / "artifacts" / "eval_LINC_clean_full.csv",
    REPO_ROOT / "artifacts" / "eval_LOGIC_LM_clean_full.csv",
]
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "verified_v1"

API_ERROR_MARKERS = ("Error:", "RateLimit", "Timeout", "APIError", "Traceback",
                      "openai.", "ConnectionError")


def load_verified_keys(path: Path) -> Set[Tuple[str, str]]:
    data = json.loads(path.read_text())
    items = data["rows"] if isinstance(data, dict) and "rows" in data else data
    keys: Set[Tuple[str, str]] = set()
    for entry in items:
        if isinstance(entry, str) and "|" in entry:
            qid, dom = entry.split("|", 1)
            keys.add((qid, dom))
        elif isinstance(entry, dict):
            keys.add((entry["id"], entry["domain"]))
    return keys


def looks_like_api_error(row: Dict[str, str]) -> bool:
    ans = row.get("answer") or ""
    return any(marker in ans for marker in API_ERROR_MARKERS)


def acc(rows: List[Dict[str, str]]) -> Tuple[int, int, float]:
    scored = [r for r in rows if (r.get("expected_contains") or "").strip()]
    k = sum(int(r.get("success", 0)) for r in scored)
    n = len(scored)
    return k, n, (k / n if n else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    ap.add_argument("--baseline", action="append", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    baselines = args.baseline or DEFAULT_BASELINES

    verified = load_verified_keys(args.ids)
    print(f"[filter] verified keys: {len(verified)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report: List[str] = ["# Baseline filter to verified subset", "",
                          f"Verified (id, domain) keys: **{len(verified)}**", ""]
    filtered_counts: List[int] = []
    per_mode_stats: Dict[str, Dict[str, object]] = {}
    failed = False

    for bpath in baselines:
        if not bpath.exists():
            print(f"[filter] ERROR: missing baseline CSV {bpath}")
            failed = True
            continue
        with bpath.open() as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            all_rows = list(reader)
        mode = all_rows[0].get("mode", bpath.stem) if all_rows else bpath.stem
        present = {(r["id"], r["domain"]) for r in all_rows}
        kept = [r for r in all_rows if (r["id"], r["domain"]) in verified]
        kept.sort(key=lambda r: (r["id"], r["domain"]))

        missing = verified - present
        api_errors = [r for r in kept if looks_like_api_error(r)]

        out_path = args.out_dir / f"eval_{mode}_verified_v1.csv"
        with out_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(kept)

        k, n, a = acc(kept)
        # by type / domain
        by_type = defaultdict(lambda: [0, 0])
        by_domain = defaultdict(lambda: [0, 0])
        for r in kept:
            if not (r.get("expected_contains") or "").strip():
                continue
            by_type[r["type"]][0] += int(r["success"])
            by_type[r["type"]][1] += 1
            by_domain[r["domain"]][0] += int(r["success"])
            by_domain[r["domain"]][1] += 1

        filtered_counts.append(len(kept))
        per_mode_stats[mode] = {
            "n": len(kept), "scored": n, "k": k, "acc": a,
            "missing": len(missing), "api_errors": len(api_errors),
            "by_type": dict(by_type), "by_domain": dict(by_domain),
        }
        status = "OK"
        if missing:
            status = f"FAIL — {len(missing)} verified keys absent from this baseline"
            failed = True
        if api_errors:
            status += f"  WARNING — {len(api_errors)} API-error rows"
        print(f"[filter] {mode:16s} kept={len(kept):5d}  acc={a:.4f}  {status}")
        report += [
            f"## {mode}",
            "",
            f"- Output: `{out_path.relative_to(REPO_ROOT)}`",
            f"- Rows kept: **{len(kept)}**  (scored: {n})",
            f"- Accuracy: **{a:.4f}**  ({k}/{n})",
            f"- Verified keys missing from this baseline: {len(missing)}",
            f"- API-error rows: {len(api_errors)}",
            f"- By type: " + ", ".join(f"{t}: {v[0]}/{v[1]}={v[0]/v[1]:.4f}"
                                       for t, v in sorted(by_type.items()) if v[1]),
            f"- By domain: " + ", ".join(f"{d}: {v[0]}/{v[1]}={v[0]/v[1]:.4f}"
                                         for d, v in sorted(by_domain.items()) if v[1]),
            "",
        ]
        if missing:
            sample = sorted(missing)[:5]
            report.append(f"- Missing-key sample: {sample}")
            report.append("")

    # Row-count parity check
    if len(set(filtered_counts)) > 1:
        print(f"[filter] ERROR: filtered row counts differ: {filtered_counts}")
        report += ["## VALIDATION", "", f"**FAIL** — row counts differ: {filtered_counts}", ""]
        failed = True
    else:
        report += ["## VALIDATION", "",
                    f"All baselines filtered to **{filtered_counts[0] if filtered_counts else 0}** rows. "
                    f"{'PASS' if not failed else 'FAIL — see above'}.", ""]

    (args.out_dir / "filter_report.md").write_text("\n".join(report))
    print(f"[filter] report → {args.out_dir / 'filter_report.md'}")

    if failed:
        print("[filter] RESULT: FAIL")
        return 1
    print("[filter] RESULT: PASS — all baselines filtered, counts match, no missing keys")
    return 0


if __name__ == "__main__":
    sys.exit(main())
