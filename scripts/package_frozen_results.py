#!/usr/bin/env python3
"""Package frozen Phase B/C result CSVs and generate an old-vs-new diff."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(frame: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    score = "success" if "success" in frame else "correct"
    rows = []
    for mode, group in frame.groupby("mode", sort=True):
        values = pd.to_numeric(group[score], errors="coerce").dropna()
        rows.append(
            {
                "benchmark": benchmark,
                "mode": mode,
                "n": len(values),
                "correct": int(values.sum()),
                "accuracy": float(values.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-results", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-tag", required=True)
    args = parser.parse_args()

    verified_files = {
        mode: args.frozen_results / "verified" / f"{mode}.csv"
        for mode in ("COMPASS", "GPT4O_LONGCTX", "LINC", "LOGIC_LM")
    }
    missing = [str(path) for path in verified_files.values() if not path.exists()]
    compose_path = args.frozen_results / "compositional" / "all_systems.csv"
    if not compose_path.exists():
        missing.append(str(compose_path))
    if missing:
        raise SystemExit(f"missing frozen result files: {missing}")

    result_dir = args.output / "results"
    table_dir = args.output / "tables"
    result_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    new_verified_frames = []
    for mode, source in verified_files.items():
        target = result_dir / f"verified_{mode}.csv"
        shutil.copy2(source, target)
        copied.append(target)
        frame = pd.read_csv(source)
        frame["mode"] = mode
        new_verified_frames.append(frame)
    new_verified = pd.concat(new_verified_frames, ignore_index=True)
    new_compose = pd.read_csv(compose_path)
    compose_target = result_dir / "compositional_all_systems.csv"
    shutil.copy2(compose_path, compose_target)
    copied.append(compose_target)

    new_summary = pd.concat(
        [summary(new_verified, "verified_release_6270"), summary(new_compose, "compositional_3000")],
        ignore_index=True,
    )
    new_summary.to_csv(table_dir / "frozen_summary.csv", index=False)

    old_verified_paths = {
        "COMPASS": args.repo_root / "artifacts/verified_v1/eval_ADAPTIVERAG_verified_v1_full.csv",
        "GPT4O_LONGCTX": args.repo_root / "artifacts/verified_v1/eval_GPT4O_LONGCTX_verified_v1.csv",
        "LINC": args.repo_root / "artifacts/verified_v1/eval_LINC_verified_v1.csv",
        "LOGIC_LM": args.repo_root / "artifacts/verified_v1/eval_LOGIC_LM_verified_v1.csv",
    }
    old_frames = []
    for mode, path in old_verified_paths.items():
        frame = pd.read_csv(path)
        frame["mode"] = mode
        old_frames.append(frame)
    old_verified = pd.concat(old_frames, ignore_index=True)
    old_compose_path = args.repo_root / "artifacts/paper_handoff_20260603/final_arch_3000_all3.csv"
    old_compose = pd.read_csv(old_compose_path)
    old_summary = pd.concat(
        [summary(old_verified, "verified_release_6270"), summary(old_compose, "compositional_3000")],
        ignore_index=True,
    )
    diff = new_summary.merge(old_summary, on=["benchmark", "mode"], how="outer", suffixes=("_new", "_old"))
    diff["accuracy_delta"] = diff["accuracy_new"] - diff["accuracy_old"]
    diff.to_csv(table_dir / "results_diff.csv", index=False)

    rendered = diff.copy()
    for column in ["accuracy_new", "accuracy_old", "accuracy_delta"]:
        rendered[column] = rendered[column].map(lambda x: "not available" if pd.isna(x) else f"{x:.4f}")
    lines = [
        "# Frozen results diff",
        "",
        f"Frozen evaluation tag: `{args.freeze_tag}`.",
        "",
        rendered[["benchmark", "mode", "n_new", "accuracy_new", "n_old", "accuracy_old", "accuracy_delta"]].to_markdown(index=False),
        "",
        "The historical 3,429-row pooled suite is excluded from this regenerated table because its retained split is missing gold for 2,084 rows; see `docs/paper/POOLED_RERUN_LIMITATION.md`.",
        "",
    ]
    (args.output / "RESULTS_DIFF.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "freeze_tag": args.freeze_tag,
        "files": [
            {"path": str(path.relative_to(args.output)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in copied
        ],
    }
    (args.output / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote frozen package: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
