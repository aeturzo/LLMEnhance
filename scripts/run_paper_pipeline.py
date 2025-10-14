#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-command paper pipeline (serial, safe).

Runs exactly these in order:
  build-corpus
  retriever-warm (skips if module missing)
  eval (per domain)
  calibrate
  selective
  fig-selective
  thresholds
  aurc
  ci
  sym-stats
  mcnemar
  export-tables

Usage:
  python scripts/run_paper_pipeline.py --domains battery
  python scripts/run_paper_pipeline.py --domains battery,lexmark --cov 0.50
  python scripts/run_paper_pipeline.py --no-warm
  python scripts/run_paper_pipeline.py --split dev --include-gold
"""

from __future__ import annotations
import argparse, importlib.util, subprocess, sys, os
from glob import glob
from pathlib import Path
from typing import List, Optional

# Repo paths
REPO    = Path(__file__).resolve().parents[1]
ART     = REPO / "artifacts"
TABLES  = REPO / "tables"
TESTS   = REPO / "tests"
CORPUS  = REPO / "backend" / "corpus" / "dpp_corpus.jsonl"

# Quiet HF tokenizer parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def run(cmd: List[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"[pipeline] failed: {' '.join(cmd)}")


def latest(pattern: str) -> Optional[Path]:
    files = [Path(p) for p in glob(pattern)]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]


def have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def trace_from_joined(joined_csv: str) -> str:
    """
    Derive 'trace_YYYYMMDD_HHMMSS.jsonl' from a joined CSV name.
    Works even if the joined file is the calibrated one with '_calibrated'.
    """
    p = Path(joined_csv)
    stem = p.stem  # e.g., "eval_joined_20250905_222657_calibrated" or "eval_joined_20250905_222657"
    ts = stem.replace("eval_joined_", "").replace("_calibrated", "")
    return str(p.parent / f"trace_{ts}.jsonl")


# ---- steps ----

def step_build_corpus(split: str = "all", include_gold: bool = False):
    cmd = [sys.executable, "-m", "scripts.build_corpus",
           "--tests", str(TESTS), "--out", str(CORPUS), "--split", split]
    if include_gold:
        cmd.append("--include-gold")
    run(cmd)


def step_retriever_warm():
    if not have("backend.retrieval.hybrid"):
        print("[retriever-warm] backend.retrieval.hybrid not found; skipping.")
        return
    code = ("from backend.retrieval.hybrid import HybridRetriever; "
            f"HybridRetriever(r'{CORPUS.as_posix()}')")
    run([sys.executable, "-c", code])


def step_eval(domain: str):
    run([sys.executable, str(REPO / "run_eval_all.py"), "--domain", domain])


def step_calibrate():
    joined = latest(f"{ART.as_posix()}/eval_joined_*.csv")
    if not joined:
        raise SystemExit("[calibrate] no eval_joined_*.csv found in artifacts/")
    run([sys.executable, "-m", "backend.eval.calibrate",
         "--joined", str(joined), "--out", str(ART)])


def step_selective_and_figs():
    run([sys.executable, "-m", "backend.eval.selective",
         "--artifacts", str(ART), "--out", str(ART)])
    # figures are optional; warn if missing
    try:
        run([sys.executable, "-m", "backend.eval.figures"])
    except SystemExit as e:
        print(f"[fig-selective] warning: {e}")


def step_thresholds(target_cov: float):
    sel = latest(f"{ART.as_posix()}/selective_*.csv")
    if not sel:
        raise SystemExit("[thresholds] no selective_*.csv found in artifacts/")
    run([sys.executable, "-m", "backend.eval.thresholds",
         "--csv", str(sel), "--out", str(TABLES), "--cov", str(target_cov)])


def step_aurc():
    sel = latest(f"{ART.as_posix()}/selective_*.csv")
    if not sel:
        raise SystemExit("[aurc] no selective_*.csv found in artifacts/")
    run([sys.executable, "-m", "backend.eval.aurc",
         "--csv", str(sel), "--out", str(TABLES)])


def step_ci():
    joined = latest(f"{ART.as_posix()}/eval_joined_*.csv")
    if not joined:
        raise SystemExit("[ci] no eval_joined_*.csv found in artifacts/")
    run([sys.executable, "-m", "backend.eval.conf_intervals",
         "--joined", str(joined), "--out", str(TABLES)])


def step_sym_stats(include_rl: bool = False):
    """
    Use the latest joined file (often the calibrated one),
    and pass an explicit --trace that strips '_calibrated' to avoid missing file errors.
    """
    joined = latest(f"{ART.as_posix()}/eval_joined_*.csv")
    if not joined:
        raise SystemExit("[sym-stats] no eval_joined_*.csv found in artifacts/")

    # Derive the expected trace path; if it doesn't exist, fallback to newest trace_*.
    trace_path = Path(trace_from_joined(str(joined)))
    if not trace_path.exists():
        fallback = latest(f"{ART.as_posix()}/trace_*.jsonl")
        if not fallback:
            raise SystemExit(f"[sym-stats] expected trace not found: {trace_path}")
        print(f"[sym-stats] derived trace missing, falling back to latest: {fallback.name}")
        trace_path = fallback

    cmd = [sys.executable, "-m", "backend.eval.sym_coverage",
           "--joined", str(joined), "--trace", str(trace_path), "--out", str(TABLES)]
    if include_rl:
        cmd.append("--include_rl")
    run(cmd)


def step_mcnemar(mode_A: str = "ADAPTIVERAG", mode_B: str = "RAG_BASE"):
    joined = latest(f"{ART.as_posix()}/eval_joined_*.csv")
    if not joined:
        raise SystemExit("[mcnemar] no eval_joined_*.csv found in artifacts/")
    run([sys.executable, "-m", "backend.eval.mcnemar",
         "--joined", str(joined),
         "--out", str(TABLES / "mcnemar.csv"),
         "--A", mode_A, "--B", mode_B])


def step_export_tables():
    run([sys.executable, str(REPO / "scripts" / "export_tables.py")])


def pipeline(domain: str, cov: float, split: str, include_gold: bool, warm: bool, sym_include_rl: bool):
    print("\n" + "="*70)
    print(f"=== PIPELINE: domain={domain} split={split} cov={cov:.2f} ===")
    print("="*70)

    step_build_corpus(split=split, include_gold=include_gold)
    if warm:
        step_retriever_warm()
    step_eval(domain)
    step_calibrate()
    step_selective_and_figs()
    step_thresholds(cov)
    step_aurc()
    step_ci()
    step_sym_stats(include_rl=sym_include_rl)
    step_mcnemar("ADAPTIVERAG", "RAG_BASE")
    step_export_tables()

    print(f"\n✔ complete: {domain}  |  artifacts: {ART}  |  tables: {TABLES}  |  paper tables: docs/paper/tables")


def main():
    ap = argparse.ArgumentParser(description="Run full paper pipeline serially for one or more domains.")
    ap.add_argument("--domains", default="battery", help="Comma-separated list, e.g. 'battery,lexmark'")
    ap.add_argument("--cov", type=float, default=0.50, help="Target coverage for thresholds")
    ap.add_argument("--split", choices=["all","dev","test"], default="all", help="Corpus split to build from")
    ap.add_argument("--include-gold", action="store_true", help="Include gold answers as docs in corpus")
    ap.add_argument("--no-warm", action="store_true", help="Skip retriever warm")
    ap.add_argument("--sym-include-rl", action="store_true", help="Include RL in overall SYM stats")
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(REPO / "docs" / "paper" / "tables", exist_ok=True)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    if not domains:
        raise SystemExit("No domains provided.")

    for d in domains:
        pipeline(
            domain=d,
            cov=args.cov,
            split=args.split,
            include_gold=args.include_gold,
            warm=not args.no_warm,
            sym_include_rl=args.sym_include_rl,
        )

    print("\nALL DONE.")


if __name__ == "__main__":
    main()
