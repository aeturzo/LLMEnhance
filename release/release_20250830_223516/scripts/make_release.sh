#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
REL="release/release_${STAMP}"
mkdir -p "$REL"

echo ">> Freezing environment"
pip freeze > "$REL/requirements_freeze.txt" || true

echo ">> Collecting minimal pack"
# Code
mkdir -p "$REL/backend/services" "$REL/backend/eval" "$REL/backend/config" "$REL/backend/ontologies" "$REL/scripts" "$REL/tests"

cp -r backend/services/*.py "$REL/backend/services/"
cp -r backend/eval/*.py "$REL/backend/eval/"
cp -r backend/config/*.py "$REL/backend/config/" || true
cp -r backend/ontologies/*.ttl "$REL/backend/ontologies/"

# Generators / boosters / pipeline
cp scripts/*.py "$REL/scripts/" || true
cp scripts/*.sh "$REL/scripts/" || true

# Scaled tests only
for d in battery lexmark viessmann; do
  mkdir -p "$REL/tests/$d"
  cp "tests/$d/tests.jsonl" "$REL/tests/$d/" || true
  for s in seed_docs.jsonl seed_mem.jsonl; do
    [ -f "tests/$d/$s" ] && cp "tests/$d/$s" "$REL/tests/$d/"
  done
done

# Artifacts summary (not heavy raw traces)
mkdir -p "$REL/artifacts"
cp artifacts/fig_*.png "$REL/artifacts/" 2>/dev/null || true
cp artifacts/selective_*.png "$REL/artifacts/" 2>/dev/null || true

# Tables
mkdir -p "$REL/tables"
cp tables/*.csv "$REL/tables/" 2>/dev/null || true
cp tables/*.md  "$REL/tables/" 2>/dev/null || true

# README pointer
cat > "$REL/README_RELEASE.md" <<'MD'
# Reproduce
1) `pip install -r requirements_freeze.txt` (or your own env meeting versions)
2) `export PYTHONPATH=.` from repo root
3) Datasets are under `tests/<domain>/tests.jsonl`.
4) Run evaluation: `python run_eval_all.py --domain battery` (and for others).
5) Generate stats/tables: `python backend/eval/stats_polish.py`
6) Reports: `python backend/eval/report_v2.py --domain battery --out artifacts/report_battery.html`
MD

echo ">> Creating ZIP"
( cd release && zip -r "release_${STAMP}.zip" "$(basename "$REL")" >/dev/null )
echo "OK: release/release_${STAMP}.zip"
