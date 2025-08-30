#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

DOMAINS="battery lexmark viessmann"

# Targets (tune freely)
R_KB=700   # recall target from KB (literal + object)
L_KB=500   # logic target from KB (inferred types)
O_DOC=800  # open from docs
R_DOC=250  # numeric recall from docs

for dom in $DOMAINS; do
  echo "== $dom =="

  # optional: ingest raw docs to seed_docs.jsonl if you put .txt/.md into tests/<dom>/docs
  if [ -d "tests/$dom/docs" ]; then
    python scripts/ingest_docs.py --domain "$dom" || true
  fi

  # generate from KB (now includes object facts)
  python scripts/autogen_kb_tests.py  --domain "$dom" --n_recall "$R_KB" --n_logic "$L_KB"

  # generate from docs/memory fallbacks
  python scripts/autogen_doc_tests.py --domain "$dom" --n_open "$O_DOC" --n_recall "$R_DOC"

  # validate silently
  python scripts/validate_dataset.py "tests/$dom/tests.jsonl" || true
done
