#!/usr/bin/env bash
set -euo pipefail

# Root + Python path
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

# Domains (override with: DOMAINS="battery" bash scripts/scale_dataset.sh)
DOMAINS="${DOMAINS:-battery lexmark viessmann}"

# Targets (override with env vars if needed)
R_KB="${R_KB:-700}"    # recall target from KB (literal + object)
L_KB="${L_KB:-500}"    # logic target from KB (inferred types)
O_DOC="${O_DOC:-800}"  # open from docs
R_DOC="${R_DOC:-250}"  # numeric recall from docs

# Seeding behavior:
# REGEN_SEEDS=1   -> force rebuild seed_docs.jsonl from docs using chunker
# REGEN_SEEDS=0   -> default; use existing seed_docs.jsonl if present
REGEN_SEEDS="${REGEN_SEEDS:-0}"

# Chunking params for make_seed_from_docs.py (only used when regenerating)
CHUNK_MIN="${CHUNK_MIN:-300}"
CHUNK_MAX="${CHUNK_MAX:-900}"

for dom in $DOMAINS; do
  echo "== $dom =="

  docs_dir="tests/$dom/docs"
  seed_path="tests/$dom/seed_docs.jsonl"

  if [ -d "$docs_dir" ]; then
    if [ "$REGEN_SEEDS" = "1" ]; then
      echo "[seed] Regenerating chunked seeds for $dom (min=$CHUNK_MIN, max=$CHUNK_MAX)"
      python scripts/make_seed_from_docs.py --domain "$dom" --min "$CHUNK_MIN" --max "$CHUNK_MAX"
    else
      if [ -f "$seed_path" ]; then
        echo "[seed] Using existing $seed_path (not overwriting)."
      else
        echo "[seed] No seed_docs.jsonl found; building once from docs (min=$CHUNK_MIN, max=$CHUNK_MAX)"
        python scripts/make_seed_from_docs.py --domain "$dom" --min "$CHUNK_MIN" --max "$CHUNK_MAX"
      fi
    fi
  else
    echo "[seed] No docs directory at $docs_dir — skipping seeding."
  fi

  # === Generate from KB (includes object facts) ===
  python scripts/autogen_kb_tests.py --domain "$dom" --n_recall "$R_KB" --n_logic "$L_KB"

  # === Generate from docs/memory (consumes seed_docs.jsonl if present) ===
  python scripts/autogen_doc_tests.py --domain "$dom" --n_open "$O_DOC" --n_recall "$R_DOC"

  # === Validate silently (non-fatal) ===
  python scripts/validate_dataset.py "tests/$dom/tests.jsonl" || true

# de-duplicate by normalized query (post-generation)
python scripts/dedup_tests.py "tests/$dom/tests.jsonl" || true

# re-validate after dedup (non-fatal if stylistic warnings only)
python scripts/validate_dataset.py "tests/$dom/tests.jsonl" || true

done
