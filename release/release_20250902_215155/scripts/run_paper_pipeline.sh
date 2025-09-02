#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   bash scripts/run_paper_pipeline.sh
#   bash scripts/run_paper_pipeline.sh battery
#   bash scripts/run_paper_pipeline.sh battery lexmark viessmann

# --- Select domains ---
if [ "$#" -gt 0 ]; then
  DOMAINS=("$@")
else
  DOMAINS=("battery" "lexmark" "viessmann")
fi

# --- repo root & env ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

# Keep SYM enabled (user wants all modules)
unset DISABLE_SYM || true

echo "== Pipeline start =="
echo "ROOT=$ROOT"
echo "DOMAINS: ${DOMAINS[*]}"

# --- deps (tabulate for publication_checks) ---
python - <<'PY' || true
import importlib, sys
try:
    importlib.import_module("tabulate")
    print("[deps] tabulate OK")
except Exception:
    sys.exit(42)
PY
rc=$?
if [ $rc -eq 42 ]; then
  echo "[deps] installing tabulate…"
  pip install -q tabulate
fi

# --- Clean outputs ---
echo ">> Clean old artifacts/tables/release"
rm -rf artifacts tables release || true
mkdir -p artifacts tables release

# --- 1) Docs (.txt) -> seed_docs.jsonl (per domain) ---
if [ -f scripts/make_seed_from_docs.py ]; then
  echo ">> Docs -> seed_docs.jsonl"
  for dom in "${DOMAINS[@]}"; do
    echo "   -- $dom --"
    python scripts/make_seed_from_docs.py --domain "$dom" --session s1 || true
  done
else
  echo "[warn] scripts/make_seed_from_docs.py not found; skipping docs->seeds"
fi

# --- 2) Scale datasets (KG + docs mining) ---
if [ -f scripts/scale_dataset.sh ]; then
  echo ">> Scale datasets (uses KG + docs)"
  bash scripts/scale_dataset.sh
else
  echo "[warn] scripts/scale_dataset.sh not found; skipping scaling"
fi

# --- Quick counts per domain ---
echo ">> Quick counts (target: 300+ per domain if possible)"
python - <<'PY'
import json, pathlib
from collections import Counter
for d in ["battery","lexmark","viessmann"]:
    p = pathlib.Path(f"tests/{d}/tests.jsonl")
    if not p.exists():
        print(f"[{d}] MISSING tests.jsonl"); continue
    n=0; c=Counter()
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            j=json.loads(line); n+=1; c[j.get("type","open")]+=1
    print(f"[{d}] total={n} breakdown={dict(c)}")
PY

# --- 3) Seed + eval + selective per domain ---
echo ">> Seed memory + run eval + selective"
for dom in "${DOMAINS[@]}"; do
  echo "== $dom =="
  # Seed (run_eval_all also seeds, but do it explicitly)
  if [ -f scripts/seed_memory.py ]; then
    python scripts/seed_memory.py --domain "$dom" || true
  else
    echo "[warn] scripts/seed_memory.py not found; skipping explicit seed"
  fi

  # Full evaluation (BASE, MEM, SYM, MEMSYM, ROUTER, ADAPTIVERAG, RL)
  python run_eval_all.py --domain "$dom"

  # Selective risk
  if [ -f run_selective.py ]; then
    python run_selective.py --domain "$dom"
  else
    echo "[warn] run_selective.py not found; skipping selective"
  fi
done

# --- 4) Aggregates, figures, ablations, calibration, qualitative, checks ---
echo ">> Figures, ablations, calibration, qualitative, checks"
python backend/eval/figures.py           --in artifacts --out artifacts || true
python backend/eval/ablate.py            --in artifacts --out artifacts || true
python backend/eval/calibration_sweep.py --in artifacts --out artifacts || true
python backend/eval/calibration.py       --artifacts artifacts --tables tables || true
python backend/eval/trace_digest.py      --in artifacts --out artifacts/qualitative.md --k 3 || true
python backend/eval/publication_checks.py || true

# Ensure tables/summary_latest.csv exists even if some steps skipped
python - <<'PY'
import glob, pandas as pd, os
os.makedirs("tables", exist_ok=True)
csvs=sorted(glob.glob("artifacts/eval_*.csv"))
if csvs:
    df=pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    df.to_csv("tables/summary_latest.csv", index=False)
    print("[tables] wrote tables/summary_latest.csv", "rows=", len(df))
else:
    print("[tables] no eval_*.csv found; skipping")
PY

# Optional polish/report, only if present
if [ -f backend/eval/stats_polish.py ]; then
  python backend/eval/stats_polish.py || true
fi
if [ -f backend/eval/report_v2.py ]; then
  python backend/eval/report_v2.py --domain battery   --out artifacts/report_battery.html   || true
  python backend/eval/report_v2.py --domain lexmark   --out artifacts/report_lexmark.html   || true
  python backend/eval/report_v2.py --domain viessmann --out artifacts/report_viessmann.html || true
fi

# --- 5) Release pack (optional) ---
if [ -x scripts/make_release.sh ]; then
  echo ">> Release pack"
  bash scripts/make_release.sh || true
else
  echo "[info] scripts/make_release.sh not found; skipping release pack"
fi

# --- 6) Sanity counts ---
echo ">> Sanity: expected vs. joined rows"
python - <<'PY'
import glob, pandas as pd, subprocess
total=0
for d in ["battery","lexmark","viessmann"]:
    try:
        n=int(subprocess.check_output(["bash","-lc",f"test -f tests/{d}/tests.jsonl && wc -l < tests/{d}/tests.jsonl || echo 0"]).decode().strip())
        print(d, n); total+=n
    except Exception: pass
js=sorted(glob.glob("artifacts/eval_joined_*.csv"))
if js:
    latest=js[-1]
    print("Expected joined rows ≈", total*7)
    print("Joined rows =", len(pd.read_csv(latest)), latest)
else:
    print("No eval_joined_*.csv yet")
PY

echo
echo "✅ Pipeline complete."
