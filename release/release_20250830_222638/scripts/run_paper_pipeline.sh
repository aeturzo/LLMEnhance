#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

DOMAINS=("battery" "lexmark" "viessmann")

echo ">> Clean old artifacts/tables/release"
rm -rf artifacts tables release || true
mkdir -p artifacts tables release

echo ">> Scale datasets (uses KG + docs/memory fallbacks)"
bash scripts/scale_dataset.sh

echo ">> Quick counts (should be ~900+ per domain after boosting)"
python - <<'PY'
import json, pathlib
from collections import Counter
for d in ["battery","lexmark","viessmann"]:
    p = pathlib.Path(f"tests/{d}/tests.jsonl")
    if not p.exists():
        print(f"[{d}] MISSING tests.jsonl"); continue
    n=0; c=Counter()
    for line in p.open(encoding="utf-8"):
        if not line.strip(): continue
        j=json.loads(line); n+=1; c[j.get("type","open")]+=1
    print(f"[{d}] total={n} breakdown={dict(c)}")
PY

echo ">> Seed memory + run eval + selective per domain (uses scaled tests)"
for dom in "${DOMAINS[@]}"; do
  echo "== $dom =="
  python scripts/seed_memory.py --domain "$dom" || true
  python run_eval_all.py --domain "$dom"
  python run_selective.py --domain "$dom"
done

echo ">> Figures, ablations, calibration, publication checks"
python backend/eval/figures.py --in artifacts --out artifacts || true
python backend/eval/ablate.py   --in artifacts --out artifacts || true
python backend/eval/calibration_sweep.py --in artifacts --out artifacts || true
python backend/eval/publication_checks.py || true

echo ">> Release pack"
make release

echo ">> Done. Key outputs:"
echo "   - artifacts/eval_joined_*.csv  (rows = total_tests x 7)"
echo "   - tables/*.csv + tables/pub_summary.md"
echo "   - artifacts/fig_selective.png, artifacts/fig_ablation.png"
echo "   - release/release_*.zip"
