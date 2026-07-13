# External-LLM Baseline Runbook

Address-the-reviewer baseline runs for the paper revision. Three baselines, run
against the release synthetic DPP-like benchmark, scored with the same
substring + canonical matchers.

## Why these three baselines

| Baseline | Maps to reviewer concern |
|---|---|
| `GPT4O_LONGCTX` | "How does a strong long-context LLM with retrieval compare?" |
| `LINC` | "How does a state-of-the-art neuro-symbolic pipeline (Olausson et al. 2023) compare?" |
| `LOGIC_LM` | "How does Logic-LM-style dispatch (Pan et al. 2023) compare?" |

The harness (`scripts/run_baselines.py`) is self-contained, resumable, and
produces CSV with the same schema as `artifacts/eval_<MODE>_*.csv`, so the
outputs plug directly into the existing aggregation pipeline (`run_paper_pipeline.py`,
`backend/eval/mcnemar.py`, etc.).

Current readiness note: use `--benchmark release-clean` for paper-grade
external-baseline runs. The clean artifact regenerates document QA rows from
the current seed documents and keeps the existing KB/fact rows. See
`artifacts/baseline_readiness_audit_20260519_clean.md`.

## Pre-flight (one-time)

```bash
cd /Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain
source temp_env.sh          # exports OPENAI_API_KEY, GEN_MODEL
pip install openai           # if not already present in the venv
```

## Smoke test (1 minute, ~$0.05)

```bash
python scripts/run_baselines.py --mode LOGIC_LM \
    --out artifacts/eval_LOGIC_LM_smoke.csv --limit 25
```

Expect: 25 rows in the CSV, `accuracy` printed at the end.

## Full run on the cleaned benchmark

```bash
# Each command writes its own CSV; runs are resumable — kill and restart safely.
python scripts/run_baselines.py --mode GPT4O_LONGCTX --benchmark release-clean \
  --out artifacts/eval_GPT4O_LONGCTX_clean_full.csv --budget-usd 3.00
python scripts/run_baselines.py --mode LINC --benchmark release-clean \
  --out artifacts/eval_LINC_clean_full.csv --budget-usd 3.00
python scripts/run_baselines.py --mode LOGIC_LM --benchmark release-clean \
  --out artifacts/eval_LOGIC_LM_clean_full.csv --budget-usd 3.00
```

Add `--sleep 0.2` if you hit rate limits.

Before running these commands for the paper, rerun the paid preflights listed in
`artifacts/baseline_readiness_audit_20260519_clean.md` if any benchmark or
retrieval code changes.

## Paper-overlap subset

The release-bundle `release/release_20250902_215155/tests/<domain>/tests.jsonl`
contains 8,093 queries (battery 2108, lexmark 2751, viessmann 3234). The
paper's archived 3,429-row pooled CSV shares 1,345 `(id, domain)` keys with the
release benchmark, but most archived paper query strings are not byte-identical
to the release query strings. Do not attach release gold answers to paper rows
by key alone.

Use `--benchmark paper-matched` only for diagnostics. In the current harness it
loads release-benchmark rows whose `(id, domain)` keys also occur in the paper
split, preserving internally consistent question/gold/context triples:

```bash
python scripts/run_baselines.py --mode LOGIC_LM --benchmark paper-matched \
    --filter-ids artifacts/paper_overlap_valid_subset_90_ids.json \
    --out artifacts/paper_overlap_subset_eval_LOGIC_LM_90.csv \
    --budget-usd 0.20
```

## Aggregation: McNemar vs ADAPTIVERAG

```bash
python -c "
import csv
from pathlib import Path
def load(path, mode):
    return {r['id']: int(r['success']) for r in csv.DictReader(open(path)) if r['mode']==mode}
adapt = load('artifacts/eval_joined_pooled_20260202_192420.csv', 'ADAPTIVERAG')
for base in ['GPT4O_LONGCTX','LINC','LOGIC_LM']:
    b = load(f'artifacts/eval_{base}_full.csv', base)
    common = adapt.keys() & b.keys()
    a_only = sum(1 for k in common if adapt[k] and not b[k])
    b_only = sum(1 for k in common if b[k] and not adapt[k])
    both = sum(1 for k in common if adapt[k] and b[k])
    print(f'{base:18s} common={len(common)}  ADAPTIVERAG-only={a_only}  {base}-only={b_only}  both={both}')
"
```

## Paper text to add (Section 5)

> "We additionally evaluated three external LLM baselines on the release
> synthetic benchmark: GPT-4o-mini with full domain-document context
> (GPT4O_LONGCTX), a LINC-style neuro-symbolic pipeline (Olausson et al., 2023),
> and Logic-LM (Pan et al., 2023). These results allow direct head-to-head
> comparison and address a reviewer concern that the published evaluation used
> only internal ablations."

(Then add a table of pooled accuracy ± Wilson 95% CI, plus McNemar vs ADAPTIVERAG, for each baseline.)
