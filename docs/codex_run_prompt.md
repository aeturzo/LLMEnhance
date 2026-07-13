# Codex Run Prompt — single, costed, one-shot

Paste the prompt below into Codex / a host shell after running the
**pre-flight section** locally. The Codex run will cost **about $5–7** on
gpt-4o-mini and produce three CSVs that drop straight into the existing
aggregation pipeline.

---

## Pre-flight (run these manually first, before pasting the Codex prompt)

```bash
cd /Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain

# 1. Source the OpenAI key
source temp_env.sh
test -n "$OPENAI_API_KEY" && echo "key OK" || { echo "MISSING KEY"; exit 1; }

# 2. Install openai client if not present
python -c "import openai" 2>/dev/null || pip install openai

# 3. (Optional) inspect what one clean-release prompt looks like — does NOT call OpenAI
python scripts/run_baselines.py --mode GPT4O_LONGCTX --benchmark release-clean \
    --filter-prefix cleandocopen --limit 1 --print-prompt --out /dev/null

# 4. (Recommended) ~$0.05 smoke test — proves the API call works end-to-end
python scripts/run_baselines.py --mode LOGIC_LM --benchmark release-clean \
    --filter-prefix cleandocopen,cleandocrec --limit 10 --budget-usd 0.20 \
    --out artifacts/baseline_smoke.csv
```

If the smoke test prints `accuracy=…` and a non-zero `total cost`, the
plumbing is good and you can paste the prompt below.

---

## The Codex prompt

> You are running the external LLM baseline experiment for the AEI paper.
> The repo root is `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain`.
> Working directory: that path. Before running the full commands, read
> `artifacts/baseline_readiness_audit_20260519_clean.md`. Run only the commands listed below. Do NOT
> modify code. Do NOT add new flags. The script `scripts/run_baselines.py`
> is already verified and supports `--budget-usd` as a hard cap.
>
> Step 1 — environment
> ```bash
> cd /Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain
> source temp_env.sh
> python -c "import openai; print('openai', openai.__version__)"
> ```
>
> Step 2 — three baseline runs on the cleaned release benchmark (6,915 rows each,
> all have gold answers). Total estimated cost ≈ $5.82 across all three modes
> on gpt-4o-mini. Each run is resumable.
> ```bash
> python scripts/run_baselines.py --mode GPT4O_LONGCTX --benchmark release-clean \
>     --out artifacts/eval_GPT4O_LONGCTX_clean_full.csv --budget-usd 3.00
>
> python scripts/run_baselines.py --mode LINC --benchmark release-clean \
>     --out artifacts/eval_LINC_clean_full.csv --budget-usd 3.00
>
> python scripts/run_baselines.py --mode LOGIC_LM --benchmark release-clean \
>     --out artifacts/eval_LOGIC_LM_clean_full.csv --budget-usd 3.00
> ```
>
> If a run aborts on its own budget cap (`ERROR: running cost ... exceeded
> budget`), report the running cost and stop — do not raise the budget on
> your own.
>
> Step 3 — aggregate into the paper table
> Note: the McNemar rows are diagnostic unless ADAPTIVERAG is also rerun on
> the cleaned release benchmark; do not treat `cleandoc*` rows as old paper-split
> head-to-head rows by identifier alone.
> ```bash
> python scripts/aggregate_baselines.py \
>     --pooled artifacts/eval_joined_pooled_20260202_192420.csv \
>     --baseline artifacts/eval_GPT4O_LONGCTX_clean_full.csv \
>     --baseline artifacts/eval_LINC_clean_full.csv \
>     --baseline artifacts/eval_LOGIC_LM_clean_full.csv \
>     --out-md artifacts/baseline_comparison.md \
>     --out-tex docs/paper/tables/baseline_comparison.tex
> ```
>
> Step 4 — verify and report
> ```bash
> echo "--- baseline CSV row counts ---"
> for f in artifacts/eval_GPT4O_LONGCTX_clean_full.csv \
>          artifacts/eval_LINC_clean_full.csv \
>          artifacts/eval_LOGIC_LM_clean_full.csv; do
>   wc -l "$f"
> done
> echo "--- final cost lines ---"
> for f in artifacts/eval_GPT4O_LONGCTX_clean_full.csv \
>          artifacts/eval_LINC_clean_full.csv \
>          artifacts/eval_LOGIC_LM_clean_full.csv; do
>   python -c "import sys,csv; p=sys.argv[1]; rows=list(csv.DictReader(open(p))); print(p, 'last_cost=$', rows[-1]['cost_usd_running'])" "$f"
> done
> cat artifacts/baseline_comparison.md
> ```
>
> Hand back the contents of `artifacts/baseline_comparison.md` plus the
> total cost reported for each of the three modes. Do not run anything else.

---

## Why this configuration and not the others

| Option | Calls | Estimated cost | Recommendation |
|---|---|---|---|
| Clean release benchmark, 6,915 × 3 | 20,745 | about $5.82 after prompt-audit estimate | **Recommended.** Document rows were regenerated against current seed docs; see `artifacts/baseline_readiness_audit_20260519_clean.md`. |
| Paper-overlap release subset, 1,345 × 3 | 4,035 | about $1.5–2 | Diagnostic only. Uses release rows whose `(id, domain)` keys also occur in the paper split; do not attach release gold to archived paper queries by key alone. |
| Paper-split, 3,429 × 3 | 10,287 | not recommended | **Do not use for scored external baselines.** The archived paper rows mostly lack byte-identical release queries; only exact query matches should be scored. |
| Stratified ~3,500 from release | 10,500 | $3–4 | Reasonable middle option if budget is the bottleneck. Not implemented; would need `--stratify-by type,domain` flag. |
| Strict mixed preflight, 60 × 3 | 180 | about $0.052 total observed | Passed at 0.8833 for all three baseline modes. |

Each run is **resumable** — if Codex aborts mid-run, restart the same
command and it will skip completed (id, domain) pairs.

## What is being addressed by this single run

1. **Reviewer concern: "no external LLM baseline."** Three published external
   methods (long-context GPT-4o, LINC, Logic-LM) on the same release-set
   queries that have gold answers.
2. **Reviewer concern: "synthetic-benchmark dominance."** Diagnostic McNemar
   can be reported on paper-overlap `(id, domain)` keys, but the safest scored
   external-baseline claim is on internally consistent release rows.
3. **Reviewer concern: "p = $1.7 \times 10^{-80}$ looks performative."**
   `scripts/format_paper_stats.py` now caps reported p-values at
   $p < 10^{-10}$ and emits Wilson 95 % CIs in paper-ready format.

After the run, drop `artifacts/baseline_comparison.md` into Section 5 of the
paper and cite `docs/synthetic_benchmark_data_card.md` from the limitations
section to defuse the leakage objection.
