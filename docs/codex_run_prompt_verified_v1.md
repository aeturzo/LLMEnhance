# Codex Run Prompt — verified-subset, ADAPTIVERAG-only, one paid run

The three external baselines (GPT4O_LONGCTX, LINC, LOGIC_LM) are **already
run**. This is the **only remaining paid run**: ADAPTIVERAG on the
objectively-verified 6,457-row subset. Estimated cost ≈ **$2–4** on gpt-4o-mini.

## What is already done (no rerun)

| Artifact | State |
|---|---|
| `artifacts/release_clean_verified_v1.json` | 6,457 verified rows (6,915 − 458 objectively excluded) |
| `artifacts/release_clean_verified_v1_ids.json` | 6,457 `id\|domain` keys |
| `artifacts/verified_v1/eval_GPT4O_LONGCTX_verified_v1.csv` | filtered, acc 0.9614 |
| `artifacts/verified_v1/eval_LINC_verified_v1.csv` | filtered, acc 0.9427 |
| `artifacts/verified_v1/eval_LOGIC_LM_verified_v1.csv` | filtered, acc 0.9664 |
| `artifacts/verified_v1/eval_ADAPTIVERAG_verified_v1_full.csv` | **header-only placeholder** — the run below fills it |

## Pre-flight (run manually first)

```bash
cd /Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain
source temp_env.sh
test -n "$OPENAI_API_KEY" && echo "key OK" || { echo "MISSING KEY"; exit 1; }
python -c "import openai, fastapi, rdflib, owlrl; print('deps OK')"
```

## The Codex prompt

> You are running the final ADAPTIVERAG evaluation for the AEI paper. Repo
> root: `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain`. Working
> directory: that path. Run ONLY the commands below, in order. Do NOT modify
> code. Do NOT rerun GPT4O_LONGCTX, LINC, or LOGIC_LM. If any step fails its
> acceptance gate, STOP and report — do not continue to the next step.
>
> Step 1 — environment
> ```bash
> cd /Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain
> source temp_env.sh
> python -c "import openai, fastapi, rdflib, owlrl; print('deps OK')"
> ```
>
> Step 2 — no-cost retrieval audit of the ADAPTIVERAG adapter on the verified
> subset. This makes NO API calls.
> ```bash
> python scripts/run_adaptiverag_clean.py \
>   --filter-ids artifacts/release_clean_verified_v1_ids.json \
>   --sample-mixed 90 --audit-only --force-corpus \
>   --out artifacts/verified_v1/audit_dummy.csv
> ```
> Acceptance gate: overall gold-in-context rate ≥ 0.97 (the script exits 0 when
> its internal rate ≥ 0.85; additionally inspect the printed JSON — `open` and
> `recall` should be ≥ 0.97). If `logic` is lower, that is acceptable because
> KB-logic answers are derived by reasoning, not literal retrieval. If overall
> open/recall are below 0.95, STOP.
>
> Step 3 — paid 90-row smoke (~$0.05). Resumable.
> ```bash
> ADAPTIVERAG_CLEAN_MAX_PASSAGES=3 \
> python scripts/run_adaptiverag_clean.py \
>   --filter-ids artifacts/release_clean_verified_v1_ids.json \
>   --sample-mixed 90 \
>   --out artifacts/verified_v1/eval_ADAPTIVERAG_verified_v1_smoke90.csv \
>   --sleep 0.2
> ```
> Acceptance gate: printed accuracy ≥ 0.956 (= max external-baseline verified
> accuracy 0.9664 − 0.01), 0 API-error rows, no single domain at 0. If the
> smoke fails, STOP and report — do NOT start the full run.
>
> Step 4 — full ADAPTIVERAG run on all 6,457 verified rows (~$2–4). Resumable;
> the output file currently holds only a header, so this fills it.
> ```bash
> ADAPTIVERAG_CLEAN_MAX_PASSAGES=3 \
> python scripts/run_adaptiverag_clean.py \
>   --filter-ids artifacts/release_clean_verified_v1_ids.json \
>   --out artifacts/verified_v1/eval_ADAPTIVERAG_verified_v1_full.csv \
>   --sleep 0.2
> ```
> If the process is interrupted, rerun the exact same command — it skips
> completed `(id, domain)` rows.
>
> Step 5 — aggregate the paper table.
> ```bash
> python scripts/aggregate_verified_v1.py
> ```
>
> Step 6 — verify and report.
> ```bash
> wc -l artifacts/verified_v1/eval_ADAPTIVERAG_verified_v1_full.csv
> python -c "import csv; rows=list(csv.DictReader(open('artifacts/verified_v1/eval_ADAPTIVERAG_verified_v1_full.csv'))); errs=[r for r in rows if any(t in (r['answer'] or '') for t in ('RateLimit','APITimeout','APIConnection','APIStatus','InternalServer'))]; print('rows',len(rows),'api_errors',len(errs))"
> cat artifacts/verified_v1/baseline_comparison_verified_v1.md
> ```
>
> Hand back: the contents of
> `artifacts/verified_v1/baseline_comparison_verified_v1.md`, the full-run row
> count, the API-error count, and the smoke-test accuracy. Do not run anything
> else.

## Expected shape of the result

- All four modes evaluated on the SAME 6,457 verified rows.
- External baselines (already known): GPT4O_LONGCTX 0.9614, LINC 0.9427,
  LOGIC_LM 0.9664.
- ADAPTIVERAG smoke (90-row) must clear ≥ 0.956 before the full run.
- The McNemar block compares ADAPTIVERAG vs each baseline on identical rows.

## Honesty note for the paper

This is a **matched external-baseline comparison on a cleaned subset**, not a
re-run of the original paper's 3,429-row pooled benchmark. The paper text
should say: external baselines and ADAPTIVERAG were evaluated on the same
objectively-verified release-clean subset; rows with contradictory ontology
facts, non-atomic gold, or unsupported evidence were removed by the rules in
`scripts/build_release_clean_verified_subset.py` (see
`artifacts/release_clean_verified_v1_report.md`). The original Table 3
synthetic numbers remain as-is; this verified comparison is the new Section 5
external-baseline evidence.
