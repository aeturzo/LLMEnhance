# AUTO_COMPOSE 3000 v1 Data Preparation and Run Plan

## Objective

Evaluate the intended LLMEnhance architecture:

1. Interpret the user question and infer route/source needs.
2. Gather relevant context from memory, base documents/search, and symbolic KG reasoning.
3. Send the composed context to the LLM as the final answerer.
4. Compare against two strong external baselines on the same 3,000-row subset.

This is different from the already completed `ADAPTIVERAG` verified run, where many structured rows were answered directly by symbolic or deterministic extractors. The previous result is preserved and should not be overwritten.

## Locked Previous Result

Do not modify these files:

- Snapshot directory: `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/locked_results/verified_v1_20260521_173612_CEST`
- Snapshot archive: `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/locked_results/verified_v1_20260521_173612_CEST.tar.gz`
- Checksum file: `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/locked_results/verified_v1_20260521_173612_CEST/SHA256SUMS.txt`

That locked result contains the 6,270-row verified `ADAPTIVERAG` comparison and the filtered external baseline outputs.

## Prepared 3,000-Row Subset

Prepared files:

- Data: `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/auto_compose_v1/auto_compose_3000_v1.json`
- Filter IDs: `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/auto_compose_v1/auto_compose_3000_v1_ids.json`
- Counts: `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/auto_compose_v1/auto_compose_3000_v1_counts.csv`
- Data report: `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/auto_compose_v1/auto_compose_3000_v1_report.md`

Parent dataset:

- `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/release_clean_verified_v1.json`

The subset is stratified proportionally by `domain` and `type` from the 6,270-row verified parent. Every row has a materialized `product` / `solver_product` field so `AUTO_COMPOSE` can call symbolic reasoning when needed.

Selected strata:

| Domain | Type | Parent rows | Selected rows |
|---|---:|---:|---:|
| battery | logic | 684 | 327 |
| battery | open | 205 | 98 |
| battery | recall | 380 | 182 |
| lexmark | logic | 1105 | 529 |
| lexmark | open | 213 | 102 |
| lexmark | recall | 1102 | 527 |
| viessmann | logic | 860 | 411 |
| viessmann | open | 453 | 217 |
| viessmann | recall | 1268 | 607 |

Total: 3,000 rows.

## Baselines

Run only two competing baselines to control cost:

1. `GPT4O_LONGCTX`: strong long-context LLM with retrieved domain context.
2. `LOGIC_LM`: strongest previous external baseline and closest logic-dispatch competitor.

Do not run `LINC` unless we explicitly decide to add a third baseline later.

## Required Claude Work

### 1. Validate Data

Check:

- 3,000 rows.
- 3,000 unique `id|domain` keys.
- No missing `query`, `domain`, `type`, `expected_contains`, `product`, or `session`.
- All 3,000 keys exist in `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/release_benchmark_clean_docs.json`, so `run_baselines.py --benchmark release-clean --filter-ids ...` evaluates the same rows.

### 2. Add an AUTO_COMPOSE Evaluation Harness

Create or update:

- `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/scripts/run_auto_compose_subset.py`

The harness must:

- Load `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/auto_compose_v1/auto_compose_3000_v1.json`.
- Call `/solve_auto` or `solve_auto_query`, not `/solve`.
- Pass `query`, `product`, `domain`, and `session` from each row.
- Configure the same clean corpus setup used by `scripts/run_adaptiverag_clean.py`.
- Be resumable by skipping completed `(id, domain)` rows.
- Support `--limit`, `--sample-mixed`, `--sleep`, `--budget-usd`, and `--audit-only`.
- Write CSV with the existing baseline-compatible columns plus these extra columns:
  - `answer_trace`
  - `llm_used`
  - `provider`
  - `model`
  - `api`
  - `included_source_types`
  - `included_passage_ids`
  - `fallback_path`
  - `composed_context_chars`

Primary output:

- `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/auto_compose_v1/eval_AUTO_COMPOSE_3000_v1.csv`

### 3. Enforce Final LLM Answering

For this experiment, the system claim requires composed context to reach the LLM final answerer.

Add a strict mode if needed:

- Environment variable: `AUTO_COMPOSE_DISABLE_DIRECT_FALLBACK=1`
- CLI flag: `--strict-llm-final`

In strict mode:

- The harness must count a row as invalid or failed if `answer_trace.llm_used` is not `true`.
- Deterministic fallbacks such as `memory_direct`, `symbolic_direct`, and `memory_symbolic_direct` must not replace the primary final answer.
- The run report must include final `llm_used` count and fallback count.

It is acceptable for symbolic reasoning to produce a context passage. It is not acceptable for symbolic reasoning to be the final answerer in this strict architecture test.

### 4. No-Cost Context Audit

Before paid calls, run a no-cost audit over all 3,000 rows or at least a stratified 300 rows.

Acceptance gates:

- Missing product rows: 0.
- Open and recall rows with gold in composed context: at least 0.97.
- Logic rows with symbolic passage included or symbolic evidence available: at least 0.95.
- No domain/type stratum with zero context coverage.

Suggested output:

- `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/auto_compose_v1/auto_compose_context_audit_3000_v1.md`
- `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/auto_compose_v1/auto_compose_context_audit_3000_v1.csv`

### 5. Paid Smoke Test

After the no-cost audit passes:

```bash
cd /Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain
source temp_env.sh
AUTO_COMPOSE_DISABLE_DIRECT_FALLBACK=1 \
python scripts/run_auto_compose_subset.py \
  --data artifacts/auto_compose_v1/auto_compose_3000_v1.json \
  --sample-mixed 90 \
  --strict-llm-final \
  --budget-usd 0.30 \
  --sleep 0.2 \
  --out artifacts/auto_compose_v1/eval_AUTO_COMPOSE_3000_v1_smoke90.csv
```

Acceptance gates:

- 90 rows completed.
- `llm_used = 90`.
- API error rows: 0.
- Accuracy should be at least 0.90. If it is below 0.85, stop and debug context construction before full run.
- No domain/type stratum should be at 0 accuracy.

### 6. Full AUTO_COMPOSE Run

Only after the smoke passes:

```bash
cd /Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain
source temp_env.sh
AUTO_COMPOSE_DISABLE_DIRECT_FALLBACK=1 \
python scripts/run_auto_compose_subset.py \
  --data artifacts/auto_compose_v1/auto_compose_3000_v1.json \
  --strict-llm-final \
  --budget-usd 2.50 \
  --sleep 0.2 \
  --out artifacts/auto_compose_v1/eval_AUTO_COMPOSE_3000_v1.csv
```

### 7. Two Baseline Runs on the Same 3,000 Rows

```bash
cd /Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain
source temp_env.sh

python scripts/run_baselines.py --mode GPT4O_LONGCTX --benchmark release-clean \
  --filter-ids artifacts/auto_compose_v1/auto_compose_3000_v1_ids.json \
  --out artifacts/auto_compose_v1/eval_GPT4O_LONGCTX_3000_v1.csv \
  --budget-usd 2.50 --sleep 0.2

python scripts/run_baselines.py --mode LOGIC_LM --benchmark release-clean \
  --filter-ids artifacts/auto_compose_v1/auto_compose_3000_v1_ids.json \
  --out artifacts/auto_compose_v1/eval_LOGIC_LM_3000_v1.csv \
  --budget-usd 2.50 --sleep 0.2
```

Expected total calls:

- `AUTO_COMPOSE`: 3,000
- `GPT4O_LONGCTX`: 3,000
- `LOGIC_LM`: 3,000
- Total: 9,000 calls

Expected cost on `gpt-4o-mini`: approximately 3 to 6 USD total, depending on final context size.

### 8. Aggregation

Create:

- `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/scripts/aggregate_auto_compose_v1.py`

Outputs:

- `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/auto_compose_v1/auto_compose_comparison_3000_v1.md`
- `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/docs/paper/tables/auto_compose_comparison_3000_v1.tex`

The table must include:

- `n`
- accuracy
- Wilson 95 percent CI
- McNemar versus `AUTO_COMPOSE`
- `llm_used` count and rate for `AUTO_COMPOSE`
- fallback count and rate for `AUTO_COMPOSE`
- average latency if available

## Paper Claim This Test Can Support

If the run succeeds, the defensible claim is:

> On a 3,000-row verified DPP benchmark, the intended LLMEnhance architecture composes context from symbolic KG reasoning, document retrieval, and memory/search evidence, then uses an LLM final answerer. Under matched evaluation, this architecture outperforms strong long-context and logic-dispatch LLM baselines while preserving explicit evidence traces.

Do not use this experiment to claim that deterministic symbolic execution alone is superior. That is the separate locked `ADAPTIVERAG` verified result.

