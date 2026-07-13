# AUTO_COMPOSE Architecture Baseline Strategy

## Direct Answer

No, we cannot guarantee that strict `AUTO_COMPOSE` will beat `GPT4O_LONGCTX` and `LOGIC_LM` on raw accuracy before running a pilot.

The current 3,000-row subset is mostly atomic:

- one fact from a document,
- one fact from the KG,
- or one logic/type-membership answer.

That subset is useful for compatibility and statistical comparison, but it does not strongly test the intended architecture. A long-context LLM baseline can perform very well when every answer is present in one retrieved context block.

## Smoke-Test Update

Smoke results are recorded here:

- `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/auto_compose_v1/auto_compose_smoke_decision_report.md`

Observed results:

- Strict `AUTO_COMPOSE` on 30 atomic rows: 29 / 30 = 0.9667, with LLM used on 30 / 30 rows.
- Same 30 rows from already-completed baseline CSVs:
  - `GPT4O_LONGCTX`: 30 / 30 = 1.0000
  - `LOGIC_LM`: 30 / 30 = 1.0000
  - `LINC`: 29 / 30 = 0.9667
- Multi-source architecture smoke: 9 / 10 exact-scored, LLM used on 10 / 10 rows. The single failure was a gold ambiguity: the query asked for one compliance standard and the answer returned another valid supported standard.
- Same-row multi-source baseline smoke with accepted alternatives:
  - `AUTO_COMPOSE`: 10 / 10 = 1.0000
  - `GPT4O_LONGCTX`: 8 / 10 = 0.8000
  - `LOGIC_LM`: 8 / 10 = 0.8000
  - Output: `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/auto_compose_v1/eval_arch_smoke10_all3_comparison.csv`

Decision: use the architecture-focused benchmark below. Do not use the current atomic-only 3,000-row subset as the main superiority experiment.

To prove the intended architecture, the benchmark must include questions that require composed evidence from multiple sources:

- memory + symbolic KG,
- document + symbolic KG,
- memory + document,
- memory + document + symbolic KG.

## Publication-Safe Principle

Do not choose weak baselines just to force a win. That will hurt a top-tier submission.

The stronger strategy is:

1. Use prominent, fair baselines.
2. Define the architecture-relevant benchmark before paid runs.
3. Show where the architecture wins: multi-source correctness, evidence traceability, calibrated routing, and lower context/cost.
4. Include a small stronger-model stress test as a credibility check, not necessarily as the main comparison.

## Recommended Main Benchmark: `auto_compose_arch_3000_v2`

Build a new 3,000-row benchmark:

| Bucket | Rows | Purpose |
|---|---:|---|
| Atomic verified rows | 1,500 | Maintains comparability with previous release-clean benchmark |
| Document + symbolic KG | 500 | Requires retrieved product facts plus KG-derived compliance or component facts |
| Memory + symbolic KG | 400 | Requires user/session preference plus KG-derived product reasoning |
| Memory + document | 300 | Requires session memory plus product/document attribute lookup |
| Memory + document + symbolic KG | 300 | Tests full intended architecture |

Total: 3,000 rows.

The current file can remain as the atomic-only fallback:

- `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/auto_compose_v1/auto_compose_3000_v1.json`

The new architecture-focused file should be:

- `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/auto_compose_v2/auto_compose_arch_3000_v2.json`
- `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/auto_compose_v2/auto_compose_arch_3000_v2_ids.json`
- `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/auto_compose_v2/auto_compose_arch_3000_v2_report.md`

## Required Schema

Each row should include:

```json
{
  "id": "arch-msym-000001",
  "domain": "battery",
  "type": "compose",
  "subtype": "memory_symbolic",
  "query": "For ProductA, combine my preferred packaging with the compliance standard required by the product record.",
  "product": "ProductA",
  "session": "arch_s001",
  "expected_contains": ["recycled cardboard", "EN 62133-2"],
  "expected_all": ["recycled cardboard", "EN 62133-2"],
  "required_sources": ["memory", "symbolic"],
  "gold_evidence": [
    {"source": "memory", "field": "preferred_packaging", "value": "recycled cardboard"},
    {"source": "symbolic", "field": "requiresCompliance", "value": "EN 62133-2"}
  ]
}
```

For backward compatibility, `expected_contains` may be stored as a delimiter-joined string if the old scorer requires it, but the new `AUTO_COMPOSE` scorer should require all values in `expected_all`.

## Data Preparation Rules

1. Sample atomic rows from the already verified parent:
   - `/Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain/artifacts/release_clean_verified_v1.json`

2. Use only evidence that exists in project assets:
   - clean product/document rows,
   - symbolic KG facts and derived rules,
   - seeded session memory facts.

3. Do not create gold answers that are not supported by evidence.

4. Split by product/session so the same exact composed question template is not repeated with the same product in both pilot and full runs.

5. Include paraphrases, but keep them answerable:
   - direct wording,
   - typo/noisy wording,
   - compliance wording,
   - user-preference wording,
   - evidence-combination wording.

6. Add a no-cost verifier that checks every gold value appears in at least one required source before any paid calls.

## Main Baselines

Use two baselines for the 3,000-call comparison:

1. `GPT4O_MINI_LONGCTX`
   - Implemented using the current `GPT4O_LONGCTX` harness with `--model gpt-4o-mini`.
   - This is a prominent low-cost production LLM baseline and matches the intended architecture's final answerer.

2. `LOGIC_LM_GPT4O_MINI`
   - Implemented using the current `LOGIC_LM` harness with `--model gpt-4o-mini`.
   - This is the closest logic-dispatch competitor.

This is fair because the main question is whether orchestration plus memory/KG/document composition beats LLM-only or logic-dispatch use of the same base model.

## Optional Credibility Stress Test

Run a smaller 300-row or 500-row audit, not the full 3,000 rows:

1. `GPT4.1_MINI_LONGCTX`
   - Use `--model gpt-4.1-mini`.
   - This is a strong current long-context baseline with a 1M token context window.

2. Optional only if budget permits: `GPT4O_LONGCTX_FULL`
   - Use `--model gpt-4o`.
   - Report as a stronger-model stress test or upper-bound, not the main matched-budget baseline.

Do not make the paper depend on beating `gpt-4o` full on raw accuracy. If it loses, the paper can still claim better cost, evidence traceability, and architecture-specific multi-source behavior under matched-model conditions.

## Pilot Before Full Run

Run a 300-row pilot first:

- 150 atomic rows,
- 50 document + symbolic rows,
- 40 memory + symbolic rows,
- 30 memory + document rows,
- 30 memory + document + symbolic rows.

Acceptance gates:

- `AUTO_COMPOSE` strict final LLM usage: 100 percent.
- Gold values available in composed context: at least 0.97.
- `AUTO_COMPOSE` accuracy: at least 0.92.
- `AUTO_COMPOSE` should beat both main baselines by at least 1 percentage point on the multi-source subset.
- If atomic rows tie the baselines, that is acceptable; the architecture claim should be driven by multi-source rows.

## Full Run Calls

If the pilot passes:

- `AUTO_COMPOSE`: 3,000 calls.
- `GPT4O_MINI_LONGCTX`: 3,000 calls.
- `LOGIC_LM_GPT4O_MINI`: 3,000 calls.

Optional stress test:

- `GPT4.1_MINI_LONGCTX`: 300 to 500 calls.

## Paper Claim If Successful

Use this claim:

> On a pre-specified 3,000-row DPP benchmark containing both atomic and multi-source questions, LLMEnhance `AUTO_COMPOSE` outperforms matched-model long-context and logic-dispatch baselines, with the largest gains on questions requiring memory, document, and symbolic KG evidence composition.

Avoid this claim:

> LLMEnhance always beats all frontier LLMs.

That is too broad and not needed for a top-tier paper.
