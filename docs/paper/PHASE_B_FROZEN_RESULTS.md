# Phase B frozen-results record

Finalization date: 2026-07-20

## Evaluation identity

- Release tag: `ijckg2026-phase-b-frozen-20260719`
- Final harness commit: `a4e623402a37e2276d2e680633144d5c0564966e`
- Requested OpenAI model alias: `gpt-4o-mini`
- API-resolved model recorded on every final LLM row:
  `gpt-4o-mini-2024-07-18`
- Direct requests to the dated name returned HTTP 403 for this account. The
  supported alias was therefore used and the API-resolved snapshot was stored
  from every response.

The external request manifest records commit
`b3da0efcb33025a739a784324b844af10ebe8a35`; AUTO_COMPOSE records final harness
commit `a4e62340`. The intervening changes add result packaging, request pacing,
model-response tracing, resume-compatible CSV materialization, and isolated
benchmark memory cleanup. They do not change the frozen prompts, benchmark
rows, scoring rules, or external request bodies. This distinction must remain
visible in the release rather than being described as identical execution
commit strings.

## Final results

| Benchmark | System | Correct | Accuracy |
|---|---|---:|---:|
| Verified release | COMPASS | 6,269 / 6,270 | 0.9998 |
| Verified release | GPT-4o LongCtx | 6,058 / 6,270 | 0.9662 |
| Verified release | LINC | 5,964 / 6,270 | 0.9512 |
| Verified release | Logic-LM | 6,083 / 6,270 | 0.9702 |
| Compositional | AUTO_COMPOSE | 3,000 / 3,000 | 1.0000 |
| Compositional | GPT-4o LongCtx | 2,878 / 3,000 | 0.9593 |
| Compositional | LINC | 2,672 / 3,000 | 0.8907 |
| Compositional | Logic-LM | 2,822 / 3,000 | 0.9407 |

All expected mode/row keys are present and unique. Exact paired McNemar results
and machine-readable validation are in
`paper_results_20260719_frozen/tables/`.

## Scope limitation

The historical 3,429-row pooled suite was not regenerated because 2,084 rows
in the retained split lack recoverable gold labels. See
`docs/paper/POOLED_RERUN_LIMITATION.md`. No labels or results were fabricated.
