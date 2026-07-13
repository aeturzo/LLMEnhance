# Frozen rerun plan

Phase B will run only after `scripts/preflight_frozen_rerun.py` passes with
`--require-clean --require-api-key` from the tagged Phase A commit.

Safety rules:

1. Never write to an existing `artifacts/`, `paper_results_*`, memory, search,
   or locked-results path.
2. Use a new timestamped output directory and refuse to run if it exists.
3. Set `MEMORY_META_PATH` and all generated corpus/index paths inside that new
   directory.
4. Use the explicit `gpt-4o-mini-2024-07-18` snapshot for every model call.
5. Retain full answers and record model, temperature, token limits, and harness
   commit per row.
6. Apply a total API budget cap of USD 12. Stop before the first paid call if
   the estimated cost exceeds the cap.
7. Run a small mixed smoke sample and parity check before full execution.
8. Fit/select routing and calibration from development artifacts only. The
   current router source is `artifacts/dev/eval_joined_20260201_205635.csv`, and
   the current calibrator manifest also identifies an `artifacts/dev` input.

Planned evaluations:

| Benchmark | Rows | Systems |
|---|---:|---|
| Pooled reliability suite | 3,429 | COMPASS and internal ablations |
| Verified release | 6,270 | COMPASS, GPT-4o-mini + retrieved context, LINC-style prompted, Logic-LM-style prompted |
| Compositional benchmark | 3,000 | COMPASS and the same three prompted external baselines |

The historical labels “oracle context,” “LINC,” and “Logic-LM” must not imply
faithful implementations. Release documentation and the paper will use the
qualified names shown above.
