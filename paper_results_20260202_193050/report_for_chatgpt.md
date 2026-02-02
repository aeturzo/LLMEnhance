# LLMEnhance Paper Update Report (for ChatGPT)

## Snapshot
- Date: 2026-02-02
- Results package: paper_results_20260202_193050/
- Pooled eval: artifacts/eval_joined_pooled_20260202_192420.csv
- Pooled trace: artifacts/trace_20260202_192420.jsonl
- Publication checks: PASSED

## Dataset size
- Total questions: 3,429
- By domain: battery 1,070; lexmark 936; viessmann 1,423
- By type: logic 295; open 2,297; recall 837

## Main results (overall accuracy)
- ADAPTIVERAG: 0.9749 (best)
- RAG_BASE: 0.8973
- RL: 0.8965
- MEMSYM: 0.5777
- MEM: 0.5001
- SYM_ONLY: 0.0965
- ROUTER: 0.0869
- BASE: 0.0169

## By type (key patterns)
- Logic: SYM_ONLY 0.9051, ROUTER 0.7932, ADAPTIVERAG 0.7085
- Open/Recall: ADAPTIVERAG 1.000/1.000, RAG_BASE 1.000/0.9283, RL 1.000/0.9271

## Symbolic coverage
- Coverage ~7.96% (273/3429) with precision 1.0 when fired

## Calibration
- ECE mean ~0.3278 (confidence not well calibrated; mention as limitation)

## Notes on router
- Router retrain + re-eval did not improve overall; router remains weak on open/recall.
- Treat router/symbolic as specialized logic modules; main contribution is ADAPTIVERAG.

## What to update in the paper
1) Abstract/Contributions:
   - Emphasize hybrid ADAPTIVERAG achieves ~97.5% overall accuracy across 3 domains.
   - Note consistent gains over RAG_BASE (~+7.8 points).

2) Results section:
   - Use tables from paper_results_20260202_193050/tables:
     - acc_overall.csv, acc_by_type.csv, acc_ci.csv
     - mcnemar.csv (significance)
     - aurc.csv, risk_coverage.csv (selective risk)
     - sym_coverage.csv (symbolic coverage)
   - Include figures from paper_results_20260202_193050/figures:
     - risk_coverage.png, reliability.png, memory_scaling.png

3) Limitations:
   - Calibration (ECE ~0.33) not ideal.
   - Router not good on open/recall; best used as a logic-specialized tool.

4) Methods/Setup:
   - Datasets: 3 domains, 3 types (logic/open/recall).
   - Pooled evaluation across domains.

## Files to cite in paper (from package)
- tables/acc_overall.csv
- tables/acc_by_type.csv
- tables/acc_ci.csv
- tables/mcnemar.csv
- tables/risk_coverage.csv
- tables/aurc.csv
- tables/sym_coverage.csv
- figures/risk_coverage.png
- figures/reliability.png
- figures/memory_scaling.png

