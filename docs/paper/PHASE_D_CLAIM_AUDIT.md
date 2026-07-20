# Phase D manuscript claim audit

Audit date: 2026-07-20

This audit checks the edited manuscript against retained code and
per-instance artifacts. It does not alter any prediction or gold-label file.

## Corrections made in the manuscript

- Memory is session-scoped text retrieval. It does not provide product
  isolation, provenance validation, immutable updates, or correction history.
- Retrieval combines standardized TF--IDF and dense E5 scores with equal
  weight. The runtime does not add reranking or save an immutable full prompt.
- Citation wrapping enforces an identifier format. It does not prove that each
  generated claim follows from the cited passage.
- The symbolic service expands the static graph with OWL 2 RL and then applies
  each enabled SPARQL rule once. It does not run those custom rules to a fixed
  point. Saved traces retain derived facts, but not complete premises and exact
  rule identifiers.
- Runtime confidence is heuristic. The retained `confidence_cal` value equals
  `confidence_raw` on every pooled row, and no fitted calibration map is
  recoverable. Calibration metrics and threshold sweeps therefore evaluate
  the stored raw scores. The online endpoint does not load a calibrator or
  enforce a shared threshold.
- The 6,270-row verified set is a template-generated, DPP-like benchmark over
  the released ontology and curated seed records. It is not a natural-query or
  regulatory DPP corpus.
- The verified COMPASS file contains 5,265 symbolic outputs, 1,001
  deterministic field extractions, and four LLM outputs. The four LLM traces
  retain only the `gpt-4o-mini` alias. Their dated snapshot and temperature are
  missing from the top-level rows.
- `AUTO_COMPOSE` attempted an LLM call on all 3,000 compositional rows.
  Deterministic checks changed or supplemented 198 final answers; 20 final
  traces discard the LLM answer. Its 1.0000 score belongs to the full pipeline.
- The archived 7.96% figure describes rows marked as symbolic-path cases. The
  retained table does not identify exact SPARQL rule executions or premises,
  so the manuscript no longer calls this rule coverage or rule precision.
- The Open Food Facts experiment is a separate row-context evaluation. It
  receives the relevant row and the names of required evidence fields. It does
  not test end-to-end retrieval or ontology transfer. A text-based rule applied
  after generation during rescoring reclassifies 702 answers containing
  missing-evidence language. The raw JSON flag marks 385 abstentions; the
  effective rule marks 1,087. Its outputs
  retain only the `gpt-4o-mini` alias.
- Simple identifier-and-number normalization yields 289 question forms in the
  verified benchmark and 497 in the compositional benchmark. The manuscript
  therefore treats Wilson intervals and row-level McNemar tests as descriptive
  of benchmark items, not as population inference for natural queries.
- The 3,429-row pooled run is historical. It cannot be regenerated in full
  because 2,084 retained rows lack recoverable gold labels.

## Main evidence retained

- Frozen verified release: 6,270 shared rows and exact paired tests.
- Frozen controlled composition benchmark: 3,000 shared rows and exact paired
  tests.
- Open Food Facts row-context test: 4,021 rows with answerability labels.
- Archived pooled ablation: retained only as secondary evidence, with its
  reconstruction and confidence limitations stated in the paper.

## Build status

The edited source compiles with the official LNCS class. The current PDF is 13
pages including references. Author names, affiliations, emails, and ORCIDs are
still required before submission.
