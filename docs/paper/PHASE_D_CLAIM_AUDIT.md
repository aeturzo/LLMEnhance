# Phase D manuscript claim audit

Audit date: 2026-07-22

This audit checks the edited manuscript against retained code and
per-instance artifacts. It does not alter any prediction or gold-label file.

## Corrections made in the manuscript

- Memory is session-scoped text retrieval. It does not provide product
  isolation, provenance validation, immutable updates, or correction history.
- The general retriever can combine standardized TF--IDF and dense E5 scores
  with equal weight. Both frozen benchmark runners set `use_dense=False` and
  therefore use TF--IDF alone. The runtime does not add reranking or save an
  immutable full prompt.
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
- The 5,265 symbolic outputs are mainly graph access: 2,649 class-membership
  lookups, 2,594 knowledge-base recalls, 14 component lookups, and eight rows
  carrying both compliance and required-step labels. They must not be
  described as 5,265 disjoint rule firings.
- Every verified COMPASS row stores confidence 0.5. The composition package
  has no common confidence field. Selective-risk and calibration claims are
  therefore limited to the archived pooled run and Open Food Facts.
- `AUTO_COMPOSE` attempted an LLM call on all 3,000 compositional rows.
  Deterministic checks changed or supplemented 198 final answers; 20 final
  traces discard the LLM answer. Its 1.0000 score belongs to the full pipeline.
- The composition harness supplies benchmark-declared evidence sections. The
  external baseline configurations mark this as oracle context and use gold
  source groups for selection. This study tests composition given relevant
  evidence, not end-to-end retrieval or source routing.
- The archived 7.96% figure describes rows marked as symbolic-path cases. The
  retained table does not identify exact SPARQL rule executions or premises,
  so the manuscript no longer calls this rule coverage or rule precision.
- The Open Food Facts experiment is a separate row-context evaluation. It
  gives the answerer the relevant row and names of required evidence fields.
  It does not test end-to-end retrieval or ontology transfer. A text-based rule
  applied after generation during rescoring reclassifies 702 answers containing
  missing-evidence language. The raw JSON flag marks 385 abstentions; the
  effective rule marks 1,087. Its outputs
  retain only the `gpt-4o-mini` alias.
- Simple identifier-and-number normalization yields 289 question forms in the
  verified benchmark and 497 in the compositional benchmark. The manuscript
  therefore treats Wilson intervals and row-level McNemar tests as descriptive
  of benchmark items, not as population inference for natural queries.
- The 3,429-row pooled run is historical. It cannot be regenerated in full
  because 2,084 retained rows lack recoverable gold labels.
- The two frozen DPP benchmarks contain supported, answerable questions and do
  not test native abstention outcomes. Selective thresholds are applied
  offline; Open Food Facts abstention counts use later text rescoring.
- Trace fields vary across releases. Verified rows retain document identifiers
  or graph evidence and a constant score; composition rows retain LLM path and
  model metadata but no common score and often no selected evidence IDs.
- The architecture figure now reflects the retained implementation: TF--IDF
  retrieval with E5 marked as disabled in the frozen runs, same-session text
  memory, static RDF with OWL 2 RL and one SPARQL-rule pass, the two frozen
  answer paths, and conditional offline confidence analysis. It does not show
  a reranker, durable verified memory, complete proofs, a fitted calibrator, or
  a deployed threshold.
- The earlier Open Food Facts AURC values were removed. They do not match the
  retained tie-aware calculation and have no canonical frozen summary. The
  row-level decision, answer, coverage, ECE, and abstention results reproduce.
- The filtering paragraph now uses the same-output Phase C audit. It does not
  combine an unfiltered run with the later Phase B baseline rerun.

## Main evidence retained

- Frozen verified release: 6,270 shared rows and exact paired tests.
- Frozen controlled composition benchmark: 3,000 shared rows and exact paired
  tests.
- Open Food Facts row-context test: 4,021 rows with answerability labels.
- Archived pooled ablation: retained only as secondary evidence, with its
  reconstruction and confidence limitations stated in the paper.

## Build status

The edited source compiles with the official LNCS class. The current PDF is 15
pages including references and has no overfull boxes. Author names,
affiliations, emails, and ORCIDs remain author-supplied fields. Recheck the
page count if they are added to a non-anonymous version.
