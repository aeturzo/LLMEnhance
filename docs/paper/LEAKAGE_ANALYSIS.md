# Leakage analysis

The verified subset removes 645 of 6,915 rows (9.3%) using evidence-only rules that do not inspect any system prediction.

## Before and after objective filtering

| system                          |   unfiltered_n |   unfiltered_accuracy |   verified_n |   verified_accuracy |   delta |
|:--------------------------------|---------------:|----------------------:|-------------:|--------------------:|--------:|
| COMPASS                         |           6915 |                0.9721 |         6270 |              0.9998 |  0.0278 |
| GPT-4o-mini + retrieved context |           6915 |                0.9424 |         6270 |              0.9657 |  0.0233 |
| LINC-style prompted             |           6915 |                0.9202 |         6270 |              0.9488 |  0.0286 |
| Logic-LM-style prompted         |           6915 |                0.9503 |         6270 |              0.9699 |  0.0196 |

These are the same-output pairs retained in
`artifacts/phase_c_20260714/leakage_before_after.csv`. The final frozen
baseline table comes from a later API rerun and has slightly different
verified scores (0.9662, 0.9512, and 0.9702). Those later values must not be
mixed with the earlier unfiltered outputs when estimating a filtering delta.
Not every source CSV for the same-output diagnostic is packaged in the frozen
branch, so the retained summary is not fully recomputable there.

## Validator-neutrality audit

A deterministic, mode-balanced sample of 100 baseline errors is exported to `artifacts/phase_c_20260714/validator_audit_sample.csv` (seed 20260714).
Current deterministic rescoring reproduced the archived outcome on all 100
rows. Manual answer-versus-question review nevertheless judged 12/100 to be
likely false negatives: the answer supplied the requested core value, but the
exact-span validator required an additional qualifier or adjacent field. Every
audit decision and note is included in the CSV.

## Generator/runtime separation

Question generation and evaluated symbolic execution use separate Python modules. They share ontology and seed-document data, which is intentional, but no generator module is imported by the runtime rule service and no runtime rule module is imported by the generators.
Hashes and the exact file inventory are in `artifacts/phase_c_20260714/generator_runtime_codepaths.json`.

## Repeated question forms

Both frozen benchmarks are template-generated and reuse source entities. A
normalizer that replaces seed IDs, product IDs, and numbers finds 289 question
forms among 6,270 verified rows and 497 among 3,000 compositional rows. The
paper therefore treats row-level Wilson intervals and McNemar tests as
descriptive benchmark statistics, not population inference for independent
natural queries.

## Draft for Section 7: Leakage considerations

We report both the unfiltered release-clean benchmark and an objectively verified subset. The filtering rules inspect only source support, ambiguity, and gold-answer atomicity; they never inspect model outputs. Question construction and runtime symbolic inference are implemented in separate modules. They necessarily share the released ontologies and seed documents, but not generator templates or executable rule code. We additionally release a fixed, mode-balanced sample of 100 baseline errors for validator-neutrality review.
