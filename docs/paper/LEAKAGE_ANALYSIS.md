# Leakage analysis

The verified subset removes 645 of 6,915 rows (9.3%) using evidence-only rules that do not inspect any system prediction.

## Before and after objective filtering

| system                          |   unfiltered_n |   unfiltered_accuracy |   verified_n |   verified_accuracy |   delta |
|:--------------------------------|---------------:|----------------------:|-------------:|--------------------:|--------:|
| COMPASS                         |           6915 |                0.9721 |         6270 |              0.9998 |  0.0278 |
| GPT-4o-mini + retrieved context |           6915 |                0.9424 |         6270 |              0.9657 |  0.0233 |
| LINC-style prompted             |           6915 |                0.9202 |         6270 |              0.9488 |  0.0286 |
| Logic-LM-style prompted         |           6915 |                0.9503 |         6270 |              0.9699 |  0.0196 |

## Validator-neutrality audit

A deterministic, mode-balanced sample of 100 baseline errors is exported to `artifacts/phase_c_20260714/validator_audit_sample.csv` (seed 20260714).
Current-validator rescoring disagreed with the archived label on 0 rows. Manual answer-versus-question review found 12/100 false negatives: the answer supplied the requested core value, but the exact-span validator required an additional qualifier or adjacent field. Every audit decision and note is included in the CSV.

## Generator/runtime separation

Question generation and evaluated symbolic execution use separate Python modules. They share ontology and seed-document data, which is intentional, but no generator module is imported by the runtime rule service and no runtime rule module is imported by the generators.
Hashes and the exact file inventory are in `artifacts/phase_c_20260714/generator_runtime_codepaths.json`.

## Draft for Section 7: Leakage considerations

We report both the unfiltered release-clean benchmark and an objectively verified subset. The filtering rules inspect only source support, ambiguity, and gold-answer atomicity; they never inspect model outputs. Question construction and runtime symbolic inference are implemented in separate modules. They necessarily share the released ontologies and seed documents, but not generator templates or executable rule code. We additionally release a fixed, mode-balanced sample of 100 baseline errors for validator-neutrality review.
