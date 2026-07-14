# Pooled-suite rerun limitation

## Finding

The historical pooled suite cannot be rescored honestly for all 3,429 rows
from the retained repository artifacts.

- `artifacts/paper_split.json` contains 3,429 `(id, domain, query)` rows and the
  historical `success_adaptiverag` label, but it does not contain
  `expected_contains` (gold answers).
- Only 3 of those queries are byte-identical to current rows for which gold can
  be attached from `artifacts/release_benchmark.json`.
- `artifacts/paper_split_with_gold.json` retains gold for 1,345 rows, not all
  3,429.
- The historical pooled evaluation CSV and trace retain predictions and success
  labels, but not the missing gold answers. Using a historical model answer as
  replacement gold would introduce outcome leakage and is not acceptable.
- The checked-in `tests/{battery,lexmark,viessmann}/tests.jsonl` files contain
  only 3, 2, and 2 smoke rows. The 8,093-row release test files are a different
  benchmark: only one query per domain is byte-identical to the pooled split.

## Consequence for Phase B / T4

The 6,270-row verified release and 3,000-row compositional benchmark can be
rerun and rescored from complete retained gold. The pooled 3,429 table must
remain explicitly historical unless the original gold-bearing suite is
recovered from an external archive. It must not be described as regenerated
from the final tag.

Recommended paper wording:

> We reran the verified-release and compositional evaluations from the frozen
> tag and manifest. The earlier 3,429-query pooled ablation remains a historical
> artifact because its archived split retained per-row outcomes but not the gold
> answers required for independent rescoring; we therefore do not present it as
> regenerated evidence.

For the strongest defensible submission, make the newly frozen verified and
compositional studies the primary evidence and move the historical pooled
ablation to an explicitly qualified appendix or limitations paragraph.

## Recovery condition

T4's pooled acceptance criterion can be closed only if an external archive
supplies all 3,429 original rows with gold answers. Before use, that file must
match `paper_split.json` on every `(id, domain, query)` key and be hashed into
`release/FREEZE_MANIFEST.json`.
