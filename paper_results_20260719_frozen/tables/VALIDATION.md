# Frozen Phase B validation

Status: **PASSED**.

Top-level model fields in the external-baseline and compositional LLM rows
resolve to `gpt-4o-mini-2024-07-18`. The validator did not inspect nested
answer traces in the verified COMPASS file. A later audit found four LLM paths
there with only the mutable `gpt-4o-mini` alias. Row parity, scores, and paired
tests below remain valid; full snapshot parity does not.

These are item-level tests on template-generated benchmarks. Normalizing
identifiers and numbers yields 289 question forms in the verified set and 497
in the compositional set. The p-values describe the frozen benchmark rows and
are not population-level inference for independent natural queries.

All 3,000 `AUTO_COMPOSE` rows attempted an LLM call. Deterministic output
checks then changed or supplemented 198 answers; 20 final traces discard the
LLM answer. The compositional score is for the full pipeline.

| Benchmark | Reference | Comparison | n | Reference only | Comparison only | Exact p |
|---|---|---|---:|---:|---:|---:|
| verified_release_6270 | COMPASS | GPT4O_LONGCTX | 6270 | 212 | 1 | 3.25e-62 |
| verified_release_6270 | COMPASS | LINC | 6270 | 306 | 1 | 2.36e-90 |
| verified_release_6270 | COMPASS | LOGIC_LM | 6270 | 187 | 1 | 9.64e-55 |
| compositional_3000 | AUTO_COMPOSE | GPT4O_LONGCTX | 3000 | 122 | 0 | 3.76e-37 |
| compositional_3000 | AUTO_COMPOSE | LINC | 3000 | 328 | 0 | 3.66e-99 |
| compositional_3000 | AUTO_COMPOSE | LOGIC_LM | 3000 | 178 | 0 | 5.22e-54 |
