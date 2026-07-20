# Frozen Phase B validation

Status: **PASSED**.

All evaluated LLM rows must resolve to `gpt-4o-mini-2024-07-18`.

| Benchmark | Reference | Comparison | n | Reference only | Comparison only | Exact p |
|---|---|---|---:|---:|---:|---:|
| verified_release_6270 | COMPASS | GPT4O_LONGCTX | 6270 | 212 | 1 | 3.25e-62 |
| verified_release_6270 | COMPASS | LINC | 6270 | 306 | 1 | 2.36e-90 |
| verified_release_6270 | COMPASS | LOGIC_LM | 6270 | 187 | 1 | 9.64e-55 |
| compositional_3000 | AUTO_COMPOSE | GPT4O_LONGCTX | 3000 | 122 | 0 | 3.76e-37 |
| compositional_3000 | AUTO_COMPOSE | LINC | 3000 | 328 | 0 | 3.66e-99 |
| compositional_3000 | AUTO_COMPOSE | LOGIC_LM | 3000 | 178 | 0 | 5.22e-54 |
