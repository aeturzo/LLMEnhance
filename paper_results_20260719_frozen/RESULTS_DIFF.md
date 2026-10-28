# Frozen results diff

Frozen evaluation tag: `ijckg2026-phase-b-frozen-20260719`.

| benchmark             | mode          |   n_new |   accuracy_new |   n_old | accuracy_old   | accuracy_delta   |
|:----------------------|:--------------|--------:|---------------:|--------:|:---------------|:-----------------|
| compositional_3000    | AUTO_COMPOSE  |    3000 |         1      |    3000 | 1.0000         | 0.0000           |
| compositional_3000    | GPT4O_LONGCTX |    3000 |         0.9593 |    3000 | 0.9577         | 0.0017           |
| compositional_3000    | LINC          |    3000 |         0.8907 |     nan | not available  | not available    |
| compositional_3000    | LOGIC_LM      |    3000 |         0.9407 |    3000 | 0.9400         | 0.0007           |
| verified_release_6270 | COMPASS       |    6270 |         0.9998 |    6270 | 1.0000         | -0.0002          |
| verified_release_6270 | GPT4O_LONGCTX |    6270 |         0.9662 |    6270 | 0.9657         | 0.0005           |
| verified_release_6270 | LINC          |    6270 |         0.9512 |    6270 | 0.9488         | 0.0024           |
| verified_release_6270 | LOGIC_LM      |    6270 |         0.9702 |    6270 | 0.9699         | 0.0003           |

The historical 3,429-row pooled suite is excluded from this regenerated table because its retained split is missing gold for 2,084 rows; see `docs/paper/POOLED_RERUN_LIMITATION.md`.
