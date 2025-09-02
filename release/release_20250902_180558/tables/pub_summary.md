# Publication Checks Summary


## Counts per domain × type


| domain | logic | open | recall |
| --- | --- | --- | --- |
| battery | 696.0000 | 269.0000 | 1143.0000 |
| lexmark | 1118.0000 | 253.0000 | 1380.0000 |
| viessmann | 871.0000 | 523.0000 | 1840.0000 |


## Accuracy (overall)


| mode | accuracy |
| --- | --- |
| MEMSYM | 0.0629 |
| ADAPTIVERAG | 0.0628 |
| MEM | 0.0628 |
| ROUTER | 0.0628 |
| RL | 0.0346 |
| BASE | 0.0014 |
| SYM | 0.0010 |


## Accuracy by type


| mode | logic | open | recall |
| --- | --- | --- | --- |
| ADAPTIVERAG | 0.1296 | 0.0067 | 0.0351 |
| BASE | 0.0041 | 0.0000 | 0.0000 |
| MEM | 0.1296 | 0.0067 | 0.0351 |
| MEMSYM | 0.1296 | 0.0067 | 0.0353 |
| RL | 0.0756 | 0.0067 | 0.0160 |
| ROUTER | 0.1296 | 0.0067 | 0.0351 |
| SYM | 0.0000 | 0.0038 | 0.0009 |


## McNemar vs BASE (exact, two-sided)


| mode | b | c | n_paired | p_value |
| --- | --- | --- | --- | --- |
| ADAPTIVERAG | 11 | 508 | 519 | 0.0000 |
| MEM | 11 | 508 | 519 | 0.0000 |
| MEMSYM | 11 | 509 | 520 | 0.0000 |
| RL | 11 | 280 | 291 | 0.0000 |
| ROUTER | 11 | 508 | 519 | 0.0000 |
| SYM | 11 | 8 | 19 | 0.6476 |