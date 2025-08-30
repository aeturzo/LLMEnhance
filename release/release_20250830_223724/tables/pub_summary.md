# Publication Checks Summary


## Counts per domain × type


| domain | logic | open | recall |
| --- | --- | --- | --- |
| battery | 416.0000 | 5.0000 | 372.0000 |
| lexmark | 308.0000 | 3.0000 | 273.0000 |
| viessmann | 231.0000 | 3.0000 | 274.0000 |


## Accuracy (overall)


| mode | accuracy |
| --- | --- |
| MEMSYM | 0.1792 |
| ADAPTIVERAG | 0.1787 |
| MEM | 0.1787 |
| ROUTER | 0.1787 |
| RL | 0.1214 |
| BASE | 0.0058 |
| SYM | 0.0042 |


## Accuracy by type


| mode | logic | open | recall |
| --- | --- | --- | --- |
| ADAPTIVERAG | 0.2723 | 0.5833 | 0.0762 |
| BASE | 0.0115 | 0.0000 | 0.0000 |
| MEM | 0.2723 | 0.5833 | 0.0762 |
| MEMSYM | 0.2723 | 0.5833 | 0.0773 |
| RL | 0.1832 | 0.5833 | 0.0511 |
| ROUTER | 0.2723 | 0.5833 | 0.0762 |
| SYM | 0.0000 | 0.3333 | 0.0044 |


## McNemar vs BASE (exact, two-sided)


| mode | b | c | n_paired | p_value |
| --- | --- | --- | --- | --- |
| ADAPTIVERAG | 11 | 337 | 348 | 0.0000 |
| MEM | 11 | 337 | 348 | 0.0000 |
| MEMSYM | 11 | 338 | 349 | 0.0000 |
| RL | 11 | 229 | 240 | 0.0000 |
| ROUTER | 11 | 337 | 348 | 0.0000 |
| SYM | 11 | 8 | 19 | 0.6476 |