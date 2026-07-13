# Baseline configurations

This file renders `baseline_configs.json`. Unknown historical values are left unknown; proposed frozen values have not yet been run.

| System ID | Recommended paper label | Historical snapshot | Context | Fidelity |
|---|---|---|---|---|
| `GPT4O_LONGCTX` | GPT-4o-mini + retrieved context | unknown (alias only) | retrieved, 12000 chars; not oracle | prompted retrieval-context baseline; not an oracle long-context baseline |
| `LINC` | LINC-style prompted baseline | unknown (alias only) | retrieved, 12000 chars; not oracle | prompted approximation; it does not execute the published LINC theorem-prover pipeline |
| `LOGIC_LM` | Logic-LM-style prompted baseline | unknown (alias only) | retrieved, 12000 chars; not oracle | prompted approximation; it does not execute the full published Logic-LM pipeline |
| `COMPASS` | COMPASS | unknown (alias only) | 6 passages / 12000 chars | project system |

## Required paper corrections

- Replace “oracle context” with “retrieval-selected context” unless a genuinely gold-selected oracle is implemented and rerun.
- Refer to LINC and Logic-LM as prompted, style-based approximations; the harness does not run their published solver pipelines.
- Do not claim exact snapshots were exported for historical runs. The exact snapshot is unknown and the archived CSVs record no model field.
- Phase B must use the explicit `gpt-4o-mini-2024-07-18` snapshot and record it per row/manifest.

## Known gaps

- Exact historical model snapshots cannot be recovered from the verified-release CSVs.
- The compositional LINC-style path is newly added and has no historical result; it must be run in Phase B.
- The verified and compositional harnesses use different prompt templates.
- The paper's oracle-context label is not supported by the implemented evidence selector.
