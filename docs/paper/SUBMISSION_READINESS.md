# IJCKG 2026 submission readiness

Initial audit: 2026-07-13. Phase D manuscript update: 2026-07-22.

The supplied draft was 17 pages. The revised LNCS source now compiles to 15
pages, including references. This is within the IJCKG 2026 limit of 12--15
pages, but it uses the full allowance. The upload build was also checked with
Springer `llncs.cls` v2.22 (2022-09-05), `splncs04.bst`, and Tectonic 0.16.9.
Recheck the page count after adding any author metadata required for a
non-anonymous version.

Recommended track: **Research Track**. The current paper reports unlogged live
demonstrations but no quantitative target-user study. The IJCKG In-Use Track
asks for convincing evidence of use by the target group, preferably outside
the development team. Submit there only if the authors can document that
evidence before the deadline.

## Current layout and content choices

The current version restores the explanatory material requested after the
first Phase D reduction. Repetition was shortened to keep that material within
the page limit.

| Choice | Status | Rationale |
|---|---|---|
| Detailed problem formulation | Restored | Defines answer/abstain thresholding, coverage, risk, AURC, ECE, Brier score, Wilson intervals, and paired McNemar tests in plain language. |
| Architecture figure | Replaced | The new vector figure shows the two frozen paths, marks E5 as disabled in those runs, and separates offline analysis from runtime. It removes unsupported reranking, durable memory, exact proof, and online-calibration claims. |
| Related work and repeated interpretation | Compressed | Makes room for the formulation and figure without dropping the main evidence. |
| Dataset counts | Kept in prose | Preserves the verified, composition, Open Food Facts, and historical counts without another float. |
| Historical results table | Merged and restored | One compact table retains overall, AURC, and query-type values, including the negative ablation evidence. |
| Open Food Facts AURC | Removed | The old values do not reproduce from a retained canonical summary; decision accuracy, answer accuracy, coverage, ECE, and abstention results remain. |

The verified-release, compositional, Open Food Facts, and merged historical
tables remain in the paper. The corpus counts remain in the text. The
architecture source and vector PDF are stored under `figures/`.

## Blocking author-supplied fields

- Author names, affiliations, emails, and ORCIDs.
- Confirm whether a Zenodo DOI will supplement the public repository URL.
- Confirm any partner acknowledgement text required by the consortium.
- Truthfully documented workbench demonstrations: dates/session count,
  approximate practitioner count or roles, and the source record supporting
  each number. No historical query or abstention counts are recoverable from
  the current logs.

## Technical corrections already prepared

- The memory lifecycle paragraph now describes session-scoped timestamped text
  memory and no longer claims validation, product scoping, provenance, or
  immutable supersession that the implementation does not provide.
- External baselines are now named as GPT-4o-mini with retrieval-selected
  context, Logic-LM-style prompted, and LINC-style prompted.
- `release/baseline_configs.json` exports exact prompts and known configuration
  gaps without inventing an historical snapshot.
- `scripts/check_baseline_parity.py` verifies all four systems share exactly
  6,270 row keys and benchmark metadata. Twenty historical LINC rows remain
  independently unverifiable because their stored answer was truncated at 500
  characters; the Phase B harness now preserves full answers.

## Remaining paper changes by phase

Phase B supplied the frozen verified and compositional results. Phase C added
leakage and calibration checks plus the publication-name risk--coverage
figure. Phase D reduced and rewrote the paper, compiled it, and completed a
claim-to-artifact audit. Author metadata and any consortium-specific partner
acknowledgement remain author-supplied.
