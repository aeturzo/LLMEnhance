# IJCKG 2026 submission readiness

Initial audit: 2026-07-13. Phase D update: 2026-07-20.

The supplied draft was 17 pages. The overhauled LNCS source compiles to 13
pages, including references. This is within the IJCKG 2026 limit of 12--15
pages. The Phase D build used the official 2024 LNCS class and bibliography
style with Tectonic 0.16.9.

Recommended track: **Research Track**. The current paper reports unlogged live
demonstrations but no quantitative target-user study. The IJCKG In-Use Track
asks for convincing evidence of use by the target group, preferably outside
the development team. Submit there only if the authors can document that
evidence before the deadline.

## Completed page reduction

The reduction removed repetition without dropping empirical evidence.

| Change | Approximate saving | Rationale |
|---|---:|---|
| Compress Related Work from four long paragraphs to three comparison-focused paragraphs | 0.4–0.5 page | Ingredient-level background is repeated in the introduction and positioning paragraph. |
| Fold the basic accuracy/coverage/ECE definitions from Problem Formulation into Experimental Setting | 0.3–0.4 page | Standard metric exposition can be shortened while retaining equations and abstention definitions. |
| Replace Table 1 with one compact count sentence or a two-line total table | 0.3–0.4 page | The per-corpus counts are useful release metadata but not central to the claim. |
| Remove repeated interpretations across Results, Discussion, and Conclusion | 0.5–0.7 page | Saturated open/recall results, the routing-fix qualification, and calibration are each explained multiple times. |
| Tighten workbench, reproducibility, ethics, acknowledgement, and AI-disclosure prose | 0.2–0.3 page | Preserve every disclosure and evidence claim in fewer sentences. |
| Reduce Figure 1 height from 0.24 to about 0.19 text height after checking readability; tighten Figure 2 whitespace | 0.2–0.3 page | Page 6 and page 12 contain substantial float-driven whitespace. |

The verified-release, compositional, and Open Food Facts tables remain in the
paper. The corpus-count table and repeated historical type table were removed.
Their data remain in the release. The archived ablation table was narrowed to
the columns used in the discussion.

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
