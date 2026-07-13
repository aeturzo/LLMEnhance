# IJCKG 2026 submission readiness

Audit date: 2026-07-13. The supplied `LLMEnhanceThaiLand-5.pdf` is 17 pages.
IJCKG 2026 permits 12–15 pages including references, so the submission must
lose at least two pages and should target 14.5 pages to tolerate float movement.

## Page-reduction plan

The reduction should remove repetition, not empirical evidence.

| Change | Target saving | Rationale |
|---|---:|---|
| Compress Related Work from four long paragraphs to three comparison-focused paragraphs | 0.4–0.5 page | Ingredient-level background is repeated in the introduction and positioning paragraph. |
| Fold the basic accuracy/coverage/ECE definitions from Problem Formulation into Experimental Setting | 0.3–0.4 page | Standard metric exposition can be shortened while retaining equations and abstention definitions. |
| Replace Table 1 with one compact count sentence or a two-line total table | 0.3–0.4 page | The per-corpus counts are useful release metadata but not central to the claim. |
| Remove repeated interpretations across Results, Discussion, and Conclusion | 0.5–0.7 page | Saturated open/recall results, the routing-fix qualification, and calibration are each explained multiple times. |
| Tighten workbench, reproducibility, ethics, acknowledgement, and AI-disclosure prose | 0.2–0.3 page | Preserve every disclosure and evidence claim in fewer sentences. |
| Reduce Figure 1 height from 0.24 to about 0.19 text height after checking readability; tighten Figure 2 whitespace | 0.2–0.3 page | Page 6 and page 12 contain substantial float-driven whitespace. |

Expected total saving: 1.9–2.6 pages. Do not remove the verified-release,
compositional, or OFF result tables; they carry the main reviewer-facing evidence.

## Blocking author-supplied fields

- Author names, affiliations, emails, and ORCIDs.
- Resolvable public repository URL and/or Zenodo DOI.
- CE-RISE Horizon Europe grant agreement number and approved partner acknowledgement text.
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

Phase B replaces the routing-fix disclosure and every affected table with
single-tag frozen results. Phase C adds leakage/calibration evidence and the
publication-name risk–coverage figure. Phase D performs the two-page reduction,
fills author-supplied fields, resolves every remaining TODO, compiles the LNCS
source, and performs a final claim-to-artifact audit.
