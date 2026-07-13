# Phase A report

Status: complete. Paid API calls: **0**.

## Safety inventory

- `EXISTING_ARTIFACT_INVENTORY.json` records SHA-256 hashes for 35/35
  paper-critical benchmark, result, ontology/rule, prompt, validator, and
  calibration files (23,173,836 bytes).
- Existing artifacts, experiments, runtime memory, and locked results were not
  modified or copied over.
- The proposed Phase B output is a new `paper_results_20260713_frozen`
  directory, and the preflight refuses to reuse an existing directory.

## T3 configuration and parity

- `release/baseline_configs.json` and `release/BASELINES.md` export exact
  current prompts, limits, retry behavior, evidence selection, validators, and
  all unknown historical fields.
- The verified GPT-4o baseline uses retrieval-selected context, not oracle
  context. The LINC and Logic-LM baselines are prompted approximations, not
  executions of the published solver pipelines.
- All four verified-release systems contain exactly the same 6,270 row keys
  and identical benchmark metadata.
- Twenty archived LINC answers were truncated at exactly 500 characters after
  scoring, so those stored scores cannot be independently recomputed. Phase B
  now retains full responses and records the model/configuration per row.
- A LINC-style compositional path has been added for the Phase B same-system,
  same-row comparison.

## T1 memory lifecycle

- `docs/paper/MEMORY_LIFECYCLE.md` records four corrected claims and file-level
  evidence.
- The paper now accurately describes session-scoped timestamped text memory
  and no longer claims enforced validation, product identity, immutable
  supersession, or provenance that the store does not implement.

## T4 dry-run

`FROZEN_RERUN_PREFLIGHT.json` passes all non-paid checks:

- benchmark counts: 3,429 / 6,270 / 3,000;
- protected artifact hashes unchanged;
- router source: development evaluation;
- calibrator source: development evaluation;
- explicit proposed snapshot: `gpt-4o-mini-2024-07-18`;
- compositional LINC-style path present;
- sufficient disk space.

Before Phase B, the committed tree must be clean and `OPENAI_API_KEY` must be
loaded. The paid preflight will enforce both requirements.

## Submission readiness

- The supplied PDF is 17 pages; the target is 14.5–15 pages.
- `docs/paper/SUBMISSION_READINESS.md` provides a 1.9–2.6 page reduction plan.
- Author/institute/ORCID, repository or DOI, grant number, and documented usage
  evidence remain author-supplied blockers.

## Phase B approval gate

Estimated API spend: USD 6–10, hard stop at USD 12. Estimated wall time:
8–24 hours depending on API tier and retries. No Phase B model call will be
made without explicit approval after the final clean preflight.
