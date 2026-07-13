# Codex tasks — close the remaining COMPASS paper TODOs (IJCKG 2026)

Each task lists: goal, where to look, steps, output artifacts, acceptance
criteria, and the paper sentence it unblocks in `COMPASS_main_v2.tex`.
Tasks T1–T3 close in-text TODO comments. T4–T8 close the reviewer risks that
cannot be fixed by wording (routing-fix fairness, leakage, calibration
diagnostics, figures). Do them in order of priority: T4 > T3 > T1 > T6 > T7 > T8 > T2 > T5.

---

## T1. Verify the memory lifecycle description (paper §4.2 TODO)

**Goal:** confirm or correct the paper's claim: "extraction → normalization →
provenance linking → validation before storage; stored facts immutable with
provenance and timestamps; corrections supersede rather than overwrite;
retrieval scoped by product identity."

**Where:** `scripts/seed_memory.py`, the memory service under `backend/`
(search for the code that writes/reads `memory.meta.jsonl`), and the
`VERIFIED_CLEAN` fact format.

**Steps:**
1. Trace the write path: what fields does a stored fact carry (provenance id,
   timestamp, product id, validation flag)? Is any fact stored without
   passing validation?
2. Trace the update path: when a fact changes, is the old record mutated or
   is a new record appended (supersession)? Is there any delete?
3. Trace the read path: is retrieval filtered by product/session identity, or
   can facts from product A be retrieved for a query about product B?
4. Write `tests/test_memory_lifecycle.py` with unit tests for: (a) rejected
   storage of an unvalidated fact, (b) supersession not mutation,
   (c) product-scoped retrieval returns no cross-product facts.

**Output:** `docs/paper/MEMORY_LIFECYCLE.md` (one page: actual pipeline,
field schema, test results) + the passing test file.

**Acceptance:** each of the four claims in the paper sentence is marked
CONFIRMED or CORRECTED, with the file/line evidence. If any is CORRECTED,
propose the replacement sentence for §4.2.

---

## T2. Extract usage evidence from the CE-RISE workbench (paper §1 TODO)

**Goal:** concrete, anonymized numbers behind "demonstrated live to
consortium practitioners": number of demo sessions, queries served, abstain
rate, distinct query topics.

**Where:** `CE-RISE-Demo/backend/` logging (check `logs/`, uvicorn access
logs, any request logging in `/api/search`); the project root `logs/` dir.

**Steps:**
1. Inventory what was actually logged during the consortium demo (grep for
   log writes in the four API endpoints).
2. If logs exist: write `scripts/summarize_workbench_usage.py` that outputs
   session count, query count, per-window usage, abstention count. No user
   identities, no query text in the summary.
3. If logs do not exist: add lightweight anonymized logging (timestamp,
   endpoint, abstain flag, latency) behind a config flag, so the next demo
   produces evidence. Do NOT fabricate numbers for past demos.

**Output:** `docs/paper/USAGE_EVIDENCE.md` with whatever is truthfully
recoverable, clearly dated.

**Acceptance:** every number in the file is derived from a log file that is
referenced by path. If nothing is recoverable, the file says so and the
paper keeps the qualitative sentence only.

---

## T3. Export baseline configurations into the release (paper §5.2 TODO)

**Goal:** make "exact model snapshots, prompts, decoding parameters, and
evidence budgets for every baseline are exported with the release" true.

**Where:** `scripts/run_baselines.py`, `scripts/run_arch_smoke_comparison.py`,
`artifacts/verified_v1/`, shard configs under `artifacts/baseline_linc_shards/`.

**Steps:**
1. Collect for GPT4O_LONGCTX, LOGIC_LM, LINC, and COMPASS: model/API snapshot
   string, system prompt + exemplars, temperature, seed (if set), max context
   and output tokens, retry policy, answer parsing/normalization rules, and
   how evidence was selected per question (state explicitly whether gold
   annotations selected the evidence for GPT4O_LONGCTX — if yes, keep the
   "oracle context" label).
2. Serialize to `release/baseline_configs.json` (one object per system) and
   add a short `release/BASELINES.md` rendering.
3. Assert in a script that all systems were scored by the same validators on
   identical row keys (reuse `scripts/filter_eval_to_verified_subset.py`).

**Output:** `release/baseline_configs.json`, `release/BASELINES.md`,
`scripts/check_baseline_parity.py` (exits 0).

**Acceptance:** every field either filled from code/config or marked
`unknown` (never guessed). Parity check passes on the verified release rows.

---

## T4. Frozen-version rerun (kills the biggest reviewer risk, paper §6.3)

**Goal:** eliminate the "two versions of COMPASS" problem. One frozen system,
all benchmarks, all baselines, identical rows.

**Where:** `scripts/run_paper_pipeline.py`, `scripts/run_baselines.py`,
`artifacts/locked_results/`.

**Steps:**
1. Freeze: git tag the current code (router post-fix), hash the ontology,
   rules, prompts, validators, calibrator, and both benchmarks
   (`sha256sum` manifest → `release/FREEZE_MANIFEST.json`).
2. Rerun COMPASS on: (a) the pooled 3,429 suite, (b) the 6,270 verified
   release, (c) the 3,000 compositional benchmark — all from the frozen tag.
3. Rerun (not filter) GPT4O_LONGCTX, LOGIC_LM, LINC on (b) and (c) with the
   T3 configs.
4. Re-export all paper tables via `scripts/export_tables.py`; produce a diff
   report against the current paper numbers.
5. Any routing/threshold choice must be justified from dev seeds only —
   record which split selected it in the manifest.

**Output:** new `paper_results_<date>/` package + `RESULTS_DIFF.md`.

**Acceptance:** every number in the paper regenerable from one tagged
commit; the §6.3 disclosure sentence about the routing fix can then be
replaced by "all systems were evaluated with a single frozen configuration
(tag, manifest hash)".

---

## T5. Deterministic non-LLM baseline (reviewer: "is an LLM even needed?")

**Goal:** a SPARQL/rule-engine baseline for structured questions and a
field-lookup baseline for OFF.

**Steps:**
1. For logic questions: evaluate rules directly over the fact graph (rdflib
   forward chaining, no LLM) → answer or "no rule applies".
2. For OFF: a pandas/JSONPath lookup that answers attribute/nutrition
   queries from the row and abstains when the field is null.
3. Score both with the standard validators; add one row each to the
   relevant tables.

**Output:** `bias-free` deterministic baseline rows in the results package +
a paragraph draft for §6 stating where deterministic lookup suffices and
where it fails (open/recall/multi-source).

**Acceptance:** results reproducible by script; the paper can state the LLM
is needed for X% of the workload with evidence.

---

## T6. Leakage analysis (reviewer: perfect scores)

**Goal:** show performance before vs. after benchmark filtering, and that
validators are not COMPASS-specific.

**Steps:**
1. From `artifacts/verified_v1/filter_report.md` inputs, recompute all four
   systems' accuracy on the UNFILTERED clean benchmark and on the verified
   subset; tabulate side by side (COMPASS's pre-filter accuracy included —
   whatever it is).
2. Quantify validator neutrality: for 100 random baseline errors, manually
   audit (script-assisted sample export) whether the validator, not the
   answer, caused the failure; report the count.
3. Confirm question generators and COMPASS rules do not share code paths:
   list the generator templates vs. the rule set; state overlaps explicitly.

**Output:** `docs/paper/LEAKAGE_ANALYSIS.md` + a short subsection draft
("Leakage considerations") for §7.

**Acceptance:** the before/after filtering table exists for all systems; the
audit sample is exported for reviewers.

---

## T7. Calibration diagnostics (paper §7 calibration paragraph)

**Goal:** back the calibration discussion with standard diagnostics.

**Where:** per-instance confidences in `artifacts/eval_*_full.csv` and
`paper_results_*/tables/calibration.csv`; OFF bins in
`artifacts/paper_handoff_20260603/off_calibration_bins.csv`.

**Steps:** compute per mode (pooled suite + OFF): reliability diagram
(matplotlib, paper style), Brier score, adaptive/equal-mass ECE, ECE with
bootstrap CI, calibration-set size, per-domain vs. pooled calibration split.

**Output:** `figures/reliability_diagrams.pdf`, a small table
(`tables/calibration_diagnostics.tex`).

**Acceptance:** in-domain ECE 0.525 reproduced by the script (sanity check),
all metrics generated from per-instance files, not hand-entered.

---

## T8. Regenerate Figure 2 (risk–coverage) with publication names

**Where:** `paper_results_20260202_193050/tables/risk_coverage.csv` (per
domain) and `selective_calibrated.csv`.

**Steps:**
1. Rebuild pooled curves by sorting all instances by confidence and
   computing cumulative risk at every unique threshold (not a coarse grid).
2. Rename series: ADAPTIVERAG→COMPASS, RAG_BASE→RAG-Base, MEMSYM→Mem+Sym,
   SYM_ONLY→Sym-Only, RL→RL, ROUTER→Router; legend outside the axes;
   vector output (PDF), consistent fonts with the paper.
3. While there: shrink Figure 1 to ~6 blocks if it is regenerated from
   source (otherwise leave for manual redraw).

**Output:** `artifacts/risk_coverage.pdf` + the plotting script.

**Acceptance:** curve for COMPASS monotone-consistent with AURC 0.0117; no
internal mode names anywhere in the figure.

---

## Not for Codex (author actions)

- Fill `\author{}` / `\institute{}` with ORCIDs (LNCS style).
- Insert the public repository URL / Zenodo DOI in §8 once T4's package is
  uploaded.
- Insert the CE-RISE grant agreement number in the acknowledgements.
- Sync the fixed `refs.bib` (duplicate `cirpass2` and `openfoodfacts`
  removed) into Overleaf.
