# Carbon Feature Implementation Report

## Purpose

This report is a running implementation log for the carbon-footprint and
recyclability feature. It is intended to be reusable later as source material
for the research paper methodology and implementation sections.

The report is cumulative. Each implementation step should append:

- what was done
- what challenges appeared
- how those challenges were resolved
- what files or outputs were produced

## Roadmap Snapshot

1. Normalize raw carbon data into machine-readable assets.
2. Build a reproducible raw-to-normalized asset generation script.
3. Add the deterministic carbon calculation service.
4. Add typed request/response schemas for carbon calculations.
5. Route carbon and recyclability questions through `/solve`.
6. Add a raw debugging endpoint for direct calculation inspection.
7. Add a carbon ontology sidecar for provenance and validation.
8. Generate answer assets and context passages for totals, raw materials, transportation, use phase, and end-of-life.
9. Add tests for the calculator and solve-path integration.
10. Add honest estimate fallback and provenance disclosure.
11. Update the frontend to call `/solve` and display carbon outputs.

## Step 1: Normalized Carbon Data Layer

### Goal

Create a stable machine-readable data layer under `backend/data/carbon/` while
keeping the original folder `totalCarbonfootprintcalculation/` untouched as the
raw-source area.

### Tasks Completed

- Created the normalized data directories:
  - `backend/data/carbon/products/`
  - `backend/data/carbon/factors/`
  - `backend/data/carbon/mappings/`
- Added a starter product profile for `Lexmark MX431adn`.
- Added a grounded starter electricity factor CSV from the CoM workbook.
- Added placeholder factor CSVs for:
  - raw materials
  - transportation
  - end-of-life
- Added a manual EF workbook mapping template for later curation.
- Added documentation in `backend/data/carbon/README.md`.
- Updated `.gitignore` to ignore Office lock files such as `~$tology.dotx`.

### Challenges

1. The official `Lexmark MX431adn.pdf` was encrypted and could not be text-extracted with the current local PDF stack.
2. The EF LCIA workbook was very large and not organized as a ready-to-query product-factor table.
3. The raw source folder mixed official files, notes, example files, and temporary Office artifacts.

### Resolution

1. Product-level numeric fields were kept explicit as `pending_official_extraction` instead of inventing values.
2. A clean separation was established between:
   - grounded factors extracted directly from source tables
   - placeholder factors that still require manual curation
3. Temporary bootstrap estimates were isolated from official product inputs so they can be replaced cleanly later.

### Outputs Produced

- `backend/data/carbon/README.md`
- `backend/data/carbon/products/lexmark_mx431adn.json`
- `backend/data/carbon/factors/electricity_lc_factors.csv`
- `backend/data/carbon/factors/raw_material_factors.csv`
- `backend/data/carbon/factors/transport_factors.csv`
- `backend/data/carbon/factors/end_of_life_factors.csv`
- `backend/data/carbon/mappings/ef_flow_map.yml`

### Step Status

Completed successfully.

## Step 24: Fix Symbolic Compliance Queries That Collapsed to `Insufficient context.`

### Objective

Resolve a regression in the auto orchestrator where a compliance-style query
could have valid symbolic evidence available, but still end up with the final
answer `Insufficient context.` because the LLM rejected the composed prompt and
no symbolic-only deterministic fallback existed.

### Tasks Completed

- Updated `backend/api/solve_auto.py` so pure compliance queries now suppress
  unrelated memory context when symbolic evidence is already available.
- Added a `symbolic_direct` fallback path for cases where:
  - symbolic evidence exists
  - the query is compliance-like
  - the final LLM answer is `Insufficient context.` or fails to include the
    symbolic content
- Extended tests in `tests/test_solve_auto.py` to verify:
  - compliance queries include only the symbolic passage in the composed
    passage list
  - an LLM-side `Insufficient context.` response now downgrades to an honest
    deterministic symbolic answer instead of surfacing the failure to the user

### Challenges

1. The user-visible failure happened only under certain live LLM behaviors, so
   the regression needed a deterministic test harness rather than only a manual
   reproduction.
2. Memory retrieval could still score above threshold for the same product id,
   even when the question was strictly about compliance and the memory note was
   irrelevant.
3. The fix needed to preserve honest audit behavior rather than silently
   pretending the LLM answered correctly.

### Resolution

1. Added a dedicated compliance guard so memory is excluded for symbolic-only
   compliance questions unless the query is explicitly memory-led.
2. Added `symbolic_direct` as a visible answer-trace path for symbolic fallback
   answers.
3. Added a regression test that mocks an LLM `Insufficient context.` result and
   verifies the symbolic fallback is returned instead.

### Outputs Produced

- updated `backend/api/solve_auto.py`
- updated `tests/test_solve_auto.py`

### Validation Results

- Syntax validation succeeded with:
  - `python3 -m py_compile backend/api/solve_auto.py tests/test_solve_auto.py`
- Regression tests succeeded with:
  - `.venv/bin/python -m unittest tests.test_solve_auto -v`

### Step Status

Completed successfully.

## Step 23: Produce a Full System and Carbon User Guide Document

### Objective

Create a standalone reader-friendly document that explains:

- the overall system in plain language
- how the carbon and recyclability feature works
- the data pipeline
- the calculation and estimation logic
- how to run the system
- which demo questions to ask

The goal was to make the project understandable to someone with little or no
background and to provide a directly reusable document file for reading and
sharing.

### Tasks Completed

- Authored a full guide in `docs/llmmain_system_and_carbon_guide.md`.
- Covered:
  - overall system architecture
  - search, memory, symbolic reasoning, orchestrator, and RL roles
  - carbon feature scope and methodology
  - normalized data layer and raw-source separation
  - stage-by-stage carbon formulas
  - estimate fallback logic
  - provenance and uncertainty disclosure
  - startup instructions for backend and frontend
  - verified demo questions for battery, Lexmark, Viessmann, and carbon
- Exported the guide to a shareable Word document:
  - `docs/llmmain_system_and_carbon_guide.docx`

### Challenges

1. The guide needed to be understandable to a non-expert reader without
   becoming technically vague or inaccurate.
2. The document had to explain both the general QA system and the carbon
   feature, which use different reasoning styles.
3. The request explicitly asked for a document file, not just a chat summary,
   so the output needed to be exported into a standard document format.

### Resolution

1. Structured the guide from general concepts to specific implementation
   details and included formulas and examples where useful.
2. Described the carbon path separately from the general QA path so the reader
   can clearly see why deterministic calculation is used there.
3. Authored the master version in Markdown for maintainability and exported it
   to `.docx` with `pandoc` for direct use outside the repo.

### Outputs Produced

- added `docs/llmmain_system_and_carbon_guide.md`
- added `docs/llmmain_system_and_carbon_guide.docx`

### Validation Results

- Markdown guide written successfully.
- Word document export succeeded with:
  - `pandoc docs/llmmain_system_and_carbon_guide.md -o docs/llmmain_system_and_carbon_guide.docx --toc`

### Step Status

Completed successfully.

## Step 22: Apply CE-RISE Glassmorphism Frontend Theme

### Objective

Restyle the existing frontend so the entire experience follows a CE-RISE-like
blue-violet-magenta-orange palette and glassmorphism design language, while
preserving all existing features, information blocks, and layout placement.

### Tasks Completed

- Added a shared visual token layer in `frontend/src/theme.js`.
- Defined:
  - CE-RISE-inspired page gradients
  - translucent glass surfaces
  - dark and light glass panels
  - shared button, chip, field, and code-block treatments
- Updated the main page shell in `frontend/src/pages/HomePage.jsx` to:
  - use the new CE-RISE gradient background
  - add soft blurred decorative orbs
  - preserve the existing workspace switcher and page structure
- Updated `frontend/src/components/SolveWorkspace.jsx` to use the new glass
  hero surface and themed example chips.
- Updated `frontend/src/components/OrchestratedWorkspace.jsx` to use:
  - themed glass cards
  - CE-RISE-colored fields and buttons
  - the new shared visual tokens without changing the domain, session, memory,
    or query workflows
- Updated `frontend/src/components/SearchBar.jsx` so the carbon solve inputs
  match the same glassmorphism treatment.
- Updated `frontend/src/components/ResultsList.jsx` so:
  - answer panels
  - carbon metrics
  - chips
  - provenance cards
  - source cards
  all follow the new translucent theme while keeping their existing content and
  ordering intact.
- Updated `frontend/src/components/AdvancedAuditView.jsx` so the audit page
  visually matches the main answer views.
- Applied global body-level styling in `frontend/src/index.js` so the frontend
  background and typography are consistent from the first paint.

### Challenges

1. The frontend used inline styles in each component instead of a shared theme
   layer, so the risk of inconsistent updates was high.
2. The user explicitly asked to preserve feature behavior, audit visibility,
   and layout placement, which meant the work had to remain visual only.
3. Glassmorphism can easily reduce readability if the surfaces become too
   translucent on a high-saturation background.

### Resolution

1. Introduced a shared theme module and mapped the existing components onto it
   rather than rebuilding the layouts.
2. Kept all component hierarchies, fields, buttons, answer sections, and audit
   sections in the same order and positions.
3. Used stronger blur, slightly denser translucent fills, and controlled text
   contrast so the CE-RISE palette remains readable across answer panels and
   audit surfaces.

### Outputs Produced

- added `frontend/src/theme.js`
- updated `frontend/src/pages/HomePage.jsx`
- updated `frontend/src/components/SolveWorkspace.jsx`
- updated `frontend/src/components/OrchestratedWorkspace.jsx`
- updated `frontend/src/components/SearchBar.jsx`
- updated `frontend/src/components/ResultsList.jsx`
- updated `frontend/src/components/AdvancedAuditView.jsx`
- updated `frontend/src/index.js`

### Validation Results

- Frontend production build succeeded with:
  - `npm --prefix frontend run build`
- No frontend feature code paths were removed or rewired during the theme pass.
- The workspace split, advanced audit view, carbon answer layout, and auto
  orchestrator layout remained intact.

### Step Status

Completed successfully.

## Step 21: Add Multi-Domain Orchestration for Battery, Lexmark, and Viessmann

### Objective

Extend the auto orchestrator so users can explicitly select a domain, symbolic
reasoning can run against the correct ontology for that domain, and the audit
trail shows which domain was actually used. Carbon handling remains separate
and limited to the current supported product data.

### What Was Implemented

1. Added per-domain symbolic reasoner support in
   `backend/services/symbolic_reasoning_service.py`.
2. Updated startup in `backend/main.py` to initialize reasoners for
   `battery`, `textiles`, `lexmark`, and `viessmann`, while preserving the old
   default `app.state.reasoner` for compatibility.
3. Extended `backend/api/solve_auto.py` with:
   - domain selection in the request
   - domain normalization and domain inference
   - domain-aware symbolic reasoning calls
   - domain information in the audit trace
4. Updated the orchestrated frontend workspace to include a domain selector and
   domain-specific example prompts.
5. Added a domain chip in the result view so the UI shows which domain the
   answer used.
6. Added regression coverage for Lexmark and Viessmann symbolic routing.

### Challenges Encountered

1. The existing architecture assumed one ontology at process startup, which made
   battery symbolic reasoning work but left Lexmark and Viessmann symbolic
   reasoning unavailable in the same runtime.
2. Search already worked across multiple document domains, so the user-facing
   issue was specifically that the ontology-backed path did not match that same
   multi-domain behavior.
3. The orchestrator audit trail needed to distinguish between the selected
   domain and the effective domain after inference.

### How The Challenges Were Resolved

1. Replaced the single implicit symbolic-reasoner lookup with a domain-aware
   cache and startup-initialized reasoner map.
2. Added domain and product inference logic in the orchestrator while still
   allowing explicit user override from the UI.
3. Surfaced `selected_domain`, `effective_domain`, and the inferred product in
   the orchestration trace so the advanced audit explains why a particular
   symbolic path was used.

### Outputs Produced

- updated `backend/services/symbolic_reasoning_service.py`
- updated `backend/main.py`
- updated `backend/api/solve_auto.py`
- updated `frontend/src/services/api.js`
- updated `frontend/src/components/OrchestratedWorkspace.jsx`
- updated `frontend/src/components/ResultsList.jsx`
- updated `tests/test_solve_auto.py`
- updated `docs/carbon_feature_implementation_report.md`

### Validation Results

- Regression tests succeeded with:
  - `.venv/bin/python -m unittest tests.test_solve_auto -v`
- Frontend production build succeeded with:
  - `npm --prefix frontend run build`
- Live verification succeeded for:
  - battery retrieval, memory, and symbolic questions
  - Lexmark retrieval, memory, and symbolic questions
  - Viessmann retrieval, memory, and symbolic questions
  - supported carbon question for `Lexmark MX431adn`
  - unsupported carbon question for `ProductV1`
- Verified runtime behavior now shows:
  - the selected domain in the UI
  - the effective domain in the audit trace
  - symbolic reasoning firing for the chosen domain
  - unsupported carbon products abstaining honestly

### Step Status

Completed successfully.

## Step 20: Infer Products in Orchestration and Add Honest Memory+Symbolic Fallback

### Objective

Fix the orchestrated workspace so ProductA-style questions work even when the
 optional product field is left blank, and make blended memory-plus-symbolic
 answers honest in the audit trail when the final response is assembled
 deterministically rather than by the LLM.

### What Was Implemented

1. Added query-based product inference in `backend/api/solve_auto.py` for
   `ProductA`-style identifiers and `Lexmark MX431adn`.
2. Propagated the inferred product into the orchestration trace as
   `effective_product` and `product_inferred_from_query`.
3. Added a deterministic blended fallback path,
   `memory_symbolic_direct`, for questions that clearly ask for both a remembered
   preference and a compliance standard.
4. Updated the orchestrated workspace helper text in
   `frontend/src/components/OrchestratedWorkspace.jsx` to make it clear that the
   product can often be inferred automatically.
5. Added regression tests covering:
   - symbolic product inference without filling the product field
   - blended memory plus symbolic composition without filling the product field

### Challenges Encountered

1. The symbolic compliance screenshot returned a plausible answer, but the audit
   trail was misleading because the product box had been left empty and symbolic
   reasoning never fired.
2. The blended memory-plus-symbolic screenshot inferred the remembered
   packaging correctly but still collapsed to a pure memory answer when the LLM
   result was incomplete or unstable.
3. The fix needed to preserve honest provenance instead of simply claiming the
   LLM was used when the final answer actually came from deterministic assembly.

### How The Challenges Were Resolved

1. Implemented product inference directly from the question text so the
   orchestrator can trigger symbolic reasoning without requiring a separate UI
   field entry.
2. Added a blended deterministic fallback that explicitly combines the top
   memory entry with the top symbolic standard when the query asks for both and
   the generated answer is incomplete.
3. Recorded the final path as `memory_symbolic_direct` so the frontend trail now
   distinguishes:
   - `llm` answers
   - `memory_direct` answers
   - `memory_symbolic_direct` answers

### Outputs Produced

- updated `backend/api/solve_auto.py`
- updated `frontend/src/components/OrchestratedWorkspace.jsx`
- updated `tests/test_solve_auto.py`
- updated `docs/carbon_feature_implementation_report.md`

### Validation Results

- Regression tests succeeded with:
  - `.venv/bin/python -m unittest tests.test_solve_auto -v`
- Live route verification succeeded for:
  - `POST /solve_auto` with `Name two compliance standards that apply to ProductA`
    and no product field
  - `POST /solve_auto` with
    `What packaging did I say ProductA prefers, and name one compliance standard for ProductA.`
    and no product field
- Verified runtime behavior now shows:
  - `product: ProductA`
  - `product_inferred_from_query: ProductA`
  - symbolic-only question can use `gpt-5`
  - blended question returns a complete answer with path
    `memory_symbolic_direct` when the final answer is assembled deterministically

### Step Status

Completed successfully.

## Step 19: Fix GPT-5 Answer-Synthesis Compatibility for Live Verification

### Objective

Repair the general answer-synthesis path so the verification set can exercise a
real GPT-5-backed curation flow for search and symbolic questions instead of
dropping to deterministic snippet fallback because of model-parameter
incompatibilities.

### What Was Implemented

1. Updated the OpenAI Responses API path in `backend/api/answerer_ctx.py` to
   request low reasoning effort while keeping the existing text-only answer
   contract.
2. Corrected the OpenAI chat-completions fallback to use
   `max_completion_tokens` instead of `max_tokens`, which GPT-5 rejects.
3. Removed the explicit `temperature=0.2` setting from the OpenAI
   chat-completions fallback because GPT-5 accepts only the default temperature
   value in this environment.
4. Re-ran live verification queries across memory, retrieval, symbolic, blended,
   and carbon routes to confirm which paths are now truly LLM-backed and which
   remain deterministic by design.

### Challenges Encountered

1. The initial GPT-5 route appeared configured correctly in the UI, but the
   trace showed `llm_used: false` for general queries. The failure was caused by
   model-specific API incompatibilities rather than missing credentials.
2. The Responses API can legally return an incomplete reasoning-only result when
   the output budget is too low, which the extractor treats as no answer text.
3. The fallback chat path then failed for GPT-5 because it still used parameters
   accepted by older chat models but rejected by GPT-5.

### How The Challenges Were Resolved

1. Verified the installed OpenAI SDK in `.venv` and confirmed the Responses API
   works with the local key and `gpt-5`.
2. Reproduced direct low-cost calls to distinguish between environment problems
   and parameter problems.
3. Patched the fallback path to use GPT-5-compatible parameters and re-ran live
   route checks until general retrieval and symbolic answers showed
   `llm_used: true`.

### Outputs Produced

- updated `backend/api/answerer_ctx.py`
- updated `docs/carbon_feature_implementation_report.md`

### Validation Results

- Syntax validation succeeded with:
  - `python3 -m py_compile backend/api/answerer_ctx.py`
- Live route verification succeeded for:
  - `POST /solve_auto` with a symbolic compliance query for `ProductA`
  - `POST /solve_auto` with blended memory and symbolic context
  - `POST /solve` with retrieval questions such as UN transport numbers and
    battery-cell materials
- Verified runtime behavior now shows:
  - `llm_used: true` for general search and symbolic composition cases
  - `provider: openai`
  - `model: gpt-5`
  - `api: responses`
- Deterministic paths intentionally remain:
  - `memory_direct` for direct memory answers
  - `carbon_rule_fallback` for carbon answers grounded in deterministic
    calculation output

### Step Status

Completed successfully.

## Step 2: Raw-to-Normalized Asset Builder

### Goal

Create a reproducible build script that regenerates normalized carbon assets
from the raw PDFs and workbooks instead of relying on one-off manual setup.

### Tasks Completed

- Add `scripts/build_carbon_assets.py`.
- Extract readable PDF text into normalized text outputs.
- Convert the CoM electricity workbook into a normalized long-format CSV.
- Inspect the EF workbook and record its sheet structure for later mapping work.
- Regenerate the starter `Lexmark MX431adn` product JSON from reproducible inputs.
- Produce a manifest that records extraction status and unresolved blockers.

### Challenges

1. Some PDFs are readable, but the official Lexmark product PDF remains encrypted.
2. The environment did not have `openpyxl`, so direct workbook parsing needed a dependency-light approach.
3. Step 2 needed to preserve the cautious stance from Step 1 and avoid silently filling unresolved official values.
4. The electricity workbook includes non-numeric placeholders such as `-` in some cells, which initially polluted the normalized CSV.

### Resolution

1. `pypdf` was used when available for readable PDFs, and unreadable sources were recorded explicitly in the manifest instead of being silently skipped.
2. `.xlsx` files were parsed directly with `zipfile` and XML so the builder does not depend on spreadsheet libraries.
3. Official product fields were only regenerated when the source was actually readable; otherwise placeholders were preserved in the product JSON.
4. The builder was tightened to skip non-numeric workbook cells and record how many were excluded.

### Outputs Produced

- `scripts/build_carbon_assets.py`
- extracted source text files under `backend/data/carbon/extracted/`
- `backend/data/carbon/extracted/source_extracts.json`
- `backend/data/carbon/build_manifest.json`
- refreshed normalized product and factor files

### Build Results

- Command executed: `python3 scripts/build_carbon_assets.py`
- Readable PDF extracts written:
  - `Calculation process .pdf`
  - `Ontology.pdf`
  - `env-epd_21_1683665824.pdf`
  - `LCD monitor.pdf`
- Unreadable PDF recorded:
  - `Lexmark MX431adn.pdf`
- Electricity-factor normalization result:
  - 958 numeric rows written
  - 30 countries covered
  - years 1990 to 2021
  - 2 non-numeric placeholder cells skipped
- EF workbook inspection result:
  - sheet names captured for later curation

### Step Status

Completed successfully.

## Step 3: Deterministic Carbon Calculation Service

### Goal

Implement a deterministic calculation service that consumes the normalized
carbon data layer and produces:

- stage-by-stage carbon results
- an overall total when all requested stages are computable
- partial results plus explicit missing-input diagnostics when they are not
- recyclability outputs derived from end-of-life inputs

### Tasks Completed

- Added `backend/services/carbon_calculation_service.py`.
- Implemented a `CarbonCalculationService` over the normalized product and factor files.
- Added deterministic resolution for:
  - raw materials
  - transportation
  - use-phase electricity
  - end-of-life
- Added provenance-rich trace items per stage.
- Added recyclability summary outputs.
- Added module-level wrappers for later API integration.
- Validated the service with:
  - a full override-based smoke test covering all four stages
  - a no-override smoke test against the current real repo state

### Challenges

1. The normalized product profile still lacks official MX431adn masses, transport details, use-phase values, and end-of-life splits because the official PDF is still unreadable.
2. The raw-material, transportation, and end-of-life factor tables are intentionally placeholders at this stage and therefore cannot support a complete official calculation yet.
3. The service needed to be useful immediately without hiding the fact that official curation is still incomplete.
4. The first verification pass exposed an interface mismatch: the service logic was present, but the override payload shape used in the Step 3 smoke test did not line up with all of the field names the service accepted.

### Resolution

1. The calculator was designed to prefer normalized official data when present, but also accept explicit scenario overrides.
2. Bootstrap estimates were supported only as explicit opt-in behavior so approximate inputs do not silently contaminate official results.
3. The service returns:
   - `status = complete` only when every requested stage is computable
   - `status = partial` with `total_kg_co2e = null` when a full total would be misleading
4. Missing masses, factors, route splits, and unresolved electricity inputs are surfaced directly in `missing_inputs`.
5. The override-handling layer was tightened so the service now accepts the planned Step 3 payload aliases directly, including:
   - `factor_value_kg_co2e_per_kg`
   - `factor_value_kg_co2e_per_ton_km`
   - `mode_key`
   - `electricity_country_code`
   - `electricity_year`
   - `recyclability_pct`
   - nested `route_factors`

### Outputs Produced

- `backend/services/carbon_calculation_service.py`
- updated `backend/data/carbon/README.md`

### Validation Results

- Syntax validation succeeded with:
  - `python3 -m py_compile backend/services/carbon_calculation_service.py`
- Full deterministic smoke test succeeded using explicit scenario overrides:
  - product: `lexmark_mx431adn`
  - status: `complete`
  - total: `87.02222052620104 kg CO2e`
  - stage totals:
    - raw materials: `68.6435`
    - transportation: `2.538`
    - use phase: `10.376970526201035`
    - end of life: `5.46375`
  - recyclability: `85%`
  - missing inputs: `0`
- Real-state no-override smoke test also succeeded behaviorally:
  - status: `partial`
  - total: `null`
  - partial total: `0.0`
  - missing inputs reported: `22`
  - result did not fabricate a footprint from incomplete official data

### Step Status

Completed successfully.

## Step 4: Typed Carbon Request and Response Schemas

### Goal

Add a typed schema layer for carbon calculations so later API work can:

- validate incoming calculator payloads
- normalize aliases consistently before execution
- serialize structured calculator outputs in a stable response format

### Tasks Completed

- Added `backend/api/carbon_models.py`.
- Added typed request models for:
  - raw-material inputs
  - transport-leg inputs
  - use-phase inputs
  - end-of-life inputs
- Added a top-level carbon calculation request model with:
  - stage selection
  - bootstrap toggle
  - shared report-year and country overrides
  - trace-inclusion control
- Added typed response models for:
  - per-trace outputs
  - per-stage outputs
  - recyclability outputs
  - full calculation outputs
- Added conversion helpers that map:
  - request models into the deterministic calculator scenario shape
  - service dataclass results into API response models
- Added import-friendly aliases so later endpoints can use short names such as `CarbonRequest` and `CarbonResult`.

### Challenges

1. The deterministic calculator from Step 3 already uses internal dataclasses, so Step 4 needed to add API schemas without duplicating or forking the actual calculation logic.
2. The repo already shows mixed Pydantic-version compatibility patterns, so the schema layer needed to avoid locking the implementation to only one serialization method.
3. The current shell environment does not have `pydantic` installed, which limits local runtime validation even though the backend declares it as a dependency.

### Resolution

1. The schema layer was built as a thin adapter over the existing deterministic service, with explicit `to_service_scenario()` and `from_service_result()` conversion methods.
2. The request model uses a compatibility helper that falls back from `model_dump()` to `dict()` so it can work cleanly with either Pydantic v2 or v1 style serialization.
3. Validation for this step was kept to static syntax compilation in the current environment, while the code itself remains aligned with the backend dependency declarations and existing FastAPI model style.
4. Stage aliases were normalized in the schema layer so later API routes do not need to repeat the same alias-cleaning logic.

### Outputs Produced

- `backend/api/carbon_models.py`
- updated `backend/data/carbon/README.md`

### Validation Results

- Static syntax validation succeeded with:
  - `python3 -m py_compile backend/api/carbon_models.py backend/services/carbon_calculation_service.py`
- The new schema layer now supports structured request payloads for:
  - raw materials
  - transportation
  - use phase
  - end of life
- The new response layer can serialize:
  - trace outputs
  - stage totals
  - recyclability outputs
  - top-level calculation metadata

### Step Status

Completed successfully.

## Step 5: Route Carbon and Recyclability Questions Through `/solve`

### Goal

Route user-facing carbon and recyclability questions through the existing
`/solve` endpoint so the system can:

- detect carbon intent from natural-language questions
- resolve supported products from the query or product field
- run deterministic carbon calculations
- return a curated answer plus structured calculation trace data

### Tasks Completed

- Added `backend/services/carbon_query_service.py`.
- Added carbon-intent detection for:
  - carbon-footprint questions
  - emissions questions
  - recyclability and end-of-life questions
- Added supported-product resolution for `Lexmark MX431adn`.
- Added stage-intent routing so `/solve` can target:
  - full footprint
  - raw materials
  - transportation
  - use phase
  - end of life / recyclability
- Added synthetic context-passage generation from the deterministic calculator outputs.
- Added a deterministic fallback answer layer so carbon responses remain usable even when no LLM backend is available.
- Updated `backend/api/solve.py` so carbon questions are intercepted early and routed to the carbon query service.
- Added explicit `CARBON` mode support in the solve request model.

### Challenges

1. The existing `/solve` route already multiplexes search, memory, symbolic, and router-driven paths, so the carbon feature needed to integrate without destabilizing those flows.
2. Carbon calculations currently remain partial for the real MX431adn product because the official product PDF is unreadable and several factor tables still need manual curation.
3. Importing `backend.api.answerer_ctx` directly from the new service triggered `backend.api.__init__`, which requires FastAPI and prevented dynamic validation in the current shell environment.
4. The solve path still needed to produce useful user-facing answers even when the context-bound LLM answerer is unavailable.

### Resolution

1. Carbon detection was placed ahead of the existing router/adaptive/classic branches in `/solve`, so carbon questions are handled by a dedicated path before they fall into generic retrieval.
2. The new carbon query service builds synthetic passages from deterministic calculator outputs, preserving the same “compute first, answer second” pattern used elsewhere in the system.
3. The answerer module is loaded lazily by file path inside the carbon query service, which avoids importing the FastAPI router package during pure-Python validation.
4. A deterministic fallback answer layer now handles:
   - unsupported products
   - partial calculations
   - stage-specific missing inputs
   - missing recyclability splits

### Outputs Produced

- `backend/services/carbon_query_service.py`
- updated `backend/api/solve.py`
- updated `backend/data/carbon/README.md`

### Validation Results

- Static syntax validation succeeded with:
  - `python3 -m py_compile backend/services/carbon_query_service.py backend/api/solve.py backend/services/carbon_calculation_service.py`
- Dynamic validation succeeded for the pure-Python carbon query service with these query types:
  - total carbon-footprint question for `Lexmark MX431adn`
  - recyclability question for `Lexmark MX431adn`
  - transportation-emissions question for `Lexmark MX431adn`
  - unsupported-product question
- Current observed runtime behavior:
  - carbon queries are routed to `mode = CARBON`
  - supported product resolves to `lexmark_mx431adn`
  - answers are currently honest `partial` responses because official MX431adn inputs are still incomplete
  - unsupported products are rejected with a supported-product answer instead of falling through to generic retrieval

### Step Status

Completed successfully.

## Step 6: Add a Raw Debugging Endpoint for Direct Calculation Inspection

### Goal

Add a direct carbon-calculation endpoint so developers can inspect structured
calculation outputs without going through the natural-language `/solve` path.

### Tasks Completed

- Added `backend/api/carbon.py`.
- Added `POST /carbon/calculate` for direct deterministic carbon calculations.
- Added `GET /carbon/products` to list currently supported normalized carbon products.
- Wired the carbon router into:
  - `backend/main.py`
  - `backend/api/__init__.py`
- Reused the Step 4 typed schema layer for:
  - request validation
  - response serialization
  - optional trace suppression via `include_trace`

### Challenges

1. The raw debug endpoint needed to expose the structured calculator cleanly without duplicating the calculation logic from Step 3.
2. The current system Python still lacks FastAPI and Pydantic, so endpoint-level verification could not be done there directly.
3. The endpoint needed to support both:
   - real-state partial calculations from the current normalized repo data
   - explicit override-based complete calculations for debugging and evaluation

### Resolution

1. The new endpoint delegates straight to `calculate_carbon_footprint()` and wraps the result with the Step 4 response model, so the endpoint is only a thin transport layer.
2. No new libraries were installed because the repo already includes a working `.venv` with `fastapi`, `pydantic`, and `uvicorn`; verification was done safely inside that virtual environment.
3. The endpoint accepts the full Step 4 request schema, so developers can inspect:
   - partial official-data runs
   - targeted single-stage runs
   - full override-based deterministic runs

### Outputs Produced

- `backend/api/carbon.py`
- updated `backend/main.py`
- updated `backend/api/__init__.py`
- updated `backend/data/carbon/README.md`

### Validation Results

- Static syntax validation succeeded with:
  - `.venv/bin/python -m py_compile backend/api/carbon.py backend/main.py backend/api/__init__.py backend/api/carbon_models.py backend/services/carbon_calculation_service.py`
- HTTP-layer validation succeeded with FastAPI `TestClient` inside the project `.venv`:
  - `GET /carbon/products` returned `['lexmark_mx431adn']`
  - `POST /carbon/calculate` with a transportation-only request returned `status = partial`
  - `include_trace = false` correctly returned an empty trace list for that stage
  - `POST /carbon/calculate` with full override payload returned `status = complete`
  - full override result total was `87.02222052620104 kg CO2e`
  - full override recyclability was `85%`

### Step Status

Completed successfully.

## Step 7: Add a Carbon Ontology Sidecar for Provenance and Validation

### Goal

Add a carbon-specific ontology sidecar that can:

- represent calculation outputs as RDF instances
- capture source provenance for activities and factors
- validate the structural completeness of the generated carbon graph
- remain separate from the deterministic arithmetic engine

### Tasks Completed

- Added `backend/ontologies/carbon_ontology.ttl`.
- Added `backend/services/carbon_ontology_service.py`.
- Extended the typed carbon request model so debug calls can request:
  - ontology validation only
  - ontology validation plus Turtle serialization
- Extended the typed carbon response model with optional ontology-sidecar output.
- Updated `backend/api/carbon.py` so `POST /carbon/calculate` can return:
  - ontology triple counts
  - validation status
  - detailed validation issues
  - optional Turtle serialization
- Kept the ontology sidecar out of `/solve`, so the user-facing answer path remains lightweight while the debug path carries the richer provenance payload.

### Challenges

1. The ontology needed to be rich enough for provenance and validation, but it could not become a second arithmetic engine competing with the deterministic calculator from Step 3.
2. The raw ontology note from the source folder was conceptual and needed to be adapted to the repo’s current product/stage/result structure.
3. The sidecar needed to support both:
   - partial official-data calculations with known missing inputs
   - fully override-based debug calculations
4. Validation had to distinguish true graph errors from expected data-quality warnings caused by incomplete official MX431adn inputs.

### Resolution

1. The ontology file was kept declarative: it defines product, scenario, stage, activity, factor, result, provenance, status, and unit vocabulary, while numeric multiplication and aggregation remain in Python.
2. The ontology service builds RDF instance graphs directly from `CarbonCalculationResult`, using the normalized product profile and source references to populate provenance links.
3. A custom validation layer was added on top of the RDF graph to check:
   - product-to-scenario linkage
   - scenario functional unit
   - requested-stage coverage
   - stage-result structure
   - computed-trace completeness
   - recyclability and total-result presence when expected
4. Validation now reports:
   - `valid` for structurally complete graphs
   - `warning` for graphs that are structurally sound but reflect missing official inputs
   - `invalid` only for actual ontology-sidecar construction failures
5. No new libraries were installed because the repo `.venv` already contained `rdflib` and `owlrl`, which were enough for parsing, serialization, and OWL-RL expansion.

### Outputs Produced

- `backend/ontologies/carbon_ontology.ttl`
- `backend/services/carbon_ontology_service.py`
- updated `backend/api/carbon_models.py`
- updated `backend/api/carbon.py`
- updated `backend/data/carbon/README.md`

### Validation Results

- Static syntax validation succeeded with:
  - `.venv/bin/python -m py_compile backend/services/carbon_ontology_service.py backend/api/carbon_models.py backend/api/carbon.py`
- Partial official-data endpoint validation succeeded:
  - `POST /carbon/calculate` with `include_ontology_sidecar = true`
  - returned ontology validation status `warning`
  - warning correctly reflected the current missing-input count
- Complete override-based ontology validation succeeded:
  - calculator status: `complete`
  - ontology validation status: `valid`
  - validation errors: `0`
  - validation warnings: `0`
  - ontology graph size through the debug endpoint response: `1115` triples
  - Turtle serialization was produced successfully and contained `lexmark_mx431adn`

### Step Status

Completed successfully.

## Step 8: Generate Answer Assets and Context Passages

### Goal

Generate reusable answer assets from the deterministic carbon outputs so the
system has curated context passages for:

- total footprint
- raw materials
- transportation
- use phase
- end of life
- recyclability

These assets are intended both for runtime answer composition and for later
evaluation or offline corpus inspection.

### Tasks Completed

- Added `backend/services/carbon_answer_assets_service.py`.
- Centralized generated carbon answer-asset creation in a shared service.
- Materialized stage-specific answer assets for:
  - overview
  - total footprint
  - raw materials
  - transportation
  - use phase
  - end of life
  - recyclability
  - missing inputs
- Refactored `backend/services/carbon_query_service.py` to reuse the shared
  answer-asset builder instead of keeping a separate local passage generator.
- Added `scripts/build_carbon_answer_assets.py` to produce a reusable JSONL
  corpus plus a machine-readable manifest.
- Generated:
  - `backend/data/carbon/corpus/carbon_docs.jsonl`
  - `backend/data/carbon/corpus/carbon_docs_manifest.json`
- Updated `backend/data/carbon/README.md` so the generated answer corpus and
  build command are documented with the rest of the carbon pipeline.

### Challenges

1. Step 5 already had a runtime-only synthetic passage builder, but Step 8
   required a reusable corpus generator without letting the solve path and the
   offline build path drift apart.
2. The generated corpus needed to support stage-specific questions for raw
   materials and transportation, not just a generic total-footprint summary.
3. The builder script initially failed when run directly from the repo root
   because Python could not resolve the `backend` package imports.
4. The current official MX431adn dataset is still incomplete, so the generated
   answer assets could not pretend to contain a final validated total.

### Resolution

1. A shared answer-asset service was introduced and both the runtime
   `/solve` carbon path and the offline corpus builder now call that same code.
2. Separate asset kinds were created for raw materials, transportation, use
   phase, and end of life so the system can answer stage-specific questions
   with targeted context instead of generic summaries.
3. The builder script now inserts the repo root into `sys.path` before loading
   backend modules, which makes it runnable as a normal repo script.
4. The generated docs preserve the calculator status honestly:
   - complete totals are emitted only when available
   - otherwise the corpus records partial totals and explicit missing-input
     diagnostics

### Outputs Produced

- `backend/services/carbon_answer_assets_service.py`
- updated `backend/services/carbon_query_service.py`
- `scripts/build_carbon_answer_assets.py`
- `backend/data/carbon/corpus/carbon_docs.jsonl`
- `backend/data/carbon/corpus/carbon_docs_manifest.json`
- updated `backend/data/carbon/README.md`

### Validation Results

- Static syntax validation succeeded with:
  - `python3 -m py_compile backend/services/carbon_answer_assets_service.py backend/services/carbon_query_service.py scripts/build_carbon_answer_assets.py`
- Corpus generation succeeded with:
  - `python3 scripts/build_carbon_answer_assets.py`
- Generated-corpus result for the current repo state:
  - product count: `1`
  - doc count: `8`
  - supported product: `lexmark_mx431adn`
  - asset kinds: `overview`, `total`, `raw_materials`, `transportation`, `use_phase`, `end_of_life`, `recyclability`, `missing_inputs`
- Runtime validation after the refactor also succeeded:
  - a transportation-emissions solve query resolved to `mode = CARBON`
  - the answer cited the generated transportation asset id
  - returned sources included runtime overview, total, transportation, recyclability, and missing-input docs
- Current observed corpus status:
  - `status = partial` for `lexmark_mx431adn`
  - `missing_input_count = 22`
  - the generated corpus now exposes raw-material and transportation passages explicitly, as planned

### Step Status

Completed successfully.

## Step 9: Add Calculator and Solve-Path Tests

### Goal

Add regression tests for the new carbon feature before frontend work, with
coverage for:

- total carbon-footprint questions
- stage-breakdown questions
- recyclability questions
- insufficient-data behavior

### Tasks Completed

- Added deterministic calculator tests in `tests/test_carbon_service.py`.
- Added solve-path integration tests in `tests/test_carbon_solve.py`.
- Added a reusable carbon question set in `tests/carbon/test.jsonl`.
- Covered a complete override-based calculator scenario with explicit expected
  stage totals and recyclability outputs.
- Covered the current official-data repo state where `Lexmark MX431adn`
  remains partial because the normalized product inputs are still incomplete.
- Covered `/solve` behavior for:
  - total footprint
  - stage breakdown
  - recyclability
  - unsupported product handling
- Updated `backend/data/carbon/README.md` with the Step 9 test files and the
  recommended command to run them.

### Challenges

1. The current project `.venv` does not include `pytest`, even though the repo
   already contains pytest-style tests in other areas.
2. Importing `backend/api/solve.py` directly in this environment pulled in
   optional non-carbon dependencies such as `pandas` through the policy-router
   path, which blocked a straightforward solve-route test.
3. The real official `Lexmark MX431adn` data is still incomplete, so the solve
   tests had to assert honest partial behavior instead of a complete final
   footprint value.

### Resolution

1. Step 9 tests were written with `unittest`, so they run in the current
   environment and will also remain compatible with future pytest-based runs.
2. The solve-route test loads `solve.py` through a lightweight import harness
   that stubs only the non-carbon dependencies. This preserves the real carbon
   interception path while avoiding unrelated optional imports.
3. The test set was split into:
   - a complete deterministic override scenario for numerical validation
   - real-state `/solve` cases that validate partial answers, recyclability
     behavior, and unsupported-product responses

### Outputs Produced

- `tests/test_carbon_service.py`
- `tests/test_carbon_solve.py`
- `tests/carbon/test.jsonl`
- updated `backend/data/carbon/README.md`

### Validation Results

- Test execution succeeded with:
  - `.venv/bin/python -m unittest tests.test_carbon_service tests.test_carbon_solve -v`
- Deterministic calculator validation covered:
  - complete result status
  - expected total `36.585 kg CO2e`
  - explicit stage totals for raw materials, transportation, use phase, and end of life
  - recyclability `80%`
- Solve-path validation covered:
  - total footprint query routed to `mode = CARBON`
  - stage-breakdown query exposed all four carbon stages
  - recyclability query narrowed to `requested_stages = ['end_of_life']`
  - unsupported product query returned the supported-product guidance response

### Step Status

Completed successfully.

## Step 10: Add Honest Estimate Fallback and Provenance Disclosure

### Goal

Make the carbon feature useful even when official product-internal data is
missing by:

- producing an estimate when the system has enough fallback data to do so
- keeping an exact-only path for strict questions
- surfacing provenance, estimated fields, and uncertainty in the returned result

### Tasks Completed

- Enriched `backend/data/carbon/products/lexmark_mx431adn.json` with:
  - curated exact observations from official web sources
  - conflict-tracked TEC observations
  - an explicit `estimation_profile`
- Extended the calculator result schema to include:
  - `quality_status`
  - `estimated_fields`
  - `provenance`
  - `uncertainty_pct`
  - `uncertainty_kg_co2e`
  - `uncertainty_range_kg_co2e`
- Added estimate-aware stage quality and uncertainty handling in the carbon
  calculation service.
- Added exact-mass resolution from curated official observations.
- Added estimate fallback for:
  - raw materials
  - transportation
  - use phase
  - end of life
- Updated `/solve` to use an exact-first strategy with estimate fallback.
- Added strict-query handling so phrases such as `exact only`, `official only`,
  and `no estimate` keep the result partial instead of forcing an estimate.
- Updated answer assets so they explicitly state:
  - whether the result is estimated
  - approximate uncertainty
  - estimated inputs
  - provenance summary
- Updated tests so the default user-facing solve path now expects labeled
  estimated answers, while strict exact-only queries still expect partial
  responses.

### Challenges

1. The official MX431adn PDF remains unreadable, so the exact product BOM,
   transport route, use profile, and declared end-of-life split still cannot be
   extracted directly from the original source folder.
2. Some official sources conflict. In particular, the MX431adn TEC value
   differed across the curated sources, so the system needed a source-priority
   rule instead of pretending those values were identical.
3. The feature needed to stay honest: adding estimates could not silently
   replace the existing exact/partial logic or make strict users lose the
   ability to ask for exact-only answers.

### Resolution

1. Exact observations were separated from fallback estimates inside the product
   JSON, so the system can tell the user which inputs came from official sources
   and which came from explicit estimate defaults.
2. A conflict-aware observed-facts layer was added for TEC, and the estimate
   profile now records why the ENERGY STAR value was preferred over the other
   observed values.
3. `/solve` now performs:
   - exact official calculation first
   - estimate fallback only when needed and only when strict exact terms are
     absent
4. The result schema was extended so the final answer can disclose not just a
   number, but also whether it is exact or estimated, how large the uncertainty
   is, and which fields were estimated.

### Outputs Produced

- updated `backend/data/carbon/products/lexmark_mx431adn.json`
- updated `backend/services/carbon_calculation_service.py`
- updated `backend/services/carbon_query_service.py`
- updated `backend/services/carbon_answer_assets_service.py`
- updated `backend/api/carbon_models.py`
- updated `tests/test_carbon_service.py`
- updated `tests/test_carbon_solve.py`
- updated `tests/carbon/test.jsonl`
- updated `backend/data/carbon/README.md`

### Validation Results

- Static syntax validation succeeded with:
  - `python3 -m py_compile backend/services/carbon_calculation_service.py backend/services/carbon_query_service.py backend/services/carbon_answer_assets_service.py backend/api/carbon_models.py tests/test_carbon_service.py tests/test_carbon_solve.py`
- Runtime calculator validation succeeded:
  - exact-only repo state remained `partial`
  - estimate-enabled repo state returned `status = complete`
  - estimate-enabled result returned `quality_status = hybrid_estimate`
  - estimate-enabled result total was `100.48496509845116 kg CO2e`
  - estimate-enabled result uncertainty was about `26.66%`
- Solve-path validation succeeded:
  - default carbon-footprint query returned a labeled estimated answer
  - stage-breakdown query returned a labeled estimated breakdown
  - recyclability query returned an estimated recyclability result
  - exact-only query remained partial and cited missing inputs
- Unit-test validation succeeded with:
  - `.venv/bin/python -m unittest tests.test_carbon_service tests.test_carbon_solve -v`
  - `4` tests passed

### Step Status

Completed successfully.

## Step 11: Integrate the Frontend with the Solve Pipeline

### Goal

Complete the remaining roadmap item by updating the frontend so the user-facing
UI:

- calls `/solve` instead of the legacy flat `/search` path
- works for both generic passport questions and carbon-specific questions
- displays structured carbon results, estimate disclosure, provenance, and
  cited sources

### Tasks Completed

- Replaced the frontend query flow in `frontend/src/services/api.js` with a
  reusable JSON request helper and a new `solveQuery(...)` client.
- Added endpoint fallback handling so the frontend can work whether the backend
  is exposed as `/solve` or `/api/solve`.
- Reworked `frontend/src/pages/HomePage.jsx` to:
  - maintain answer/loading/error state
  - call the solve endpoint
  - expose example carbon queries
  - support an optional product input
- Rebuilt `frontend/src/components/SearchBar.jsx` into a query form designed for
  question-style input instead of one-line semantic search strings.
- Replaced the old flat hit-list renderer in
  `frontend/src/components/ResultsList.jsx` with a solve-response view that
  shows:
  - answer text
  - mode, product, confidence, and estimate-fallback chips
  - carbon total, quality, uncertainty, and recyclability metrics
  - raw-material, transportation, use-phase, and end-of-life stage cards
  - estimated inputs
  - field-level provenance
  - source snippets

### Challenges

1. The existing frontend was still wired to `REACT_APP_API_URL =
   http://localhost:8000/api`, while the current FastAPI app mounts `/solve` at
   the root and does not include the aggregated `/api` router in `backend/main.py`.
2. The old UI only knew how to render `search.results[]`, but the solve route
   returns an answer-centric payload with `answer`, `steps`, `sources`, `mode`,
   and optional nested `carbon` data.
3. The user-facing carbon result needed to remain honest about estimation while
   still being readable and useful in a browser UI.

### Resolution

1. The frontend API layer now tries both base-url shapes, so it remains usable
   in environments configured for either `/api/...` or root-mounted routes.
2. The page was moved to a solve-first state model rather than a search-hit
   list model.
3. The new result renderer separates:
   - the top-level answer
   - carbon summary metrics
   - stage-level details
   - provenance and source evidence
4. Estimate fallback, uncertainty, and missing-input information are now exposed
   directly in the rendered response rather than being hidden in raw JSON.

### Outputs Produced

- updated `frontend/src/services/api.js`
- updated `frontend/src/pages/HomePage.jsx`
- updated `frontend/src/components/SearchBar.jsx`
- updated `frontend/src/components/ResultsList.jsx`

### Validation Results

- Frontend production build succeeded with:
  - `npm --prefix frontend run build`
- The new frontend flow now matches the backend solve payload shape used by:
  - general `/solve` answers
  - carbon estimate answers
  - strict exact-only carbon answers
- The UI now renders:
  - estimated totals with uncertainty
  - stage breakdown cards for raw materials, transportation, use phase, and end of life
  - recyclability metrics
  - provenance entries and source snippets

### Step Status

Completed successfully.

## Step 12: Add an Advanced Audit Drill-Down Page

### Goal

Expose the raw backend audit artifacts for users who want deeper inspection,
without cluttering the main answer view. The drill-down should show:

- the raw `steps` object as JSON
- per-trace calculation formulas and items when trace data is available
- ontology or debug sidecar payloads when the backend includes them

### Tasks Completed

- Added `frontend/src/components/AdvancedAuditView.jsx`.
- Added a conditional `Open advanced audit` button to
  `frontend/src/components/ResultsList.jsx`.
- Updated `frontend/src/pages/HomePage.jsx` to switch between:
  - the main answer page
  - the advanced audit page
- Added a `Back to answer` button on the advanced audit page.
- Added a copy-to-clipboard action for the full response JSON.

### Challenges

1. The frontend is still a small single-page React app without a router, so the
   audit page needed to behave like a distinct page without introducing a full
   routing dependency.
2. Not every backend response contains all advanced payloads. In particular,
   ontology sidecar data is currently available through the debug endpoint path,
   but not the standard `/solve` response.
3. The result object contains nested stage traces, so the audit page needed to
   flatten those traces into a readable inspection surface.

### Resolution

1. A lightweight page-state toggle was added in `HomePage.jsx` rather than
   adding routing infrastructure.
2. The advanced-audit button is shown only when advanced data is available from
   the current response.
3. The new drill-down page now separates:
   - raw step metadata
   - trace-level formula and emissions items
   - ontology/debug payloads or an explicit absence message

### Outputs Produced

- `frontend/src/components/AdvancedAuditView.jsx`
- updated `frontend/src/components/ResultsList.jsx`
- updated `frontend/src/pages/HomePage.jsx`

### Validation Results

- Frontend production build succeeded with:
  - `npm --prefix frontend run build`
- The main answer page now exposes an advanced-audit button when backend audit
  data is present.
- The advanced audit page now shows:
  - `steps` JSON
  - trace-level formulas/items
  - ontology/debug payloads if returned
  - a back button to return to the main answer view

### Step Status

Completed successfully.

## Step 13: Make Backend Startup Tolerant of Optional Router Dependencies

### Goal

Allow the backend and frontend demo flow to start even when optional router
training dependencies such as `pandas` are not installed in the local virtual
environment.

### Tasks Completed

- Updated `backend/api/solve.py` so the policy-router import is optional at
  module-load time.
- Kept the existing runtime fallback behavior, where missing router support
  degrades to the adaptive route instead of crashing the process.

### Challenges

1. The backend imported `backend.services.policy_router` at module import time,
   which pulled in `pandas` even for normal `/solve` and carbon requests that do
   not require the supervised router.
2. This caused local `uvicorn` startup to fail before the app could serve the
   frontend.

### Resolution

1. Wrapped the policy-router import in `backend/api/solve.py` with a safe
   fallback.
2. Updated `_get_router()` so it returns `None` immediately when the optional
   router stack is unavailable.
3. This preserves the existing `ADAPTIVERAG` fallback path for router-less
   environments while keeping `ROUTER` mode available when the full dependency
   set is installed.

### Outputs Produced

- updated `backend/api/solve.py`

### Validation Results

- Static validation succeeded with:
  - `python3 -m py_compile backend/api/solve.py`
- Runtime startup validation succeeded with:
  - `.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8010`
- Verified result:
  - the backend passed the earlier `ModuleNotFoundError: No module named 'pandas'`
    startup failure
  - the symbolic reasoner initialized and the app reached `Application startup complete`

### Step Status

Completed successfully.

## Step 14: Fix Frontend Blank Page on Browser Startup

### Goal

Resolve the frontend runtime issue where `localhost:3000` served a blank page
instead of rendering the React application.

### Tasks Completed

- Updated `frontend/src/services/api.js` to read `REACT_APP_API_URL` safely in a
  plain browser context.
- Kept the existing fallback base URL of `http://localhost:8000/api`.

### Challenges

1. The webpack setup in this repo does not inject a browser `process` object by
   default.
2. The frontend directly referenced `process.env.REACT_APP_API_URL`, which
   caused a browser-side `process is not defined` failure before React mounted.
3. Because the app crashed during module initialization, the browser showed a
   fully blank white page.

### Resolution

1. Replaced the direct `process.env...` access with a guarded expression:
   - check `typeof process !== "undefined"`
   - then read `process.env.REACT_APP_API_URL` only if available
   - otherwise fall back to the default backend URL
2. Verified that the served dev bundle no longer contains the unsafe direct
   `process.env.REACT_APP_API_URL` runtime dependency.

### Outputs Produced

- updated `frontend/src/services/api.js`

### Validation Results

- Frontend production build succeeded with:
  - `npm --prefix frontend run build`
- Live dev bundle validation succeeded:
  - the bundle served from `http://127.0.0.1:3000` no longer contained the
    unsafe `process.env.REACT_APP_API_URL` reference
  - the dev server continued serving the current app bundle and the React app
    strings remained present

### Step Status

Completed successfully.

## Step 15: Add a Separate Auto-Orchestrated Workspace Without Regressing Carbon Solve

### Goal

Restore the user-facing behavior where the system can automatically inspect
memory, retrieval, and symbolic evidence for non-carbon questions, while
keeping the newer carbon solve workspace unchanged.

### Tasks Completed

- Reviewed the frontend history and confirmed the original React UI was only a
  plain semantic-search screen and did not previously perform multi-source
  orchestration.
- Added a new backend orchestration endpoint at `POST /solve_auto`.
- Implemented automatic evidence composition that:
  - checks session memory
  - checks retrieval/search evidence
  - checks symbolic reasoning when a product is provided
  - filters weak search context for memory-style questions
  - passes the retained evidence to the LLM context answerer
- Added a separate frontend workspace toggle so the current carbon solve flow
  remains available as-is and the new orchestration behavior is accessed as a
  parallel feature.
- Added session-aware memory seeding in the UI so memory queries can be
  demonstrated without leaving the frontend.
- Added backend tests for the new route and verified the frontend build.

### Challenges

1. The current solve frontend was intentionally simplified around `/solve` and
   carbon-specific answer rendering, so memory-sensitive questions were falling
   through to `BASE` retrieval and returning irrelevant search snippets.
2. The old frontend did not contain the orchestration logic the user expected;
   it only called `/search`.
3. Memory retrieval is session-scoped, so a frontend that hides the session
   value makes memory behavior look inconsistent even when the backend is
   correct.
4. The new feature needed to coexist with the carbon solve work instead of
   replacing it.

### Resolution

1. Introduced a new route in `backend/api/solve_auto.py` instead of changing
   the current `/solve` carbon flow.
2. Reused the existing retrieval and answerer utilities from `backend/api/solve.py`
   so the new path can synthesize search, memory, and symbolic evidence into a
   single answer with one audit trail.
3. Added heuristics so memory-style questions prefer session memory and only
   include search evidence when it is strong enough to help rather than pollute
   the context.
4. Split the frontend into two workspaces:
   - `Carbon Solve`
   - `Auto Orchestrator`
5. Added a session input plus a memory save panel in the orchestration
   workspace so stored memory and follow-up questions remain aligned.

### Outputs Produced

- `backend/api/solve_auto.py`
- updated `backend/main.py`
- updated `backend/api/__init__.py`
- updated `frontend/src/services/api.js`
- `frontend/src/components/SolveWorkspace.jsx`
- `frontend/src/components/OrchestratedWorkspace.jsx`
- updated `frontend/src/pages/HomePage.jsx`
- `tests/test_solve_auto.py`

### Validation Results

- Backend route validation succeeded with:
  - `.venv/bin/python -m unittest tests.test_solve_auto -v`
- Verified route behavior:
  - memory-style questions now return seeded session memory from `/solve_auto`
  - carbon questions sent to `/solve_auto` still delegate to the existing carbon
    flow
- Frontend production build succeeded with:
  - `npm --prefix frontend run build`

### Step Status

Completed successfully.

## Step 16: Tighten Orchestrated Memory Routing for Query Variants

### Goal

Make the new auto-orchestrated workspace robust to small wording changes in
memory-oriented questions so the system still prefers session memory when it is
the strongest evidence.

### Tasks Completed

- Reproduced the failure case where a phrasing variant such as
  `What packaging ProductA prefers?` still surfaced retrieval-first output even
  though the memory evidence scored higher.
- Expanded the memory-intent heuristic in `backend/api/solve_auto.py` to catch
  packaging and supplier preference variants beyond the literal `did I say`
  phrasing.
- Added a memory-dominance rule so high-scoring session memory can suppress weak
  search passages even when the wording is imperfect.
- Added a direct memory-answer fallback for packaging and supplier questions so
  the answer remains useful even when the generic context answerer would
  otherwise echo an irrelevant top search snippet.
- Added a regression test for the exact packaging-variant question.

### Challenges

1. The orchestration path originally treated the packaging-only variant as not
   explicitly memory-like, so search context still entered the composed passage
   list ahead of memory.
2. In environments without a live LLM backend, the fallback answerer returns
   the first passage snippet, so evidence ordering matters a great deal.
3. The source list already showed memory as the strongest signal, which made
   the answer look inconsistent and untrustworthy from the user’s perspective.

### Resolution

1. Broadened `_is_memory_like_query(...)` so preference questions about
   packaging and suppliers count as memory-led.
2. Added a `memory_dominant` gate based on relative memory and search scores.
3. When memory is dominant, search evidence is filtered out unless it is strong
   enough to help.
4. Added `_derive_memory_answer(...)` so packaging/supplier questions can be
   answered directly from the top memory record with citation when needed.

### Outputs Produced

- updated `backend/api/solve_auto.py`
- updated `tests/test_solve_auto.py`

### Validation Results

- Route verification succeeded for:
  - `POST /solve_auto` with `What packaging ProductA prefers?`
- Verified result:
  - the answer now cites the memory result instead of the unrelated retrieval
    snippet
  - search evidence is marked as checked but excluded for that case
- Regression tests succeeded with:
  - `.venv/bin/python -m unittest tests.test_solve_auto -v`

### Step Status

Completed successfully.

## Step 17: Secure Local OpenAI Key Handling and Enable GPT-5 Answer Synthesis

### Goal

Allow the backend answerer to use a local OpenAI API key safely for curated
answers, while keeping the secret out of version control and improving GPT-5
compatibility.

### Tasks Completed

- Added the local key file pattern to `.gitignore`, including `openApikey.rtf`.
- Removed the credential-like hardcoded string from `backend/api/answerer_ctx.py`.
- Updated the answerer so the modern OpenAI SDK path prefers the Responses API
  before falling back to Chat Completions.
- Added `scripts/start_backend_with_openai.sh` to:
  - extract the key from `openApikey.rtf`
  - export `OPENAI_API_KEY`
  - set `GEN_MODEL` to a GPT-5 model by default
  - start the backend with the required environment
- Updated `.env.example` to reflect GPT-5-based answer synthesis settings.

### Challenges

1. The local key was stored in an `.rtf` file rather than a plain `.env` file,
   so the startup path needed to extract the `sk-...` token safely.
2. The existing answerer supported OpenAI via Chat Completions, but GPT-5 is
   better aligned with the Responses API.
3. The repository still contained a credential-like string in source code,
   which is a security risk even if the code no longer depends on it.

### Resolution

1. Added explicit ignore rules for the local key file.
2. Implemented key extraction in the startup helper with a Python regex, with a
   `textutil` fallback for RTF conversion when needed.
3. Updated the answerer to use `client.responses.create(...)` first and only
   fall back to Chat Completions if necessary.
4. Removed the hardcoded credential-like string from the code file.

### Outputs Produced

- updated `.gitignore`
- updated `.env.example`
- updated `backend/api/answerer_ctx.py`
- `scripts/start_backend_with_openai.sh`

### Validation Results

- Syntax validation succeeded with:
  - `python3 -m py_compile backend/api/answerer_ctx.py`
- Startup helper dry-run succeeded with:
  - `DRY_RUN=1 scripts/start_backend_with_openai.sh`
- Verified result:
  - the helper loaded the local key file path
  - the helper defaulted the answer model to `gpt-5.4`

### Step Status

Completed successfully.

## Step 18: Add Explicit LLM Usage Trace to Answer Responses

### Goal

Make it obvious from both the backend response and the frontend UI whether a
live LLM was used for the final answer, which provider/model/API path were
involved, and when the system stayed on a deterministic fallback path.

### Tasks Completed

- Added structured answer-trace metadata in `backend/api/answerer_ctx.py`.
- Extended the answerer to return both:
  - the answer text
  - a structured trace with provider/model/API/path metadata
- Propagated `answer_trace` through:
  - `/solve` search answers
  - `/solve_auto` orchestrated answers
  - carbon answer synthesis
- Added an `ANSWER` step in the orchestrated trail so advanced audit clearly
  shows the final synthesis path.
- Updated the frontend answer panel to display chips for:
  - whether an LLM was used
  - provider
  - API path
  - final answer path
- Updated the advanced audit page to render the full answer-synthesis trace.
- Added regression assertions for the new `answer_trace` object in
  `tests/test_solve_auto.py`.

### Challenges

1. The existing answerer returned only a plain string, so there was no stable
   place to expose runtime metadata about LLM usage.
2. The system can legitimately override an LLM-generated candidate with a
   deterministic memory answer, so `LLM attempted` and `LLM used for final
   answer` are not always the same thing.
3. The answer-trace feature needed to work consistently across orchestrated,
   retrieval, and carbon flows.

### Resolution

1. Added `answer_with_context_detailed(...)` and `answerer_config()` to the
   answerer module.
2. Introduced a normalized `answer_trace` object carrying:
   - configured provider/model
   - whether an LLM attempt happened
   - whether the final answer actually used the LLM
   - provider/API/model used when applicable
   - fallback path and reason when not
3. In the orchestration flow, deterministic memory overrides now explicitly set
   the answer path to `memory_direct` while preserving the fact that an LLM may
   have been attempted earlier.
4. Surfaced the trace in the frontend main view and advanced audit view.

### Outputs Produced

- updated `backend/api/answerer_ctx.py`
- updated `backend/api/solve.py`
- updated `backend/api/solve_auto.py`
- updated `backend/services/carbon_query_service.py`
- updated `frontend/src/components/ResultsList.jsx`
- updated `frontend/src/components/AdvancedAuditView.jsx`
- updated `tests/test_solve_auto.py`

### Validation Results

- Syntax validation succeeded with:
  - `python3 -m py_compile backend/api/answerer_ctx.py backend/api/solve.py backend/api/solve_auto.py backend/services/carbon_query_service.py`
- Regression tests succeeded with:
  - `.venv/bin/python -m unittest tests.test_solve_auto -v`
- Frontend production build succeeded with:
  - `npm --prefix frontend run build`
- Live route verification succeeded for:
  - `POST /solve_auto` with `what is the prefered packaging of productA?does it help environment?`
- Verified result:
  - the response now includes `answer_trace`
  - the trail shows `configured_model: gpt-5`, `configured_provider: openai`
  - the final answer path is marked as `memory_direct`
  - `llm_used` is explicitly `false` for that answer

### Step Status

Completed successfully.
