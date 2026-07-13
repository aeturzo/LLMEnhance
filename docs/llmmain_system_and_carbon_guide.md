---
title: "LLMMain System and Carbon Calculator Guide"
author: "Codex Implementation Notes"
date: "2026-03-26"
---

# LLMMain System and Carbon Calculator Guide

## 1. Purpose of This Document

This guide explains the current `llmmain` system in plain language and then
describes, in detail, the carbon footprint and environmental indicator feature
that was added to it.

The document is written for a reader with little or no background in:

- LLM systems
- symbolic reasoning
- carbon-footprint calculations
- Digital Product Passport (DPP) systems

By the end of this document, you should understand:

1. what the system does in general
2. how the different modules work together
3. how the carbon and recyclability feature works
4. where the data comes from
5. how the system estimates missing values honestly
6. how to run the backend and frontend
7. which demo questions to ask to verify each major capability

---

## 2. What the System Is in General

`llmmain` is a hybrid question-answering system for Digital Product Passport
style use cases. It does not rely on only one technique. Instead, it combines
multiple reasoning and retrieval mechanisms:

- `BASE / SEARCH`
  Finds relevant text passages from the document corpus.
- `MEMORY`
  Retrieves session-specific notes or facts stored earlier by the user.
- `SYMBOLIC`
  Uses ontology-backed symbolic reasoning rules to answer compliance and
  structured product questions.
- `AUTO ORCHESTRATOR`
  Combines memory, search, and symbolic reasoning and then gives the combined
  context to the LLM for answer curation.
- `RL`
  A research endpoint that uses feature-based confidence and policy-style
  behavior for adaptive answering.
- `CARBON`
  A deterministic carbon and recyclability calculation path that is now routed
  through the same frontend.

In simple terms, the system tries to answer a question using the most suitable
combination of:

- documents
- remembered notes
- symbolic product knowledge
- a large language model
- deterministic carbon math

This is important because different questions need different treatment.

Examples:

- A factual document question like “What is the UN number for lithium-ion
  batteries?” is best answered from the document corpus.
- A personal/session question like “What note did I save about PrinterMemo1?”
  is best answered from memory.
- A rule/compliance question like “Which standards apply to ProductA?” is best
  answered from the symbolic reasoner.
- A carbon question like “What is the carbon footprint of Lexmark MX431adn?”
  should not be guessed by the LLM. It should be calculated deterministically,
  then explained by the answer layer.

---

## 3. High-Level System Architecture

At runtime, the system has the following major parts.

### 3.1 Frontend

The frontend has two main workspaces:

- `Carbon Solve`
  For general `/solve` questions, including carbon and recyclability.
- `Auto Orchestrator`
  For domain-aware composition of memory, search, and symbolic reasoning.

The frontend also shows an audit trail so you can inspect:

- which mode answered the question
- which domain was selected
- whether an LLM was used
- which provider/model/API were used
- which sources contributed
- the raw backend steps and trace information

### 3.2 Backend Endpoints

The main backend endpoints are:

- `POST /solve`
  The general solve pipeline.
- `POST /solve_auto`
  The multi-domain orchestrator pipeline.
- `POST /solve_rl`
  The RL/research-oriented adaptive endpoint.
- `GET /carbon/products`
  Lists supported carbon calculation products.
- `POST /carbon/calculate`
  Debug endpoint for direct structured carbon calculations.
- `POST /memory/put`
  Saves a memory note to a session.

### 3.3 Symbolic Reasoners

The backend now loads multiple symbolic reasoners at startup, one per domain.

The currently supported symbolic domains are:

- `battery`
- `lexmark`
- `viessmann`
- `textiles` (loaded for compatibility, but not the focus of the current demo)

This means the orchestrator can now choose the correct ontology-backed
reasoning path for different product families instead of using a single
default ontology for everything.

### 3.4 LLM Answer Layer

The system uses an LLM to curate answers when a curated answer is useful and
safe.

The LLM is used for:

- retrieval-backed answers
- orchestrated memory + search + symbolic answers
- some carbon answers, after the deterministic calculation has already
  produced the numeric result

The LLM is **not** trusted to invent the carbon math itself.

The system records an `answer_trace` so the user can see:

- whether an LLM was configured
- whether it was attempted
- whether it was actually used for the final answer
- which provider/model/API path were involved
- whether a deterministic fallback answered instead

This is especially important because some direct memory questions are answered
deterministically on purpose, even if an LLM is available.

---

## 4. General Answer Flow

This section explains how a normal non-carbon question moves through the
system.

### 4.1 Search / Base Flow

For a document-driven question:

1. the question is sent to the backend
2. the retrieval layer finds the best matching passages
3. those passages are assembled as context
4. the LLM curates an answer from that context
5. the UI shows the answer and the source snippets

Example:

- Question: `What is the UN number for lithium-ion batteries?`
- Expected module: `BASE / SEARCH`
- Expected result type: document-grounded answer with LLM curation

### 4.2 Memory Flow

For a session-specific note:

1. a note is saved under a session id
2. the user asks a follow-up question in the same session
3. the system retrieves the relevant note
4. depending on the phrasing, the system either:
   - answers directly from memory, or
   - uses the LLM to phrase a memory-grounded answer

Example:

- Saved note: `For PrinterMemo1, refurbish the unit before replacement and keep toner returns on the green lane.`
- Question: `Remind me what note I saved about PrinterMemo1 green lane and replacement.`

### 4.3 Symbolic Flow

For compliance or rule questions:

1. the system determines the domain
2. it identifies the product
3. it queries the symbolic reasoner for applicable rules/triples
4. the symbolic evidence is converted to answerable context
5. the LLM curates the final response

Example:

- Question: `Name two compliance standards that apply to ProductA.`
- Domain: `battery`
- Expected result type: symbolic answer with LLM curation

### 4.4 Auto Orchestrator Flow

The `Auto Orchestrator` combines multiple evidence types:

1. session memory
2. document retrieval
3. symbolic reasoning

It then decides:

- which evidence to include
- which evidence to suppress as weak/noisy
- whether to let the LLM produce the final answer
- whether a deterministic fallback is more honest

This is the closest part of the system to “automatic mode selection” for
multi-source product questions.

Example blended question:

- `What note did I save about ProductA packaging, and name one compliance standard for ProductA?`

This can combine:

- memory evidence
- symbolic evidence
- LLM curation, if appropriate

---

## 5. Where the Carbon Feature Fits Into the System

The carbon feature is a **new deterministic sub-system** that was integrated
into the existing QA architecture.

This was done carefully so that:

- the old QA features continue to work
- carbon answers do not depend on LLM guessing
- carbon answers can still be explained in natural language
- unsupported products are rejected honestly

### 5.1 Why Carbon Uses a Deterministic Engine

Carbon-footprint answers are different from ordinary QA answers.

If a user asks:

- `What is the carbon footprint of Lexmark MX431adn?`

the system should not simply retrieve a snippet and let the LLM improvise a
number. Instead, it should:

1. resolve the product
2. load the product profile and factor tables
3. run explicit stage-by-stage calculations
4. determine whether the result is exact, estimated, or partial
5. disclose provenance and uncertainty
6. only then present the answer

This is why the carbon path is implemented as:

- `compute first`
- `answer second`

### 5.2 What Environmental Indicators Are Currently Supported

At the moment, the environmental feature supports:

- total carbon footprint
- stage-level emissions
  - raw materials
  - transportation
  - use phase
  - end of life
- recyclability / end-of-life split
- provenance and uncertainty reporting

So when this document says “carbon/environment indicator,” the currently
implemented indicators are:

- `kg CO2e`
- recyclability percentage
- recoverable mass
- end-of-life route split implications

### 5.3 Current Product Scope

The currently supported carbon product is:

- `Lexmark MX431adn`

For other products, the system should respond honestly that carbon calculation
support is not available yet. This is already the intended behavior.

---

## 6. Carbon Data Pipeline

The carbon calculator depends on a normalized data layer under:

- `backend/data/carbon/`

The raw source material remains in:

- `totalCarbonfootprintcalculation/`

This separation is critical. It means:

- raw source files remain untouched
- the runtime reads only normalized, machine-readable assets
- preprocessing is reproducible
- future source updates can be rebuilt cleanly

### 6.1 Raw Source Inputs

The original raw folder contains:

- product files and notes
- emission factor workbooks
- ontology notes
- PDFs and spreadsheets

Important raw sources used in the current implementation include:

- `Lexmark MX431adn.pdf`
- `env-epd_21_1683665824.pdf`
- `Calculation process .pdf`
- `Ontology.pdf`
- `CoM-Emission-factors-for-national-electricity-2024.xlsx`
- `EF-LCIAMethod_CF(EF-v3.1).xlsx`

### 6.2 Normalized Runtime Assets

The runtime assets created from the raw sources include:

- product profiles in `backend/data/carbon/products/`
- factor tables in `backend/data/carbon/factors/`
- mapping files in `backend/data/carbon/mappings/`
- extracted source summaries in `backend/data/carbon/extracted/`
- generated answer assets in `backend/data/carbon/corpus/`

### 6.3 Important Data Files for Lexmark MX431adn

The main normalized product file is:

- `backend/data/carbon/products/lexmark_mx431adn.json`

This file contains:

- product identity
- current calculation scope
- exact observed facts
- empty official placeholders when data is still missing
- estimate-ready defaults
- uncertainty defaults
- provenance source references

### 6.4 Exact Observed Facts Already Curated

The current product profile includes curated exact or derived observations such
as:

- product mass: `12.8 kg`
- packaged mass: `14.7 kg`
- packaging mass: `1.9 kg` (derived)
- recommended monthly pages: `800–8000`
- maximum duty cycle: `80000 pages/month`
- preferred TEC value: `0.46 kWh/week`

The TEC field is especially important because the sources disagreed:

- ENERGY STAR: `0.46 kWh/week`
- official product page: `0.47 kWh/week`
- official brochure: `0.44 kWh/week`

The system keeps this conflict visible and explicitly records why the ENERGY
STAR value was preferred.

### 6.5 Missing Official Inputs

The current official `life_cycle_inputs` still have gaps because the raw
`Lexmark MX431adn.pdf` is encrypted and could not be extracted automatically.

The currently missing official product-internal data includes:

- material masses
- declared transport route
- declared use-phase lifetime inputs
- declared end-of-life split

These are stored as placeholders rather than guessed official values.

This is important because the system distinguishes between:

- `official data we truly have`
- `values we still need`
- `values we estimate temporarily`

---

## 7. Carbon Calculation Methodology

The carbon calculator is implemented in:

- `backend/services/carbon_calculation_service.py`

It calculates four main life-cycle stages.

### 7.1 Stage 1: Raw Materials

For each material entry, the system applies:

`emissions = mass_kg × factor_kgCO2e_per_kg`

Example:

- plastics mass = `5.76 kg`
- plastics factor = `2.7 kg CO2e/kg`
- plastics emissions = `5.76 × 2.7 = 15.552 kg CO2e`

The stage total is the sum of all material entries.

### 7.2 Stage 2: Transportation

For each transport leg, the system applies:

`emissions = (mass_kg / 1000) × distance_km × factor_kgCO2e_per_ton_km`

Example:

- mass = `14.7 kg`
- distance = `18000 km`
- factor = `0.009 kg CO2e/ton-km`
- activity = `(14.7 / 1000) × 18000 = 264.6 ton-km`
- emissions = `264.6 × 0.009 = 2.3814 kg CO2e`

The stage total is the sum of all transport legs.

### 7.3 Stage 3: Use Phase

The use phase is based on lifetime electricity use:

`emissions = lifetime_energy_kwh × electricity_factor_kgCO2e_per_kWh`

Example in the current estimate profile:

- lifetime energy = `119.6 kWh`
- electricity factor = `0.438488002495411 kg CO2e/kWh`
- use-phase emissions = `119.6 × 0.438488... ≈ 52.44 kg CO2e`

### 7.4 Stage 4: End of Life

For each disposal route, the system calculates route mass first:

`route_mass_kg = total_mass_kg × route_rate_pct / 100`

Then:

`emissions = route_mass_kg × factor_kgCO2e_per_kg`

This is done separately for:

- recycling
- incineration
- landfill

The end-of-life stage total is the sum of the route emissions.

### 7.5 Recyclability

Recyclability is derived from the end-of-life split.

The main formula is:

`recoverable_mass_kg = total_mass_kg × recycling_rate_pct / 100`

Example with the current estimate profile:

- total mass = `12.8 kg`
- recycling rate = `80%`
- recoverable mass = `12.8 × 0.80 = 10.24 kg`

The system also derives:

- incineration mass
- landfill mass

### 7.6 Result Status Logic

The calculator returns one of the following high-level states:

- `complete`
  All requested stages were computable.
- `partial`
  Some stages were computed, but a full honest total could not be guaranteed.
- `missing`
  No usable calculation could be produced.

The output also carries a `quality_status`, such as:

- `exact`
- `hybrid_estimate`
- `scenario_override`
- `partial`

---

## 8. How Estimation Works

Because the official MX431adn product-internal data is still incomplete, the
system uses an **exact-first, estimate-second** strategy.

### 8.1 Exact-First Principle

Whenever possible, the system first tries to use:

- official product observations
- official registry observations
- normalized electricity factors
- official declared inputs, if available

Only the still-missing fields are estimated.

### 8.2 Estimate Profile

The product file contains an explicit `estimation_profile`.

This profile currently provides estimated defaults for:

- raw material mix and masses
- transportation legs
- use-phase lifetime and electricity factor context
- end-of-life split
- uncertainty percentages

These estimated values are not hidden. They are labeled and tracked.

### 8.3 Current Lexmark Estimate Inputs

The current estimated raw-material profile uses:

- plastics: `45%`
- steel: `35%`
- electronics: `12%`
- elastomers: `4%`
- other: `4%`

The current estimated transport profile uses:

- ship leg: `18000 km`
- truck leg: `1000 km`

The current estimated use phase uses:

- TEC-based lifetime estimate
- default lifetime: `5 years`
- Germany 2021 electricity factor

The current estimated end-of-life profile uses:

- recycling: `80%`
- incineration: `15%`
- landfill: `5%`

### 8.4 Uncertainty Estimation

Each estimated stage has a default uncertainty percentage:

- raw materials: `35%`
- transportation: `30%`
- use phase: `20%`
- end of life: `25%`
- total default fallback: `32%`

The total uncertainty is calculated as a weighted average of stage
uncertainties based on each stage’s contribution to the total.

The system then derives:

- `uncertainty_pct`
- `uncertainty_kg_co2e`
- `uncertainty_range_kg_co2e`

### 8.5 Strict Exact Mode

If the user writes phrases such as:

- `exact only`
- `official only`
- `no estimate`

the system does **not** silently estimate the missing values.

Instead, it returns:

- a partial result
- missing-input explanation
- honest disclosure that a full exact footprint is not yet available

This is a key research and trust feature.

---

## 9. Provenance and Audit Trail

One of the main strengths of the current implementation is that the system
does not only give an answer. It also explains how it produced the answer.

### 9.1 Provenance in Carbon Results

The carbon results include provenance items such as:

- product mass
- packaged mass
- packaging mass
- TEC value
- estimated transport route
- estimated raw-material mix
- estimated use-phase profile
- estimated end-of-life split

Each provenance item can include:

- label
- value
- unit
- status
- method
- source references
- notes
- uncertainty

### 9.2 Advanced Audit Page

The frontend includes an advanced audit page that can show:

- `answer_trace`
- raw `steps`
- trace-level carbon formulas and items
- ontology/debug sidecar payloads, when present

This allows a user to inspect not only the final answer, but also the internal
reasoning trace.

### 9.3 LLM Usage Transparency

The audit trail shows:

- which LLM was configured
- which provider was used
- which API path was used
- whether the LLM was actually used for the final answer
- whether a deterministic fallback took over instead

This matters because:

- direct memory answers may intentionally bypass the LLM
- symbolic and retrieval answers usually use LLM curation
- carbon math is deterministic even when the wording is LLM-curated

---

## 10. Carbon Answer Behavior in Practice

### 10.1 Supported Carbon Question

Question:

`What is the carbon footprint of Lexmark MX431adn?`

Current expected behavior:

- product resolves to `lexmark_mx431adn`
- system computes all stages using exact observations plus estimate defaults
- result quality is `hybrid_estimate`
- answer includes total, uncertainty, and stage breakdown

The current verified total in the implemented system is approximately:

- `100.485 kg CO2e`

with approximate uncertainty around:

- `26.7%`

### 10.2 Supported Recyclability Question

Question:

`How recyclable is Lexmark MX431adn?`

Current expected behavior:

- recyclability about `80%`
- recoverable mass about `10.24 kg`
- answer explicitly says the end-of-life split is estimated

### 10.3 Strict Exact Carbon Question

Question:

`What is the exact carbon footprint of Lexmark MX431adn with no estimate?`

Current expected behavior:

- system refuses to fabricate a full exact total
- answer remains partial
- missing inputs are listed
- user is told that exact official calculation is not yet possible

### 10.4 Unsupported Product Carbon Question

Question:

`What is the carbon footprint of ProductV1?`

Current expected behavior:

- system abstains honestly
- answer says carbon calculation currently supports `Lexmark MX431adn` only

---

## 11. How the LLM Is Used in the System

The LLM is used for answer curation, not for replacing symbolic reasoning or
carbon arithmetic.

### 11.1 In Base/Search Questions

The LLM receives:

- the user question
- the retrieved passages

It then produces a concise grounded answer.

### 11.2 In Auto Orchestrator Questions

The LLM can receive:

- memory evidence
- search evidence
- symbolic evidence

It then combines them into one answer.

### 11.3 In Carbon Questions

The LLM may be used to phrase the answer, but only **after** the carbon engine
has already computed:

- stage totals
- total emissions
- quality label
- uncertainty
- provenance

This means the LLM helps with:

- explanation
- readability
- answer composition

but not with:

- carbon arithmetic
- factor multiplication
- uncertainty aggregation

### 11.4 Current OpenAI Startup

The project contains a helper script:

- `scripts/start_backend_with_openai.sh`

It loads the API key from `openApikey.rtf`, exports `OPENAI_API_KEY`, and
starts the backend.

If you want to explicitly use `gpt-5`, start the backend like this:

```bash
cd /Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain
GEN_MODEL=gpt-5 scripts/start_backend_with_openai.sh
```

---

## 12. How to Run the System

This section explains how to run the backend, the frontend, and the main demo
paths.

### 12.1 Start the Backend

Recommended command:

```bash
cd /Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain
GEN_MODEL=gpt-5 scripts/start_backend_with_openai.sh
```

This will:

- read the OpenAI key from `openApikey.rtf`
- set `OPENAI_API_KEY`
- start `uvicorn`
- use `gpt-5` for the answer layer

If you only want to test whether the key loads:

```bash
cd /Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain
GEN_MODEL=gpt-5 DRY_RUN=1 scripts/start_backend_with_openai.sh
```

### 12.2 Start the Frontend

Open a second terminal and run:

```bash
cd /Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain
npm --prefix frontend start
```

Then open:

```text
http://localhost:3000
```

### 12.3 Open the API Docs

The backend Swagger UI is available at:

```text
http://localhost:8000/docs
```

This is useful for:

- direct endpoint testing
- seeing request/response schemas
- checking the raw carbon endpoint

### 12.4 Workspaces in the Frontend

The frontend now has two workspaces:

#### Carbon Solve

Use this for:

- carbon footprint questions
- recyclability questions
- general solve questions

#### Auto Orchestrator

Use this for:

- domain-aware memory demos
- domain-aware search demos
- domain-aware symbolic reasoning demos

### 12.5 Advanced Audit

After you run a question, click:

- `Open advanced audit`

Use this page to inspect:

- LLM usage
- raw steps
- symbolic trace
- carbon formulas and trace items
- ontology/debug sidecar when available

---

## 13. How to Use the Auto Orchestrator

The orchestrator is easiest to understand if you think of it as a controlled
multi-source assistant.

### 13.1 Choose a Domain

In the `Auto Orchestrator` workspace, select one of:

- `Battery`
- `Lexmark`
- `Viessmann`
- `Auto`

This domain selection mainly affects:

- symbolic reasoning
- product inference
- example prompts

### 13.2 Use the Same Session for Memory

Memory is session-scoped. This means:

- first you save a note
- then you ask the follow-up question in the same session

If the session changes, the memory query may not find the saved note.

### 13.3 Product Inference

In many questions, you do not need to fill the product field manually.

The system can often infer products like:

- `ProductA`
- `PrinterL1`
- `ProductV1`
- `Lexmark MX431adn`

from the question text.

---

## 14. Verified Demo Questions

This section gives a practical question set you can use to verify the system.

For the memory demos, first save the note shown in the same session.

### 14.1 Battery Domain

#### Battery Base Demo

- Domain: `Battery`
- Question: `What is the UN number for lithium-ion batteries?`
- Expected module: `search/base`
- Expected answer type: document-grounded answer
- Expected audit:
  - `LLM: gpt-5`
  - `Provider: Openai`
  - `API: Responses`
  - `Answer Path: Llm`
  - source includes search evidence

#### Battery Memory Demo

Save this note first:

`For BatteryMemo1, use reusable metal trays for shipping and schedule service with NordCells every Tuesday.`

Then ask:

`Remind me what note I saved about BatteryMemo1 shipping and service schedule.`

Expected behavior:

- answer summarizes the saved note
- source is memory
- audit shows LLM curation over memory evidence

Expected audit:

- `LLM: gpt-5`
- `Answer Path: Llm`
- source includes memory evidence

#### Battery Symbolic Demo

- Domain: `Battery`
- Question: `Name two compliance standards that apply to ProductA.`
- Expected module: `symbolic`
- Expected answer shape:
  - `EN 62133-2`
  - `RoHS`
- Expected audit:
  - `LLM: gpt-5`
  - `Answer Path: Llm`
  - source includes symbolic evidence

### 14.2 Lexmark Domain

#### Lexmark Base Demo

- Domain: `Lexmark`
- Question: `What is the brand of Lexmark MS521dn?`
- Expected answer: `Lexmark`
- Expected audit:
  - `LLM: gpt-5`
  - `Answer Path: Llm`
  - source includes search evidence

#### Lexmark Memory Demo

Save this note first:

`For PrinterMemo1, refurbish the unit before replacement and keep toner returns on the green lane.`

Then ask:

`Remind me what note I saved about PrinterMemo1 green lane and replacement.`

Expected behavior:

- answer summarizes the note
- source is memory
- audit shows LLM curation over memory evidence

#### Lexmark Symbolic Demo

- Domain: `Lexmark`
- Question: `List two compliance standards for PrinterL1.`
- Expected answer shape:
  - `Wireless Compliance`
  - `RoHS`
- Expected audit:
  - `LLM: gpt-5`
  - `Answer Path: Llm`
  - source includes symbolic evidence

### 14.3 Viessmann Domain

#### Viessmann Base Demo

- Domain: `Viessmann`
- Question: `What is the website of Viessmann Climate Solutions SE?`
- Expected answer: the company website
- Expected audit:
  - `LLM: gpt-5`
  - `Answer Path: Llm`
  - source includes search evidence

#### Viessmann Memory Demo

Save this note first:

`For HeatMemo1, record the leak-check result and review the wireless module during commissioning.`

Then ask:

`Remind me what note I saved about HeatMemo1 commissioning and checks.`

Expected behavior:

- answer summarizes the note
- source is memory
- audit shows LLM curation over memory evidence

#### Viessmann Symbolic Demo

- Domain: `Viessmann`
- Question: `Name two compliance standards that apply to ProductV1.`
- Expected answer shape:
  - `EU F-Gas Regulation`
  - `RoHS`
- Expected audit:
  - `LLM: gpt-5`
  - `Answer Path: Llm`
  - source includes symbolic evidence

### 14.4 Carbon Solve Demo

Use the `Carbon Solve` workspace for the following questions.

#### Carbon Supported Product

- Question: `What is the carbon footprint of Lexmark MX431adn?`
- Expected behavior:
  - result is computed for `lexmark_mx431adn`
  - total is around `100.485 kg CO2e`
  - quality is `hybrid_estimate`
  - uncertainty is shown
  - stage breakdown is shown
  - provenance is shown

#### Carbon Stage Breakdown

- Question: `Give me a stage breakdown for Lexmark MX431adn carbon footprint.`
- Expected behavior:
  - stage cards appear for:
    - raw materials
    - transportation
    - use phase
    - end of life

#### Carbon Recyclability

- Question: `How recyclable is Lexmark MX431adn?`
- Expected behavior:
  - recyclability shown at about `80%`
  - recoverable mass shown
  - answer states this is estimated

#### Carbon Strict Exact Mode

- Question: `What is the exact carbon footprint of Lexmark MX431adn with no estimate?`
- Expected behavior:
  - result stays partial
  - system explains missing inputs
  - no silent estimation is done

#### Carbon Unsupported Product

- Question: `What is the carbon footprint of ProductV1?`
- Expected behavior:
  - system abstains honestly
  - says current carbon calculation support is limited to `Lexmark MX431adn`

---

## 15. Reading the Audit Trail Correctly

When you inspect a result, pay attention to the following fields.

### 15.1 `Mode`

This tells you which top-level path answered:

- `CARBON`
- `AUTO_COMPOSE`
- `BASE`
- other internal modes

### 15.2 `Domain`

In orchestrated mode, this shows which domain the system used:

- `battery`
- `lexmark`
- `viessmann`

### 15.3 `LLM`

This tells you whether the final answer used the LLM and which model.

Common cases:

- `LLM: gpt-5`
  The final answer was curated by the LLM.
- `LLM: Not used`
  A deterministic fallback answered instead.

### 15.4 `Answer Path`

This is especially important.

Typical values include:

- `llm`
- `memory_direct`
- `memory_symbolic_direct`
- `unsupported_product`

Interpretation:

- `llm`
  The final answer was written through the answer layer using the LLM.
- `memory_direct`
  The system answered directly from memory because that was more reliable.
- `memory_symbolic_direct`
  The system assembled the answer deterministically from memory plus symbolic
  evidence.
- `unsupported_product`
  The system abstained honestly because the requested carbon product is not yet
  supported.

### 15.5 Carbon Quality

For carbon, pay special attention to:

- `exact`
- `hybrid_estimate`
- `partial`

These labels are there so a user does not confuse:

- a fully official result
- a hybrid estimated result
- an incomplete result

---

## 16. Direct API Examples

If you want to test the system without the frontend, these examples are useful.

### 16.1 Save Memory

```bash
curl -X POST http://localhost:8000/memory/put \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-session",
    "content": "For PrinterMemo1, refurbish the unit before replacement and keep toner returns on the green lane."
  }'
```

### 16.2 Run the Auto Orchestrator

```bash
curl -X POST http://localhost:8000/solve_auto \
  -H "Content-Type: application/json" \
  -d '{
    "query": "List two compliance standards for PrinterL1.",
    "domain": "lexmark",
    "session": "demo-session"
  }'
```

### 16.3 Run a Carbon Solve Query

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the carbon footprint of Lexmark MX431adn?",
    "session": "demo-session"
  }'
```

### 16.4 Inspect Raw Carbon Calculation Output

```bash
curl -X POST http://localhost:8000/carbon/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "lexmark_mx431adn",
    "use_bootstrap_estimates": true,
    "include_trace": true,
    "include_ontology_sidecar": true,
    "include_ontology_turtle": true
  }'
```

This is useful when you want:

- the raw stage outputs
- trace formulas
- ontology validation metadata

---

## 17. Known Limitations

The system is usable now, but it is important to understand the current
limitations.

### 17.1 Carbon Scope Is Still Narrow

At present, carbon calculation support is implemented for:

- `Lexmark MX431adn`

Other products should return an honest unsupported answer.

### 17.2 Official Product PDF Is Still Encrypted

The raw `Lexmark MX431adn.pdf` could not be extracted automatically, which is
why several official life-cycle inputs are still missing.

### 17.3 Some Carbon Factors Are Still Seed Values

The factor tables for:

- raw materials
- transportation
- end of life

still contain seed or placeholder values where manual curation is pending.

This is why the current user-facing result is correctly labeled as a hybrid
estimate rather than a final certified footprint.

### 17.4 Memory Can Be Deterministic

Very direct memory questions may intentionally bypass the LLM and answer
directly from the top memory entry. This is not a bug. It is a design choice
for reliability.

---

## 18. Recommended Interpretation for Research and Demos

If you are using this system in a research setting or presentation, the most
important message is:

the system does not treat carbon as ordinary QA.

Instead, it:

1. uses structured product and factor data
2. calculates stage emissions deterministically
3. estimates only when allowed
4. records provenance and uncertainty
5. discloses when a result is estimated
6. uses the LLM only to curate the explanation layer

This gives you a stronger methodology than a simple retrieval chatbot because:

- the reasoning path is inspectable
- the carbon math is reproducible
- unsupported cases abstain honestly
- multi-domain QA remains available through the same overall system

---

## 19. Short Practical Checklist

If you just want to get started quickly, do this:

1. Start backend:

```bash
cd /Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain
GEN_MODEL=gpt-5 scripts/start_backend_with_openai.sh
```

2. Start frontend:

```bash
cd /Users/turzo/Desktop/StudyUIO/Research/LLMENHANCE/llmmain
npm --prefix frontend start
```

3. Open:

```text
http://localhost:3000
```

4. In `Auto Orchestrator`, test one domain at a time.

5. In `Carbon Solve`, ask:

`What is the carbon footprint of Lexmark MX431adn?`

6. Click `Open advanced audit` after each query.

7. Check:

- mode
- domain
- LLM usage
- answer path
- sources
- provenance
- uncertainty

---

## 20. Final Summary

The current system is a hybrid DPP QA platform that combines:

- retrieval
- memory
- symbolic reasoning
- LLM answer curation
- deterministic carbon and recyclability calculation

The carbon feature was added in a way that preserves the original QA system
while introducing a more trustworthy environmental calculation path.

The most important design decisions are:

- deterministic carbon arithmetic
- exact-first estimate-second policy
- strict exact mode
- provenance disclosure
- uncertainty disclosure
- honest abstention for unsupported products
- visible audit of whether the LLM was used

That combination makes the current system suitable both for demonstration and
for research-oriented explanation of how LLM, symbolic reasoning, and
structured environmental calculation can work together in one interface.
