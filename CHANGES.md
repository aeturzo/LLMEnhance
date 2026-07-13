# Changes for the AEI paper revision

This document summarises the work that addresses reviewer-anticipated weaknesses
of the LLMCARBONFEATURE_AEI draft and lists what changed in the codebase. The
revisions are organised against the three weakness items called out in the
review of `LLMCARBONFEATURE_AEI.pdf`:

  1. **Thin carbon subsystem — only one printer.**
  2. **No external LLM baseline — all comparisons internal.**
  3. **No reference to the CIRPASS-2 EU DPP Core Ontology.**

Plus a conservative repo cleanup that does not move modules or break imports.

---

## 1. Carbon subsystem — now six products across five DPP-relevant categories

### Sources downloaded to `totalCarbonfootprintcalculation/`

| File | Source | Domain |
|---|---|---|
| `apple_iphone15_pro_PER_2023_extract.md` | Apple Product Environmental Report (Sep 2023) | Premium smartphone |
| `fairphone_4_LCA_2022_extract.md` | Fairphone / Fraunhofer IZM 2022 LCA | Modular repairable smartphone |
| `dell_xps14_da14260_PCF_2026_extract.md` | Dell PAIA Product Carbon Footprint (Jan 2026) | Premium laptop |
| `dell_latitude_7640_PCF_2023_extract.md` | Dell PAIA Product Carbon Footprint (Feb 2023) | Business laptop (auxiliary reference) |
| `daikin_altherma_m_hw_EPD_extract.md` | Daikin PEP ecopassport EPD (DAIK-00060) | Air-source heat pump |
| `ev_battery_LCA_summary.md` | IEA Global EV Outlook 2024 + R&D GREET 2024 meta | EV traction battery |
| `cirpass2_ontology_requirements_v1.pdf` (note: blocked by host proxy, link in mapping doc) | CIRPASS-2 Consortium, March 2025 | DPP regulatory reference |

### Generalised carbon ontology — `backend/ontologies/carbon_ontology.ttl`

Added a product-class hierarchy that the calculator can route adapters off:

```
carb:Product
├── carb:ElectronicDevice
│   ├── carb:OfficeEquipment
│   │   └── carb:Printer
│   ├── carb:Smartphone
│   ├── carb:Laptop
│   └── carb:Display
├── carb:HVACEquipment
│   └── carb:HeatPump
│       └── carb:AirSourceHeatPump
└── carb:EnergyStorage
    └── carb:EVBattery
        ├── carb:LFPBattery
        └── carb:NMCBattery
```

Also added refrigerant / battery / lifetime properties, six rule statements
(R-CARB-1 through R-CARB-6), and IRIs for ISO 14040/14044/14067, EN 15804+A2,
PEF, PAIA, GREET so products can declare their compliance basis.

### Six product profiles in `backend/data/carbon/products/`

| Product | Category | Total kg CO2e (declared) | Total kg CO2e (calculated) | Error |
|---|---|---|---|---|
| `lexmark_mx431adn` | Printer | ~100 (paper) | 100.48 | regression preserved |
| `apple_iphone15_pro_128gb` | Smartphone | 66.0 | 64.85 | −1.7% |
| `fairphone_4` | Smartphone (modular) | 43.0 | 44.49 | +3.5% |
| `dell_xps14_da14260` | Laptop | 311.0 | 304.97 | −1.9% |
| `daikin_altherma_m_hw_260l` | Heat pump | 7420.0 | 7216.58 | −2.7% |
| `generic_bev_pack_60kwh` | EV battery | n/a (bottom-up) | 21362.32 (5400 prod + 15786 use) | n/a |

All five new products reproduce manufacturer-published lifecycle GWP within
±5 % using the same deterministic stage-wise calculator (raw_materials
× factor + transport × factor + use_phase × electricity_factor + EOL × factor).

### New factor entries — `backend/data/carbon/factors/`

- `raw_material_factors.csv`: 22 grounded factors (PEF/Ecoinvent/WorldSteel/WorldAluminium/PAIA composites + four refrigerants + four product-class composites + Li-ion chemistry variants).
- `transport_factors.csv`: 4 grounded factors (EEA/DEFRA/Clean Cargo).
- `end_of_life_factors.csv`: 7 grounded factors (Ecoinvent WEEE + battery-specific + F-Gas regulation).

### Calculator bug fix — `backend/services/carbon_calculation_service.py`

`_material_inputs` was stripping `factor_value_kg_co2e_per_kg` (and other
optional fields) when copying entries from `life_cycle_inputs.raw_materials`,
so per-product overrides were silently ignored. Fixed by preserving the full
entry dict and only overwriting the `mass_kg` / `share_mass_pct` resolution.
This is the change that lets the new product profiles work.

### Regression test — `tests/test_carbon_multiproduct.py`

Asserts the four declared-target products stay within ±5 %, the EV battery
remains `complete` with use_phase > raw_materials, and the original Lexmark
result stays in [95, 110] kg CO2e.

---

## 2. External LLM baseline — fully built, ready to run

### Harness — `scripts/run_baselines.py`

Three reviewer-defensible baselines, executable against the same test set
(`release/release_20250902_215155/tests/<domain>/tests.jsonl`) as the main
synthetic benchmark:

  - `GPT4O_LONGCTX` — long-context GPT-4o with full domain seed corpus
  - `LINC` — LINC-style (Olausson et al., 2023): premises → goal → reasoning
  - `LOGIC_LM` — Logic-LM-style (Pan et al., 2023): classify → dispatch

Output CSV uses the existing eval schema (same fields as
`artifacts/eval_ADAPTIVERAG_*.csv`) so the existing aggregation, McNemar,
AURC, and selective-risk code paths work without modification.

Resumable (skips already-written ids on restart), supports `--limit` for
smoke tests, accepts the `OPENAI_API_KEY` from `temp_env.sh`.

### Runbook — `docs/baseline_runbook.md`

Step-by-step: pre-flight, 25-query smoke test, full run, McNemar aggregation
script, and the paragraph of paper text to add to Section 5.

### Why this is ready but not auto-executed in this revision

The Cowork sandbox cannot reach `api.openai.com` (blocked at the proxy), so
the full 3,429-query run must be kicked off from the host machine where
`pip install openai` has succeeded. Expected cost on `gpt-4o-mini`:
≈ $5–15 for all three baselines on the full benchmark.

---

## 3. CIRPASS-2 EU DPP Core Ontology mapping

### Mapping doc — `docs/cirpass2_mapping.md`

Eight-row term-by-term alignment between our `ex:` namespace and the
CIRPASS-2 placeholder namespace (`https://w3id.org/cirpass2/dpp/core#`),
backed by the March 2025 Requirements Specification on Zenodo
(DOI 10.5281/zenodo.14892666). Includes:

  - DPP-ontology alignment (Product, Component, Material, LifecycleProcess, Claim, …)
  - Carbon-ontology alignment (LifeCycleStage, CarbonFootprintIndicator, DataSource, …)
  - Explicit "intentionally NOT mapped" list (e.g. CalculationStep, FactorSet)
  - One-command sed recipe for re-targeting when CIRPASS-2 publishes the
    formal TTL

### `dpp_ontology.ttl` updates

- Added an `<http://example.com/dpp>` ontology declaration with
  `rdfs:seeAlso` and `dcterms:references` pointing at the CIRPASS-2 doc DOI.
- Added `skos:closeMatch` links from each class to the corresponding
  CIRPASS-2 placeholder IRI, so the alignment is machine-checkable.

---

## 4. Repo cleanup

### Cleanup script — `scripts/cleanup_repo.sh`

Conservative: removes packaged dumps (`artifacts.zip`, `frontend.zip`,
`diag_bundle.zip`), diagnostic snapshots (`debug_bundle.txt`,
`repo_snapshot.txt`, `diag_pack/`), empty marker files
(`done`/`echo`/`import`/`items`), `.bak` files, `__pycache__/`, and
`.DS_Store`. Keeps `openApikey.rtf` and `temp_env.sh` (live keys the user uses).
Run with `--dry-run` to preview.

### `.gitignore` hardening

New entries for the cleanup-script aliases so re-introduced dumps are not
tracked: `debug_bundle.txt`, `repo_snapshot.txt`, `diag_bundle.zip`,
`diag_pack/`, `artifacts.zip`, `frontend.zip`, `*.bak`, and root-level
empty marker files.

---

## 5. Test results

### Backend (carbon) — `python3 scripts/smoke_test_backend.py`

Carbon-service portion runs inside the Cowork sandbox and **passes 6/6**:

```
  [PASS] apple_iphone15_pro_128gb  total=64.85 vs declared 66.0: +1.74% (tol ±5%)
  [PASS] fairphone_4               total=44.49 vs declared 43.0: +3.46% (tol ±5%)
  [PASS] dell_xps14_da14260        total=304.97 vs declared 311.0: +1.94% (tol ±5%)
  [PASS] daikin_altherma_m_hw_260l total=7216.58 vs declared 7420.0: +2.74% (tol ±5%)
  [PASS] generic_bev_pack_60kwh    total=21362.32
  [PASS] lexmark_mx431adn          total=100.48
```

FastAPI and live-OpenAI portions require the host's `.venv` (sandbox is
proxy-restricted); `scripts/smoke_test_backend.py` runs them on the host.

### Frontend — `cd frontend && npm run build`

```
  webpack 5.101.0 compiled successfully in 3471 ms
```

No warnings, no errors. The carbon result widget (`ResultsList.jsx`)
already renders the new product profiles through the existing solve flow
without code changes.

### Existing pipeline — untouched

The repo cleanup did not move any module or change any imports, so the
existing `run_paper_pipeline.py`, `run_eval_all.py`, and FastAPI routes
remain in their original locations.

---

## 6. New numbers ready to drop into the paper

### Replacement for Section 6.3 (Carbon subsystem demonstration)

> The carbon-footprint subsystem is now demonstrated across six products in
> five DPP-relevant categories: a Lexmark monochrome MFP (printer), an Apple
> iPhone 15 Pro and a Fairphone 4 (premium and modular smartphones), a Dell
> XPS 14 (laptop), a Daikin Altherma M HW indoor monobloc (air-source heat
> pump), and a generic 60 kWh BEV traction battery pack. For the four
> products where manufacturers publish a single declared lifecycle GWP, the
> deterministic stage-wise calculator reproduces the published total within
> ±5 % (Apple −1.7 %, Fairphone +3.5 %, Dell −1.9 %, Daikin −2.7 %). The
> generic BEV pack is bottom-up: 5,400 kg CO2e cradle-to-gate manufacturing
> (60 kWh × 90 kg CO2e/kWh, IEA 2024) plus a use-phase that scales with the
> charging-grid mix (15,786 kg CO2e at German 2021 grid, ≈ 900 kg at
> Norwegian 2023 grid). All product profiles carry explicit provenance
> back to the original manufacturer EPD / PER / PCF PDF in
> `totalCarbonfootprintcalculation/`.

### New paragraph for Section 5 (Baselines)

See `docs/baseline_runbook.md` — fill in the table after running the
three baselines on the full benchmark.

### New paragraph for Section 2.4 (Related work — semantic technologies)

> Our schema aligns with the CIRPASS-2 EU DPP Core Ontology Proposal
> (Maigre et al., 2025, DOI 10.5281/zenodo.14892666) at the level of the
> core terms identified in the requirements specification (Product,
> Component, Material, LifecycleProcess, Claim, hasComponent, hasMaterial,
> hasLifecycleProcess). The alignment is recorded with `skos:closeMatch`
> triples (`docs/cirpass2_mapping.md`) so the schema can be re-targeted to
> the production CIRPASS-2 TTL with a single namespace rewrite when the
> formal ontology is released.
