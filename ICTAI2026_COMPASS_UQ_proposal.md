# Bias-Aware Selective QA for Digital Product Passports
### ICTAI 2026 proposal + experiment plan + 8-page draft outline
**Combination:** COMPASS (LLM selective-prediction QA) × Constraint-Calibrated Latent-Bias MCMC (fairness-aware UQ)
**Venue:** IEEE ICTAI 2026 — Boca Raton, FL, 2–4 Nov 2026. 8 pages IEEE 2-column (up to 10 paid), double-blind. **Deadline: 21 July 2026 AoE.**

---

## PART A — Proposal & Experiment Plan

### A.1 Thesis (one paragraph)
LLM question-answering over Digital Product Passports (DPPs) must withhold answers not only when evidence is *missing*, but when recorded attribute values are *systematically biased* — carbon footprint, repairability, and durability are frequently reported with uneven quality across regions and suppliers, so a naive RAG/LLM returns confident-but-biased values. We couple COMPASS (hybrid retrieval + persistent memory + targeted symbolic checks + selective abstention) with a constraint-calibrated latent-bias MCMC layer that infers the *clean* value of a queried attribute, its uncertainty, and its policy-target sensitivity. The result is a trustworthy QA tool that returns debiased answers with calibrated, policy-sensitive intervals and provenance — or abstains when the data cannot be trusted.

### A.2 Why ICTAI (category mapping)
- **AI Synergistic Models — Bridging AI Models for Advanced AI** (primary): LLM RAG ⨝ Bayesian/MCMC UQ.
- **Uncertainty in AI**: approximate probabilistic inference; KDD for uncertain data.
- **AI & Societal Impact**: Fairness, Accountability, Transparency; Trustworthy AI.
- **NLP**: Large language models; digital assistants.
- **KR/Reasoning & Semantic Web**: ontology-backed validation.

This multi-category fit is a genuine advantage; lead the paper as a *synergistic trustworthy-AI tool*, not an application.

### A.3 The gap and the core novelty
COMPASS abstains on **missing** evidence. The two MCMC works show DPP attribute values are often **present yet systematically biased**. No prior work makes an LLM QA system abstain (or correct) based on *measurement-bias uncertainty about the recorded value itself*. Core novelty claims:

- **N1 — Bias-aware selective prediction.** A new abstention criterion: abstain on *untrustworthy-but-present* evidence, driven by a fairness-aware latent-bias posterior over the queried attribute. First coupling of LLM selective prediction with a latent-bias UQ layer over the knowledge store.
- **N2 — Unified λ-tilted reliability.** One transparent trade-off dial links selective answering, debiasing, and calibration (see A.4). Elevates the work from "integration" to a *principle*.
- **N3 — Policy-sensitivity-aware answers.** QA outputs carry intervals that reflect regulatory/target sensitivity, making the fairness–fidelity trade-off auditable *at answer time*.
- **N4 — Benchmark + evidence.** A biased-attribute DPP+Open Food Facts QA benchmark, and evidence that bias-aware abstention removes confident-biased answers and restores calibration at no accuracy cost.

### A.4 The unifying spine: one λ-tilted distribution
All three of your works are the same object — a base distribution tilted by a transparent knob:

| Work | Tilted object | Form | Knob |
|---|---|---|---|
| COMPASS | answers | selective threshold τ over calibrated `c(x)` | operating point (coverage vs risk) |
| PD-MCMC | records | `π_λ(i) ∝ P(i)·exp(−λ B(i))` | fairness vs fidelity |
| UQ (this) | beliefs | `π_λ(z\|D) ∝ p₀(z\|D)·exp(−Σ_k λ_k C_k(z))` | constraint strength (fairness/policy) |

**Framing for the paper:** COMPASS tilts *what to say*, the latent-bias posterior tilts *what to believe about the data*; one dial exposes a joint **risk–coverage–bias operating envelope**. This is the theoretical contribution that makes N1–N4 cohere.

### A.5 Method (how the two couple)
1. **Evidence layer (COMPASS, unchanged):** retrieval + memory + ontology checks produce a context pack, a candidate answer, and an evidence-confidence `c_ev(x)` (retrieval margins, snippet agreement, symbolic fire flag, gen-prob).
2. **Data-trust layer (new):** for attribute/value queries, run the constraint-calibrated latent-bias sampler on the queried `(record, feature)` (and its group) to obtain the clean-value posterior `p(x_clean)`. From it derive:
   - posterior center `x̂_clean` (debiased answer),
   - credible-interval half-width `w(x)` (uncertainty),
   - target-sensitivity `s(x)` — movement of `x̂_clean` across a policy-target/λ grid (from the UQ paper's sensitivity analysis),
   - estimated group-bias magnitude `b̂_g`.
3. **Fusion + bias-aware abstention:** calibrate a fused confidence `c(x) = g(c_ev(x), −w(x), −s(x))` with a monotone combiner on dev data. Abstain if `c(x) < τ` — i.e., abstain when evidence is missing **or** the value is present but untrustworthy (wide/target-sensitive posterior).
4. **Answer object:** debiased value `x̂_clean` + credible interval + provenance (citations, rule trace) + the active λ/constraints. Fully auditable.

*Algorithm 1* = COMPASS inference with a data-trust branch on attribute queries; the frozen-chain MCMC protocol (adapt λ in warm-up, freeze, then sample) keeps a clean stationary-target guarantee.

### A.6 Benchmark design
**Common real testbed = Open Food Facts** (already used by *both* COMPASS as OOD and PD-MCMC as case study): systematically missing/biased sustainability fields (eco-score present 16.6%, nutri-score 30.4%); use `countries_tags` as EU vs Global-South region proxy. **Controlled testbed = DPP corpora** (battery, Lexmark, Viessmann): inject group-dependent bias on carbon/repairability/durability with known clean references (UQ protocol) for ground-truth recovery.

Query types: (a) **attribute-value** ("carbon footprint of product X"), (b) **group-gap** ("is repairability lower for GS suppliers?"), (c) **compliance/logic** (existing COMPASS rules), (d) **abstain-trigger** (missing or heavily biased). Ground truth via clean references (synthetic DPP) and held-out complete OFF rows treated as clean (semi-synthetic, mirroring the UQ NHANES protocol).

### A.7 Metrics
- **Selective risk:** AURC overall and **on the biased subset**; full risk–coverage curves.
- **Calibration:** ECE (10-bin), reliability diagram — in-domain (COMPASS's weak point).
- **Trust/bias (new):** *confidently-wrong-on-biased* rate; clean-value interval coverage @90%; distance-to-clean of reported group gaps; reported target-sensitivity.
- **Standard QA:** accuracy, Wilson 95% CIs, McNemar vs naive COMPASS.
- **MCMC diagnostics:** rank-normalized R̂, bulk/tail ESS for the posterior layer.

### A.8 Baselines and ablations
Baselines: naive COMPASS (evidence-only abstention); COMPASS + bootstrap CI; COMPASS + multiple imputation; COMPASS + unconstrained latent (no fairness constraint); COMPASS + inverse-propensity reweighting.
Ablations: constraint on/off (is the group-gap constraint the driver?); target-sensitivity channel on/off; fusion method (monotone vs learned combiner); target-misspecification (wrong clean target → predictable degradation, not leakage).

### A.9 Projected results (targets, reasoned from the source papers — not yet run)
| Metric | Naive COMPASS (published) | Bias-aware COMPASS (projected) | Basis |
|---|---|---|---|
| Confidently-wrong on biased attributes | ~35–40% | **< 8%** | UQ recovers clean gap; abstain/correct |
| Clean-value interval coverage @90% | ~0.55 (bootstrap/MI) | **~0.90+** | UQ distance-to-clean 20.1 → 0.264 (≈76×) |
| In-domain ECE | 0.5247 | **~0.10–0.15** | posterior-width confidence channel; OFF already 0.021 |
| AURC on biased subset | ~0.35 | **~0.04** | COMPASS abstention (overall AURC 0.0117) + bias signal |
| Group-gap distance-to-clean | — | **≈0.26** | UQ 0.264 (DPP); PD-MCMC 4.4–13.9× at dataset level |
| Overall accuracy / OOD OFF | 0.9749 / 0.972 | **maintained** | data-trust layer fires only on attribute queries |
| Posterior mixing (R̂ / ESS) | — | **R̂ ≤ 1.05** | UQ constrained variants |

Headline: *bias-aware abstention nearly eliminates confident-biased answers and restores calibration, at no cost to accuracy.*

### A.10 Risks and mitigations
- **Identifiability of latent bias** → rely on explicit clean/policy targets as *auditable assumptions* (UQ framing); report target-sensitivity rather than claim a single truth.
- **Scope over-claim** → keep "targeted": the data-trust layer fires only on attribute queries; report its coverage honestly (as COMPASS did with 7.96% symbolic).
- **Compute/time** → reuse both existing codebases; the frozen-chain protocol is cheap per query on a small feature set.
- **Double-blind** → anonymize CE-RISE/partners; anonymous code+ontology+benchmark release.

### A.11 Timeline to 21 July (18 days)
- **D1–3:** build biased-attribute QA benchmark (DPP injected-bias + OFF; clean refs, query generation).
- **D3–7:** implement data-trust layer + confidence fusion + interval answers; wire posterior into COMPASS answerer.
- **D7–11:** run main experiments, baselines, ablations; MCMC diagnostics.
- **D11–14:** figures/tables; draft (reuse COMPASS + UQ prose).
- **D14–17:** internal review, calibrate claims, anonymize, prep code release.
- **D18:** buffer + submit.

---

## PART B — ICTAI 8-page draft outline (IEEE 2-column)

**Title options**
1. *Knowing When the Data Lies: Bias-Aware Selective Question Answering for Digital Product Passports*
2. *Trustworthy DPP Question Answering under Systematic Measurement Bias: An LLM–MCMC Synergistic Tool*
3. *λ-Tilted Reliability: Unifying Selective Abstention and Latent-Bias Uncertainty for Product-Record QA*

**Abstract (~180 words)** — problem (missing vs biased evidence) → COMPASS×latent-bias-UQ coupling → benchmark → projected headline numbers (confident-biased 38%→<8%, interval coverage ~0.9, ECE 0.52→~0.12, accuracy maintained) → code release.

**1. Introduction (~0.75 pg).** DPP QA is trust-critical; distinguish *missing* from *biased* evidence; a confident-biased failure example (Fig. 1 teaser: naive answer vs bias-aware abstention/interval). Contributions N1–N4.

**2. Related work (~0.75 pg, compressed).** RAG + selective prediction + LLM calibration; fairness-aware UQ + measurement-error models; neuro-symbolic/KG QA. State the gap: no LLM QA abstains on measurement-bias uncertainty.

**3. Unified λ-tilted reliability (~1 pg).** Define the three tilts (Table: A.4); the joint risk–coverage–bias operating envelope; formal problem statement; metric definitions (AURC, ECE, interval coverage, distance-to-clean, confidently-wrong-on-biased).

**4. Method (~2 pg).** COMPASS recap (brief); constraint-calibrated latent-bias posterior `π_λ(z|D)`; frozen-chain MCMC protocol; data-trust signals `(x̂_clean, w, s, b̂_g)`; confidence fusion + bias-aware abstention rule; policy-sensitive interval answer object. **Fig. 2** architecture; **Algorithm 1**.

**5. Experimental setup (~0.75 pg).** Benchmark (DPP injected-bias + OFF), groups, query types, ground truth, baselines, metrics, MCMC diagnostics. **Table 1** benchmark counts.

**6. Results (~2 pg).** **Table 2** main (bias-aware vs naive vs bootstrap/MI/IPW/unconstrained); **Fig. 3** risk–coverage(–bias) curves; **Fig. 4** reliability diagram (ECE 0.52→~0.12); clean-value interval coverage + distance-to-clean; **Table 3** ablations (constraint on/off = main driver; sensitivity channel; target-misspecification); **Table 4** R̂/ESS diagnostics.

**7. Discussion & limitations (~0.5 pg).** Identifiability; targets as auditable assumptions; targeted coverage of the data-trust layer; abstention-driven reliability stated honestly; scope.

**8. Conclusion + reproducibility/ethics + references (~0.25 pg + refs).** Anonymous release of code, ontology, and benchmark; human oversight for compliance-critical use.

**Assets to produce:** Fig 1 teaser, Fig 2 architecture, Fig 3 risk–coverage–bias, Fig 4 reliability; Tables 1–4. Reuse COMPASS figures/harness and the UQ MCMC diagnostics tooling.
