# Synthetic DPP-like Benchmark — Data Card

This data card documents the synthetic benchmark used in the main results
(Table 3 of the AEI draft) and answers, in writing, the reviewer concern that
the 97.49 % vs 89.73 % pooled-accuracy gap (266 McNemar wins vs 0 losses) might
be a benchmark-leakage artifact. It is the single document a hostile reviewer
will be pointed to when raising that objection.

## 1. Summary statistics

| | Battery | Lexmark | Viessmann | Pooled |
|---|---:|---:|---:|---:|
| Logic | 93 | 76 | 126 | 295 |
| Open | 646 | 749 | 902 | 2,297 |
| Recall | 331 | 111 | 395 | 837 |
| **Total** | 1,070 | 936 | 1,423 | 3,429 |

(Reproduces Table 1 in the paper.)

## 2. Provenance and generation

The 3,429 test items were produced by `scripts/gen_dataset.py` /
`scripts/gen_synth.py` over the per-domain seed-document corpora
(`release/release_20250902_215155/tests/<domain>/seed_docs.jsonl`, 56 docs in
each of battery / lexmark / viessmann). For each seed doc, the generator
produces:

  - `docopen-NNNNNN` — open-extraction questions of the form *"According to
    seed_XXXX, what is the model name?"* with `expected_contains` set to the
    canonical model name appearing in the doc.
  - `docrec-NNNNNN` — recall questions that quote a span near a field and ask
    for its value (gold = value adjacent to the quoted span).
  - `docfix-NNNNNN` — fixed-template extraction over the same seed (used to
    test paraphrase invariance).
  - `log-NNNNNN` / `rec-NNNNNN` / `opn-NNNNNN` — older shorter-template
    versions retained for backward compatibility.
  - `<domain>.kb.<type>.NNNNN` — knowledge-base-style questions that bypass
    document retrieval and rely on the symbolic layer.

All questions are deterministically derived from the seed docs; the
generator's RNG seed is fixed (`set_global_seed(42)` in `run_eval_all.py`).

## 3. What "synthetic" means here — and what it does not mean

The benchmark is **synthetic in question construction** (templated questions
over real seed-document content) but **not synthetic in content**: every seed
doc is hand-curated from public manufacturer documentation
(battery passports, Lexmark brochures, Viessmann datasheets). The seed docs
were never passed through ChatGPT for generation.

This matters because the most common reviewer model of "synthetic-benchmark
leakage" is *"the same LLM generated the questions and is evaluated on them,
so it has memorised the answers."* That mechanism does not apply here: no LLM
participated in seed-doc curation; the question generator is a deterministic
template over canonical field locations.

## 4. The 266-McNemar-wins-vs-0-losses observation

ADAPTIVERAG dominates RAG-BASE by 266 paired wins and 0 paired losses
(p ≈ $1.7 \times 10^{-80}$, exact two-sided McNemar). The asymmetry is real
but its mechanism is **mode capability**, not benchmark leakage:

  * RAG-BASE in our implementation is a strictly retrieval-only mode with no
    adaptive re-ranking, no memory access, and no symbolic invocation. It
    fails almost completely on the 295 logic queries (RAG-BASE logic
    accuracy = 0.0102; see Table 4 in the paper). ADAPTIVERAG can fire the
    OWL-RL rule layer on these queries, and the symbolic layer is **perfect**
    on the subset where it fires (conditional precision = 1.0, see Section
    5.4 of the paper).
  * On the 837 recall queries, both modes have access to the same memory
    store but ADAPTIVERAG additionally uses adaptive re-ranking. The result
    is RAG-BASE = 0.9283, ADAPTIVERAG = 1.0000 — a small but consistent gain.
  * Therefore the 266 wins concentrate on the **logic subset**
    (RAG-BASE 0.0102 → ADAPTIVERAG 0.7085, a swing of ≈ 70 pp) and the
    **recall subset** (small but uniformly positive). Open extraction is
    saturated at 1.0 for both modes, so it contributes 0 paired flips.

The McNemar p-value $1.7 \times 10^{-80}$ should be read as
"the binomial null is rejected so strongly that the test is no longer the
limiting factor." For paper presentation we recommend reporting it as
**$p < 10^{-10}$** and emphasising the **effect-size delta**
(+7.76 pp pooled, +69.8 pp on the logic subset) and the **Wilson 95 % CI**
([0.9691, 0.9796]) rather than the extreme exponent.

## 5. Reviewer-defence-in-depth: external baselines

To address the "all comparisons internal" concern, we additionally evaluated
three external LLM baselines on the same benchmark:

  - **GPT4O_LONGCTX** — long-context GPT-4o-mini given the per-domain seed
    corpus as in-prompt context.
  - **LINC** — neuro-symbolic FOL-style chain (Olausson et al., 2023).
  - **LOGIC_LM** — classify-then-dispatch (Pan et al., 2023).

See `scripts/run_baselines.py`, `docs/baseline_runbook.md`, and the resulting
`artifacts/eval_{GPT4O_LONGCTX,LINC,LOGIC_LM}_full.csv` for the numbers. The
McNemar comparisons against ADAPTIVERAG use the **release-benchmark** test
set, not the paper's original 3,429-row pooled-CSV split, because the paper's
in-pipeline generator was not archived to disk and its exact query strings
cannot be reproduced byte-identically. The release benchmark has 8,093 rows,
all with gold answers; 1,345 of those share `(id, domain)` keys with the
paper-split and are used for the head-to-head McNemar.

## 6. What still cannot be claimed

The benchmark is DPP-**like**, not a regulatory DPP corpus. Until a public,
audited DPP dataset exists (CIRPASS-2 may release one), the conservative
claim is that the framework's **architecture and reliability metrics**
generalise, not that its **numbers** would carry over to a regulatory
benchmark without recalibration.
