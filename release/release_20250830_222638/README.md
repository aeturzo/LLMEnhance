# Neural–Symbolic + RL for DPP QA

Hybrid **LLM + Symbolic Reasoning + Memory + RL** system for Digital Product Passport (DPP) QA with 7 modes:
**BASE, MEM, SYM, MEMSYM, ROUTER, ADAPTIVERAG, RL**.

This README covers:
1) environment & quickstart for paper reproduction,
2) **code structure and how components connect**,
3) **end-to-end flows** (inference, training, evaluation),
4) **artifacts & results layout**,
5) **frontend contract** (API you can wire a UI to later).

---

## 0) TL;DR Reproduce the paper
```bash
# Conda (recommended)
conda activate llmrl
python -c "import fastapi, rdflib, owlrl, faiss, pandas; print('env OK')"

# Run everything end-to-end
make all
