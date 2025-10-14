# ================================
# Project-wide Makefile (paper-ready)
# ================================

# ---- Variables ----
PY          ?= python
DOMAINS     ?= battery textiles viessmann lexmark
DOMAIN      ?= battery
TESTS_DIR   ?= tests
ART         ?= artifacts
TABLES      ?= tables
CORPUS      ?= backend/corpus/dpp_corpus.jsonl
COV         ?= 0.50

LATEST_JOINED   = $(shell ls -t $(ART)/eval_joined_*.csv 2>/dev/null | head -n1)
LATEST_SELECTIVE= $(shell ls -t $(ART)/selective_*.csv 2>/dev/null | head -n1)

# For portability on macOS: ensure head exists
SHELL := /bin/bash

.PHONY: help
help:
	@echo "Targets:"
	@echo "  lint-tests         Validate tests JSONL format"
	@echo "  build-corpus       Build retrieval corpus from tests"
	@echo "  retriever-warm     (Optional) warm retriever index (skips if module missing)"
	@echo "  eval               Run full eval spine for a domain (use: make eval DOMAIN=battery)"
	@echo "  calibrate          Calibrate confidences -> *_calibrated.csv"
	@echo "  selective          Build selective risk/coverage curves"
	@echo "  fig-selective      Plot selective curve(s)"
	@echo "  thresholds         Pick abstention thresholds near coverage COV (default $(COV))"
	@echo "  aurc               Compute AURC per mode"
	@echo "  ci                 95% Wilson CIs per mode/type"
	@echo "  sym-stats          Symbolic coverage stats"
	@echo "  mcnemar            McNemar test ADAPTIVERAG vs RAG_BASE"
	@echo "  export-tables      Emit CSV+LaTeX tables under docs/paper/tables"
	@echo "  paper-pipeline     ONE COMMAND: everything for DOMAIN=$(DOMAIN)"
	@echo "  paper-pipeline-all Run pipeline for all DOMAINS='$(DOMAINS)'"
	@echo "  clean-artifacts    Remove artifacts & tables"

# ---- Basic hygiene ----
.PHONY: lint-tests
lint-tests:
	$(PY) -m backend.eval.lint_tests --root $(TESTS_DIR)

# ---- Corpus / Retrieval ----
.PHONY: build-corpus
build-corpus:
	$(PY) -m scripts.build_corpus --tests $(TESTS_DIR) --out $(CORPUS)

# This target is optional. It will NO-OP if the retriever module is absent.
.PHONY: retriever-warm
retriever-warm:
	@set -e; \
	if $(PY) - <<'PY' >/dev/null 2>&1; then \
	    print("ok"); \
	else \
	    pass; \
	end
	import importlib.util, sys
	spec = importlib.util.find_spec("backend.retrieval.hybrid")
	if spec is None:
	    print("[retriever-warm] backend.retrieval.hybrid not found; skipping.")
	else:
	    from backend.retrieval.hybrid import HybridRetriever
	    HybridRetriever("$(CORPUS)")
	PY
	@true

# ---- Evaluation ----
.PHONY: eval
eval:
	@echo ">> Running evaluation (classic + router/adaptiverag + RL) for DOMAIN=$(DOMAIN)"
	$(PY) run_eval_all.py --domain $(DOMAIN)

# ---- Calibration / Selective ----
.PHONY: calibrate
calibrate:
	$(PY) -m backend.eval.calibrate --joined $(LATEST_JOINED) --out $(ART)

.PHONY: selective
selective:
	$(PY) -m backend.eval.selective --artifacts $(ART) --out $(ART)

.PHONY: fig-selective
fig-selective:
	$(PY) -m backend.eval.figures

.PHONY: thresholds
thresholds:
	$(PY) -m backend.eval.thresholds --csv $(LATEST_SELECTIVE) --out $(TABLES) --cov $(COV)

.PHONY: aurc
aurc:
	$(PY) -m backend.eval.aurc --csv $(LATEST_SELECTIVE) --out $(TABLES)

.PHONY: ci
ci:
	$(PY) -m backend.eval.conf_intervals --joined $(LATEST_JOINED) --out $(TABLES)

.PHONY: sym-stats
sym-stats:
	$(PY) -m backend.eval.sym_coverage --joined $(LATEST_JOINED) --out $(TABLES)

.PHONY: mcnemar
mcnemar:
	$(PY) -m backend.eval.mcnemar --joined $(LATEST_JOINED) --out $(TABLES)/mcnemar.csv --A ADAPTIVERAG --B RAG_BASE

.PHONY: export-tables
export-tables:
	$(PY) scripts/export_tables.py

# ---- One-command pipelines ----
.PHONY: paper-pipeline
paper-pipeline: lint-tests build-corpus retriever-warm eval calibrate selective fig-selective thresholds aurc ci sym-stats mcnemar export-tables
	@echo ">> Pipeline complete for DOMAIN=$(DOMAIN)"
	@echo "Artifacts under $(ART); tables under $(TABLES) and docs/paper/tables"

.PHONY: paper-pipeline-all
paper-pipeline-all:
	@set -e; \
	for D in $(DOMAINS); do \
	  echo "===== PIPELINE $$D ====="; \
	  $(MAKE) paper-pipeline DOMAIN=$$D || exit $$?; \
	done

# ---- Clean ----
.PHONY: clean-artifacts
clean-artifacts:
	rm -rf $(ART) $(TABLES) docs/paper/tables
	mkdir -p $(ART) $(TABLES)
