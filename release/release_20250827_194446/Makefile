# Makefile — Day 7
# One-command pipeline for evals, RL Pareto, faithfulness, and env freeze.

# --- Tooling knobs -----------------------------------------------------------

DOMAIN ?= battery
export DPP_DOMAIN := $(DOMAIN)


PY ?= python
PIP ?= pip
export PYTHONPATH := $(PWD)

# RL cost weight (used by `make rl` and `make pareto`)
RL_ALPHA ?= 0.4

# Adaptive-RAG thresholds (tweak as you like)
ADAPTIVE_MEM_T ?= 0.45
ADAPTIVE_SEARCH_T ?= 0.55

# Paths
ARTIFACTS_DIR := artifacts
TESTS_CANON := tests/dpp_rl/tests.jsonl
SEED_FILE := tests/dpp_rl/seed_mem.jsonl

# Default goal
.DEFAULT_GOAL := help

# --- Targets ----------------------------------------------------------------
.PHONY: help seed router eval pareto faithful rl freeze all clean

help:
	@echo "Targets:"
	@echo "  make seed        - seed episodic memory from $(SEED_FILE)"
	@echo "  make router      - train supervised router -> artifacts/policy_router.json"
	@echo "  make eval        - run full evaluation (BASE/MEM/SYM/MEMSYM/ROUTER/ADAPTIVERAG + RL)"
	@echo "  make pareto      - sweep RL_ALPHA and write artifacts/pareto_*.csv/png"
	@echo "  make faithful    - run faithfulness (memory knockout + rule ablations)"
	@echo "  make rl          - quick RL run with RL_ALPHA=$(RL_ALPHA) via Pareto script"
	@echo "  make freeze      - write environment.yml and requirements.txt"
	@echo "  make all         - seed -> router -> eval -> pareto -> faithful -> freeze"
	@echo "  make clean       - remove artifacts/*.csv/.jsonl/.png/.html/.json"

seed:
	@echo ">> Seeding memory from $(SEED_FILE)"
	$(PY) scripts/seed_memory.py

router:
	@echo ">> Training supervised router"
	$(PY) scripts/train_router.py

eval: seed router
	@echo ">> Running evaluation (classic + router/adaptiverag + RL)"
	$(PY) run_eval_all.py

pareto:
	@echo ">> RL Pareto sweep (alpha in script); current env RL_ALPHA=$(RL_ALPHA)"
	RL_ALPHA=$(RL_ALPHA) $(PY) run_pareto.py || true

faithful:
	@echo ">> Running faithfulness (knockouts + rule ablations)"
	$(PY) run_faithfulness.py

rl:
	@echo ">> Quick RL run: Pareto with RL_ALPHA=$(RL_ALPHA)"
	RL_ALPHA=$(RL_ALPHA) $(PY) run_pareto.py || true

freeze:
	@echo ">> Freezing environment"
	- conda env export --no-builds > environment.yml 2>/dev/null || true
	$(PIP) freeze > requirements.txt

all: seed router eval pareto faithful freeze
	@echo "✅ make all complete."

clean:
	@echo ">> Cleaning artifacts"
	- rm -f $(ARTIFACTS_DIR)/*.csv $(ARTIFACTS_DIR)/*.jsonl $(ARTIFACTS_DIR)/*.png $(ARTIFACTS_DIR)/*.html $(ARTIFACTS_DIR)/*.json

.PHONY: robustness
robustness:
	@echo ">> Running robustness (augmentations + mem scale)"
	python run_robustness.py

.PHONY: report
report:
	@echo ">> Building HTML report"
	python -m backend.eval.report --artifacts artifacts

.PHONY: selective
selective:
	@echo ">> Running selective risk (Day 10)"
	python run_selective.py --domain $(DOMAIN)

.PHONY: stats figures release

stats:
	@echo ">> Statistical tests (Day 11)"
	python -m backend.eval.stats_eval

figures:
	@echo ">> Building figures & tables (Day 12/14)"
	python -m backend.eval.figures

release:
	@echo ">> Making reproducible release pack (Day 13)"
	bash scripts/make_release.sh
	
.PHONY: camera_ready
camera_ready: all figures stats report release
	@echo "✅ Camera-ready assets generated."