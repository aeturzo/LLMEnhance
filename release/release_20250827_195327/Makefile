# ===========
# Makefile (repo root)
# Orchestrates the full pipeline in correct order:
# seed → eval → router → pareto → faithful → figures → stats → report → release
# ===========

# ---- Config (override on the command line, e.g., make camera_ready DOMAIN=lexmark RL_ALPHA=0.6) ----
DOMAIN ?= battery
RL_ALPHA ?= 0.4
PY ?= python

# ---- Phony targets ----
.PHONY: seed eval router pareto faithful robustness selective stats figures report release all camera_ready clean

# Seed episodic memory from tests/<domain>/seed_mem.jsonl
seed:
	@echo ">> Seeding memory from tests/..."
	$(PY) scripts/seed_memory.py

# Run the full evaluation to produce eval_joined_*.csv (needed before router training)
eval:
	@echo ">> Running evaluation (classic + router/adaptiverag + RL)"
	$(PY) run_eval_all.py --domain $(DOMAIN)

# Train the supervised router AFTER eval so labels exist
router: eval
	@echo ">> Training supervised router"
	$(PY) scripts/train_router.py

# RL Pareto sweep (alpha passed from env/var)
pareto:
	@echo ">> RL Pareto sweep (alpha = $(RL_ALPHA))"
	RL_ALPHA=$(RL_ALPHA) $(PY) run_pareto.py --domain $(DOMAIN) || true

# Faithfulness (memory knockout + rule ablations)
faithful:
	@echo ">> Running faithfulness (knockouts + rule ablations)"
	$(PY) run_faithfulness.py --domain $(DOMAIN)

# Optional: robustness sweeps (paraphrase/noise + memory scale)
robustness:
	@echo ">> Running robustness sweeps"
	$(PY) run_robustness.py --domain $(DOMAIN)

# Optional: selective risk curve (uncertainty/abstain)
selective:
	@echo ">> Building selective risk curve"
	$(PY) run_selective.py --domain $(DOMAIN)

# Statistical testing & effect sizes (Day 11)
stats:
	@echo ">> Statistical tests (Day 11)"
	$(PY) -m backend.eval.stats_eval

# Figures & tables (Day 12/14)
figures:
	@echo ">> Building figures & tables (Day 12/14)"
	$(PY) -m backend.eval.figures

# HTML report (uses latest artifacts & traces)
report:
	@echo ">> Building HTML report"
	$(PY) -m backend.eval.report --artifacts artifacts

# Reproducible release pack (code+data+env+artifacts)
release:
	@echo ">> Making reproducible release pack (Day 13)"
	bash scripts/make_release.sh

# Convenience bundles
all: seed eval router pareto faithful
	@echo "✅ make all complete."

camera_ready: seed eval router pareto faithful figures stats report release
	@echo "✅ Camera-ready assets generated."

# Clean convenience
clean:
	@echo ">> Cleaning transient files"
	rm -f artifacts/eval_*.csv artifacts/eval_joined_*.csv artifacts/eval_summary_*.csv
	rm -f artifacts/pareto_*.csv artifacts/pareto_*.png
	rm -f artifacts/selective_*.csv artifacts/selective_*.png
	rm -f artifacts/faithfulness_*.csv
	rm -f artifacts/fig_*.png
	rm -f artifacts/trace_*.jsonl
	rm -f tables/*.csv || true
