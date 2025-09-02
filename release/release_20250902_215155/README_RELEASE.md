# Reproduce
1) `pip install -r requirements_freeze.txt` (or your own env meeting versions)
2) `export PYTHONPATH=.` from repo root
3) Datasets are under `tests/<domain>/tests.jsonl`.
4) Run evaluation: `python run_eval_all.py --domain battery` (and for others).
5) Generate stats/tables: `python backend/eval/stats_polish.py`
6) Reports: `python backend/eval/report_v2.py --domain battery --out artifacts/report_battery.html`
