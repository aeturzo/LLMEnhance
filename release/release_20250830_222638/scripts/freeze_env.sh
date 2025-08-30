#!/usr/bin/env bash
set -euo pipefail
echo ">> Freezing environment"
conda env export --no-builds > environment.yml 2>/dev/null || true
pip freeze > requirements.txt
echo "Wrote environment.yml and requirements.txt"
