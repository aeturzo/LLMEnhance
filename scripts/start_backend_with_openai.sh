#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY_FILE="${1:-$ROOT_DIR/openApikey.rtf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL="${GEN_MODEL:-gpt-5.4}"

if [[ ! -f "$KEY_FILE" ]]; then
  echo "Key file not found: $KEY_FILE" >&2
  echo "Pass the file path as the first argument or place openApikey.rtf in the repo root." >&2
  exit 1
fi

OPENAI_KEY="$(
python3 - "$KEY_FILE" <<'PY'
from pathlib import Path
import re
import shutil
import subprocess
import sys

path = Path(sys.argv[1])
raw = path.read_text(errors="ignore")
match = re.search(r"sk-[A-Za-z0-9._\-]+", raw)

if not match and shutil.which("textutil"):
    try:
        txt = subprocess.check_output(
            ["textutil", "-convert", "txt", "-stdout", str(path)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        match = re.search(r"sk-[A-Za-z0-9._\-]+", txt)
    except Exception:
        match = None

print(match.group(0) if match else "")
PY
)"

if [[ -z "$OPENAI_KEY" ]]; then
  echo "No OpenAI API key pattern was found in $KEY_FILE" >&2
  exit 1
fi

export OPENAI_API_KEY="$OPENAI_KEY"
export GEN_MODEL="$MODEL"
unset OPENROUTER_API_KEY
unset LLM_DISABLED

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "OPENAI_API_KEY=<loaded from $KEY_FILE>"
  echo "GEN_MODEL=$GEN_MODEL"
  echo "HOST=$HOST"
  echo "PORT=$PORT"
  exit 0
fi

cd "$ROOT_DIR"
exec .venv/bin/uvicorn backend.main:app --reload --host "$HOST" --port "$PORT"
