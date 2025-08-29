#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/release"
STAMP="$(date +%Y%m%d_%H%M%S)"
PKG="$OUT/release_$STAMP"
mkdir -p "$PKG"

echo ">> Freezing environment"
conda env export --no-builds > "$ROOT/environment.yml" 2>/dev/null || true
pip freeze > "$ROOT/requirements.txt"

echo ">> Collecting minimal pack"
mkdir -p "$PKG/artifacts" "$PKG/backend/ontologies" "$PKG/tests" "$PKG/scripts"

# Code (exclude big caches / venvs)
rsync -a --exclude '.git' --exclude '__pycache__' --exclude '.mypy_cache' \
  --exclude '.pytest_cache' --exclude '.idea' --exclude '.vscode' \
  --exclude '*.pt' --exclude '*.bin' \
  "$ROOT/backend" "$PKG/"

# Ontologies & tests (both domains)
rsync -a "$ROOT/backend/ontologies/" "$PKG/backend/ontologies/"
rsync -a "$ROOT/tests/dpp_rl" "$PKG/tests/" 2>/dev/null || true
rsync -a "$ROOT/tests/dpp_textiles" "$PKG/tests/" 2>/dev/null || true
rsync -a "$ROOT/tests/dpp_lexmark" "$PKG/tests/" 2>/dev/null || true
rsync -a "$ROOT/tests/dpp_viessmann" "$PKG/tests/" 2>/dev/null || true

# Scripts
rsync -a "$ROOT/scripts/" "$PKG/scripts/"

# Env & docs
cp "$ROOT/requirements.txt" "$PKG/" 2>/dev/null || true
cp "$ROOT/environment.yml" "$PKG/" 2>/dev/null || true
cp "$ROOT/Makefile" "$PKG/" 2>/dev/null || true
[ -f "$ROOT/README.md" ] && cp "$ROOT/README.md" "$PKG/" || true

echo ">> Creating ZIP"
cd "$OUT"
zip -r "release_$STAMP.zip" "release_$STAMP" >/dev/null
sha256sum "release_$STAMP.zip" > "release_$STAMP.SHA256SUMS" 2>/dev/null || shasum -a 256 "release_$STAMP.zip" > "release_$STAMP.SHA256SUMS"

echo "OK: $OUT/release_$STAMP.zip"
