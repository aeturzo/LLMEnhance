# backend/config/run_modes.py
from __future__ import annotations
from enum import Enum

class RunMode(str, Enum):
    # Single-module routes
    BASE = "BASE"          # plain search/LLM
    MEM = "MEM"            # memory-only
    SEARCH = "SEARCH"      # search-only (explicit)
    SYM = "SYM"            # symbolic-only

    # Composed / learned routes
    MEMSYM = "MEMSYM"      # memory + symbolic (composer)
    ROUTER = "ROUTER"      # supervised router picks a module
    ADAPTIVERAG = "ADAPTIVERAG"  # heuristic, feature-threshold router

    # NOTE: RL is served via /solve_rl, not as a /solve mode.
    # Keep aliases below for backward compatibility with old env settings.

def mode_from_env(env: str | None) -> "RunMode":
    """
    Backward-compatible resolver for environment strings.
    - "RL"   → ROUTER  (closest modern analogue for /solve)
    - "ALL"  → MEMSYM
    - Unknown → MEMSYM (safe default)
    """
    raw = (env or "MEMSYM").upper()
    alias = {
        "RL": "ROUTER",  # legacy setting
        "ALL": "MEMSYM",
    }
    name = alias.get(raw, raw)
    try:
        return RunMode(name)
    except Exception:
        return RunMode.MEMSYM
