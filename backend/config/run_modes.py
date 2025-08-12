from enum import Enum

class RunMode(str, Enum):
    BASE = "BASE"       # LLM/search only; no memory, no symbolic
    MEM = "MEM"         # Memory enabled, no symbolic
    SYM = "SYM"         # Symbolic enabled, no memory
    MEMSYM = "MEMSYM"   # Memory + Symbolic

def mode_from_env(env: str | None) -> RunMode:
    try:
        return RunMode((env or "MEMSYM").upper())
    except Exception:
        return RunMode.MEMSYM
