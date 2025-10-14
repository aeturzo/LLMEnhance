# backend/config/run_modes.py
from __future__ import annotations
from enum import Enum

class RunMode(str, Enum):
    BASE="BASE"; 
    MEM="MEM"; 
    SEARCH="SEARCH"; 
    SYM="SYM"; 
    MEMSYM="MEMSYM"
    ROUTER="ROUTER"; 
    ADAPTIVERAG="ADAPTIVERAG"; 
    RL="RL"
    RAG_BASE="RAG_BASE"; 
    SYM_ONLY="SYM_ONLY"

def parse_modes_csv(csv_string: str) -> tuple[RunMode, ...]:
    """
    Robust parser for '--modes' like 'BASE,MEM,RL'.
    Unknown names are ignored.
    """
    out: list[RunMode] = []
    if not csv_string:
        return tuple(out)
    for name in (x.strip().upper() for x in csv_string.split(",")):
        if not name:
            continue
        try:
            out.append(RunMode[name])
        except KeyError:
            print(f"[run_modes] WARN: unknown mode '{name}' (ignored)")
    return tuple(out)
