# backend/services/policy_confidence.py
from __future__ import annotations

def confidence_from_features(f: dict) -> float:
    """
    Map cheap features -> [0,1] confidence used by Day-10 selective abstention.

    Inputs we expect in f:
      - mem_top, search_top: float similarity scores in [0,1] (roughly)
      - sym_fired: 0/1 flag (True if KG/rules have relevant facts)
    """
    mem_top = float(f.get("mem_top", 0.0) or 0.0)
    sch_top = float(f.get("search_top", 0.0) or 0.0)
    sym     = int(f.get("sym_fired", 0) or 0)

    # Baseline: max signal from retrievals (smooth-ish)
    base = max(mem_top, sch_top)
    conf = 0.20 + 0.60 * base  # in [0.2, 0.8] for typical 0..1 scores

    # If both retrieval signals are strong, bump a bit
    if mem_top >= 0.45 and sch_top >= 0.35:
        conf = max(conf, 0.80)

    # If the symbolic path is relevant, floor high (it's precise & cheap)
    if sym:
        conf = max(conf, 0.95)

    # Clamp
    return float(min(1.0, max(0.0, conf)))
