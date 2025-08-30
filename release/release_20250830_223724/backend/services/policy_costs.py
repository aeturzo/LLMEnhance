# backend/services/policy_costs.py
from __future__ import annotations
from typing import List, Dict, Any

# Cheap, explicit costs per module step. Tune later.
COSTS = {
    "BASE":   0.10,
    "MEM":    0.30,
    "SEARCH": 0.50,
    "SYM":    0.80,
    "COMPOSE":0.10,  # if you emit a separate compose step
}

def episode_cost(steps: List[Dict[str, Any]] | None) -> float:
    steps = steps or []
    cost = 0.0
    for s in steps:
        src = (s.get("source") or "").upper()
        cost += COSTS.get(src, 0.0)
    return float(cost)
