# backend/services/confidence.py
import math

def _sigmoid(x):  # safe-ish squashing
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def from_retrieval(top_score=None):
    if top_score is None:  # unknown
        return 0.5
    # scale guess; adjust if your scores are in another range
    return _sigmoid((float(top_score) / 2.0))

def from_memory(top_score=None):
    if top_score is None:
        return 0.5
    # assume memory scores in [0,1] or similar
    s = max(0.0, min(1.0, float(top_score)))
    return s

def from_router(prob=None):
    if prob is None:
        return 0.5
    return max(0.0, min(1.0, float(prob)))

def from_sym(proved=None, refuted=None):
    if proved:  return 1.0
    if refuted: return 0.0
    return 0.5

def fallback():
    return 0.5
