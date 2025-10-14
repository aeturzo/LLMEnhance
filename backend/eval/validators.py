# backend/eval/validators.py
import re
import yaml, os
ALIASES = {}

try:
    with open("backend/eval/aliases.yaml", "r", encoding="utf-8") as f:
        ALIASES = yaml.safe_load(f) or {}
except FileNotFoundError:
    ALIASES = {}

def _normalize(s: str) -> str:
    return (s or "").strip().lower()

def recall_canonical(pred: str, gold: str) -> bool:
    p = _normalize(pred); g = _normalize(gold)
    if p == g: return True
    # check aliases of gold
    for k, vals in ALIASES.items():
        if _normalize(k) == g:
            if p in {_normalize(v) for v in vals}:
                return True
    return False

_WS = re.compile(r"\s+")
_PUNC = re.compile(r"[^\w\s]")

def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip().lower())

def _canon(s: str) -> str:
    s = _PUNC.sub(" ", (s or "").strip().lower())
    return _WS.sub(" ", s)

def recall_canonical(answer: str, gold_value: str, aliases=None) -> bool:
    a = _canon(answer)
    cands = {_canon(gold_value)} | {_canon(x) for x in (aliases or [])}
    # exact canonical match or lenient contains of canonical target
    return a in cands or any(x and x in a for x in cands)

def open_with_citation(answer: str, span: str) -> bool:
    a, s = _norm(answer), _norm(span)
    return bool(s) and s in a

def logic_yesno(answer: str, gold_label: str) -> bool:
    a, g = _norm(answer), _norm(gold_label)
    return (a == g) or a.startswith(g + " ")
