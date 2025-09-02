# backend/eval/validators.py
import re
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
