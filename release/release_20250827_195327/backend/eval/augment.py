# backend/eval/augment.py
from __future__ import annotations

import random
import re
from typing import List

_WORD = re.compile(r"\w+|\S")
_INT  = re.compile(r"(?<!\w)(\d+)(?!\w)")

# tiny synonym pool (domain-agnostic + DPP-ish)
SYNONYMS = {
    "standard": ["spec", "specification", "standard"],
    "standards": ["specs", "specifications", "standards"],
    "label": ["marking", "label"],
    "labels": ["markings", "labels"],
    "battery": ["battery", "accumulator"],
    "wireless": ["wireless", "radio"],
    "test": ["test", "inspection", "check"],
    "testing": ["testing", "inspection"],
    "apply": ["apply", "pertain", "govern"],
    "applies": ["applies", "pertains", "governs"],
    "required": ["required", "needed", "necessary"],
    "steps": ["steps", "procedures"],
    "compliance": ["compliance", "conformance"],
    "safety": ["safety", "protection"],
    "emc": ["emc", "electromagnetic compatibility"],
    "weee": ["weee", "waste electrical"],
    "rohs": ["rohs", "restriction of hazardous substances"],
}

NOISE_TOKENS = ["uh", "please", "btw", "kindly", "hmm", "…", "###", "(?)", "[noise]"]

def _tok(s: str) -> List[str]:
    return _WORD.findall(s)

def synswap(query: str, intensity: int = 1, rng: random.Random | None = None) -> str:
    """
    Randomly replace up to `intensity` content words with synonyms.
    """
    rng = rng or random
    toks = _tok(query)
    idxs = list(range(len(toks)))
    rng.shuffle(idxs)
    swaps = 0
    for i in idxs:
        w = toks[i]
        key = w.lower()
        if key in SYNONYMS:
            cand = SYNONYMS[key]
            choice = rng.choice(cand)
            if w[0].isupper():
                choice = choice.capitalize()
            toks[i] = choice
            swaps += 1
            if swaps >= intensity:
                break
    return "".join(toks).strip()

def insert_noise(query: str, intensity: int = 1, rng: random.Random | None = None) -> str:
    """
    Insert `intensity` noise tokens into random positions.
    """
    rng = rng or random
    toks = _tok(query)
    for _ in range(intensity):
        pos = rng.randint(0, len(toks))
        toks.insert(pos, " " + rng.choice(NOISE_TOKENS) + " ")
    return "".join(toks).strip()

def numshift(query: str, shift: int = 1) -> str:
    """
    Shift each integer in text by +/- `shift` (default +1).
    """
    def _rep(m):
        n = int(m.group(1))
        return str(max(0, n + shift))
    return _INT.sub(_rep, query)

def augment(query: str, kind: str, intensity: int, rng_seed: int | None = 13) -> str:
    """
    kind ∈ {'synswap','noise','num+1','num-1'}
    """
    rng = random.Random(rng_seed)
    if kind == "synswap":
        return synswap(query, intensity=intensity, rng=rng)
    if kind == "noise":
        return insert_noise(query, intensity=intensity, rng=rng)
    if kind == "num+1":
        return numshift(query, shift=+intensity)
    if kind == "num-1":
        return numshift(query, shift=-intensity)
    return query
