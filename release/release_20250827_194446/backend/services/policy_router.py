# backend/services/policy_router.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Optional deps (graceful if missing)
try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

from backend.services.policy_costs import COSTS
from backend.services.symbolic_reasoning_service import sym_fire_flags  # type: ignore
from backend.services.policy_costs import episode_cost

# Day-2 features; if missing, we synthesize a minimal set
try:
    from backend.services.policy_features import extract_features  # type: ignore
except Exception:
    extract_features = None  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True, parents=True)
TESTS = ROOT / "tests" / "dpp_rl" / "tests.jsonl"

ACTIONS = ["BASE", "MEM", "SEARCH", "SYM", "MEMSYM"]
MODEL_PATH = ART / "policy_router.json"  # JSON (portable) instead of pickle


def _safe_features(query: str, product: Optional[str], session: str) -> Dict[str, float | int]:
    if extract_features is None:
        has_num = int(any(ch.isdigit() for ch in query))
        return {
            "len_query": len(query),
            "has_number": has_num,
            "has_product": int(bool(product)),
            "mem_top": 0.0,
            "mem_max3": 0.0,
            "search_top": 0.0,
            "search_max3": 0.0,
            "sym_fired": int(sym_fire_flags(query, product)),
        }
    try:
        return extract_features(query, product, session) or {}
    except Exception:
        has_num = int(any(ch.isdigit() for ch in query))
        return {
            "len_query": len(query),
            "has_number": has_num,
            "has_product": int(bool(product)),
            "mem_top": 0.0,
            "mem_max3": 0.0,
            "search_top": 0.0,
            "search_max3": 0.0,
            "sym_fired": int(sym_fire_flags(query, product)),
        }


def _latest_eval_joined() -> Path:
    cands = sorted(ART.glob("eval_joined_*.csv"))
    if not cands:
        raise FileNotFoundError("No eval_joined_*.csv found in artifacts/. Run run_eval_all.py first.")
    return cands[-1]


def _load_tests() -> Dict[str, Dict[str, Any]]:
    # id -> {query, product, session}
    import json as _json
    path = TESTS if TESTS.exists() else None
    if path is None:
        raise FileNotFoundError("tests/dpp_rl/tests.jsonl not found.")
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = _json.loads(s)
            out[row["id"]] = {
                "query": row["query"],
                "product": row.get("product"),
                "session": row.get("session", "s1"),
                "type": row.get("type"),
            }
    return out


def _label_best_action(eval_joined_path: Path) -> List[Tuple[Dict[str, Any], str]]:
    """
    Build (features, label) where label is the best action per id:
      - Prefer modes with success == 1
      - Break ties by LOWER approximate cost, then by LOWER latency
      - If none successful, pick the one with lowest cost
    Uses eval_joined.csv rows where mode ∈ ACTIONS.
    """
    import pandas as pd
    df = pd.read_csv(eval_joined_path)
    df = df[df["mode"].isin(ACTIONS)].copy()

    tests = _load_tests()
    rows: List[Tuple[Dict[str, Any], str]] = []

    for id_, group in df.groupby("id"):
        spec = tests.get(id_)
        if not spec:
            continue
        feats = _safe_features(spec["query"], spec.get("product"), spec.get("session", "s1"))

        # compute approximate action cost
        def approx_cost(mode: str) -> float:
            if mode == "MEMSYM":
                return COSTS.get("MEM", 0.0) + COSTS.get("SYM", 0.0)
            return COSTS.get(mode, 0.0)

        # choose label
        g = group.copy()
        g["approx_cost"] = g["mode"].apply(approx_cost)
        winners = g[g["success"] == 1]
        if not winners.empty:
            winners = winners.sort_values(by=["approx_cost", "latency_ms"], ascending=[True, True])
            label = str(winners.iloc[0]["mode"])
        else:
            # pick cheapest (tie -> lowest latency)
            losers = g.sort_values(by=["approx_cost", "latency_ms"], ascending=[True, True])
            label = str(losers.iloc[0]["mode"])

        rows.append((feats, label))

    return rows


def _stack_xy(rows: List[Tuple[Dict[str, Any], str]]):
    # fixed feature order for portability
    feat_names = ["len_query","has_number","has_product","mem_top","mem_max3","search_top","search_max3","sym_fired"]
    X, y = [], []
    for feats, label in rows:
        X.append([float(feats.get(k, 0.0)) for k in feat_names])
        y.append(ACTIONS.index(label))
    return feat_names, X, y


@dataclass
class RouterModel:
    feat_names: List[str]
    classes: List[str]
    coef: List[List[float]]    # shape [n_classes, n_features]
    intercept: List[float]     # shape [n_classes]

    @classmethod
    def train(cls) -> "RouterModel":
        rows = _label_best_action(_latest_eval_joined())
        if not rows:
            raise RuntimeError("No training rows found. Run run_eval_all.py to create eval_joined_*.csv first.")
        feat_names, X, y = _stack_xy(rows)

        if _HAVE_SK:
            import numpy as np
            lr = LogisticRegression(max_iter=2000, multi_class="ovr")
            lr.fit(np.asarray(X, dtype="float32"), np.asarray(y, dtype="int64"))
            coef = lr.coef_.astype("float32").tolist()
            intercept = lr.intercept_.astype("float32").tolist()
            classes = [ACTIONS[i] for i in lr.classes_.tolist()]
            return cls(feat_names, classes, coef, intercept)

        # Minimal softmax (one-vs-rest) fallback
        import numpy as np
        Xn = np.asarray(X, dtype="float32")
        yn = np.asarray(y, dtype="int64")
        C = len(ACTIONS)
        D = Xn.shape[1]
        W = np.zeros((C, D), dtype="float32")
        b = np.zeros((C,), dtype="float32")
        lr = 0.05
        for _ in range(500):
            # softmax
            scores = Xn @ W.T + b
            e = np.exp(scores - scores.max(axis=1, keepdims=True))
            P = e / np.clip(e.sum(axis=1, keepdims=True), 1e-6, None)
            # gradients
            Y = np.zeros_like(P); Y[np.arange(len(yn)), yn] = 1.0
            G = (P - Y) / len(yn)  # [N, C]
            dW = G.T @ Xn          # [C, D]
            db = G.sum(axis=0)     # [C]
            W -= lr * dW
            b -= lr * db
        return cls(feat_names, ACTIONS, W.tolist(), b.tolist())

    def predict(self, feats: Dict[str, Any]) -> str:
        import numpy as np
        x = np.asarray([[float(feats.get(k, 0.0)) for k in self.feat_names]], dtype="float32")  # [1,D]
        W = np.asarray(self.coef, dtype="float32")      # [C,D]
        b = np.asarray(self.intercept, dtype="float32") # [C]
        scores = x @ W.T + b
        # argmax
        idx = int(scores.argmax(axis=1)[0])
        return self.classes[idx]

    def save(self, path: Path = MODEL_PATH) -> Path:
        data = {
            "feat_names": self.feat_names,
            "classes": self.classes,
            "coef": self.coef,
            "intercept": self.intercept,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "RouterModel":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            feat_names=d["feat_names"],
            classes=d["classes"],
            coef=d["coef"],
            intercept=d["intercept"],
        )
