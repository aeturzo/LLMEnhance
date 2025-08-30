#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backend/services/policy_router.py

Supervised router over Day-2 features.
- Finds the newest artifacts/eval_joined_*.csv robustly
- Builds labels from {BASE, MEM, SYM, MEMSYM} (must be success==1; tie-break by latency, then steps)
- Trains LogisticRegression (or a linear fallback if sklearn not installed)
- Saves artifacts/policy_router.json
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    SK_OK = True
except Exception:
    SK_OK = False

from backend.services.policy_features import extract_features

# ---- Paths ----
# .../llmmain/backend/services/policy_router.py  -> parents:
# 0 file, 1 services, 2 backend, 3 llmmain (repo root)
THIS = Path(__file__).resolve()
BACKEND_DIR = THIS.parents[2]
REPO_ROOT = BACKEND_DIR.parent
ART = REPO_ROOT / "artifacts"
MODEL_PATH = ART / "policy_router.json"

LABELS = ["BASE", "MEM", "SYM", "MEMSYM"]  # prediction space

@dataclass
class TrainStats:
    n_queries: int
    n_train: int
    label_counts: Dict[str, int]
    loaded_from: str

def _candidate_eval_files(hint_dir: Optional[Path] = None) -> List[Path]:
    """Search several likely places for eval_joined_*.csv and return sorted by mtime."""
    seen: Dict[Path, Path] = {}
    dirs_to_check: List[Path] = []

    # Priority: explicit hint -> repo artifacts -> cwd/artifacts
    if hint_dir:
        dirs_to_check.append(hint_dir)
    dirs_to_check.append(ART)
    dirs_to_check.append(Path.cwd() / "artifacts")

    for d in dirs_to_check:
        if d and d.exists():
            for p in d.glob("eval_joined_*.csv"):
                seen[p.resolve()] = p

    # Fallback: recursive scan under repo root (cheap, project is small)
    for p in REPO_ROOT.glob("**/eval_joined_*.csv"):
        seen[p.resolve()] = p

    files = sorted(seen.values(), key=lambda p: p.stat().st_mtime)
    return files

def _select_label(rows: pd.DataFrame) -> Optional[str]:
    """Pick best label among single-module rows: success==1, lowest latency, then lowest steps."""
    cand = rows[rows["mode"].isin(LABELS)].copy()
    if cand.empty:
        return None
    cand["succ"] = cand["success"].astype(int)
    # order: success desc, latency asc, steps asc
    best = cand.sort_values(by=["succ", "latency_ms", "steps"], ascending=[False, True, True]).head(1)
    if best.empty or int(best["succ"].iloc[0]) != 1:
        return None
    return str(best["mode"].iloc[0])

def _rows_to_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], TrainStats]:
    need = {"id","mode","success","latency_ms","steps","query","product","session"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"eval_joined is missing columns: {missing}")

    X: List[List[float]] = []
    Y: List[int] = []
    label_counts: Dict[str,int] = {}
    feat_names: List[str] | None = None

    for qid, g in df.groupby("id", as_index=False):
        label = _select_label(g)
        if not label:
            continue
        row0 = g.iloc[0]
        q = str(row0["query"])
        p = (row0.get("product") if "product" in row0 else None)
        s = str(row0.get("session") or "s1")

        feats = extract_features(q, p, s) or {}
        keys = ["len_query","has_number","has_product","mem_top","mem_max3","search_top","search_max3","sym_fired"]
        x = [float(feats.get(k, 0.0) or 0.0) for k in keys]

        if feat_names is None:
            feat_names = keys

        X.append(x)
        Y.append(LABELS.index(label))
        label_counts[label] = label_counts.get(label, 0) + 1

    X_arr = np.asarray(X, dtype="float32")
    y_arr = np.asarray(Y, dtype="int64")
    stats = TrainStats(
        n_queries=int(df["id"].nunique()),
        n_train=int(len(y_arr)),
        label_counts=label_counts,
        loaded_from="",  # set by caller
    )
    return X_arr, y_arr, (feat_names or []), stats

class RouterModel:
    def __init__(self, coef: np.ndarray, bias: np.ndarray, feat_names: List[str]):
        self.coef_ = coef.astype("float32")       # (C, D)
        self.intercept_ = bias.astype("float32")  # (C,)
        self.feat_names = feat_names

    @staticmethod
    def train(artifacts_dir: Optional[str] = None) -> Tuple["RouterModel", TrainStats]:
        # Accept CLI arg or env var ARTIFACTS_DIR
        hint = Path(artifacts_dir) if artifacts_dir else None
        if not hint:
            env_dir = os.getenv("ARTIFACTS_DIR")
            if env_dir:
                hint = Path(env_dir)

        files = _candidate_eval_files(hint)
        if not files:
            checks = [
                str(hint) if hint else "<none>",
                str(ART),
                str(Path.cwd() / "artifacts"),
                f"{REPO_ROOT}/**/eval_joined_*.csv",
            ]
            raise RuntimeError(
                "No eval_joined_*.csv found. Looked in:\n  - " + "\n  - ".join(checks) +
                "\nRun `python run_eval_all.py` first."
            )

        p = files[-1]
        df = pd.read_csv(p)
        X, y, feat_names, stats = _rows_to_dataset(df)
        if len(y) == 0:
            raise RuntimeError("No training rows with success==1 among {BASE,MEM,SYM,MEMSYM}. Re-run eval with solvable tests.")

        # Train
        if SK_OK:
            clf = LogisticRegression(max_iter=1000, multi_class="ovr")
            clf.fit(X, y)
            coef = clf.coef_
            bias = clf.intercept_
        else:
            # simple least-squares one-vs-rest fallback
            C, D = len(LABELS), X.shape[1]
            coef = np.zeros((C, D), dtype="float32")
            bias = np.zeros((C,), dtype="float32")
            Xb = np.hstack([X, np.ones((X.shape[0],1), dtype="float32")])
            for c in range(C):
                y_c = (y == c).astype("float32")
                w, *_ = np.linalg.lstsq(Xb, y_c, rcond=None)
                coef[c, :] = w[:-1]
                bias[c] = w[-1]

        # Save
        ART.mkdir(exist_ok=True, parents=True)
        MODEL_PATH.write_text(json.dumps({
            "coef": coef.tolist(),
            "bias": bias.tolist(),
            "feat_names": feat_names,
            "labels": LABELS,
            "source_eval": str(p),
            "n_train": int(len(y)),
            "label_counts": stats.label_counts,
        }, indent=2), encoding="utf-8")

        model = RouterModel(coef=coef, bias=bias, feat_names=feat_names)
        stats.loaded_from = str(p)
        return model, stats

    @staticmethod
    def load(path: str | Path = MODEL_PATH) -> "RouterModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        coef = np.asarray(d["coef"], dtype="float32")
        bias = np.asarray(d["bias"], dtype="float32")
        feat_names = list(d["feat_names"])
        return RouterModel(coef, bias, feat_names)

    def predict_proba(self, feats: Dict[str, float]) -> np.ndarray:
        keys = self.feat_names
        x = np.asarray([float(feats.get(k, 0.0) or 0.0) for k in keys], dtype="float32")
        logits = (self.coef_ @ x) + self.intercept_
        m = float(logits.max())
        e = np.exp(logits - m)
        return e / np.maximum(e.sum(), 1e-12)

    def predict_label(self, feats: Dict[str, float]) -> str:
        p = self.predict_proba(feats)
        return LABELS[int(np.argmax(p))]
    # --- Compatibility shim for solve.py ---
def predict(self, features):
    """
    Compatibility shim for callers that expect .predict(...).
    It tries common method names without ever calling self(...).
    Returns a plain string like 'MEM', 'SYM', 'ADAPTIVERAG'.
    """
    # Prefer explicit methods; DO NOT call self(features) here.
    if hasattr(self, "route") and callable(self.route):
        out = self.route(features)
    elif hasattr(self, "choose") and callable(self.choose):
        out = self.choose(features)
    elif hasattr(self, "decide") and callable(self.decide):
        out = self.decide(features)
    else:
        return "ADAPTIVERAG"

    # Normalize outputs like ("MEM", score) or Enum to plain str
    if isinstance(out, (tuple, list)) and out:
        out = out[0]
    if hasattr(out, "value"):  # Enum
        out = out.value
    return str(out).upper()
