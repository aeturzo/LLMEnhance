from __future__ import annotations

import json, math, os, random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

from backend.services import memory_service, search_service
try:
    from backend.services.symbolic_reasoning_service import SymbolicReasoner
except Exception:
    SymbolicReasoner = None  # type: ignore

# ------------------ Actions & State ------------------

class Action(IntEnum):
    CALL_MEMORY = 0
    CALL_SEARCH = 1
    CALL_SYMBOLIC = 2
    ANSWER = 3

ACTIONS = [Action.CALL_MEMORY, Action.CALL_SEARCH, Action.CALL_SYMBOLIC, Action.ANSWER]

@dataclass
class Obs:
    query: str
    product: Optional[str]
    mem_hits: int = 0
    doc_hits: int = 0
    has_sym: bool = False
    steps_taken: int = 0
    last_added_info: int = 0

@dataclass
class StepTrace:
    action: Action
    detail: str

@dataclass
class Trajectory:
    obs: List[Obs] = field(default_factory=list)
    acts: List[Action] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    steps: List[StepTrace] = field(default_factory=list)
    answer: str = ""

# ------------------ Policy ------------------

class LinearSoftmaxPolicy:
    """
    π(a|s) ∝ exp(w_a · φ(s)) over simple features.
    """
    def __init__(self, feat_dim: int, n_actions: int, seed: int = 0):
        random.seed(seed)
        self.W = [[(random.random()-0.5)*0.1 for _ in range(feat_dim)] for _ in range(n_actions)]

    @staticmethod
    def featurize(o: Obs) -> List[float]:
        qlen = len(o.query)
        return [
            1.0,                              # bias
            1.0 if o.product else 0.0,       # product present
            float(o.mem_hits > 0),           # any memory hits
            float(o.doc_hits > 0),           # any search hits
            float(o.has_sym),                # symbolic available
            min(qlen/200.0, 1.0),            # normalized query length
            float(o.steps_taken)/6.0,        # steps so far
            min(o.last_added_info/5.0, 1.0), # fruitful last step
        ]

    def _logits(self, phi: List[float]) -> List[float]:
        return [sum(wi*fj for wi, fj in zip(w, phi)) for w in self.W]

    @staticmethod
    def _softmax(xs: List[float]) -> List[float]:
        m = max(xs)
        exps = [math.exp(x-m) for x in xs]
        Z = sum(exps) + 1e-12
        return [e/Z for e in exps]

    def action_dist(self, obs: Obs) -> List[float]:
        phi = self.featurize(obs)
        return self._softmax(self._logits(phi))

    def sample(self, obs: Obs) -> Tuple[Action, List[float], List[float]]:
        phi = self.featurize(obs)
        probs = self._softmax(self._logits(phi))
        r, cum = random.random(), 0.0
        for a, p in enumerate(probs):
            cum += p
            if r <= cum:
                return Action(a), probs, phi
        return Action(len(probs)-1), probs, phi

    def update_reinforce(self, grads: List[Tuple[int, List[float], float]], lr: float = 0.05):
        # Δw_a = lr * G * φ(s) for sampled action a (vanilla REINFORCE one-hot)
        for a_idx, phi, G in grads:
            for j, f in enumerate(phi):
                self.W[a_idx][j] += lr * G * f

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"W": self.W}, f)

    @classmethod
    def load(cls, path: str) -> "LinearSoftmaxPolicy":
        data = json.load(open(path, "r", encoding="utf-8"))
        obj = cls(feat_dim=len(data["W"][0]), n_actions=len(data["W"]))
        obj.W = data["W"]
        return obj

# ------------------ Environment ------------------

class DPPRLEnv:
    """
    Environment that lets the agent call: memory, search, symbolic; then ANSWER.
    Reward comes from a task-specific evaluator.
    """
    def __init__(self, reasoner: Optional[SymbolicReasoner], evaluator, max_steps: int = 6):
        self.reasoner = reasoner
        self.evaluator = evaluator
        self.max_steps = max_steps

    def run_episode(self, sample: Dict[str, Any], policy: LinearSoftmaxPolicy) -> Trajectory:
        traj = Trajectory()
        session = sample.get("session") or "s0"
        query = sample["query"]
        product = sample.get("product")

        mem_hits, doc_hits = [], []
        sym_payload: Optional[Dict[str, Any]] = None

        def mkobs(last_added_info: int) -> Obs:
            return Obs(
                query=query,
                product=product,
                mem_hits=len(mem_hits),
                doc_hits=len(doc_hits),
                has_sym=self.reasoner is not None,
                steps_taken=len(traj.steps),
                last_added_info=last_added_info,
            )

        for _ in range(self.max_steps):
            o = mkobs(last_added_info=0)
            traj.obs.append(o)

            act, _, _ = policy.sample(o)
            traj.acts.append(act)

            if act == Action.CALL_MEMORY:
                hits = memory_service.retrieve(session_id=session, query=query, top_k=3)
                mem_hits = hits or []
                traj.steps.append(StepTrace(action=act, detail=f"memory(k={len(mem_hits)})"))
                mkobs(last_added_info=len(mem_hits))

            elif act == Action.CALL_SEARCH:
                hits = search_service.search(query_text=query, top_k=3)
                doc_hits = hits or []
                traj.steps.append(StepTrace(action=act, detail=f"search(k={len(doc_hits)})"))
                mkobs(last_added_info=len(doc_hits))

            elif act == Action.CALL_SYMBOLIC and self.reasoner and product:
                reqs = [r.split("#")[-1] for r in self.reasoner.check_compliance_requirements(product)]
                miss = [r.split("#")[-1] for r in self.reasoner.suggest_missing_steps(product)]
                sym_payload = {"requires": reqs, "missing": miss}
                traj.steps.append(StepTrace(action=act, detail="symbolic()"))
                mkobs(last_added_info=len(reqs)+len(miss))

            elif act == Action.ANSWER:
                if mem_hits:
                    ans = getattr(mem_hits[0], "content", str(mem_hits[0]))
                elif doc_hits:
                    top = doc_hits[0]
                    snippet = getattr(top, "snippet", None)
                    name = getattr(top, "document_name", None) or getattr(top, "name", None)
                    ans = snippet or name or "Found a relevant document."
                elif isinstance(sym_payload, dict):
                    reqs = ", ".join(sym_payload.get("requires", []))
                    miss = ", ".join(sym_payload.get("missing", [])) or "none"
                    ans = f"requires: {reqs}; missing: {miss}"
                else:
                    ans = "No relevant information found."
                traj.answer = ans
                R = float(self.evaluator(sample, ans, sym_payload))
                traj.rewards.append(R)
                return traj
            else:
                traj.steps.append(StepTrace(action=act, detail="noop"))

        traj.rewards.append(-0.5)  # never answered
        traj.answer = "No answer."
        return traj

# ------------------ Training ------------------

class RLTrainer:
    def __init__(self, reasoner: Optional[SymbolicReasoner], policy: Optional[LinearSoftmaxPolicy] = None):
        self.reasoner = reasoner
        self.policy = policy or LinearSoftmaxPolicy(feat_dim=8, n_actions=len(ACTIONS), seed=0)

    @staticmethod
    def default_evaluator():
        """
        (sample, answer, sym_payload) -> reward in [-1, 1]
        - gold.contains: exact/substring check on final answer
        - gold.requires/missing: Jaccard similarity on symbolic sets
        """
        def _eval(sample, answer: str, sym_payload):
            gold = sample.get("gold", {})
            contains = gold.get("contains")
            if contains is not None:
                return 1.0 if (contains.lower() in (answer or "").lower()) else -1.0

            exp_req = set(gold.get("requires", []))
            exp_miss = set(gold.get("missing", []))
            got_req = set((sym_payload or {}).get("requires", []))
            got_miss = set((sym_payload or {}).get("missing", []))

            if exp_req or exp_miss:
                jacc = 0.0
                if exp_req or got_req:
                    inter = len(exp_req & got_req); uni = max(1, len(exp_req | got_req))
                    jacc += inter/uni
                if exp_miss or got_miss:
                    inter = len(exp_miss & got_miss); uni = max(1, len(exp_miss | got_miss))
                    jacc += inter/uni
                return 2.0*(jacc/2.0) - 0.5  # map [0,1] -> [-0.5, 0.5]
            return -0.2
        return _eval

    def fit(self, episodes: List[Dict[str, Any]], epochs: int = 40, lr: float = 0.05, shuffle: bool = True) -> Dict[str, Any]:
        env = DPPRLEnv(self.reasoner, evaluator=self.default_evaluator())
        history = {"avg_return": []}
        for _ in range(epochs):
            if shuffle:
                random.shuffle(episodes)
            returns = []
            for sample in episodes:
                traj = env.run_episode(sample, self.policy)
                G = sum(traj.rewards)
                returns.append(G)
                grads = [(int(a), self.policy.featurize(o), G) for a, o in zip(traj.acts, traj.obs)]
                self.policy.update_reinforce(grads, lr=lr)
            history["avg_return"].append(sum(returns)/max(1, len(returns)))
        return history
