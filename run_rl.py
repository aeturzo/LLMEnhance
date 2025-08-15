import json, pathlib
from backend.services.rl_agent import RLTrainer
from backend.services.symbolic_reasoning_service import build_reasoner
from backend.main import app

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "tests" / "dpp_rl" / "episodes.jsonl"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

def load_episodes():
    episodes = []
    with open(DATA, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes

if __name__ == "__main__":
    app.state.reasoner = build_reasoner(run_owl_rl=True)
    trainer = RLTrainer(reasoner=app.state.reasoner)
    episodes = load_episodes()
    hist = trainer.fit(episodes, epochs=50, lr=0.05)
    policy_path = ART / "rl_policy.json"
    trainer.policy.save(str(policy_path))
    print(f"Saved policy to {policy_path}")
    print("Training curve (first/last 5):", hist["avg_return"][:5], "...", hist["avg_return"][-5:])
