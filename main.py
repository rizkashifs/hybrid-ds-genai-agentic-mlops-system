"""
Hybrid ML + LLM + Agent demo.

Flow:
  1. Train an ML model on synthetic data.
  2. OrchestratorAgent predicts + LLM explains (if configured).
  3. RetrainingAgent detects feature drift and triggers retraining.

Run:
  python main.py
  ANTHROPIC_API_KEY=sk-... python main.py   # for real LLM explanations

Customise behaviour in configs/config.yaml — no code changes needed.
"""

import numpy as np
from src.core import load_config
from src.ml.model import train
from src.agents.orchestrator import OrchestratorAgent
from src.agents.retraining_agent import RetrainingAgent


def main():
    cfg = load_config()

    # ── Step 1: Train ─────────────────────────────────────────────────────────
    print("=== Step 1: Train ML model ===")
    model = train(cfg)
    print(f"Trained model → saved to {cfg['ml']['model_path']}\n")

    # ── Step 2: Predict + Explain ─────────────────────────────────────────────
    print("=== Step 2: Predict + Explain (OrchestratorAgent) ===")
    agent = OrchestratorAgent(model, cfg)

    features = [0.5, -1.2, 0.8, 1.1]
    result = agent.run(features)

    pred = result["prediction"]
    label_name = cfg["llm"]["label_names"].get(str(pred["label"]), str(pred["label"]))
    print(f"Features          : {features}")
    print(f"ML prediction     : {label_name} ({pred['probability']:.0%} confidence)")
    print(f"LLM explanation   : {result['explanation']}\n")

    # ── Step 3: Drift → Retraining ────────────────────────────────────────────
    print("=== Step 3: Drift detection → RetrainingAgent ===")
    rng = np.random.default_rng(0)
    baseline = rng.standard_normal((100, cfg["ml"]["n_features"])).tolist()
    shifted  = (rng.standard_normal((100, cfg["ml"]["n_features"])) + 0.6).tolist()

    retraining_agent = RetrainingAgent(cfg)
    outcome = retraining_agent.check_and_act(
        baseline_data=baseline,
        new_data=shifted,
        retrain_fn=lambda: train(cfg),
    )
    print(f"Drift score   : {outcome['drift_score']:.3f}")
    print(f"Drift detected: {outcome['drifted']}")
    print(f"Action taken  : {outcome['action'] or 'none'}")


if __name__ == "__main__":
    main()
