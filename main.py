"""
Hybrid ML + LLM + Agent demo.

Flow:
  1. Train an ML model on synthetic customer data.
  2. OrchestratorAgent predicts conversion + LLM explains why.
  3. RetrainingAgent detects feature drift and triggers retraining.

Run:
  python main.py
  ANTHROPIC_API_KEY=sk-... python main.py   # for real LLM explanations
"""

import numpy as np
from src.ml.model import train, MODEL_PATH
from src.agents.orchestrator import OrchestratorAgent
from src.agents.retraining_agent import RetrainingAgent


def main():
    # ── Step 1: Train ────────────────────────────────────────────────────────
    print("=== Step 1: Train ML model ===")
    model = train()
    print(f"Trained LogisticRegression → saved to {MODEL_PATH}\n")

    # ── Step 2: Predict + Explain ─────────────────────────────────────────────
    print("=== Step 2: Predict + Explain (OrchestratorAgent) ===")
    agent = OrchestratorAgent(model)

    customer = [0.5, -1.2, 0.8, 1.1]   # age_norm, spend_norm, visits_norm, recency_norm
    result = agent.run(customer)

    pred = result["prediction"]
    outcome = "Convert" if pred["label"] == 1 else "No Convert"
    print(f"Customer features : {customer}")
    print(f"ML prediction     : {outcome} ({pred['probability']:.0%} confidence)")
    print(f"LLM explanation   : {result['explanation']}\n")

    # ── Step 3: Drift → Retraining ────────────────────────────────────────────
    print("=== Step 3: Drift detection → RetrainingAgent ===")
    rng = np.random.default_rng(0)
    baseline = rng.standard_normal((100, 4)).tolist()
    shifted  = (rng.standard_normal((100, 4)) + 0.6).tolist()   # simulated drift

    retraining_agent = RetrainingAgent(drift_threshold=0.1)
    outcome = retraining_agent.check_and_act(
        baseline_data=baseline,
        new_data=shifted,
        retrain_fn=lambda: train(),
    )
    print(f"Drift score   : {outcome['drift_score']:.3f}")
    print(f"Drift detected: {outcome['drifted']}")
    print(f"Action taken  : {outcome['action'] or 'none'}")


if __name__ == "__main__":
    main()
