"""
Hybrid ML + LLM + Agent demo.

Flow:
  1. Train an ML model on synthetic data.
  2. OrchestratorAgent shows both routing paths:
       - High confidence  →  LLM skipped (model is certain, explanation adds little)
       - Low confidence   →  LLM called  (model is uncertain, explanation adds value)
  3. RetrainingAgent detects feature drift and triggers retraining.

Run:
  python main.py
  ANTHROPIC_API_KEY=sk-... python main.py   # for real LLM explanations

Customise behaviour in configs/config.yaml — no code changes needed.
"""

import numpy as np
from src.core import load_config
from src.ml.model import train, predict
from src.agents.orchestrator import OrchestratorAgent
from src.agents.retraining_agent import RetrainingAgent


def _label(cfg, label_int):
    return cfg["llm"]["label_names"].get(str(label_int), str(label_int))


def main():
    cfg = load_config()
    threshold = cfg["agents"]["explain_confidence_threshold"]

    # ── Step 1: Train ─────────────────────────────────────────────────────────
    print("=== Step 1: Train ML model ===")
    model = train(cfg)
    print(f"Trained model → saved to {cfg['ml']['model_path']}\n")

    # ── Step 2: Routing demo ──────────────────────────────────────────────────
    print(f"=== Step 2: OrchestratorAgent routing (threshold={threshold}) ===")
    agent = OrchestratorAgent(model, cfg)

    # 2a: high-confidence features — LLM should be skipped
    high_conf_features = [0.5, -1.2, 0.8, 1.1]
    result_a = agent.run(high_conf_features)
    pred_a = result_a["prediction"]
    print(f"[Path A — high confidence]")
    print(f"  Features    : {high_conf_features}")
    print(f"  Prediction  : {_label(cfg, pred_a['label'])} ({pred_a['probability']:.0%})")
    print(f"  Routing     : {result_a['routing']['reason']}")
    if "explanation" in result_a:
        print(f"  Explanation : {result_a['explanation']}")
    print()

    # 2b: force LLM with explain_result=True to show the override path
    result_b = agent.run(high_conf_features, explain_result=True)
    print(f"[Path B — forced explanation]")
    print(f"  Features    : {high_conf_features}")
    print(f"  Prediction  : {_label(cfg, result_b['prediction']['label'])} ({result_b['prediction']['probability']:.0%})")
    print(f"  Routing     : {result_b['routing']['reason']}")
    print(f"  Explanation : {result_b['explanation']}\n")

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
