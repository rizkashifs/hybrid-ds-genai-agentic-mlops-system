# Architecture

## Overview

The system has three layers that run in sequence for every request:

```
Input (customer features)
        ↓
  OrchestratorAgent          ← agent layer
   ├── ML Model               ← prediction layer
   └── LLM Reasoner           ← reasoning layer
        ↓
  Combined result
        ↓
  RetrainingAgent             ← MLOps automation layer
   └── DriftDetector
```

---

## Layer Breakdown

### ML Layer (`src/ml/model.py`)

- Model: `LogisticRegression` (scikit-learn)
- Training data: synthetic 4-feature customer dataset (500 samples)
- Output: `{label, probability, features}`
- The model is serialised to `model.pkl` after training and reloaded on prediction

### LLM Layer (`src/llm/reasoner.py`)

- Provider: Anthropic Claude (`claude-sonnet-4-6`)
- Input: ML prediction dict
- Output: 2-sentence plain-English explanation
- The system prompt is cached with `cache_control: ephemeral` to reduce token cost on repeated calls
- Falls back to a mock response if `ANTHROPIC_API_KEY` is not set

### Agent Layer (`src/agents/`)

**OrchestratorAgent**
- Receives raw customer features
- Calls `predict()` from the ML layer
- Passes the result to `explain()` from the LLM layer
- Returns both together as a single combined dict

**RetrainingAgent**
- Accepts baseline data and new incoming data
- Calls `detect_drift()` to compute a drift score
- If score exceeds the configured threshold, marks `drifted=True` and fires the provided `retrain_fn` callback
- The callback is injected at call time, so retraining logic stays outside the agent

### Monitoring (`src/monitoring/drift.py`)

- Drift metric: mean absolute difference in per-feature means between baseline and current data
- Simple and interpretable — not tied to any specific statistical test
- Threshold is configurable per `RetrainingAgent` instance (default: `0.1`)

---

## Data Flow (Step by Step)

1. Caller passes a feature vector to `OrchestratorAgent.run()`
2. Agent calls `predict(model, features)` → gets `{label, probability}`
3. Agent calls `explain(prediction)` → gets a string from Claude (or mock)
4. Agent returns `{prediction, explanation}` to the caller
5. Separately, `RetrainingAgent.check_and_act()` receives baseline vs. new data
6. `detect_drift()` computes the score
7. If drifted, the agent calls the injected `retrain_fn` (which re-runs `train()`)

---

## Extension Points

| What to extend | Where |
|---|---|
| Swap ML model | `src/ml/model.py` — replace `LogisticRegression` |
| Change LLM prompt | `src/llm/reasoner.py` — edit `_SYSTEM` |
| Add an evaluation agent | `src/agents/` — new class, same pattern |
| Replace drift metric | `src/monitoring/drift.py` — swap the function body |
| Add a real retraining pipeline | Pass a real `retrain_fn` into `RetrainingAgent.check_and_act()` |
