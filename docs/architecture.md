# Architecture

## Overview

The system has three layers that run in sequence for every request:

```
Input (feature vector)
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

- Model: `LogisticRegression` (scikit-learn) — swap for any model in this file
- Training data: synthetic dataset generated from config; replace with your real data loader
- Output: `{label, probability, features}`
- Model artifact path, sample count, and feature count all come from `configs/config.yaml`

**Log events emitted:**

| Event | Fields |
|---|---|
| `model.trained` | `save_path`, `n_samples` |
| `model.loaded` | `path` |
| `model.predicted` | `label`, `probability` |

### LLM Layer (`src/llm/reasoner.py`)

- Provider: Anthropic Claude (model configurable via `llm.model` in config)
- Input: ML prediction dict
- Output: plain-English explanation string
- System prompt, label names, and feature names are all read from config — no domain-specific text in code
- The system prompt is cached with `cache_control: ephemeral` to reduce token cost on repeated calls
- Falls back to a clearly-labelled mock response if `ANTHROPIC_API_KEY` is not set

**Log events emitted:**

| Event | Fields |
|---|---|
| `llm.explain.mock` | `label` |
| `llm.explain.start` | `model`, `label` |
| `llm.explain.done` | `input_tokens`, `output_tokens` |

### Agent Layer (`src/agents/`)

**OrchestratorAgent**

Routing decision tree:

```
run(features, explain_result=None)
        ↓
  ML predict()  →  {label, probability}
        ↓
  explain_result is False?  →  skip LLM  (caller override)
  explain_result is True?   →  call LLM  (caller override)
  explain_result is None?   →  confidence < threshold?
                                   yes  →  call LLM  (uncertain, explanation adds value)
                                   no   →  skip LLM  (confident, explanation adds little)
        ↓
  return {prediction, routing, explanation?}
```

- `routing` key is always present: `{"explain_called": bool, "reason": str}`
- `explanation` key is only present when the LLM was called
- Threshold configured via `agents.explain_confidence_threshold` in `config.yaml`
- Logs wall-clock duration and routing decision on every run

**Log events emitted:**

| Event | Fields |
|---|---|
| `agent.orchestrator.run.start` | `n_features` |
| `agent.orchestrator.run.done` | `duration_ms`, `explain_called`, `reason` |

**RetrainingAgent**
- Accepts baseline data and new incoming data
- Calls `detect_drift()` to compute a drift score
- If score exceeds `monitoring.drift_threshold` (from config), raises `RuntimeError` unless `warn_only=True`
- Fires the injected `retrain_fn` callback — retraining logic stays outside the agent
- `action="retraining_triggered"` only when the callback actually ran

**Log events emitted:**

| Event | Fields |
|---|---|
| `agent.retraining.check` | `drift_score`, `threshold`, `drifted` |
| `agent.retraining.triggered` | `drift_score` |

### Monitoring (`src/monitoring/drift.py`)

- Drift metric: mean absolute difference in per-feature means between baseline and current data
- Threshold is set in `configs/config.yaml` under `monitoring.drift_threshold`

**Log events emitted:**

| Event | Fields |
|---|---|
| `monitoring.drift.computed` | `score` |

---

## Observability

All layers emit structured JSON logs to **stderr** using Python's stdlib `logging` with a custom `_JSONFormatter` in `src/core/__init__.py`. Each log line is a self-contained JSON object:

```json
{
  "ts": "2026-05-04T16:37:53.951585+00:00",
  "level": "INFO",
  "logger": "src.ml.model",
  "event": "model.predicted",
  "label": 1,
  "probability": 0.9278
}
```

**Usage:**

```bash
# Human-readable demo output only
python main.py

# Logs only (pipe stderr, discard stdout)
python main.py 2>&1 1>/dev/null | jq .

# Filter to a specific event
python main.py 2>&1 1>/dev/null | jq 'select(.event == "agent.retraining.check")'

# Change log level (default: INFO)
LOG_LEVEL=DEBUG python main.py
```

Logs are designed to ship directly to any JSON-capable sink: Datadog, CloudWatch Logs, Loki, or a local file via shell redirection.

---

## Data Flow (Step by Step)

1. Caller passes a feature vector to `OrchestratorAgent.run()`
2. Agent logs `agent.orchestrator.run.start`
3. Agent calls `predict(model, features)` → gets `{label, probability}`, logs `model.predicted`
4. Agent evaluates the routing decision: compare `probability` to `explain_confidence_threshold`
5. If LLM is called: `explain(prediction, cfg)` → explanation string, logs `llm.explain.done`
6. Agent returns `{prediction, routing, explanation?}`, logs `agent.orchestrator.run.done` with duration and reason
7. Separately, `RetrainingAgent.check_and_act()` receives baseline vs. new data
8. `detect_drift()` computes the score, logs `monitoring.drift.computed`
9. Agent logs `agent.retraining.check`, and if drifted and callback provided, fires `retrain_fn` and logs `agent.retraining.triggered`

---

## Extension Points

| What to extend | Where |
|---|---|
| Swap ML model | `src/ml/model.py` — replace `LogisticRegression` |
| Change LLM prompt / domain | `configs/config.yaml` — edit `llm.system_prompt`, `llm.label_names`, `llm.feature_names` |
| Add an evaluation agent | `src/agents/` — new class, same pattern |
| Replace drift metric | `src/monitoring/drift.py` — swap the function body, signature stays the same |
| Add a real retraining pipeline | Pass a real `retrain_fn` into `RetrainingAgent.check_and_act()` |
| Ship logs to a sink | Redirect stderr or add a second handler in `src/core/__init__.py` |
