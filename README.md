# Hybrid DS + GenAI + Agentic MLOps System

[![CI](https://github.com/rizkashifs/hybrid-ds-genai-agentic-mlops-system/actions/workflows/ci.yml/badge.svg)](https://github.com/rizkashifs/hybrid-ds-genai-agentic-mlops-system/actions/workflows/ci.yml)

A working demonstration of how ML, LLMs, and agents combine into a single unified system — not three separate tools.

---

## The Idea

Most teams build either an ML pipeline **or** an LLM app. This repo shows what it looks like when they work together, with agents connecting the two and automating the operations layer.

This repository explores how agentic systems can automate the machine learning lifecycle, moving beyond static pipelines to adaptive, self-improving systems.

```
User / System Trigger
        ↓
  OrchestratorAgent
   ├── ML Model       →  prediction + confidence score
   └── LLM (conditional)
        ├── confidence < threshold  →  call LLM  (uncertain, explain)
        └── confidence ≥ threshold  →  skip LLM  (confident, no cost)
        ↓
  Combined Output  {prediction, routing, explanation?}
        ↓
  RetrainingAgent  →  monitors drift → triggers retraining automatically
```

---

## Design Principles

- Standardization over ad-hoc pipelines
- Observability as a first-class concern
- Reproducibility over experimentation speed
- Clear separation of concerns across lifecycle stages

---

## Layers

### 1. ML Layer — `src/ml/model.py`
Any scikit-learn compatible model — the template ships with `LogisticRegression` as a working example. Swap it for `XGBClassifier`, `RandomForestRegressor`, or any model that exposes `predict` and `predict_proba`. Returns a structured prediction:
```python
{"label": 1, "probability": 0.93, "features": [0.5, -1.2, 0.8, 1.1]}
```

### 2. LLM Layer — `src/llm/reasoner.py`
Takes the ML prediction and asks Claude to explain it in plain English. The system prompt, label names, and feature names are all configured in `config.yaml` — no code changes needed to adapt this to a new domain. Uses prompt caching so repeated calls are cheap.

### 3. Agent Layer — `src/agents/`

| Agent | File | Responsibility |
|---|---|---|
| `OrchestratorAgent` | `orchestrator.py` | Calls ML + LLM, combines result |
| `RetrainingAgent` | `retraining_agent.py` | Detects feature drift, fires retraining |

The monitoring logic lives in `src/monitoring/drift.py` — a simple mean-shift score across features.

---

## Agent Responsibilities

- **Orchestrator Agent** → decides ML vs LLM vs hybrid
- **Retraining Agent** → triggers retraining based on signals
- **Evaluation Agent** → compares and selects models
- **Debug Agent** → analyzes failures and suggests fixes

### `OrchestratorAgent` — `src/agents/orchestrator.py`

The entry point for every prediction request. It sequences the ML and LLM layers and applies routing logic so the LLM is only called when it adds value.

| Step | What happens |
|---|---|
| 1. Receive features | Accepts a feature vector as a Python list |
| 2. ML prediction | Calls `predict()` → returns `{label, probability, features}` |
| 3. Routing decision | If `probability < explain_confidence_threshold`, flag for LLM; otherwise skip |
| 4. LLM call (conditional) | Calls `explain()` with the prediction and config — only when flagged |
| 5. Return result | Returns `{prediction, routing, explanation?}` — `routing` is always present |

Callers can override routing with `explain_result=True` (always explain) or `explain_result=False` (never explain).

### `RetrainingAgent` — `src/agents/retraining_agent.py`

The MLOps automation layer. It monitors for data drift and triggers retraining when the incoming distribution has shifted enough to degrade model performance.

| Step | What happens |
|---|---|
| 1. Receive data | Accepts `baseline_data` (training distribution) and `new_data` (recent production data) |
| 2. Drift computation | Calls `detect_drift()` → mean absolute difference in per-feature means |
| 3. Threshold check | Compares drift score to `monitoring.drift_threshold` from config |
| 4. Act | If drifted: calls `retrain_fn()` if provided; raises `RuntimeError` if not (unless `warn_only=True`) |
| 5. Return result | Returns `{drift_score, drifted, action}` — `action` is `"retraining_triggered"` or `None` |

The `retrain_fn` is injected by the caller — in the demo it's a simple `train()` call; in production it can be a pipeline trigger, Airflow DAG kick, or any callable.

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python main.py
```

**With real LLM explanations (Claude API):**
```bash
ANTHROPIC_API_KEY=sk-... python main.py
```

Without an API key the system still runs — the LLM step returns a clearly labelled mock response.

---

## Tests

```bash
python3 -m pytest tests/ -v
```

19 tests covering every public function across all layers. No test touches the network or writes outside `pytest`'s `tmp_path`.

| File | What it covers |
|---|---|
| `tests/test_config.py` | Config loads, required keys present, missing file raises |
| `tests/test_model.py` | Train/load round-trip, prediction structure, wrong feature count |
| `tests/test_drift.py` | Zero drift, non-zero drift, known-value arithmetic |
| `tests/test_retraining_agent.py` | No drift, drift + callback, drift + no callback raises, warn_only |
| `tests/test_orchestrator.py` | explain on/off, prediction structure, LLM is mocked |

---

## CI

Every push and pull request runs two jobs via GitHub Actions (`.github/workflows/ci.yml`):

| Job | Command | What it checks |
|---|---|---|
| `lint` | `ruff check src/ tests/` | Style, unused imports, common errors |
| `test` | `python3 -m pytest tests/ -v` | All 24 unit tests |

To run the same checks locally before pushing:
```bash
ruff check src/ tests/
python3 -m pytest tests/ -v
```

---

## Logging

Every layer emits structured JSON logs to **stderr**. The demo `print()` output goes to **stdout**, so they never mix.

```bash
# View logs alongside demo output
python main.py

# Logs only, formatted
python main.py 2>&1 1>/dev/null | jq .

# Filter to a specific event
python main.py 2>&1 1>/dev/null | jq 'select(.event == "agent.retraining.check")'

# Raise or lower verbosity
LOG_LEVEL=DEBUG python main.py
```

Sample log line:
```json
{"ts": "2026-05-04T16:37:53.952557+00:00", "level": "INFO", "logger": "src.agents.retraining_agent", "event": "agent.retraining.check", "drift_score": 0.6208, "threshold": 0.1, "drifted": true}
```

See `docs/architecture.md` for the full list of events and fields emitted by each layer.

---

## Serving (HTTP API)

```bash
uvicorn src.services.api:app --reload
```

The model loads once at startup. Three endpoints are available:

**`GET /health`**
```bash
curl http://localhost:8000/health
# {"status": "ok", "model_loaded": true}
```

**`POST /predict`**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, -1.2, 0.8, 1.1]}'
# {"prediction": {...}, "routing": {"explain_called": false, "reason": "confidence 0.93 >= threshold 0.85"}, "explanation": null}

# Force LLM regardless of confidence:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, -1.2, 0.8, 1.1], "force_explain": true}'
```

**`POST /drift-check`**
```bash
curl -X POST http://localhost:8000/drift-check \
  -H "Content-Type: application/json" \
  -d '{"baseline": [[0,0,0,0],[0,0,0,0]], "new_data": [[1,1,1,1],[1,1,1,1]]}'
# {"drift_score": 1.0, "drifted": true, "action": null}
```

The drift-check endpoint is monitoring-only — it reports drift but does not trigger retraining. To trigger retraining from the API, wire `retrain_fn` in a background task.

> **Note:** The API has no authentication. Add auth middleware before exposing it outside a trusted network.

---

## Example Output

```
=== Step 1: Train ML model ===
Trained model → saved to model.pkl

=== Step 2: Predict + Explain (OrchestratorAgent) ===
Features          : [0.5, -1.2, 0.8, 1.1]
ML prediction     : positive (93% confidence)
LLM explanation   : [Mock LLM] Predicted 'positive' with 93% confidence.
                    Set ANTHROPIC_API_KEY to get a real explanation from Claude.

=== Step 3: Drift detection → RetrainingAgent ===
Drift score   : 0.621
Drift detected: True
Action taken  : retraining_triggered
```

Label names (`positive`/`negative`) and feature names are set in `config.yaml`.

---

## Example Workflow

Two execution paths depending on model confidence. Both go through `OrchestratorAgent`.

**Path A: High-confidence prediction (LLM skipped)**

```
features = [0.5, -1.2, 0.8, 1.1]
         ↓
OrchestratorAgent.run(features)
         ↓
ML prediction: label=1, probability=0.93
         ↓
Routing: 0.93 >= threshold 0.85  →  skip LLM
         ↓
Result: {
  "prediction":  {"label": 1, "probability": 0.93, "features": [...]},
  "routing":     {"explain_called": false, "reason": "confidence 0.93 >= threshold 0.85"},
  "explanation": null
}
```

**Path B: Low-confidence prediction (LLM called)**

```
features = [0.1, 0.0, -0.1, 0.2]
         ↓
OrchestratorAgent.run(features)
         ↓
ML prediction: label=1, probability=0.61
         ↓
Routing: 0.61 < threshold 0.85  →  call LLM
         ↓
LLM: Claude explains the prediction in 2 sentences
         ↓
Result: {
  "prediction":  {"label": 1, "probability": 0.61, "features": [...]},
  "routing":     {"explain_called": true, "reason": "confidence 0.61 < threshold 0.85"},
  "explanation": "The model predicted positive with moderate confidence ..."
}
```

**Background: Drift detection + retraining**

```
RetrainingAgent.check_and_act(baseline_data, new_data, retrain_fn=train)
         ↓
drift_score = mean |baseline_feature_means − new_feature_means| = 0.62
         ↓
0.62 > threshold 0.10  →  drifted
         ↓
retrain_fn() called  →  model retrained on new distribution
         ↓
Result: {"drift_score": 0.62, "drifted": true, "action": "retraining_triggered"}
```

---

## Project Structure

```
hybrid-ds-genai-agentic-mlops-system/
├── main.py                        # End-to-end demo
├── requirements.txt
├── src/
│   ├── ml/
│   │   └── model.py               # Train, load, predict
│   ├── llm/
│   │   └── reasoner.py            # Claude API explanation
│   ├── agents/
│   │   ├── orchestrator.py        # OrchestratorAgent
│   │   └── retraining_agent.py    # RetrainingAgent
│   └── monitoring/
│       └── drift.py               # Drift detection
├── configs/
│   └── config.yaml
└── docs/
    ├── architecture.md
    └── decisions.md
```

---

## Why a Hybrid System

### ML alone is not enough

A model score with no explanation is a black box. A data scientist can interpret a 93% confidence prediction; a business stakeholder, compliance team, or end user cannot. Without reasoning, predictions are hard to audit, hard to trust, and hard to act on.

Model performance also degrades silently. Without active monitoring, a model trained on last quarter's data quietly becomes stale — no alert fires, no retraining happens. Teams discover drift by noticing downstream metrics fall off weeks later.

### LLM alone is not enough

An LLM reasoning from scratch on structured data is expensive, slow, and unreliable. It has no grounded prediction to reason from, and calling it on every request is unnecessary cost — a 97% confident prediction needs no explanation.

### Together, they cover each other's weaknesses

| Challenge | ML alone | LLM alone | Hybrid (this system) |
|---|---|---|---|
| Structured prediction | ✓ Fast, accurate | ✗ Unreliable on tabular data | ✓ ML handles prediction |
| Plain-English reasoning | ✗ None | ✓ Natural language | ✓ LLM called when it adds value |
| Cost control | ✓ Cheap at scale | ✗ Expensive at scale | ✓ LLM skipped on confident calls |
| Drift detection | ✗ Manual | ✗ None | ✓ Agent monitors automatically |
| Retraining | ✗ Manual trigger | ✗ None | ✓ Agent fires callback on drift |

### Concrete use cases

The same system adapts to any classification or regression problem by changing `config.yaml`:

- **Fraud detection** — Model flags a transaction; LLM explains which features drove suspicion; retraining agent picks up distribution shift as fraud patterns evolve.
- **Customer churn** — Model scores churn risk; LLM generates a retention message; drift agent triggers retraining when seasonal patterns shift.
- **Credit risk** — Model outputs a risk score; LLM produces an audit-ready explanation; monitoring ensures the model stays calibrated as economic conditions change.
- **Medical triage** — Model ranks case priority; LLM explains clinical reasoning; retraining fires when patient population demographics shift.

The code changes in none of these cases. Only `config.yaml` changes.

---

## Configuration

All tuneable values live in `configs/config.yaml`. To adapt this template to a new domain, only the config needs to change:

| Key | What it controls |
|---|---|
| `ml.model_path` | Where the trained model artifact is saved |
| `ml.n_features` | Number of input features |
| `llm.system_prompt` | The instruction sent to the LLM — make it domain-specific here |
| `llm.label_names` | Human-readable names for each class (e.g. `"1": "churn"`) |
| `llm.feature_names` | Names for each feature position (e.g. `["age", "spend"]`); leave `[]` to use `feature_0`, `feature_1`, ... |
| `monitoring.drift_threshold` | Drift score above which retraining fires |
| `agents.explain_confidence_threshold` | Confidence below which the LLM is called |

---

## Dependencies

| Package | Purpose |
|---|---|
| `scikit-learn` | ML model training and prediction |
| `numpy` | Feature arrays and drift computation |
| `anthropic` | Claude API for LLM reasoning |
| `pyyaml` | Config loading |

---

## Part of AI Platform

This repository is part of a modular AI platform:

- [ds-mlops-enterprise-system](https://github.com/rizkashifs/ds-mlops-enterprise-system) → defines standards and best practices
- [mlops-control-plane](https://github.com/rizkashifs/mlops-control-plane) → manages model lifecycle and governance
- [enterprise-rag-agent-system](https://github.com/rizkashifs/enterprise-rag-agent-system) → GenAI application layer
- [hybrid-ds-genai-agentic-mlops-system](https://github.com/rizkashifs/hybrid-ds-genai-agentic-mlops-system) → ML + LLM + agentic workflows
- [ai-observability-and-drift-platform](https://github.com/rizkashifs/ai-observability-and-drift-platform) → monitoring and reliability
- [multi-model-routing-engine](https://github.com/rizkashifs/multi-model-routing-engine) → model selection and optimization

These repositories together represent an enterprise-grade AI system.
