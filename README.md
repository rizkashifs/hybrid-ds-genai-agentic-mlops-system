# Hybrid DS + GenAI + Agentic MLOps System

A working demonstration of how ML, LLMs, and agents combine into a single unified system — not three separate tools.

---

## The Idea

Most teams build either an ML pipeline **or** an LLM app. This repo shows what it looks like when they work together, with agents connecting the two and automating the operations layer.

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

## Layers

### 1. ML Layer — `src/ml/model.py`
A `LogisticRegression` model trained on your data. Returns a structured prediction:
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

## Why This Matters

| Without this pattern | With this pattern |
|---|---|
| ML model gives a score with no context | Score + plain-English reasoning in one call |
| Drift monitored manually on a schedule | Agent detects drift and acts automatically |
| ML ops and LLM apps maintained separately | One system, one loop, one place to extend |

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
