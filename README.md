# Hybrid DS + GenAI + Agentic MLOps System

A working demonstration of how ML, LLMs, and agents combine into a single unified system — not three separate tools.

---

## The Idea

Most teams build either an ML pipeline **or** an LLM app. This repo shows what it looks like when they work together, with agents connecting the two and automating the operations layer.

```
User / System Trigger
        ↓
  OrchestratorAgent
   ├── ML Model  →  prediction + confidence score
   └── LLM       →  plain-English explanation of why
        ↓
  Combined Output
        ↓
  RetrainingAgent  →  monitors drift → triggers retraining automatically
```

---

## Layers

### 1. ML Layer — `src/ml/model.py`
A `LogisticRegression` model trained on synthetic customer data. Returns a structured prediction:
```python
{"label": 1, "probability": 0.93, "features": [0.5, -1.2, 0.8, 1.1]}
```

### 2. LLM Layer — `src/llm/reasoner.py`
Takes the ML prediction and asks Claude to explain it in plain English. Uses prompt caching so repeated calls are cheap.

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

## Example Output

```
=== Step 1: Train ML model ===
Trained LogisticRegression → saved to model.pkl

=== Step 2: Predict + Explain (OrchestratorAgent) ===
Customer features : [0.5, -1.2, 0.8, 1.1]
ML prediction     : Convert (93% confidence)
LLM explanation   : [Mock LLM] Predicted 'convert' with 93% confidence.
                    Set ANTHROPIC_API_KEY to get a real explanation from Claude.

=== Step 3: Drift detection → RetrainingAgent ===
Drift score   : 0.621
Drift detected: True
Action taken  : retraining_triggered
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

## Why This Matters

| Without this pattern | With this pattern |
|---|---|
| ML model gives a score with no context | Score + plain-English reasoning in one call |
| Drift monitored manually on a schedule | Agent detects drift and acts automatically |
| ML ops and LLM apps maintained separately | One system, one loop, one place to extend |

---

## Dependencies

| Package | Purpose |
|---|---|
| `scikit-learn` | ML model training and prediction |
| `numpy` | Feature arrays and drift computation |
| `anthropic` | Claude API for LLM reasoning |
