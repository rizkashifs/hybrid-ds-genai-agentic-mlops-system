from unittest.mock import patch, MagicMock
from src.agents.orchestrator import OrchestratorAgent
from src.ml.model import train


def _cfg(tmp_path, threshold=0.85):
    return {
        "ml": {"model_path": str(tmp_path / "model.pkl"), "n_samples": 100, "n_features": 4},
        "llm": {
            "model": "claude-sonnet-4-6",
            "max_tokens": 50,
            "system_prompt": "Explain this.",
            "label_names": {"0": "negative", "1": "positive"},
            "feature_names": [],
        },
        "monitoring": {"drift_threshold": 0.1},
        "agents": {"explain_confidence_threshold": threshold},
    }


def _agent(tmp_path, threshold=0.85):
    cfg = _cfg(tmp_path, threshold)
    return OrchestratorAgent(train(cfg), cfg)


def _mock_predict(probability):
    """Return a mock prediction with a controlled probability."""
    return {"label": 1, "probability": probability, "features": [0.0, 0.0, 0.0, 0.0]}


# ── Existing behaviour (unchanged) ───────────────────────────────────────────

def test_explain_false_returns_prediction_only(tmp_path):
    result = _agent(tmp_path).run([0.5, -1.2, 0.8, 1.1], explain_result=False)
    assert "prediction" in result
    assert "explanation" not in result


def test_explain_true_calls_llm_and_returns_explanation(tmp_path):
    with patch("src.agents.orchestrator.explain", return_value="test explanation") as mock_explain:
        result = _agent(tmp_path).run([0.5, -1.2, 0.8, 1.1], explain_result=True)
    assert result["explanation"] == "test explanation"
    mock_explain.assert_called_once()


def test_prediction_has_correct_structure(tmp_path):
    pred = _agent(tmp_path).run([0.5, -1.2, 0.8, 1.1], explain_result=False)["prediction"]
    assert set(pred.keys()) == {"label", "probability", "features"}
    assert pred["label"] in (0, 1)
    assert 0.0 <= pred["probability"] <= 1.0


# ── Routing key always present ────────────────────────────────────────────────

def test_routing_key_always_present(tmp_path):
    result = _agent(tmp_path).run([0.5, -1.2, 0.8, 1.1], explain_result=False)
    assert "routing" in result
    assert "explain_called" in result["routing"]
    assert "reason" in result["routing"]


# ── Confidence-based routing ──────────────────────────────────────────────────

def test_high_confidence_skips_llm(tmp_path):
    agent = _agent(tmp_path, threshold=0.85)
    with patch("src.agents.orchestrator.predict", return_value=_mock_predict(0.95)):
        with patch("src.agents.orchestrator.explain") as mock_explain:
            result = agent.run([0.0, 0.0, 0.0, 0.0])
    assert result["routing"]["explain_called"] is False
    assert "explanation" not in result
    mock_explain.assert_not_called()


def test_low_confidence_calls_llm(tmp_path):
    agent = _agent(tmp_path, threshold=0.85)
    with patch("src.agents.orchestrator.predict", return_value=_mock_predict(0.60)):
        with patch("src.agents.orchestrator.explain", return_value="uncertain — here's why") as mock_explain:
            result = agent.run([0.0, 0.0, 0.0, 0.0])
    assert result["routing"]["explain_called"] is True
    assert result["explanation"] == "uncertain — here's why"
    mock_explain.assert_called_once()


def test_force_explain_overrides_high_confidence(tmp_path):
    agent = _agent(tmp_path, threshold=0.85)
    with patch("src.agents.orchestrator.predict", return_value=_mock_predict(0.99)):
        with patch("src.agents.orchestrator.explain", return_value="forced") as mock_explain:
            result = agent.run([0.0, 0.0, 0.0, 0.0], explain_result=True)
    assert result["routing"]["explain_called"] is True
    assert result["routing"]["reason"] == "forced by caller"
    mock_explain.assert_called_once()


def test_routing_reason_contains_threshold(tmp_path):
    agent = _agent(tmp_path, threshold=0.85)
    with patch("src.agents.orchestrator.predict", return_value=_mock_predict(0.95)):
        result = agent.run([0.0, 0.0, 0.0, 0.0])
    assert "0.85" in result["routing"]["reason"]
