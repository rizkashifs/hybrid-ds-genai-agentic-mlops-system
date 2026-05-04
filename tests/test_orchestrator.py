from unittest.mock import patch
from src.agents.orchestrator import OrchestratorAgent
from src.ml.model import train


def _cfg(tmp_path):
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
        "agents": {"explain_confidence_threshold": 0.85},
    }


def test_explain_false_returns_prediction_only(tmp_path):
    cfg = _cfg(tmp_path)
    agent = OrchestratorAgent(train(cfg), cfg)
    result = agent.run([0.5, -1.2, 0.8, 1.1], explain_result=False)
    assert "prediction" in result
    assert "explanation" not in result


def test_explain_true_calls_llm_and_returns_explanation(tmp_path):
    cfg = _cfg(tmp_path)
    agent = OrchestratorAgent(train(cfg), cfg)
    with patch("src.agents.orchestrator.explain", return_value="test explanation") as mock_explain:
        result = agent.run([0.5, -1.2, 0.8, 1.1], explain_result=True)
    assert result["explanation"] == "test explanation"
    mock_explain.assert_called_once()


def test_prediction_has_correct_structure(tmp_path):
    cfg = _cfg(tmp_path)
    agent = OrchestratorAgent(train(cfg), cfg)
    pred = agent.run([0.5, -1.2, 0.8, 1.1], explain_result=False)["prediction"]
    assert set(pred.keys()) == {"label", "probability", "features"}
    assert pred["label"] in (0, 1)
    assert 0.0 <= pred["probability"] <= 1.0
