import numpy as np
import pytest
from src.agents.retraining_agent import RetrainingAgent


def _cfg(threshold=0.1):
    return {"monitoring": {"drift_threshold": threshold}}


def _stable(seed=0):
    return np.random.default_rng(seed).standard_normal((100, 4)).tolist()


def _shifted(amount=5.0):
    return (np.random.default_rng(99).standard_normal((100, 4)) + amount).tolist()


def test_no_drift_returns_false_and_no_action():
    agent = RetrainingAgent(_cfg())
    data = _stable()
    result = agent.check_and_act(baseline_data=data, new_data=data)
    assert result["drifted"] is False
    assert result["action"] is None


def test_drift_with_callback_triggers_retrain():
    called = []
    agent = RetrainingAgent(_cfg())
    result = agent.check_and_act(
        baseline_data=_stable(),
        new_data=_shifted(),
        retrain_fn=lambda: called.append(1),
    )
    assert result["drifted"] is True
    assert result["action"] == "retraining_triggered"
    assert called == [1]


def test_drift_without_callback_raises():
    agent = RetrainingAgent(_cfg())
    with pytest.raises(RuntimeError, match="retrain_fn"):
        agent.check_and_act(baseline_data=_stable(), new_data=_shifted())


def test_warn_only_does_not_raise_or_act():
    agent = RetrainingAgent(_cfg(), warn_only=True)
    result = agent.check_and_act(baseline_data=_stable(), new_data=_shifted())
    assert result["drifted"] is True
    assert result["action"] is None  # detected but not triggered
