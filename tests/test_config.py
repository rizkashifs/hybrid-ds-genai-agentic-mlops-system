import pytest
from pathlib import Path
from src.core import load_config


def test_has_required_top_level_keys():
    cfg = load_config()
    for key in ("ml", "llm", "monitoring", "agents"):
        assert key in cfg, f"Missing top-level key: {key}"


def test_ml_section_has_required_keys():
    cfg = load_config()
    for key in ("model_path", "n_samples", "n_features"):
        assert key in cfg["ml"], f"Missing ml key: {key}"


def test_llm_section_has_required_keys():
    cfg = load_config()
    for key in ("model", "max_tokens", "system_prompt", "label_names", "feature_names"):
        assert key in cfg["llm"], f"Missing llm key: {key}"


def test_monitoring_has_drift_threshold():
    cfg = load_config()
    assert "drift_threshold" in cfg["monitoring"]
    assert isinstance(cfg["monitoring"]["drift_threshold"], float)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))
