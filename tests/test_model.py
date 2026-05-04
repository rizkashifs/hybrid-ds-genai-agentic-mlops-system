import pytest
from src.ml.model import train, load, predict


def _cfg(tmp_path):
    return {"ml": {"model_path": str(tmp_path / "model.pkl"), "n_samples": 100, "n_features": 4}}


def test_train_creates_artifact(tmp_path):
    train(_cfg(tmp_path))
    assert (tmp_path / "model.pkl").exists()


def test_train_load_round_trip(tmp_path):
    cfg = _cfg(tmp_path)
    features = [0.5, -1.2, 0.8, 1.1]
    original = train(cfg)
    reloaded = load(cfg)
    assert predict(original, features) == predict(reloaded, features)


def test_predict_output_structure(tmp_path):
    model = train(_cfg(tmp_path))
    result = predict(model, [0.5, -1.2, 0.8, 1.1])
    assert set(result.keys()) == {"label", "probability", "features"}
    assert result["label"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["features"] == [0.5, -1.2, 0.8, 1.1]


def test_predict_wrong_feature_count_raises(tmp_path):
    model = train(_cfg(tmp_path))
    with pytest.raises(Exception):
        predict(model, [0.5, -1.2])  # model expects 4 features
