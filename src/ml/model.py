import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from src.core import load_config, get_logger

log = get_logger(__name__)


def train(cfg: dict = None) -> LogisticRegression:
    if cfg is None:
        cfg = load_config()
    n_features = cfg["ml"]["n_features"]
    save_path = cfg["ml"]["model_path"]

    X, y = make_classification(
        n_samples=cfg["ml"]["n_samples"],
        n_features=n_features,
        n_informative=max(2, n_features - 2),
        n_redundant=1,
        random_state=42,
    )
    model = LogisticRegression()
    model.fit(X, y)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    log.info("model.trained", extra={"save_path": save_path, "n_samples": cfg["ml"]["n_samples"]})
    return model


def load(cfg: dict = None) -> LogisticRegression:
    if cfg is None:
        cfg = load_config()
    path = cfg["ml"]["model_path"]
    with open(path, "rb") as f:
        model = pickle.load(f)
    log.info("model.loaded", extra={"path": path})
    return model


def predict(model: LogisticRegression, features: list) -> dict:
    X = np.array(features).reshape(1, -1)
    label = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][label])
    result = {"label": label, "probability": prob, "features": features}
    log.info("model.predicted", extra={"label": label, "probability": round(prob, 4)})
    return result
