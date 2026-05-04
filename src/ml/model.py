import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from src.core import load_config


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
    return model


def load(cfg: dict = None) -> LogisticRegression:
    if cfg is None:
        cfg = load_config()
    with open(cfg["ml"]["model_path"], "rb") as f:
        return pickle.load(f)


def predict(model: LogisticRegression, features: list) -> dict:
    X = np.array(features).reshape(1, -1)
    label = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][label])
    return {"label": label, "probability": prob, "features": features}
