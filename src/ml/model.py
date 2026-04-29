import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

MODEL_PATH = "model.pkl"


def train(save_path=MODEL_PATH) -> LogisticRegression:
    X, y = make_classification(n_samples=500, n_features=4, n_informative=2, n_redundant=1, random_state=42)
    model = LogisticRegression()
    model.fit(X, y)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    return model


def load(path=MODEL_PATH) -> LogisticRegression:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(model: LogisticRegression, features: list) -> dict:
    X = np.array(features).reshape(1, -1)
    label = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][label])
    return {"label": label, "probability": prob, "features": features}
