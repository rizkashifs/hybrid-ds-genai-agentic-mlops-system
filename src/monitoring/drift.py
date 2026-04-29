import numpy as np


def detect_drift(baseline: list, current: list) -> float:
    """Mean absolute difference in per-feature means between two datasets."""
    b = np.array(baseline)
    c = np.array(current)
    return float(np.mean(np.abs(b.mean(axis=0) - c.mean(axis=0))))
