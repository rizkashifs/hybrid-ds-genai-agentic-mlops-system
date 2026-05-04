import numpy as np
from src.core import get_logger

log = get_logger(__name__)


def detect_drift(baseline: list, current: list) -> float:
    """Mean absolute difference in per-feature means between two datasets."""
    b = np.array(baseline)
    c = np.array(current)
    score = float(np.mean(np.abs(b.mean(axis=0) - c.mean(axis=0))))
    log.info("monitoring.drift.computed", extra={"score": round(score, 4)})
    return score
