import numpy as np
from src.monitoring.drift import detect_drift


def test_identical_data_scores_zero():
    data = np.random.default_rng(0).standard_normal((50, 4)).tolist()
    assert detect_drift(data, data) == 0.0


def test_shifted_data_scores_above_zero():
    rng = np.random.default_rng(0)
    baseline = rng.standard_normal((50, 4)).tolist()
    shifted = (rng.standard_normal((50, 4)) + 2.0).tolist()
    assert detect_drift(baseline, shifted) > 0.0


def test_known_shift_computes_correctly():
    # baseline mean = 0, current mean = 1 per feature → drift score = 1.0
    baseline = np.zeros((100, 2)).tolist()
    current = np.ones((100, 2)).tolist()
    assert abs(detect_drift(baseline, current) - 1.0) < 1e-9
