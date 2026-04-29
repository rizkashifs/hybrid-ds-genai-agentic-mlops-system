from src.monitoring.drift import detect_drift


class RetrainingAgent:
    """Monitors for drift and triggers retraining when the threshold is exceeded."""

    def __init__(self, drift_threshold: float = 0.1):
        self.drift_threshold = drift_threshold

    def check_and_act(self, baseline_data: list, new_data: list, retrain_fn=None) -> dict:
        drift_score = detect_drift(baseline_data, new_data)
        drifted = drift_score > self.drift_threshold
        action = None
        if drifted:
            action = "retraining_triggered"
            if retrain_fn:
                retrain_fn()
        return {"drift_score": drift_score, "drifted": drifted, "action": action}
