from src.monitoring.drift import detect_drift


class RetrainingAgent:
    """Monitors for drift and triggers retraining when the threshold is exceeded."""

    def __init__(self, cfg: dict, warn_only: bool = False):
        self.drift_threshold = cfg["monitoring"]["drift_threshold"]
        self.warn_only = warn_only

    def check_and_act(self, baseline_data: list, new_data: list, retrain_fn=None) -> dict:
        drift_score = detect_drift(baseline_data, new_data)
        drifted = drift_score > self.drift_threshold
        action = None

        if drifted:
            if retrain_fn is None and not self.warn_only:
                raise RuntimeError(
                    f"Drift detected (score={drift_score:.3f} > threshold={self.drift_threshold}) "
                    f"but no retrain_fn was supplied. Pass a retrain_fn or set warn_only=True."
                )
            action = "retraining_triggered"
            if retrain_fn:
                retrain_fn()

        return {"drift_score": drift_score, "drifted": drifted, "action": action}
