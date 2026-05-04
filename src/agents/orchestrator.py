import time
from src.ml.model import predict
from src.llm.reasoner import explain
from src.core import get_logger

log = get_logger(__name__)


class OrchestratorAgent:
    """
    Calls ML, then decides whether to call the LLM based on prediction confidence.

    Routing logic (applied when explain_result is not explicitly set):
      - probability < explain_confidence_threshold  →  call LLM (model is uncertain)
      - probability >= explain_confidence_threshold →  skip LLM (model is confident)

    Callers can override routing with explain_result=True (force) or False (skip).
    """

    def __init__(self, ml_model, cfg: dict):
        self.ml_model = ml_model
        self.cfg = cfg
        self._threshold = cfg["agents"]["explain_confidence_threshold"]

    def run(self, features: list, explain_result: bool = None) -> dict:
        t0 = time.perf_counter()
        log.info("agent.orchestrator.run.start", extra={"n_features": len(features)})

        prediction = predict(self.ml_model, features)

        if explain_result is False:
            should_explain = False
            reason = "disabled by caller"
        elif explain_result is True:
            should_explain = True
            reason = "forced by caller"
        else:
            should_explain = prediction["probability"] < self._threshold
            op = "<" if should_explain else ">="
            reason = f"confidence {prediction['probability']:.2f} {op} threshold {self._threshold}"

        result = {
            "prediction": prediction,
            "routing": {"explain_called": should_explain, "reason": reason},
        }
        if should_explain:
            result["explanation"] = explain(prediction, self.cfg)

        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        log.info("agent.orchestrator.run.done", extra={
            "duration_ms": duration_ms,
            "explain_called": should_explain,
            "reason": reason,
        })
        return result
