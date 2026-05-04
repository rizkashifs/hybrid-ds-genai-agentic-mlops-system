import time
from src.ml.model import predict
from src.llm.reasoner import explain
from src.core import get_logger

log = get_logger(__name__)


class OrchestratorAgent:
    """Decides to call ML, LLM, or both, then combines the result."""

    def __init__(self, ml_model, cfg: dict):
        self.ml_model = ml_model
        self.cfg = cfg

    def run(self, features: list, explain_result: bool = True) -> dict:
        t0 = time.perf_counter()
        log.info("agent.orchestrator.run.start", extra={"n_features": len(features)})

        prediction = predict(self.ml_model, features)
        result = {"prediction": prediction}
        if explain_result:
            result["explanation"] = explain(prediction, self.cfg)

        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        log.info("agent.orchestrator.run.done", extra={
            "duration_ms": duration_ms,
            "explain_called": explain_result,
        })
        return result
