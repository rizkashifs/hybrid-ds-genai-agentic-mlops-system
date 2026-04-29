from src.ml.model import predict
from src.llm.reasoner import explain


class OrchestratorAgent:
    """Decides to call ML, LLM, or both, then combines the result."""

    def __init__(self, ml_model):
        self.ml_model = ml_model

    def run(self, features: list, explain_result: bool = True) -> dict:
        prediction = predict(self.ml_model, features)
        result = {"prediction": prediction}
        if explain_result:
            result["explanation"] = explain(prediction)
        return result
