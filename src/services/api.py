"""
FastAPI serving layer for the hybrid ML + LLM + Agent system.

Endpoints:
  GET  /health       — liveness check, reports whether model is loaded
  POST /predict      — run OrchestratorAgent on a feature vector
  POST /drift-check  — run RetrainingAgent in monitoring-only mode

The ML model is loaded once at startup via the lifespan context manager.
Start with: uvicorn src.services.api:app --reload
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.core import load_config, get_logger
from src.ml.model import train, load
from src.agents.orchestrator import OrchestratorAgent
from src.agents.retraining_agent import RetrainingAgent

log = get_logger(__name__)


# ── Request / Response models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: list[float]
    # None = auto-route by confidence, True = force LLM, False = skip LLM
    force_explain: Optional[bool] = None


class PredictionOut(BaseModel):
    label: int
    probability: float
    features: list[float]


class RoutingInfo(BaseModel):
    explain_called: bool
    reason: str


class PredictResponse(BaseModel):
    prediction: PredictionOut
    routing: RoutingInfo
    explanation: Optional[str] = None


class DriftCheckRequest(BaseModel):
    baseline: list[list[float]]
    new_data: list[list[float]]


class DriftCheckResponse(BaseModel):
    drift_score: float
    drifted: bool
    action: Optional[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    try:
        model = load(cfg)
        log.info("api.startup.model_loaded", extra={"path": cfg["ml"]["model_path"]})
    except FileNotFoundError:
        log.info("api.startup.model_not_found_training")
        model = train(cfg)

    app.state.agent = OrchestratorAgent(model, cfg)
    app.state.retraining_agent = RetrainingAgent(cfg, warn_only=True)
    app.state.model_loaded = True
    yield


app = FastAPI(title="Hybrid ML + LLM + Agent API", lifespan=lifespan)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model_loaded": app.state.model_loaded}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return app.state.agent.run(req.features, explain_result=req.force_explain)


@app.post("/drift-check", response_model=DriftCheckResponse)
def drift_check(req: DriftCheckRequest):
    # warn_only=True — API layer detects drift but does not trigger retraining
    return app.state.retraining_agent.check_and_act(
        baseline_data=req.baseline,
        new_data=req.new_data,
    )
