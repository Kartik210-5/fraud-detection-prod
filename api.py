# api.py
import contextlib
import json
import logging
import time
import uuid
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
import joblib
from pathlib import Path
import pandas as pd

# =====================================================================
# TOUCHDOWN 1: STRUCTURED LOGGING & JSON FORMATTING CONFIGURATION
# =====================================================================
class JsonFormatter(logging.Formatter):
    """
    Custom formatter to convert Python log records into structured, 
    single-line JSON strings optimal for cloud log parsers like Railway.
    """
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            # Fall back to "SYSTEM" if the log is emitted outside an HTTP request lifecycle
            "request_id": getattr(record, "request_id", "SYSTEM")
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

# Initialize standard library logging channel
logger = logging.getLogger("fraud_pipeline")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class TransactionPayload(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float; V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount: float

ml_models_registry = {}

# Clean Lifespan context manager now using structured system logging
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("⏳ Warming up local model binaries into RAM cache...")
    from model_io import ALGORITHMS, STRATEGIES, get_model_path
    for algo in ALGORITHMS:
        ml_models_registry[algo] = {}
        for strategy in STRATEGIES:
            path = get_model_path(algo, strategy)
            if path.exists():
                try:
                    ml_models_registry[algo][strategy] = joblib.load(path)
                except Exception as e:
                    logger.error(f"⚠️ Failed to load model file at {path}: {e}")
    yield
    logger.info("🛑 Draining memory caches and shutting down API routing...")


app = FastAPI(title="Fraud Inference Engine", version="1.1.0", lifespan=lifespan)

# =====================================================================
# REQUEST LIFECYCLE MIDDLEWARE (CORRELATION ID ASSIGNMENT)
# =====================================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Check if upstream already passed a tracking ID; if not, spin up a new UUIDv4
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    
    # Cache the tracking ID inside the state context so downstream endpoint logic can read it
    request.state.request_id = request_id
    
    extra = {"request_id": request_id}
    logger.info(f"Incoming request: {request.method} {request.url.path}", extra=extra)
    
    start_time = time.time()
    response = await call_next(request)
    latency = (time.time() - start_time) * 1000
    
    logger.info(f"Completed request with status {response.status_code} in {latency:.2f}ms", extra=extra)
    
    # Return it to the client via response headers for quick end-to-end debugging trace loops
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/")
def read_root():
    return {
        "message": "Credit Card Fraud Detection API is Live & Operational!",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health")
def health_check():
    count = sum(len(ml_models_registry[a]) for a in ml_models_registry)
    return {"status": "Healthy", "models_cached_count": count}


@app.post("/predict")
def predict_transaction(
    payload: TransactionPayload,
    request: Request, # Explicitly pass the request object to parse state values
    algo: str = Query("Random Forest"),
    strategy: str = Query("None"),
    threshold: float = Query(0.50, ge=0.0, le=1.0)
):
    # Extract the dynamic tracking ID assigned by the HTTP middleware layer
    req_id = getattr(request.state, "request_id", "UNKNOWN")
    extra = {"request_id": req_id}

    if algo not in ml_models_registry or strategy not in ml_models_registry[algo]:
        logger.warning(f"Prediction aborted: Setup '{algo} ({strategy})' not found in RAM cache.", extra=extra)
        raise HTTPException(status_code=404, detail=f"Requested setup '{algo} ({strategy})' not cached.")
        
    cached = ml_models_registry[algo][strategy]
    input_df = pd.DataFrame([payload.model_dump()])
    model_obj = cached["model"] if isinstance(cached, dict) and "model" in cached else cached
    
    try:
        fraud_prob = float(model_obj.predict_proba(input_df)[0, 1])
    except Exception as e:
        logger.error(f"Scoring engine encountered a execution critical crash: {str(e)}", extra=extra)
        raise HTTPException(status_code=500, detail=f"Scoring engine error: {str(e)}")
    
    prediction_label = "Fraud" if fraud_prob >= threshold else "Legit"
    
    # Log the complete prediction results out securely in structured format
    logger.info(
        f"Inference complete | Model: {algo} ({strategy}) | Amount: {payload.Amount} | "
        f"Prob: {fraud_prob:.4f} | Label: {prediction_label}", 
        extra=extra
    )
    
    return {
        "prediction": prediction_label,
        "fraud_probability": fraud_prob,
        "applied_threshold": threshold,
        "model_used": f"{algo} ({strategy})"
    }


@app.get("/models")
def get_leaderboard_matrix():
    leaderboard = []
    for algo in ml_models_registry:
        for strategy, data in ml_models_registry[algo].items():
            metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
            leaderboard.append({
                "algorithm": algo,
                "strategy": strategy,
                "f1_score_fraud": metrics.get("f1_fraud", metrics.get("f1_score_fraud", 0.0)),
                "train_rows": metrics.get("train_rows", "N/A"),
                "test_rows": metrics.get("test_rows", "N/A")
            })
    return sorted(leaderboard, key=lambda x: x.get("f1_score_fraud", 0.0), reverse=True)