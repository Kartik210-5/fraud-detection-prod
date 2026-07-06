# api.py
import contextlib
import json
import logging
import os
import time
import uuid
from fastapi import FastAPI, HTTPException, Query, Request, Depends, status, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import joblib
from pathlib import Path
import pandas as pd
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from supabase import create_client, Client

# =====================================================================
# TOUCHDOWN 1: STRUCTURED LOGGING CONFIGURATION
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


# =====================================================================
# TOUCHDOWN 1: API KEY SECURITY SCHEME & DEPENDENCY
# =====================================================================
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(header_value: str = Depends(api_key_header)):
    """
    Dependency to validate incoming requests against a secure environment token.
    Defaults to 'dev-secret-key-123' if no environment variable is set locally.
    """
    expected_key = os.getenv("INFERENCE_API_KEY", "dev-secret-key-123")
    
    if not header_value:
        logger.warning("Access Denied: Request missing X-API-Key identification header.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key validation header."
        )
        
    if header_value != expected_key:
        logger.warning("Access Denied: Unauthorized or mismatched key token supplied.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unauthorized API key."
        )
        
    return header_value


# =====================================================================
# DATABASE MANAGEMENT ENGINE (RESTORED DATABASE CONNECTIVITY)
# =====================================================================
supabase_client: Client = None

def init_supabase():
    global supabase_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if url and key:
        try:
            supabase_client = create_client(url, key)
            logger.info("📡 Successfully established a persistent channel to the Supabase Database.")
        except Exception as e:
            logger.error(f"⚠️ Initial connection to Supabase failed: {e}")
    else:
        logger.warning("⚠️ Database tracking disabled: SUPABASE_URL or SUPABASE_KEY missing from environment.")

def persist_prediction_task(payload_data: dict, latency: float, prob: float, label: str, algo: str, strategy: str, threshold: float):
    if not supabase_client:
        return
    try:
        record = {
            "algo": algo,
            "strategy": strategy,
            "amount": float(payload_data.get("Amount", 0.0)),
            "fraud_probability": prob,
            "label": label,
            "latency_ms": latency,
            "threshold": threshold  # Added this key to satisfy your Supabase schema!
        }
        supabase_client.table("predictions").insert(record).execute()
        print("🚀 Successfully pushed row to Supabase!")
    except Exception as e:
        print(f"❌ DATABASE INSERTION EXCEPTION ERROR: {str(e)}")
        logger.error(f"❌ Asynchronous Telemetry Write Failed: {e}")


# =====================================================================
# CORE APPLICATION STATE & SCHEMAS
# =====================================================================
class TransactionPayload(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float; V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount: float

ml_models_registry = {}

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("⏳ Warming up local model binaries into RAM cache...")
    init_supabase()  # Initialize connection channel on app startup
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


# =====================================================================
# TOUCHDOWN 2: RATE LIMITING CONFIGURATION
# =====================================================================
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Fraud Inference Engine", version="1.2.0", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =====================================================================
# REQUEST LIFECYCLE MIDDLEWARE (CORRELATION ID ASSIGNMENT)
# =====================================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    
    extra = {"request_id": request_id}
    logger.info(f"Incoming request: {request.method} {request.url.path}", extra=extra)
    
    start_time = time.time()
    response = await call_next(request)
    latency = (time.time() - start_time) * 1000
    
    logger.info(f"Completed request with status {response.status_code} in {latency:.2f}ms", extra=extra)
    response.headers["X-Request-ID"] = request_id
    return response


# =====================================================================
# CORE API ROUTES
# =====================================================================
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


# =====================================================================
# TOUCHDOWN 1 & 2: SECURED AND RATE-LIMITED INFERENCE ROUTE
# =====================================================================
@app.post("/predict", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
def predict_transaction(
    payload: TransactionPayload,
    request: Request, 
    background_tasks: BackgroundTasks, # Injected back to handle async db ingestion threads
    algo: str = Query("Random Forest"),
    strategy: str = Query("None"),
    threshold: float = Query(0.50, ge=0.0, le=1.0)
):
    start_process_time = time.time()
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
        logger.error(f"Scoring engine encountered an execution critical crash: {str(e)}", extra=extra)
        raise HTTPException(status_code=500, detail=f"Scoring engine error: {str(e)}")
    
    prediction_label = "Fraud" if fraud_prob >= threshold else "Legit"
    
    # Calculate real computational execution latency profile block
    elapsed_latency_ms = (time.time() - start_process_time) * 1000
    
    logger.info(
        f"Inference complete | Model: {algo} ({strategy}) | Amount: {payload.Amount} | "
        f"Prob: {fraud_prob:.4f} | Label: {prediction_label}", 
        extra=extra
    )
    
    # DISPATCH PERSISTENCE LOG ASYNC TASK (Fills database row for Streamlit UI tracking)
    # DISPATCH PERSISTENCE LOG ASYNC TASK
    background_tasks.add_task(
        persist_prediction_task,
        payload_data=payload.model_dump(),
        latency=elapsed_latency_ms,
        prob=fraud_prob,
        label=prediction_label,
        algo=algo,
        strategy=strategy,
        threshold=threshold  # Injected right here
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