# api.py
import contextlib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
from pathlib import Path
import pandas as pd

class TransactionPayload(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float; Amount: float

ml_models_registry = {}

# TD4: Clean Lifespan context manager instead of @app.on_event
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Warming up local model binaries into RAM cache...")
    from model_io import ALGORITHMS, STRATEGIES, get_model_path
    for algo in ALGORITHMS:
        ml_models_registry[algo] = {}
        for strategy in STRATEGIES:
            path = get_model_path(algo, strategy)
            if path.exists():
                ml_models_registry[algo][strategy] = joblib.load(path)
    yield
    print("🛑 Draining memory caches and shutting down API routing...")

app = FastAPI(title="Fraud Inference Engine", version="1.1.0", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "Healthy", "models_cached": sum(len(ml_models_registry[a]) for a in ml_models_registry)}

@app.post("/predict")
def predict_transaction(
    payload: TransactionPayload,
    algo: str = Query("Random Forest"),
    strategy: str = Query("None"),
    threshold: float = Query(0.50, ge=0.0, le=1.0) # TD4: Restored user control parameter
):
    if algo not in ml_models_registry or strategy not in ml_models_registry[algo]:
        raise HTTPException(status_code=404, detail="Requested pipeline setup not cached.")
        
    cached = ml_models_registry[algo][strategy]
    
    # Run rapid in-memory scoring without nested network bloat
    input_df = pd.DataFrame([payload.model_dump()])
    fraud_prob = float(cached["model"].predict_proba(input_df)[0, 1])
    
    return {
        "prediction": "Fraud" if fraud_prob >= threshold else "Legit",
        "fraud_probability": fraud_prob,
        "applied_threshold": threshold,
        "model_used": f"{algo} ({strategy})"
    }

@app.get("/models")
def get_leaderboard_matrix():
    leaderboard = []
    for algo in ml_models_registry:
        for strategy, data in ml_models_registry[algo].items():
            leaderboard.append({
                "algorithm": algo,
                "strategy": strategy,
                "f1_score_fraud": data["metrics"]["f1_fraud"],
                "train_rows": data["metrics"]["train_rows"],
                "test_rows": data["metrics"]["test_rows"]
            })
    return sorted(leaderboard, key=lambda x: x["f1_score_fraud"], reverse=True)