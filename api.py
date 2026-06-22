# api.py
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Initialize FastAPI engine
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Production-grade REST engine serving Random Forest and XGBoost inference microservices.",
    version="1.0.0"
)

MODEL_DIR = Path(__file__).parent

# Internal structures to hold models and metadata matrices in memory
pipelines_pool = {}
leaderboard_cache = []

@app.on_event("startup")
def load_all_pipelines():
    """
    Scans the directory for trained model packages, registers them into memory,
    and constructs an internal JSON leaderboard ranked by validation F1-Score.
    """
    global pipelines_pool, leaderboard_cache
    
    # Mirroring the matrix structures from model.py
    algorithms = ["Random Forest", "XGBoost"]
    strategies = ["None", "class_weight", "SMOTE"]
    
    raw_leaderboard = []

    for algo in algorithms:
        algo_slug = algo.lower().replace(" ", "_")
        pipelines_pool[algo] = {}
        
        for strategy in strategies:
            filename = f"fraud_model_{algo_slug}_{strategy}.pkl"
            filepath = MODEL_DIR / filename
            
            if filepath.exists():
                try:
                    artifacts = joblib.load(filepath)
                    # Cache the live binary pipeline for inference lookup
                    pipelines_pool[algo][strategy] = artifacts
                    
                    # Extract telemetry metrics for the leaderboard
                    f1 = artifacts["metrics"].get("f1_fraud", 0.0)
                    raw_leaderboard.append({
                        "algorithm": algo,
                        "strategy": strategy,
                        "f1_score_fraud": round(f1, 4),
                        "train_rows": artifacts["metrics"].get("train_rows", "N/A"),
                        "test_rows": artifacts["metrics"].get("test_rows", "N/A")
                    })
                except Exception as e:
                    print(f"⚠️ Failed to load artifact {filename}: {str(e)}")
            else:
                print(f"⚠️ Missing expected artifact file: {filename}")

    # Rank the configs dynamically by F1-Score in descending order
    leaderboard_cache = sorted(raw_leaderboard, key=lambda x: x["f1_score_fraud"], reverse=True)
    print(f"🚀 Loaded {len(raw_leaderboard)} models. Leaderboard ranked successfully.")


# --- DATA CONTRACT SCHEMA ---
class TransactionFeatures(BaseModel):
    V1: float = Field(..., example=-1.359807)
    V2: float = Field(..., example=-0.072781)
    V3: float = Field(..., example=2.536347)
    V4: float = Field(..., example=1.378155)
    V5: float = Field(..., example=-0.338321)
    V6: float = Field(..., example=0.462388)
    V7: float = Field(..., example=0.239599)
    V8: float = Field(..., example=0.098698)
    V9: float = Field(..., example=0.363787)
    V10: float = Field(..., example=0.090794)
    V11: float = Field(..., example=-0.551600)
    V12: float = Field(..., example=-0.617801)
    V13: float = Field(..., example=-0.991390)
    V14: float = Field(..., example=-0.311169)
    V15: float = Field(..., example=1.468177)
    V16: float = Field(..., example=-0.470401)
    V17: float = Field(..., example=0.207971)
    V18: float = Field(..., example=0.025791)
    V19: float = Field(..., example=0.403993)
    V20: float = Field(..., example=0.251412)
    V21: float = Field(..., example=-0.018307)
    V22: float = Field(..., example=0.277838)
    V23: float = Field(..., example=-0.110474)
    V24: float = Field(..., example=0.066928)
    V25: float = Field(..., example=0.128539)
    V26: float = Field(..., example=-0.189115)
    V27: float = Field(..., example=0.133558)
    V28: float = Field(..., example=-0.021053)
    Amount: float = Field(..., example=149.62)


@app.get("/health", tags=["System Health"])
def health_check():
    return {"status": "ok"}


@app.get("/models", tags=["Model Registry"])
def get_leaderboard():
    """
    Returns a live, ranked dashboard configuration matrix of all compiled 
    pipelines matching F1 validation scoring.
    """
    if not leaderboard_cache:
        raise HTTPException(status_code=404, detail="No models found in engine cache registry.")
    return leaderboard_cache


@app.post("/predict", tags=["Inference Engine"])
def predict_transaction(
    payload: TransactionFeatures,
    algo: str = Query(default="Random Forest", description="Target variant: 'Random Forest' or 'XGBoost'"),
    strategy: str = Query(default="SMOTE", description="Target handling strategy: 'None', 'class_weight', or 'SMOTE'")
):
    """
    Processes transaction records using an explicitly selected model-balancing variant combination.
    Defaults to the historical champion configuration (Random Forest + SMOTE).
    """
    # Verify the selected model configuration structure exists in memory
    if algo not in pipelines_pool or strategy not in pipelines_pool[algo]:
        raise HTTPException(
            status_code=400, 
            detail=f"Configuration variant combo '{algo}' with '{strategy}' is unavailable or not pre-trained."
        )

    artifacts = pipelines_pool[algo][strategy]
    trained_model = artifacts["model"]
    decision_threshold = 0.50

    # Format data row payload for inference
    input_data = payload.dict()
    feature_columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    features_df = pd.DataFrame([input_data])[feature_columns]

    # Generate probabilities
    probabilities = trained_model.predict_proba(features_df)[0]
    fraud_probability = float(probabilities[1])
    prediction_label = "Fraud" if fraud_probability >= decision_threshold else "Legit"

    return {
        "prediction": prediction_label,
        "fraud_probability": round(fraud_probability, 4),
        "model_used": f"{algo} ({strategy})",
        "threshold": decision_threshold
    }