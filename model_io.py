# model_io.py
from pathlib import Path

ALGORITHMS = ["Random Forest", "XGBoost"]
STRATEGIES = ["None", "class_weight", "SMOTE"]
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 28 + 1)] + ["Amount"]
TARGET_COLUMN = "Class"

def get_model_path(algo: str, strategy: str) -> Path:
    # Match the exact file combinations in your repo:
    # fraud_model_random_forest_None.pkl, fraud_model_xgboost_SMOTE.pkl, etc.
    if algo == "Random Forest":
        algo_name = "random_forest"
    else:
        algo_name = "xgboost"
        
    return Path(__file__).parent / f"fraud_model_{algo_name}_{strategy}.pkl"