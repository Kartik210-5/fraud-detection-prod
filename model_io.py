# model_io.py
from pathlib import Path

ALGORITHMS = ["Random Forest", "XGBoost"]
STRATEGIES = ["None", "class_weight", "SMOTE"]
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 28 + 1)] + ["Amount"]
TARGET_COLUMN = "Class"

def get_model_path(algo: str, strategy: str) -> Path:
    clean_algo = algo.lower().replace(" ", "_")
    return Path(__file__).parent / f"fraud_model_{clean_algo}_{strategy}.pkl"