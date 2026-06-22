# model.py (Updated functions)
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path(__file__).parent / "creditcard.csv"
TEMP_DATA_PATH = Path(__file__).parent / "temp.csv"

ALGORITHMS = ["Random Forest", "XGBoost"]
STRATEGIES = ["None", "class_weight", "SMOTE"]
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"

def get_model_path(algo: str, strategy: str) -> Path:
    clean_algo = algo.lower().replace(" ", "_")
    return Path(__file__).parent / f"fraud_model_{clean_algo}_{strategy}.pkl"

def train_single_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          y_train: pd.Series, y_test: pd.Series, 
                          algo: str, strategy: str) -> dict:
    """Trains a chosen algorithm with a specific balancing strategy and collects raw evaluation metrics."""
    # 1. Handle Data Balancing (SMOTE)
    if strategy == "SMOTE":
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    # Pinned Hyper-parameters
    rf_n_estimators, rf_max_depth = 100, 8
    xgb_n_estimators, xgb_max_depth, xgb_lr = 100, 6, 0.1
    random_state = 42

    # 2. Instantiate Classifiers
    if algo == "Random Forest":
        class_weight = "balanced" if strategy == "class_weight" else None
        model = RandomForestClassifier(
            n_estimators=rf_n_estimators, max_depth=rf_max_depth, 
            class_weight=class_weight, random_state=random_state, n_jobs=-1
        )
        params_to_log = {
            "n_estimators": rf_n_estimators,
            "max_depth": rf_max_depth,
            "random_state": random_state,
            "class_weight": str(class_weight)
        }
    elif algo == "XGBoost":
        scale_pos_weight = None
        if strategy == "class_weight":
            num_neg = (y_train_resampled == 0).sum()
            num_pos = (y_train_resampled == 1).sum()
            scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
            
        model = XGBClassifier(
            n_estimators=xgb_n_estimators, max_depth=xgb_max_depth, scale_pos_weight=scale_pos_weight,
            learning_rate=xgb_lr, random_state=random_state, n_jobs=-1, eval_metric="logloss"
        )
        params_to_log = {
            "n_estimators": xgb_n_estimators,
            "max_depth": xgb_max_depth,
            "learning_rate": xgb_lr,
            "random_state": random_state,
            "scale_pos_weight": str(scale_pos_weight)
        }

    # 3. Fit Model
    model.fit(X_train_resampled, y_train_resampled)

    # 4. Gather Metrics
    y_pred = model.predict(X_test)
    fraud_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "f1_fraud": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "precision_fraud": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall_fraud": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "report": classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]),
        "train_rows": len(X_train_resampled),
        "test_rows": len(X_test),
        "fraud_rate": float(y_train_resampled.mean()),
    }

    return {
        "model": model,
        "metrics": metrics,
        "params": params_to_log,
        "y_test": y_test.to_numpy(),
        "fraud_proba": fraud_proba,
    }

def execute_full_pipeline(df: pd.DataFrame) -> dict:
    """Loops through all algorithms and strategies, trains them, and logs telemetry to MLflow."""
    # Ensure tracking points to local background server
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Credit_Card_Fraud_Production")
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    all_artifacts = {}

    for algo in ALGORITHMS:
        all_artifacts[algo] = {}
        for strategy in STRATEGIES:
            run_name = f"{algo.replace(' ', '_')}_{strategy}"
            
            # 🚀 OPEN MLFLOW RUN FOR EXPERIMENT METRIC LOGGING
            with mlflow.start_run(run_name=run_name):
                # Train model single sequence
                artifacts = train_single_pipeline(X_train, X_test, y_train, y_test, algo, strategy)
                
                # A. Log Core Parameter Schemas
                mlflow.log_param("algorithm", algo)
                mlflow.log_param("balancing_strategy", strategy)
                for param_name, param_val in artifacts["params"].items():
                    mlflow.log_param(param_name, param_val)
                
                # B. Log Evaluation Metrics Metrics
                mlflow.log_metric("f1_score_fraud", artifacts["metrics"]["f1_fraud"])
                mlflow.log_metric("precision_fraud", artifacts["metrics"]["precision_fraud"])
                mlflow.log_metric("recall_fraud", artifacts["metrics"]["recall_fraud"])
                mlflow.log_metric("train_rows", artifacts["metrics"]["train_rows"])
                
                # C. Log the MLmodel artifact weights directly into the MLflow system
                mlflow.sklearn.log_model(artifacts["model"], "model_binary")
                
                # Maintain compatibility with your local cache fallback architecture
                save_package = {
                    "model": artifacts["model"],
                    "metrics": artifacts["metrics"],
                    "y_test": artifacts["y_test"],
                    "fraud_proba": artifacts["fraud_proba"]
                }
                joblib.dump(save_package, get_model_path(algo, strategy))
                
                all_artifacts[algo][strategy] = artifacts
                print(f"🎉 Logged Run: {run_name} -> F1: {artifacts['metrics']['f1_fraud']:.4f}")

    return all_artifacts