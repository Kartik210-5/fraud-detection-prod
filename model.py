# model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"

def train_configured_model(df: pd.DataFrame, balancing_method: str) -> dict:
    """
    Trains a Random Forest classifier using the specified dataset balancing strategy.
    Supports: "None", "class_weight", and "SMOTE".
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Configure class weights dynamically based on selection
    if balancing_method == "class_weight":
        class_weight = "balanced"
    else:
        class_weight = None

    # Apply SMOTE exclusively to the training set if selected
    if balancing_method == "SMOTE":
        # k_neighbors=5 is default; adjusted safely for minority extraction
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    # Initialize model with configurations
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(X_train_resampled, y_train_resampled)

    # Generate test set predictions and metrics
    y_pred = model.predict(X_test)
    fraud_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "f1_fraud": float(f1_score(y_test, y_pred, pos_label=1)),
        "report": classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]),
        "train_rows": len(X_train_resampled),
        "test_rows": len(X_test),
        "fraud_rate": float(y_train_resampled.mean() if balancing_method == "SMOTE" else y.mean()),
    }

    return {
        "model": model,
        "metrics": metrics,
        "y_test": y_test.to_numpy(),
        "fraud_proba": fraud_proba,
    }