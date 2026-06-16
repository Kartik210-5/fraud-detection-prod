# model.py
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE

DATA_PATH = Path(__file__).parent / "creditcard.csv"
TEMP_DATA_PATH = Path(__file__).parent / "temp.csv"

# Individual Model Paths
MODEL_PATHS = {
    "None": Path(__file__).parent / "fraud_model_none.pkl",
    "class_weight": Path(__file__).parent / "fraud_model_class_weight.pkl",
    "SMOTE": Path(__file__).parent / "fraud_model_smote.pkl"
}

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def generate_and_save_temp_sample(df: pd.DataFrame) -> None:
    """
    Samples exactly 1,000 records from the source dataset maintaining an exact 
    distribution constraint: 90% Legitimate (900 rows) and 10% Fraudulent (100 rows).
    Saves the extracted sample to temp.csv.
    """
    # Separate the classes
    legit_df = df[df[TARGET_COLUMN] == 0]
    fraud_df = df[df[TARGET_COLUMN] == 1]
    
    # Safely compute sample targets capped against actual layout availability boundaries
    legit_sample_size = min(900, len(legit_df))
    fraud_sample_size = min(100, len(fraud_df))
    
    # Extract structural random subsets
    legit_sample = legit_df.sample(n=legit_sample_size, random_state=42)
    fraud_sample = fraud_df.sample(n=fraud_sample_size, random_state=42)
    
    # Combine, shuffle, and flush to disk
    temp_sample_df = pd.concat([legit_sample, fraud_sample]).sample(frac=1, random_state=42)
    temp_sample_df.to_csv(TEMP_DATA_PATH, index=False)


def train_single_strategy(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          y_train: pd.Series, y_test: pd.Series, 
                          strategy: str) -> dict:
    """Trains a Random Forest classifier using a specific balancing strategy configuration."""
    class_weight = "balanced" if strategy == "class_weight" else None

    if strategy == "SMOTE":
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test)
    fraud_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "f1_fraud": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "report": classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]),
        "train_rows": len(X_train_resampled),
        "test_rows": len(X_test),
        "fraud_rate": float(y_train_resampled.mean()),
    }

    return {
        "model": model,
        "metrics": metrics,
        "y_test": y_test.to_numpy(),
        "fraud_proba": fraud_proba,
    }


def execute_full_pipeline(df: pd.DataFrame) -> dict:
    """Trains all three balancing methods sequentially and saves them into individual files."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    all_artifacts = {}
    strategies = ["None", "class_weight", "SMOTE"]

    for strategy in strategies:
        artifacts = train_single_strategy(X_train, X_test, y_train, y_test, strategy)
        
        save_package = {
            "model": artifacts["model"],
            "metrics": artifacts["metrics"],
            "y_test": artifacts["y_test"],
            "fraud_proba": artifacts["fraud_proba"]
        }
        joblib.dump(save_package, MODEL_PATHS[strategy])
        all_artifacts[strategy] = artifacts

    return all_artifacts


def load_all_saved_models() -> dict:
    """Loads all three pre-trained pipeline models directly from pkl disk storage."""
    all_artifacts = {}
    for strategy, path in MODEL_PATHS.items():
        if not path.exists():
            return {}
        package = joblib.load(path)
        all_artifacts[strategy] = package
    return all_artifacts


def get_or_train_all_models(cleaned_df: pd.DataFrame = None) -> dict:
    """
    Main orchestration function:
    Checks if CSV file data path asset is present to execute active training. 
    Otherwise fallback loads pre-trained pickle models.
    """
    if DATA_PATH.exists() and cleaned_df is not None:
        # Create and cache the stratified 90/10 temp.csv lookup file
        generate_and_save_temp_sample(cleaned_df)
        
        # Train all 3 versions and save individually
        return execute_full_pipeline(cleaned_df)
    else:
        loaded_models = load_all_saved_models()
        if not loaded_models:
            raise FileNotFoundError("Missing both creditcard_2023.csv and pre-trained pickle models.")
        return loaded_models