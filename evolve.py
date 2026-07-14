# evolve.py
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# =====================================================================
# 1. ENVIRONMENT INITIALIZATION
# =====================================================================
# Explicitly load .env file from the root directory
load_dotenv()

import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp
from supabase import create_client, Client

# Import modules from current project layout
from model_io import FEATURE_COLUMNS, TARGET_COLUMN, get_model_path
from model import train_single_pipeline

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evolution_pipeline")

# =====================================================================
# 2. SUPABASE HANDSHAKE
# =====================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("❌ Pipeline terminated: SUPABASE_URL or SUPABASE_KEY missing from environment.")
    sys.exit(1)

try:
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("📡 Successfully established connection to Supabase database.")
except Exception as e:
    logger.error(f"❌ Failed to initialize Supabase client: {e}")
    sys.exit(1)

# Paths
TEMP_DATA_PATH = Path("temp.csv")


# =====================================================================
# 3. HELPER FUNCTIONS: DATA ALIGNMENT & CLEANING
# =====================================================================
def fetch_telemetry_data(limit: int = 1000) -> pd.DataFrame:
    """
    Fetches the latest prediction telemetry records from the Supabase predictions table.
    """
    try:
        res = supabase_client.table("predictions").select("*").order("created_at", desc=True).limit(limit).execute()
        records = res.data
        if not records:
            logger.warning("⚠️ No telemetry records found in the database.")
            return pd.DataFrame()
        return pd.DataFrame(records)
    except Exception as e:
        logger.error(f"❌ Error fetching telemetry from Supabase: {e}")
        return pd.DataFrame()


def clean_and_align_telemetry(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns production telemetry columns and types with the offline baseline requirements.
    Translates database schema 'label' to dataset schema 'Class' and converts strings to 0/1.
    """
    if telemetry_df.empty:
        return pd.DataFrame()

    cleaned_df = pd.DataFrame()

    # 1. Map/Align Features (Handles database lowercase and system casing)
    for col in FEATURE_COLUMNS:
        db_col_name = col.lower()  # e.g., 'amount', 'v1'
        if db_col_name in telemetry_df.columns:
            cleaned_df[col] = telemetry_df[db_col_name].astype(float)
        elif col in telemetry_df.columns:
            cleaned_df[col] = telemetry_df[col].astype(float)
        else:
            # Fallback for missing features to prevent NaN crashes
            cleaned_df[col] = 0.0

    # 2. Map Target Column: DB 'label' -> ML 'Class'
    # DB has 'label' (e.g., "Fraud", "Legit") or 'label_val'
    db_target = "label" if "label" in telemetry_df.columns else "label_val"
    
    if db_target in telemetry_df.columns:
        # Convert "Fraud" / "Legit" string entries into 1 / 0 numerical values
        cleaned_df[TARGET_COLUMN] = telemetry_df[db_target].map({
            "Fraud": 1, 
            "Legit": 0,
            1: 1,
            0: 0,
            1.0: 1,
            0.0: 0
        })
    else:
        # If no target exists, default to NaN so it can be safely pruned
        cleaned_df[TARGET_COLUMN] = np.nan

    # 3. CRITICAL FIX: Drop any row that contains NaN in the target label column
    initial_count = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(subset=[TARGET_COLUMN])
    cleaned_df[TARGET_COLUMN] = cleaned_df[TARGET_COLUMN].astype(int)
    
    dropped_count = initial_count - len(cleaned_df)
    if dropped_count > 0:
        logger.info(f"🧹 Cleaned data: Dropped {dropped_count} rows containing NaN targets.")

    return cleaned_df


# =====================================================================
# 4. EVOLVE LOOP EXECUTION ENGINE
# =====================================================================
def execute_evolution_loop():
    logger.info("🔄 Initializing Closed-Loop Evolution Pipeline...")

    # 1. Fetch baseline data to serve as training and comparison baseline
    if not TEMP_DATA_PATH.exists():
        logger.error(f"❌ Base reference file {TEMP_DATA_PATH} not found. Cannot evaluate drift.")
        return
    
    baseline_df = pd.read_csv(TEMP_DATA_PATH)
    
    # 2. Fetch live telemetry from Supabase
    raw_telemetry = fetch_telemetry_data(limit=1000)
    if raw_telemetry.empty:
        logger.warning("⚠️ Telemetry pool is empty. Skipping evolution execution cycle.")
        return

    # 3. Clean and parse telemetry dataset safely
    telemetry_cleaned = clean_and_align_telemetry(raw_telemetry)
    if telemetry_cleaned.empty or len(telemetry_cleaned) < 10:
        logger.warning("⚠️ Insufficient labeled telemetry data (minimum 10 rows required). Skipped.")
        return

    # 4. Perform Kolmogorov-Smirnov (KS) Drift Analysis on Feature Distributions
    drift_detected = False
    drift_features = []
    
    for col in FEATURE_COLUMNS:
        # Compare baseline distribution vs latest telemetry distribution
        stat, p_val = ks_2samp(baseline_df[col], telemetry_cleaned[col])
        if p_val < 0.05:  # Statistically significant change in distribution
            drift_detected = True
            drift_features.append(col)

    logger.info(f"📊 Drift detection complete. Features with distribution shifts: {drift_features}")
    
    force_drift = os.getenv("FORCE_DRIFT", "false").lower() == "true"
    if drift_detected or force_drift:
        if force_drift:
            logger.info("🚀 Force-Drift Override Triggered! Initiating challenger build...")
        else:
            logger.warning("🚨 Drift breach confirmed! Constructing Challenger Model...")

        # 5. Combine baseline dataset and cleaned telemetry data
        combined_df = pd.concat([baseline_df, telemetry_cleaned], ignore_index=True)
        
        # Ensure target column remains non-null in combined dataset
        combined_df = combined_df.dropna(subset=[TARGET_COLUMN])

        # Prepare matrices
        X = combined_df[FEATURE_COLUMNS]
        y = combined_df[TARGET_COLUMN]

        # 6. Train the Challenger Model (XGBoost + SMOTE)
        algo = "XGBoost"
        strategy = "SMOTE"
        logger.info(f"🏋️ Training Challenger: {algo} ({strategy})...")

        # Create split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        try:
            # Execute standard pipeline training 
            challenger_model, challenger_metrics = train_single_pipeline(
                X_train, X_test, y_train, y_test, algo, strategy
            )
            
            # Save the challenger
            model_out_path = get_model_path(algo, strategy)
            joblib.dump({"model": challenger_model, "metrics": challenger_metrics}, model_out_path)
            
            logger.info(f"🏆 Challenger trained successfully! F1-Score: {challenger_metrics.get('f1_fraud', 0.0):.4f}")
            
            # 7. Write training execution metrics back to Supabase logs
            try:
                record = {
                    "event_type": "model_evolution",
                    "details": f"Challenger trained. Alg: {algo}, Strat: {strategy}, F1: {challenger_metrics.get('f1_fraud', 0.0):.4f}",
                    "metadata": {
                        "drifted_features": drift_features,
                        "metrics": challenger_metrics,
                        "samples_count": len(combined_df)
                    }
                }
                # Optional logs table payload deployment
                supabase_client.table("model_decisions").insert(record).execute()
                logger.info("📝 Log metrics successfully pushed to Supabase model_decisions.")
            except Exception as ex:
                logger.warning(f"⚠️ Log transaction not saved to database: {ex}")

        except Exception as e:
            logger.error(f"❌ Training of challenger pipeline failed: {e}")
            raise e
    else:
        logger.info("🟢 System status: Stable. Incoming transaction data distributions match baseline profiles.")


if __name__ == "__main__":
    execute_evolution_loop()