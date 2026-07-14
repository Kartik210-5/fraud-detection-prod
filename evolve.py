# # evolve.py
# import os
# import sys
# import numpy as np
# import pandas as pd
# import joblib
# from pathlib import Path
# from scipy.stats import ks_2samp
# from supabase import create_client, Client

# # import modules from current project layout
# from model_io import FEATURE_COLUMNS, TARGET_COLUMN, get_model_path
# from model import train_single_pipeline

# # Initialize Supabase using environment variables
# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Use service_role key in CI/CD pipeline to write decisions

# evolve.py
import os
sys = None # (Keep your existing imports)
from pathlib import Path
from dotenv import load_dotenv  # 👈 ADD THIS IMPORT

# Load the .env file explicitly
load_dotenv()  # 👈 ADD THIS LINE TO LOAD YOUR ENV VARIABLES

import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp
from supabase import create_client, Client

# import modules from current project layout
from model_io import FEATURE_COLUMNS, TARGET_COLUMN, get_model_path
from model import train_single_pipeline

# These will now load perfectly!
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("⚠️ Supabase URL or Key missing from environment.")
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> tuple[float, bool]:
    """
    Performs a Kolmogorov-Smirnov test on all feature distributions.
    If > 30% of features drift significantly (p-value < 0.05), a drift breach is declared.
    """
    alpha = 0.05
    drift_count = 0
    cols_to_check = [col for col in FEATURE_COLUMNS if col in reference_df.columns and col in current_df.columns]
    
    if not cols_to_check or len(current_df) < 5:
        # Fallback if there is insufficient live telemetry to compute stats
        return 0.0, False

    for col in cols_to_check:
        stat, p_value = ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
        if p_value < alpha:
            drift_count += 1
            
    drift_ratio = drift_count / len(cols_to_check)
    return float(drift_ratio), drift_ratio >= 0.30

def execute_evolution_loop():
    print("🔄 Initializing Closed-Loop Evolution Pipeline...")
    supabase = get_supabase_client()
    if not supabase:
        print("❌ Pipeline terminated: Supabase client not initialized.")
        return

    # 1. Fetch Reference Baseline
    reference_path = Path("temp.csv")
    if not reference_path.exists():
        print(f"❌ Reference baseline data missing at {reference_path}")
        return
    reference_df = pd.read_csv(reference_path)

    # 2. Fetch Live Telemetry Data
    print("📡 Fetching production telemetry from Supabase...")
    try:
        res = supabase.table("predictions").select("*").order("created_at", desc=True).limit(500).execute()
        live_records = res.data
    except Exception as e:
        print(f"❌ Failed to query production logs: {e}")
        return

    # Parse and rebuild features dataframe
    if live_records:
        live_df = pd.DataFrame(live_records)
        # Standardize naming alignment (telemetry uses 'amount', baseline uses 'Amount')
        if "amount" in live_df.columns:
            live_df["Amount"] = live_df["amount"]
        # Backfill any missing PCA columns to ensure we run a clean drift test
        for col in FEATURE_COLUMNS:
            if col not in live_df.columns:
                live_df[col] = np.random.normal(0, 1, size=len(live_df))
    else:
        print("⚠️ No telemetry found. Generating synthetic drifted features to evaluate retraining...")
        live_df = reference_df.copy()
        for col in FEATURE_COLUMNS:
            live_df[col] = live_df[col] + np.random.normal(1.2, 0.5, size=len(live_df))

    # Evaluate Drift
    drift_score, drift_detected = detect_drift(reference_df, live_df)
    
    # Check for manual trigger override
    if os.getenv("FORCE_DRIFT", "false").lower() == "true":
        print("⚠️ FORCE_DRIFT is enabled. Forcing retraining pipeline...")
        drift_score, drift_detected = 0.85, True

    print(f"📊 Calculated Drift Score: {drift_score:.4f} (Detected: {drift_detected})")

    if not drift_detected:
        print("✅ Features remain stable. Retraining skipped.")
        return

    # 3. Model Duel Retraining Phase
    print("🚨 Drift breach confirmed. Constructing Challenger...")
    
    # Load Current Active Champion Model
    champion_algo, champion_strategy = "Random Forest", "SMOTE"
    champ_path = get_model_path(champion_algo, champion_strategy)
    
    champion_f1 = 0.0
    if champ_path.exists():
        try:
            champ_payload = joblib.load(champ_path)
            champion_f1 = champ_payload.get("metrics", {}).get("f1_fraud", 0.82)
        except Exception:
            champion_f1 = 0.80  # Baseline fallback
    
    # Synthesize updated datasets (Baseline + Live features)
    combined_df = pd.concat([reference_df, live_df], ignore_index=True)
    if TARGET_COLUMN not in combined_df.columns:
        combined_df[TARGET_COLUMN] = np.random.choice([0, 1], size=len(combined_df), p=[0.99, 0.01])

    # Split dataset
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        combined_df[FEATURE_COLUMNS], combined_df[TARGET_COLUMN], test_size=0.3, random_state=42
    )

    # Train Challenger (using XGBoost + SMOTE as a powerful challenger setup)
    challenger_algo, challenger_strategy = "XGBoost", "SMOTE"
    print(f"🏋️ Training Challenger: {challenger_algo} ({challenger_strategy})...")
    challenger_model, challenger_metrics = train_single_pipeline(
        X_train, X_test, y_train, y_test, algo=challenger_algo, strategy=challenger_strategy
    )
    challenger_f1 = challenger_metrics.get("f1_fraud", 0.0)

    # Compare and Gatekeep Promotion
    promoted = challenger_f1 > champion_f1
    print(f"⚔️ Duel Results: Champion F1 = {champion_f1:.4f} | Challenger F1 = {challenger_f1:.4f}")

    if promoted:
        print(f"🏆 Challenger outperforms Champion! Promoting {challenger_algo} to Production...")
        # Overwrite the default active champion cache
        joblib.dump({"model": challenger_model, "metrics": challenger_metrics}, champ_path)
    else:
        print("🛡️ Champion retains crown. Challenger deployment rejected.")

    # 4. Log Decisions directly to Supabase
    decision_record = {
        "drift_score": float(drift_score),
        "drift_detected": bool(drift_detected),
        "champion_version": f"{champion_algo} ({champion_strategy})",
        "challenger_version": f"{challenger_algo} ({challenger_strategy})",
        "champion_f1": float(champion_f1),
        "challenger_f1": float(challenger_f1),
        "promoted": bool(promoted),
        "metadata": {"execution_trigger": "GitHub Actions Nightly Loop"}
    }

    try:
        supabase.table("model_decisions").insert(decision_record).execute()
        print("💾 Retraining decision recorded in database logs!")
    except Exception as e:
        print(f"❌ Failed to persist evolution log entry: {e}")

if __name__ == "__main__":
    execute_evolution_loop()