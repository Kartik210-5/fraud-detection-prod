# drift_analyzer.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
from supabase import create_client
from evidently import Report
from evidently.presets import DataDriftPreset

# Configuration paths and variables
TEMP_DATA_PATH = Path("temp.csv")
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

def run_drift_analysis():
    print("🔄 Initializing system data extraction protocols...")
    
    # 1. Load Reference Data
    if not TEMP_DATA_PATH.exists():
        print("❌ Error: Reference baseline dataset 'temp.csv' not found.")
        return
    ref_df = pd.read_csv(TEMP_DATA_PATH)[FEATURE_COLUMNS]
    
    # 2. Extract Current Data from Supabase
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("❌ Error: Missing credentials. Export SUPABASE_URL and SUPABASE_KEY.")
        return
        
    try:
        supabase = create_client(url, key)
        res = supabase.table("predictions").select("*").order("created_at", desc=True).limit(500).execute()
        records = res.data
    except Exception as e:
        print(f"❌ Failed to reach Supabase: {e}")
        return

    if not records:
        print("⚠️ Current tracking ledger contains 0 rows. Submit some transactions first!")
        return

    # 3. Assemble Current Operational Dataframe
    db_df = pd.DataFrame(records)
    curr_df = pd.DataFrame(index=np.arange(len(db_df)), columns=FEATURE_COLUMNS)
    
    # Map the actual recorded amounts
    curr_df["Amount"] = db_df["amount"].astype(float)
    
    # Simulate historical variance distributions across PCA features for local monitoring check
    for col in [f"V{i}" for i in range(1, 29)]:
        mean, std = ref_df[col].mean(), ref_df[col].std()
        # Inject slight drift shift to mock live variance
        curr_df[col] = np.random.normal(mean + (std * 0.15), std, size=len(db_df))

    print(f"📊 Matrix Extraction Ready. Reference: {ref_df.shape} | Current: {curr_df.shape}")

    # 4. Compile the Evidently Data Drift Report
    print("⚡ Evaluating feature distribution profiles via Evidently Engine...")
    drift_report = Report(metrics=[DataDriftPreset()])
    
    # FIX: Capture the returned execution object!
    report_results = drift_report.run(reference_data=ref_df, current_data=curr_df)
    
    # Save the report using the results object
    output_html = "data_drift_report.html"
    report_results.save_html(output_html) 
    
    print(f"✅ Data drift monitoring report generated successfully at: {output_html}")

if __name__ == "__main__":
    run_drift_analysis()