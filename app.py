from pathlib import Path
import os
import numpy as np
import pandas as pd
import streamlit as st
import requests
from supabase import create_client, Client

API_URL = os.getenv("API_URL", "https://fraud-detection-prod-production.up.railway.app").rstrip("/")
TEMP_DATA_PATH = Path("temp.csv")
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

@st.cache_resource
def get_supabase_client():
    url = st.secrets.get("https://mqdbfqlhjddqqncyrwpy.supabase.co") or os.getenv("https://mqdbfqlhjddqqncyrwpy.supabase.co")
    key = st.secrets.get("sb_publishable_vb9pr-pn42rp3UOhGtYRAQ_woFkX5dg") or os.getenv("sb_publishable_vb9pr-pn42rp3UOhGtYRAQ_woFkX5dg")
    if url and key:
        return create_client(url, key)
    return None

def get_api_leaderboard() -> list:
    try:
        response = requests.get(f"{API_URL}/models", timeout=30)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []

def get_api_prediction(features_dict: dict, algo: str, strategy: str, threshold: float) -> dict:
    try:
        params = {"algo": algo, "strategy": strategy, "threshold": threshold}
        response = requests.post(f"{API_URL}/predict", json=features_dict, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

def record_transaction(amount: float, fraud_proba: float, threshold: float, model_name: str) -> None:
    prediction = "Fraud" if fraud_proba >= threshold else "Legit"
    confidence = fraud_proba if prediction == "Fraud" else 1 - fraud_proba
    entry = {
        "Checked at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model Used": model_name,
        "Amount": round(amount, 2),
        "Fraud probability": round(fraud_proba * 100, 2),
        "Threshold": round(threshold, 2),
        "Prediction": prediction,
        "Confidence (%)": round(confidence * 100, 2),
    }
    st.session_state.transaction_history.insert(0, entry)
    st.session_state.transaction_history = st.session_state.transaction_history[:10]

def render_prediction_form(reference_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Enter transaction details")
    if st.button("Fill random values"):
        if not reference_df.empty:
            sample = reference_df[FEATURE_COLUMNS].sample(1).iloc[0]
            st.session_state["amount"] = float(sample["Amount"])
            for i in range(1, 29):
                st.session_state[f"v{i}"] = float(sample[f"V{i}"])
            st.rerun()

    amount_val = st.number_input("Amount", value=100.0, min_value=0.0, format="%.2f", key="amount")
    v_values = {}
    with st.expander("PCA features V1–V14"):
        l, r = st.columns(2)
        for i in range(1, 15):
            target = l if i % 2 else r
            v_values[f"V{i}"] = target.number_input(f"V{i}", value=0.0, format="%.3f", key=f"v{i}")
    with st.expander("PCA features V15–V28"):
        l, r = st.columns(2)
        for i in range(15, 29):
            target = l if i % 2 else r
            v_values[f"V{i}"] = target.number_input(f"V{i}", value=0.0, format="%.3f", key=f"v{i}")
    
    row = {**v_values, "Amount": amount_val}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)

def render_detector(cleaned_df: pd.DataFrame, model_architecture: str, balancing_method: str) -> None:
    st.subheader("Data overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", "283,726")
    col2.metric("Columns", 31)
    col3.metric("Fraud rate", "0.17%")

    st.divider()
    input_df = render_prediction_form(cleaned_df)
    threshold = st.session_state.decision_threshold

    if st.button("Predict transaction", type="primary"):
        features_dict = input_df[FEATURE_COLUMNS].iloc[0].to_dict()
        with st.spinner("Streaming transaction payload to inference engine..."):
            result = get_api_prediction(features_dict, model_architecture, balancing_method, threshold)
        
        if result:
            fraud_proba = result["fraud_probability"]
            prediction_label = result["prediction"]
            model_used = result["model_used"]
            record_transaction(float(input_df["Amount"].iloc[0]), fraud_proba, threshold, model_used)

            if prediction_label == "Fraud":
                st.error(f"**🚨 Fraud** — confidence: **{(fraud_proba*100):.2f}%**")
            else:
                st.success(f"**✅ Legit** — confidence: **{((1-fraud_proba)*100):.2f}%**")
            st.progress(fraud_proba)

def render_dashboard(cleaned_df: pd.DataFrame, current_metrics: dict) -> None:
    st.markdown("<div class='custom-box'><h3>Dataset Overview</h3></div>", unsafe_allow_html=True)
    l_c, f_c = 283253, 473
    t_c = l_c + f_c
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_col1.metric("Total transactions", f"{t_c:,}")
    stat_col2.metric("Legitimate", f"{l_c:,}")
    stat_col3.metric("Fraudulent", f"{f_c:,}")
    stat_col4.metric("Fraud rate", f"{(f_c/t_c)*100:.2f}%")
    
    st.bar_chart(pd.DataFrame({"Count": [l_c, f_c]}, index=["Legit", "Fraud"]))
    
    st.subheader("Decision threshold")
    st.slider("Fraud probability threshold", 0.0, 1.0, float(st.session_state.decision_threshold), 0.01, key="decision_threshold")

# ==========================================
# TOUCHDOWN 3: LIVE MONITOR TAB RENDER ENGINE
# ==========================================
def render_live_monitor():
    st.markdown("""
    <div class='custom-box'>
    <h3>🛡️ Live Production Telemetry & System Infrastructure Logs</h3>
    <p>Real-time analytics processing operational workloads and profiling inference delays.</p>
    </div>
    """, unsafe_allow_html=True)
    
    sb = get_supabase_client()
    if not sb:
        st.warning("⚠️ Supabase credentials missing. Wire them to environment variables to activate live analysis.")
        return

    # User-controlled data window slicing
    window_size = st.slider("Select telemetry window size (Last N requests)", min_value=10, max_value=500, value=100, step=10)

    if st.button("🔄 Refresh System Metrics", type="secondary"):
        st.rerun()

    try:
        # Fetch operational data window directly from the indexed table
        res = sb.table("predictions").select("*").order("created_at", desc=True).limit(window_size).execute()
        records = res.data
    except Exception as e:
        st.error(f"❌ Error communicating with database cluster telemetry layer: {e}")
        return

    if records:
        df = pd.DataFrame(records)
        df["created_at"] = pd.to_datetime(df["created_at"])
        
        # 1. Operational Telemetry Computations
        total_inferences = len(df)
        fraud_events = len(df[df["label"] == "Fraud"])
        fraud_rate = (fraud_events / total_inferences) * 100 if total_inferences > 0 else 0.0
        p95_latency = df["latency_ms"].quantile(0.95)

        # Render Core KPI Metrics Block
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Monitored Traffic Window", f"{total_inferences} calls")
        m_col2.metric("Detected Fraud Velocity", f"{fraud_rate:.2f}%")
        m_col3.metric("Tail Latency (p95 Profile)", f"{p95_latency:.2f} ms")

        st.markdown("---")
        
        # Chronological sort required for plotting timeseries trends cleanly
        df_chronological = df.sort_values("created_at")

        # 2. Plot: Tail Latency Profile over Time
        st.write("#### ⚡ Real-Time Server Latency Profile (ms)")
        st.line_chart(df_chronological.set_index("created_at")["latency_ms"])

        # 3. Plot: Fraud Volatility Rate Over Time Window
        st.write("#### 📊 Cumulative Fraud Velocity Trend (%)")
        # Build rolling expanding fraud rate line to track live system variations smoothly
        df_chronological["is_fraud"] = (df_chronological["label"] == "Fraud").astype(int)
        df_chronological["rolling_fraud_rate"] = (df_chronological["is_fraud"].expanding().mean()) * 100
        st.line_chart(df_chronological.set_index("created_at")["rolling_fraud_rate"])

        st.markdown("---")
        
        # 4. Live Audit Ledger
        st.write("#### 📋 Live Telemetry Audit Data Grid")
        st.dataframe(
            df[["created_at", "algo", "strategy", "amount", "fraud_probability", "label", "latency_ms"]].head(25),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("⚡ System database is initialized. Standing by for transaction execution data...")

        
def main() -> None:
    st.markdown("""
    <style>
    .stApp { background-color: #f5f7fb; color: #0f172a; }
    .custom-box { background: white; padding: 20px; border-radius: 12px; border-left: 6px solid #2a5298; box-shadow: 0px 3px 10px rgba(0,0,0,0.08); margin-bottom: 15px; color: #0f172a; }
    .stButton > button { background: linear-gradient(135deg, #2a5298, #00c6ff); color: white; border-radius: 10px; border: none; font-weight: bold; }
    div[data-testid="stMarkdownContainer"] p { color: #0f172a; }
    </style>
    """, unsafe_allow_html=True)

    if "transaction_history" not in st.session_state: st.session_state.transaction_history = []
    if "decision_threshold" not in st.session_state: st.session_state.decision_threshold = 0.5

    st.sidebar.header("Model Settings")
    balancing_method = st.sidebar.selectbox("Data Balancing Strategy", options=["None", "class_weight", "SMOTE"], index=2)
    model_architecture = st.sidebar.selectbox("Model Architecture", options=["Random Forest", "XGBoost"], index=0)

    leaderboard_data = get_api_leaderboard()
    current_metrics = next((item for item in leaderboard_data if item.get("algorithm") == model_architecture and item.get("strategy") == balancing_method), {})

    cleaned_df = pd.read_csv(TEMP_DATA_PATH) if TEMP_DATA_PATH.exists() else pd.DataFrame(columns=FEATURE_COLUMNS)

    # Updated navigation structure to mount the third tab cleanly
    detector_tab, dashboard_tab, leaderboard_tab, monitor_tab = st.tabs(["Fraud Detector", "Dashboard", "Leaderboard", "📈 Live Monitor"])

    with detector_tab: render_detector(cleaned_df, model_architecture, balancing_method)
    with dashboard_tab: render_dashboard(cleaned_df, current_metrics)
    with monitor_tab: render_live_monitor()
    with leaderboard_tab:
        if leaderboard_data:
            ld_df = pd.DataFrame(leaderboard_data)
            st.dataframe(ld_df, use_container_width=True, hide_index=True)
        else:
            st.info("No active pipeline data found.")

if __name__ == "__main__":
    main()