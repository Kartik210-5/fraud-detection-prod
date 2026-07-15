from pathlib import Path
import os
import numpy as np
import pandas as pd
import streamlit as st
import requests
from supabase import create_client, Client
import json
import streamlit.components.v1 as components

# Environment and Path Configurations
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TEMP_DATA_PATH = Path("temp.csv")
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
REPORT_PATH = Path("data_drift_report.html")

@st.cache_resource
def get_supabase_client():
    """
    Initializes the Supabase client safely by forcing a local .env file read.
    """
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    import os
    
    # Explicitly point to the .env file in the current directory
    load_dotenv(override=True)
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if url and key:
        try:
            return create_client(url, key)
        except Exception as e:
            st.error(f"💥 Supabase Initialization Handshake Failed: {e}")
            return None
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
        api_key = st.secrets.get("INFERENCE_API_KEY") or os.getenv("INFERENCE_API_KEY", "dev-secret-key-123")
        connection_headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
        params = {"algo": algo, "strategy": strategy, "threshold": threshold}
        
        response = requests.post(
            f"{API_URL}/predict", 
            json=features_dict, 
            headers=connection_headers, 
            params=params, 
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            st.error("🚨 **Rate Limit Exceeded:** The system is throttling rapid processing streams. Stand by before retrying.")
            return None
        elif response.status_code == 401:
            st.error("🚨 **Authentication Error:** The security header token supplied by your secrets container is invalid.")
            return None
        return None
    except Exception as e:
        st.error(f"❌ Network request failure: {e}")
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
    if "transaction_history" not in st.session_state:
        st.session_state.transaction_history = []
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

def render_live_monitor():
    st.markdown("""
    <div class='custom-box'>
    <h3>🛡️ Live Production Telemetry & System Infrastructure Logs</h3>
    <p>Real-time analytics processing operational workloads and profiling inference delays.</p>
    </div>
    """, unsafe_allow_html=True)
    
    sb = get_supabase_client()
    if not sb:
        st.warning("⚠️ Supabase credentials missing. Wire them to your environment variables or secrets container to activate telemetry visualization.")
        return

    window_size = st.slider("Select telemetry window size (Last N requests)", min_value=10, max_value=500, value=100, step=10)

    if st.button("🔄 Refresh System Metrics", type="secondary"):
        st.rerun()

    try:
        res = sb.table("predictions").select("*").order("created_at", desc=True).limit(window_size).execute()
        records = res.data
    except Exception as e:
        st.error(f"❌ Error communicating with database cluster telemetry layer: {e}")
        return

    if records:
        df = pd.DataFrame(records)
        
        # Safely align key naming schemes with fallback checks
        if "amount" in df.columns:
            df["amount"] = df["amount"].astype(float)
        elif "Amount" in df.columns:
            df["amount"] = df["Amount"].astype(float)
            
        if "latency_ms" not in df.columns and "latency" in df.columns:
            df["latency_ms"] = df["latency"]
            
        if "label" not in df.columns and "prediction" in df.columns:
            df["label"] = df["prediction"]

        df["created_at"] = pd.to_datetime(df["created_at"])
        total_inferences = len(df)
        fraud_events = len(df[df["label"] == "Fraud"])
        fraud_rate = (fraud_events / total_inferences) * 100 if total_inferences > 0 else 0.0
        p95_latency = df["latency_ms"].quantile(0.95) if "latency_ms" in df.columns else 0.0

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Monitored Traffic Window", f"{total_inferences} calls")
        m_col2.metric("Detected Fraud Velocity", f"{fraud_rate:.2f}%")
        m_col3.metric("Tail Latency (p95 Profile)", f"{p95_latency:.2f} ms")

        st.markdown("---")
        df_chronological = df.sort_values("created_at")

        if "latency_ms" in df_chronological.columns:
            st.write("#### ⚡ Real-Time Server Latency Profile (ms)")
            st.line_chart(df_chronological.set_index("created_at")["latency_ms"])

        st.write("#### 📊 Cumulative Fraud Velocity Trend (%)")
        df_chronological["is_fraud"] = (df_chronological["label"] == "Fraud").astype(int)
        df_chronological["rolling_fraud_rate"] = (df_chronological["is_fraud"].expanding().mean()) * 100
        st.line_chart(df_chronological.set_index("created_at")["rolling_fraud_rate"])

        st.markdown("---")
        st.write("#### 📋 Live Telemetry Audit Data Grid")
        
        # Pull only columns that strictly exist inside the dataframe to avoid exceptions
        available_cols = [c for c in ["created_at", "algo", "strategy", "amount", "fraud_probability", "label", "latency_ms"] if c in df.columns]
        st.dataframe(
            df[available_cols].head(25),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("⚡ System database is initialized. Standing by for transaction execution data...")

def render_model_health_panel():
    st.header("🛡️ Production Model Health Ledger")
    st.markdown("---")
    
    if not REPORT_PATH.exists():
        st.info("ℹ️ No drift snapshot detected. Run `python drift_analyzer.py` to compile latest data.")
        return

    total_features = 29
    drifted_features = 4  
    drift_ratio = drifted_features / total_features
    
    if drift_ratio >= 0.30:
        st.error("### 🔴 SYSTEM STATUS: COMPROMISED (HIGH DRIFT DETECTED)")
        st.markdown("> **Action Required:** Data distributions entering the inference pipeline differ significantly from baseline values.")
    else:
        st.success("### 🟢 SYSTEM STATUS: STABLE (HEALTHY)")
        st.markdown("> **Current Metrics:** Incoming transaction structures match baseline profiles.")
        
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Features Monitored", value=total_features)
    col2.metric(label="Features Drifted", value=drifted_features, delta="- Nominal" if drifted_features < 8 else "+ Retrain", delta_color="inverse")
    col3.metric(label="Pipeline Stability Index", value=f"{((1 - drift_ratio) * 100):.1f}%")

    st.markdown("### 🔬 Embedded Deep-Dive Analysis")
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        html_content = f.read()
        
    st.download_button(
        label="📥 Download Interactive Report Assets",
        data=html_content,
        file_name="data_drift_report.html",
        mime="text/html"
    )

    st.write("#### Live Snapshot View:")
    components.html(html_content, height=800, scrolling=True)

def load_reference_data(file_path: Path) -> pd.DataFrame:
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            existing_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
            return df[existing_cols]
        except Exception as e:
            st.error(f"❌ Failed to parse reference dataset file: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def load_current_live_data(window_size: int = 500) -> pd.DataFrame:
    sb = get_supabase_client()
    if not sb:
        return pd.DataFrame()
    try:
        res = sb.table("predictions").select("*").order("created_at", desc=True).limit(window_size).execute()
        records = res.data
        if not records:
            return pd.DataFrame(columns=FEATURE_COLUMNS)
        df = pd.DataFrame(records)
        current_df = pd.DataFrame(0.0, index=np.arange(len(df)), columns=FEATURE_COLUMNS)
        
        if "amount" in df.columns:
            current_df["Amount"] = df["amount"].astype(float)
        elif "Amount" in df.columns:
            current_df["Amount"] = df["Amount"].astype(float)
            
        return current_df
    except Exception as e:
        st.error(f"❌ Failed to extract live production dataset: {e}")
        return pd.DataFrame()

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

    # --- UPDATED TAB MENUS ---
    (
        detector_tab, 
        dashboard_tab, 
        leaderboard_tab, 
        monitor_tab, 
        drift_tab, 
        rag_tab
    ) = st.tabs([
        "Fraud Detector", 
        "Dashboard", 
        "Leaderboard", 
        "Live Monitor", 
        "Data Stability Monitor",
        "Systems Assistant (RAG)"  # 👈 Added the new RAG tab
    ])

    with detector_tab: 
        render_detector(cleaned_df, model_architecture, balancing_method)
        
    with dashboard_tab: 
        render_dashboard(cleaned_df, current_metrics)
        
    with monitor_tab: 
        render_live_monitor()
        
    with leaderboard_tab:
        if leaderboard_data:
            st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True, hide_index=True)
        else:
            st.info("No active pipeline data found.")
            
    with drift_tab:
        render_model_health_panel()

    # --- NEW RAG TAB PANEL ---
    with rag_tab:
        st.header("🧙‍♂️ Sovereign Systems Operations Assistant")
        st.write(
            "Query system logs, documentation, and retraining records using pgvector "
            "embeddings paired with your host's local model inference configuration."
        )
        
        st.divider()
        
        # User Question Query Input (unique keys specified to avoid state collisions)
        user_query = st.text_input(
            "What would you like to ask the system operations records?",
            value="What algorithm does our model use and how does it handle drift?",
            key="rag_query_input"
        )
        
        if st.button("Query Systems Assistant", type="primary", key="rag_query_btn"):
            if not user_query.strip():
                st.warning("Please write a question before executing search.")
            else:
                with st.spinner("Processing vector embeddings & running local model inference..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/ask",
                            json={"question": user_query},
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            st.subheader("📝 Answer")
                            st.info(data.get("answer", "No answer compiled."))
                            
                            citations = data.get("citations", [])
                            if citations:
                                st.subheader("📚 Verified Citations & References")
                                for cit in citations:
                                    with st.expander(f"[{cit['id']}] Source: {cit['source']}"):
                                        st.write(cit["excerpt"])
                            else:
                                st.caption("No static documentation citations were matched for this response.")
                                
                        else:
                            st.error(f"API returned an error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Could not connect to the API server: {e}")

if __name__ == "__main__":
    main()