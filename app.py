# app.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import requests

# FastAPI Backend Endpoint - Single Source of Truth
API_URL = "http://127.0.0.1:8000"

# Fallback path if you need to pull mock features locally
TEMP_DATA_PATH = Path("temp.csv")
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def get_api_leaderboard() -> list:
    """Fetches the live ranked model metrics matrix directly from the FastAPI endpoint."""
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        if response.status_code == 200:
            return response.json()
        return []
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to FastAPI server. Is uvicorn running on port 8000?")
        return []


def get_api_prediction(features_dict: dict, algo: str, strategy: str) -> dict:
    """Sends transaction payload over the network to the API inference engine."""
    try:
        params = {"algo": algo, "strategy": strategy}
        response = requests.post(
            f"{API_URL}/predict", 
            json=features_dict, 
            params=params,
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error from API: {response.json().get('detail')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ API server is unreachable.")
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
    history = st.session_state.transaction_history
    history.insert(0, entry)
    st.session_state.transaction_history = history[:10]


def render_prediction_form(reference_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Enter transaction details")

    if st.button("Fill random values"):
        if reference_df.empty:
            st.warning("⚠️ Cannot sample random values: No local test features available.")
        else:
            sample = reference_df[FEATURE_COLUMNS].sample(1, random_state=None).iloc[0]
            st.session_state["amount"] = float(sample["Amount"])
            for i in range(1, 29):
                st.session_state[f"v{i}"] = float(sample[f"V{i}"])
            st.rerun()

    amount_val = st.number_input("Amount", value=100.0, min_value=0.0, format="%.2f", key="amount")

    v_values: dict[str, float] = {}
    with st.expander("PCA features V1–V14", expanded=False):
        left, right = st.columns(2)
        for i in range(1, 15):
            target = left if i % 2 else right
            with target:
                v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.3f", key=f"v{i}")

    with st.expander("PCA features V15–V28", expanded=False):
        left, right = st.columns(2)
        for i in range(15, 29):
            target = left if i % 2 else right
            with target:
                v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.3f", key=f"v{i}")

    row = {**v_values, "Amount": amount_val}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def render_detector(cleaned_df: pd.DataFrame, model_architecture: str, balancing_method: str) -> None:
    st.subheader("Data overview")
    overview_col1, overview_col2, overview_col3 = st.columns(3)
    overview_col1.metric("Rows", "283,726")
    overview_col2.metric("Columns", 31)
    overview_col3.metric("Fraud rate", "0.17%")

    st.divider()
    input_df = render_prediction_form(cleaned_df)
    threshold = st.session_state.decision_threshold

    if st.button("Predict transaction", type="primary"):
        features_dict = input_df[FEATURE_COLUMNS].iloc[0].to_dict()
        
        with st.spinner("Streaming transaction payload to inference engine..."):
            result = get_api_prediction(features_dict, model_architecture, balancing_method)
        
        if result:
            fraud_proba = result["fraud_probability"]
            prediction_label = result["prediction"]
            model_used = result["model_used"]

            legit_confidence = (1 - fraud_proba) * 100
            fraud_confidence = fraud_proba * 100

            record_transaction(float(input_df["Amount"].iloc[0]), fraud_proba, threshold, model_used)

            if prediction_label == "Fraud":
                st.error(f"**🚨 Fraud** — confidence: **{fraud_confidence:.2f}%** (threshold: {threshold:.2f})")
                st.caption(f"Evaluated by backend microservice: {model_used}")
            else:
                st.success(f"**✅ Legit** — confidence: **{legit_confidence:.2f}%** (threshold: {threshold:.2f})")
                st.caption(f"Evaluated by backend microservice: {model_used}")

            st.progress(fraud_confidence / 100)
            st.write(
                f"Legit probability: **{legit_confidence:.2f}%** · "
                f"Fraud probability: **{fraud_confidence:.2f}%**"
            )


def render_dashboard(cleaned_df: pd.DataFrame, current_metrics: dict) -> None:
    st.markdown("""
    <div class='custom-box'>
    <h3>Dataset Overview</h3>
    </div>
    """, unsafe_allow_html=True)

    legit_count = 283253
    fraud_count = 473
    total_count = legit_count + fraud_count
    fraud_rate = fraud_count / total_count if total_count else 0.0

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_col1.metric("Total transactions", f"{total_count:,}")
    stat_col2.metric("Legitimate", f"{legit_count:,}")
    stat_col3.metric("Fraudulent", f"{fraud_count:,}")
    stat_col4.metric("Fraud rate", f"{fraud_rate * 100:.2f}%")

    balance_df = pd.DataFrame(
        {"Count": [legit_count, fraud_count]},
        index=["Legit (0)", "Fraud (1)"],
    )
    st.bar_chart(balance_df)

    st.subheader("Decision threshold")
    st.caption("Adjust the fraud-probability cutoff used for predictions and track metadata configurations.")

    st.slider(
        "Fraud probability threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.decision_threshold),
        step=0.01,
        key="decision_threshold",
    )

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Active Variant F1-Score", f"{current_metrics.get('f1_score_fraud', 0.0):.4f}")
    metric_col2.metric("Target Variant Training Size", f"{current_metrics.get('train_rows', 'N/A')}")

    st.subheader("Recent checks")
    if st.session_state.transaction_history:
        st.dataframe(
            pd.DataFrame(st.session_state.transaction_history),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No transactions checked yet. Run a prediction on the Fraud Detector tab.")


def main() -> None:
    st.markdown("""
    <style>
    /* Main app layout backgrounds for crisp Light Mode */
    .stApp { background-color: #f5f7fb; color: #0f172a; }
    
    /* Header branding container layout styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 25px; border-radius: 15px; text-align: center; color: white; margin-bottom: 20px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
    }
    
    /* Elegant clean light-mode box card elements */
    .custom-box {
        background: white; padding: 20px; border-radius: 12px; border-left: 6px solid #2a5298;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.08); margin-bottom: 15px; color: #0f172a;
    }
    hr { border: none; height: 2px; background: linear-gradient(to right, #2a5298, #00c6ff); margin: 20px 0; }
    
    /* Tab Controls matching Light Layout */
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { background-color: #eef2ff; border-radius: 10px; padding: 10px 20px; color: #475569 !important; }
    .stTabs [aria-selected="true"] { background-color: #2a5298 !important; color: white !important; }
    
    /* Interactivity button layouts */
    .stButton > button {
        background: linear-gradient(135deg, #2a5298, #00c6ff); color: white; border-radius: 10px;
        border: none; font-weight: bold; padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover { transform: scale(1.02); transition: 0.2s; box-shadow: 0px 4px 10px rgba(0,198,255,0.3); }
    
    /* --- SIDEBAR STYLINGS FOR HIGH-CONTRAST LIGHT MODE --- */
    section[data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #ffffff, #f1f5f9) !important;
        border-right: 1px solid #cbd5e1;
    }
    section[data-testid="stSidebar"] * { 
        color: #0f172a !important; 
    }
    section[data-testid="stSidebar"] label p, section[data-testid="stSidebar"] h2 {
        color: #0f172a !important;
        font-weight: bold !important;
    }
    div[data-baseweb="popover"] * {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* HARD LOCK: Hides settings header, footer, and the options menu entirely */
    #MainMenu, header, footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='main-header'>
        <h1>Credit Card Fraud Detection System</h1>
        <p>Machine Learning Powered Fraud Analysis Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    if "transaction_history" not in st.session_state:
        st.session_state.transaction_history = []
    if "decision_threshold" not in st.session_state:
        st.session_state.decision_threshold = 0.5

    # --- Sidebar Dropdown Controls Block ---
    st.sidebar.header("Model Settings")
    
    balancing_method = st.sidebar.selectbox(
        "Data Balancing Strategy",
        options=["None", "class_weight", "SMOTE"],
        index=2,
        help="Select how to manage dataset structural skewness during training."
    )
    
    model_architecture = st.sidebar.selectbox(
        "Model Architecture",
        options=["Random Forest", "XGBoost"],
        index=0,
        help="Choose the underlying machine learning classifier algorithm."
    )

    # --- FETCH LIVE DATA CONTRACT FROM BACKEND API ---
    leaderboard_data = get_api_leaderboard()
    
    # Locate current selected parameters inside API telemetry metrics array
    current_metrics = {}
    for item in leaderboard_data:
        if item["algorithm"] == model_architecture and item["strategy"] == balancing_method:
            current_metrics = item
            break

    # Load fallback test variables for the generator layout
    if TEMP_DATA_PATH.exists():
        cleaned_df = pd.read_csv(TEMP_DATA_PATH)
        cleaning_steps = ["Loaded sample metrics dynamically from backend API records.", "Random value assignment using temp.csv active."]
    else:
        cleaned_df = pd.DataFrame(columns=FEATURE_COLUMNS)
        cleaning_steps = ["Loaded sample metrics dynamically from backend API records.", "Warning: temp.csv missing. Sampling disabled."]

    # Diagnostic Logs Display Container
    with st.expander("System data quality & API telemetry logs", expanded=False):
        quality_col1, quality_col2 = st.columns(2)
        quality_col1.metric("API Status", "Connected" if leaderboard_data else "Disconnected")
        quality_col2.metric("Total API Models", len(leaderboard_data))

        st.subheader("System Summary")
        for step in cleaning_steps:
            st.write(f"- {step}")

        if current_metrics:
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Training Rows", f"{current_metrics.get('train_rows', 0):,}")
            metric_col2.metric("Test Rows", f"{current_metrics.get('test_rows', 0):,}")
            metric_col3.metric("F1 Score", f"{current_metrics.get('f1_score_fraud', 0.0):.3f}")

    # --- TAB NAVIGATION MATRIX ---
    detector_tab, dashboard_tab, leaderboard_tab = st.tabs(["Fraud Detector", "Dashboard", "Leaderboard"])

    with detector_tab:
        render_detector(cleaned_df, model_architecture, balancing_method)

    with dashboard_tab:
        render_dashboard(cleaned_df, current_metrics)

    with leaderboard_tab:
        st.subheader("🏆 Model Performance Leaderboard")
        st.caption("All combinations ranked automatically by their F1-Score evaluation metrics from the API service.")

        if leaderboard_data:
            ld_df = pd.DataFrame(leaderboard_data)
            
            # Re-map columns for production layout styling
            ld_df.columns = ["Algorithm", "Balancing Strategy", "F1-Score (Fraud)", "Training Matrix Size", "Test Matrix Size"]
            
            champion_algo = ld_df.iloc[0]["Algorithm"]
            champion_strategy = ld_df.iloc[0]["Balancing Strategy"]
            champion_f1 = ld_df.iloc[0]["F1-Score (Fraud)"]

            st.markdown(f"""
            <div style="background-color: #f0fdf4; border: 2px solid #16a34a; padding: 20px; border-radius: 12px; margin-bottom: 25px;">
                <h3 style="color: #16a34a; margin-top: 0;">🥇 Current Champion Variant</h3>
                <p style="color: #14532d; font-size: 1.15rem; margin-bottom: 0;">
                    The best performing setup is <strong>{champion_algo}</strong> utilizing the <strong>{champion_strategy}</strong> strategy 
                    yielding an F1-Score of <strong>{champion_f1:.4f}</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)

            def highlight_champion(row):
                if row["Algorithm"] == champion_algo and row["Balancing Strategy"] == champion_strategy:
                    return ["background-color: #bbf7d0; font-weight: bold; color: #0f172a;"] * len(row)
                return ["color: #0f172a;"] * len(row)

            styled_ld_df = ld_df.style.apply(highlight_champion, axis=1).format({"F1-Score (Fraud)": "{:.4f}", "Training Matrix Size": "{:,}"})
            st.dataframe(styled_ld_df, use_container_width=True, hide_index=True)

            st.subheader("F1-Score Comparison Across Configurations")
            chart_df = ld_df.copy()
            chart_df["Configuration"] = chart_df["Algorithm"] + " (" + chart_df["Balancing Strategy"] + ")"
            chart_df = chart_df.set_index("Configuration")[["F1-Score (Fraud)"]]
            st.bar_chart(chart_df)
        else:
            st.info("No active pipeline configuration metrics found on backend API registries.")


if __name__ == "__main__":
    main()