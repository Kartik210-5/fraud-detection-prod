# app.py
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_score, recall_score

# Import from our multi-model backend
from model import (
    get_or_train_all_models, 
    DATA_PATH, 
    FEATURE_COLUMNS, 
    TARGET_COLUMN,
    TEMP_DATA_PATH
)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at `{DATA_PATH}`. Place `creditcard_2023.csv` in the same folder as this app.")
        st.stop()
    return pd.read_csv(DATA_PATH)


def assess_data_quality(df: pd.DataFrame) -> dict:
    return {
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "infinite_values": int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum()),
        "invalid_class_values": int((~df[TARGET_COLUMN].isin([0, 1])).sum()) if TARGET_COLUMN in df.columns else 0,
    }


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cleaned = df.copy()
    steps: list[str] = []

    if "id" in cleaned.columns:
        cleaned = cleaned.drop(columns=["id"])
        steps.append("Dropped `id` column (identifier, not used for modeling).")

    if cleaned[TARGET_COLUMN].dtype == object:
        cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
        steps.append("Converted `Class` to numeric.")

    missing_before = cleaned.isnull().sum().sum()
    if missing_before:
        cleaned = cleaned.dropna()
        steps.append(f"Dropped rows with missing values ({missing_before} missing cells).")

    duplicates = cleaned.duplicated().sum()
    if duplicates:
        cleaned = cleaned.drop_duplicates()
        steps.append(f"Removed {duplicates} duplicate rows.")

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(cleaned[numeric_cols]).any(axis=1)
    inf_count = int(inf_mask.sum())
    if inf_count:
        cleaned = cleaned.loc[~inf_mask]
        steps.append(f"Removed {inf_count} rows containing infinite values.")

    invalid_class = (~cleaned[TARGET_COLUMN].isin([0, 1])).sum()
    if invalid_class:
        cleaned = cleaned[cleaned[TARGET_COLUMN].isin([0, 1])]
        steps.append(f"Removed {invalid_class} rows with invalid class labels.")

    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    if not steps:
        steps.append("No cleaning required. Dataset was already in good shape.")

    return cleaned.reset_index(drop=True), steps


# Cache the full pipeline load/train routine to keep interface changes instant
@st.cache_resource(show_spinner="Processing pipeline model matrices...")
def load_or_train_pipeline(cleaned_df=None) -> dict:
    return get_or_train_all_models(cleaned_df)


def threshold_metrics(y_true: np.ndarray, fraud_proba: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (fraud_proba >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def record_transaction(amount: float, fraud_proba: float, threshold: float) -> None:
    prediction = "Fraud" if fraud_proba >= threshold else "Legit"
    confidence = fraud_proba if prediction == "Fraud" else 1 - fraud_proba
    entry = {
        "Checked at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        # CRITICAL FIX: Check if reference data contains any rows to sample from
        if reference_df.empty:
            st.warning("⚠️ Cannot sample random values: App is running in offline cache mode with no underlying dataset loaded.")
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


def render_detector(cleaned_df: pd.DataFrame, model) -> None:
    st.subheader("Data overview")
    overview_col1, overview_col2, overview_col3 = st.columns(3)
    overview_col1.metric("Rows", f"{len(cleaned_df):,}")
    overview_col2.metric("Columns", len(cleaned_df.columns))
    overview_col3.metric("Fraud rate", f"{cleaned_df[TARGET_COLUMN].mean() * 100:.3f}%")

    st.divider()
    input_df = render_prediction_form(cleaned_df)
    threshold = st.session_state.decision_threshold

    if st.button("Predict transaction", type="primary"):
        features = input_df[FEATURE_COLUMNS]
        probabilities = model.predict_proba(features)[0]
        fraud_proba = float(probabilities[1])
        prediction = 1 if fraud_proba >= threshold else 0

        legit_confidence = (1 - fraud_proba) * 100
        fraud_confidence = fraud_proba * 100

        record_transaction(float(input_df["Amount"].iloc[0]), fraud_proba, threshold)

        if prediction == 1:
            label = "Fraud"
            confidence = fraud_confidence
            st.error(f"**{label}** — confidence: **{confidence:.2f}%** (threshold: {threshold:.2f})")
        else:
            label = "Legit"
            confidence = legit_confidence
            st.success(f"**{label}** — confidence: **{confidence:.2f}%** (threshold: {threshold:.2f})")

        st.progress(confidence / 100)
        st.write(
            f"Legit probability: **{legit_confidence:.2f}%** · "
            f"Fraud probability: **{fraud_confidence:.2f}%**"
        )


def render_dashboard(cleaned_df: pd.DataFrame, artifacts: dict) -> None:
    st.markdown("""
    <div class='custom-box'>
    <h3>Dataset Overview</h3>
    </div>
    """, unsafe_allow_html=True)

    class_counts = cleaned_df[TARGET_COLUMN].value_counts().sort_index()
    legit_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))
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
    st.caption("Adjust the fraud-probability cutoff used for predictions and see hold-out test metrics update live.")

    threshold = st.slider(
        "Fraud probability threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.decision_threshold),
        step=0.01,
        key="decision_threshold",
    )

    live_metrics = threshold_metrics(artifacts["y_test"], artifacts["fraud_proba"], threshold)
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Precision (fraud)", f"{live_metrics['precision']:.3f}")
    metric_col2.metric("Recall (fraud)", f"{live_metrics['recall']:.3f}")

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
    st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
    st.markdown("""
    <style>
    /* Main app layout backgrounds */
    .stApp { background-color: #f5f7fb; }
    
    /* Header branding container layout styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 25px; border-radius: 15px; text-align: center; color: white; margin-bottom: 20px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
    }
    
    /* Box element structuring layout styling */
    .custom-box {
        background: white; padding: 20px; border-radius: 12px; border-left: 6px solid #2a5298;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.08); margin-bottom: 15px;
    }
    hr { border: none; height: 2px; background: linear-gradient(to right,#2a5298,#00c6ff); margin: 20px 0; }
    
    /* Document Tab Layout Controls */
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { background-color: #eef2ff; border-radius: 10px; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #2a5298 !important; color: white !important; }
    
    /* Interactivity element layout styles */
    .stButton > button {
        background: linear-gradient(135deg,#2a5298,#00c6ff); color: white; border-radius: 10px;
        border: none; font-weight: bold; padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover { transform: scale(1.02); transition: 0.2s; }
    
    /* --- OVERRIDE SIDEBAR STYLINGS TO FORCE BLACK TEXT HIGHLIGHTS --- */
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
        index=1,
        help="Select how to manage dataset structural skewness during training."
    )

    # --- PIPELINE RESILIENT MULTI-MODEL HANDLING BLOCK ---
    if DATA_PATH.exists():
        df = load_data()
        quality = assess_data_quality(df)
        cleaned_df, cleaning_steps = clean_data(df)

        # Automatically processes models and exports temp.csv
        pipeline_artifacts = load_or_train_pipeline(cleaned_df)
    else:
        # FALLBACK MODE: CSV is missing. Load saved models from pkl assets
        try:
            pipeline_artifacts = load_or_train_pipeline()
        except FileNotFoundError:
            st.error("Severe Exception: Absolute system assets missing. Place `creditcard_2023.csv` in repository root.")
            st.stop()

        quality = {"missing_values": 0, "duplicate_rows": 0, "infinite_values": 0, "invalid_class_values": 0}
        cleaning_steps = ["Loaded pre-compiled strategy models from storage logs."]
        
        # FIX: Check if temp.csv exists to preserve the random layout generator feature
        if TEMP_DATA_PATH.exists():
            cleaned_df = pd.read_csv(TEMP_DATA_PATH)
            cleaning_steps.append("Loaded fallback random testing instances from temp.csv.")
        else:
            cleaned_df = pd.DataFrame(columns=FEATURE_COLUMNS + [TARGET_COLUMN])
            cleaning_steps.append("Warning: temp.csv missing. Random testing features are disabled.")

    # Unpack properties dynamically based on active selection choice
    artifacts = pipeline_artifacts[balancing_method]
    model = artifacts["model"]
    metrics = artifacts["metrics"]

    # Diagnostic Dropdown Wrapper Element
    with st.expander("Data quality & model training status logs", expanded=False):
        quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
        quality_col1.metric("Missing values", quality["missing_values"])
        quality_col2.metric("Duplicate rows", quality["duplicate_rows"])
        quality_col3.metric("Infinite values", quality["infinite_values"])
        quality_col4.metric("Invalid labels", quality["invalid_class_values"])

        st.subheader("Cleaning summary")
        for step in cleaning_steps:
            st.write(f"- {step}")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Training rows", f"{metrics['train_rows']:,}")
        metric_col2.metric("Test rows", f"{metrics['test_rows']:,}")
        metric_col3.metric("F1 (fraud class)", f"{metrics['f1_fraud']:.3f}")

        with st.expander("Classification report (hold-out test set)"):
            st.text(metrics["report"])

    # --- TAB NAVIGATION MATRIX ---
    detector_tab, dashboard_tab, compare_tab = st.tabs(["Fraud Detector", "Dashboard", "Compare Strategies"])

    with detector_tab:
        render_detector(cleaned_df, model)

    with dashboard_tab:
        render_dashboard(cleaned_df, artifacts)

    with compare_tab:
        st.subheader("Balancing Strategy Performance Metrics Table")
        
        # Build evaluation matrix summary dynamically from runtime memory registers
        comp_df = pd.DataFrame({
            "Strategy Method": ["None (Unbalanced)", "class_weight", "SMOTE Resampling"],
            "Validation F1-Score": [
                f"{pipeline_artifacts['None']['metrics']['f1_fraud']:.4f}", 
                f"{pipeline_artifacts['class_weight']['metrics']['f1_fraud']:.4f}", 
                f"{pipeline_artifacts['SMOTE']['metrics']['f1_fraud']:.4f}"
            ],
            "Training Matrix Sample Size": [
                f"{pipeline_artifacts['None']['metrics']['train_rows']:,}", 
                f"{pipeline_artifacts['class_weight']['metrics']['train_rows']:,}", 
                f"{pipeline_artifacts['SMOTE']['metrics']['train_rows']:,}"
            ]
        })
        
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        st.subheader("F1-Score Comparison Bar Chart")
        chart_data = pd.DataFrame({
            "F1-Score": [
                pipeline_artifacts["None"]["metrics"]["f1_fraud"], 
                pipeline_artifacts["class_weight"]["metrics"]["f1_fraud"], 
                pipeline_artifacts["SMOTE"]["metrics"]["f1_fraud"]
            ]
        }, index=["None", "class_weight", "SMOTE"])
        st.bar_chart(chart_data)


if __name__ == "__main__":
    main()