from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "creditcard_2023.csv"
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


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


@st.cache_resource(show_spinner="Training Random Forest model...")
def train_model(df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    fraud_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "f1_fraud": float(f1_score(y_test, y_pred, pos_label=1)),
        "report": classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "fraud_rate": float(y.mean()),
    }

    return {
        "model": model,
        "metrics": metrics,
        "y_test": y_test.to_numpy(),
        "fraud_proba": fraud_proba,
    }


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


def render_detector(cleaned_df: pd.DataFrame, model: RandomForestClassifier) -> None:
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
    st.subheader("Class balance & fraud rate")

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
    st.title("Credit Card Fraud Detection")
    st.caption("Load data, clean it, train a Random Forest, and score new transactions.")

    if "transaction_history" not in st.session_state:
        st.session_state.transaction_history = []
    if "decision_threshold" not in st.session_state:
        st.session_state.decision_threshold = 0.5

    df = load_data()
    quality = assess_data_quality(df)
    cleaned_df, cleaning_steps = clean_data(df)
    artifacts = train_model(cleaned_df)
    model = artifacts["model"]
    metrics = artifacts["metrics"]

    with st.expander("Data quality & model training", expanded=False):
        quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
        quality_col1.metric("Missing values", quality["missing_values"])
        quality_col2.metric("Duplicate rows", quality["duplicate_rows"])
        quality_col3.metric("Infinite values", quality["infinite_values"])
        quality_col4.metric("Invalid labels", quality["invalid_class_values"])

        st.subheader("Cleaning summary")
        for step in cleaning_steps:
            st.write(f"- {step}")
        if len(cleaned_df) != len(df):
            st.info(f"Rows after cleaning: {len(cleaned_df):,} (removed {len(df) - len(cleaned_df):,}).")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Training rows", f"{metrics['train_rows']:,}")
        metric_col2.metric("Test rows", f"{metrics['test_rows']:,}")
        metric_col3.metric("F1 (fraud class)", f"{metrics['f1_fraud']:.3f}")

        with st.expander("Classification report (hold-out test set)"):
            st.text(metrics["report"])

    detector_tab, dashboard_tab = st.tabs(["Fraud Detector", "Dashboard"])

    with detector_tab:
        render_detector(cleaned_df, model)

    with dashboard_tab:
        render_dashboard(cleaned_df, artifacts)


if __name__ == "__main__":
    main()
