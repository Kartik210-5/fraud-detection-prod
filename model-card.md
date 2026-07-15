# 📊 Model Card: Production Fraud Detection System

## 1. Model Details
* **Algorithm:** XGBoost (Extreme Gradient Boosting Classifier)
* **Optimization Strategy:** SMOTE (Synthetic Minority Over-sampling Technique) to handle severe class imbalance
* **Input Features:** PCA-transformed numerical features (V1–V28), transaction Amount, and relative Time
* **Target:** Binary Classification (`0`: Legit transaction, `1`: Fraudulent transaction)

## 2. Intended Use
* **Primary Use Case:** Real-time production credit card fraud classification.
* **Out of Scope:** Multi-class classification or forecasting generalized financial trends.

## 3. Training & Validation Data
* **Baseline Data:** Historical financial transaction datasets containing credit card records.
* **Live Ingestion Data:** Continuously augmented with production telemetry records pulled from Supabase.
* **Evaluation Strategy:** Stratified train-test splits (80/20) to preserve minority target ratios during pipeline retraining.

## 4. Operational Guardrails (Evolution Pipeline)
* **Drift Metric:** Two-sample Kolmogorov-Smirnov (KS) test calculated over critical features.
* **Trigger Threshold:** Drastic statistical shifts ($p\text{-value} < 0.05$) trigger the closed-loop challenger training pipeline.
* **Artifact Path:** Models are serialized and saved via joblib relative to system directories.