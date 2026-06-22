import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np

# 1. Point your script to your active local tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# 2. Define or select an experiment group
mlflow.set_experiment("Credit_Card_Fraud_Detection")

# --- MOCK DATA FOR THE PLUMBING BASELINE ---
# (Replace with your actual X_train, y_train matrix splits later)
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, size=100)
X_test = np.random.rand(20, 5)
y_test = np.random.randint(0, 2, size=20)

# 3. Wrap your execution code inside an explicit MLflow tracking run context
with mlflow.start_run(run_name="Baseline_Random_Forest"):
    
    # Define hyper-parameters
    n_estimators = 100
    max_depth = 10
    
    # Log Hyper-parameters to the dashboard
    mlflow.log_param("algorithm", "Random Forest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("balancing_strategy", "None")
    
    # Train the estimator
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate metrics
    predictions = model.predict(X_test)
    f1_fraud = f1_score(y_test, predictions, pos_label=1, zero_division=0)
    
    # Log performance metrics to the dashboard
    mlflow.log_metric("f1_fraud", float(f1_fraud))
    
    # Log the actual model binary artifact so it can be deployed directly later
    mlflow.sklearn.log_model(model, "model")
    
    print(f"✅ Baseline Run Completed. Logged f1_fraud: {f1_fraud:.4f}")