from fastapi import FastAPI

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Production-grade REST engine serving Random Forest and XGBoost inference microservices.",
    version="1.0.0"
)

@app.get("/health", tags=["System Health"])
def health_check():
    """
    Validates server availability and operational heartbeat status.
    """
    return {"status": "ok"}