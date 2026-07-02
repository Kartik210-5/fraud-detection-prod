# test_api.py
import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture(scope="module")
def client():
    """Context-managed test client that explicitly triggers FastAPI lifespan startup/shutdown events."""
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Verifies that the /health gateway is alive and reporting cached model schemas."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Healthy"
    # Now that the lifespan runs, this count will be greater than 0
    assert data["models_cached_count"] > 0


def test_predict_endpoint_valid_payload(client):
    """Verifies that the scoring engine accurately parses a standard input transaction matrix."""
    # Build a clean mock transaction payload matching your Pydantic schema precisely
    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload["Amount"] = 100.00
    
    # Send the post request over the internal memory stack
    params = {"algo": "Random Forest", "strategy": "None", "threshold": 0.50}
    response = client.post("/predict", json=payload, params=params)
    
    if response.status_code != 200:
        print(f"\n❌ Server Error Detail: {response.json()}")
        
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "fraud_probability" in data
    assert "model_used" in data
    
    # Enforce that the output label strictly matches the production contract enum
    assert data["prediction"] in ["Fraud", "Legit"]


def test_debug_paths():
    """A clean inspection helper to verify workspace asset tracking."""
    from model_io import ALGORITHMS, STRATEGIES, get_model_path
    print("\n🔍 Checking expected model paths:")
    for algo in ALGORITHMS:
        for strategy in STRATEGIES:
            p = get_model_path(algo, strategy)
            print(f"Path: {p} | Exists: {p.exists()}")