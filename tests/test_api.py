from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.main import app

client = TestClient(app)

def test_predict_endpoint_success():
    # Valid request with exactly 10 features
    payload = {
        "features": [0.5, -1.2, 3.4, 0.0, 1.1, -0.9, 2.2, 0.1, -0.5, 1.0]
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "class" in data

def test_predict_endpoint_invalid_features():
    # Invalid request (only 9 features)
    payload = {
        "features": [0.5, -1.2, 3.4, 0.0, 1.1, -0.9, 2.2, 0.1, -0.5]
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 400
    assert "Expected exactly 10 features" in response.json()["detail"]
