import pytest
from fastapi.testclient import TestClient
from serving.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_valid_prediction():
    payload = {
        "tenure": 6.0,
        "monthly_charges": 95.0,
        "total_charges": 570.0,
        "support_tickets_30d": 4,
        "contract_type": "Month-to-Month",
        "payment_method": "Electronic Check",
        "paperless_billing": "Yes"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert data["churn_prediction"] in [0, 1]
    assert data["status"] == "success"

def test_invalid_payload_validation():
    payload = {
        "tenure": -5,
        "monthly_charges": 95.0,
        "support_tickets_30d": 4,
        "contract_type": "Month-to-Month",
        "payment_method": "Electronic Check",
        "paperless_billing": "Yes"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
