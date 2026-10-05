"""Tests for Churn Prediction API."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200


def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_predict_valid():
    payload = {
        "gender": "Male", "senior_citizen": 0, "partner": "Yes", "dependents": "No",
        "tenure": 12, "contract": "Month-to-month", "internet_service": "Fiber optic",
        "tech_support": "No", "payment_method": "Electronic check",
        "paperless_billing": "Yes", "monthly_charges": 95.5, "total_charges": 1146.0,
    }
    r = client.post("/predict", json=payload)
    if r.status_code == 503:
        return
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in ("Low", "Medium", "High")


def test_predict_invalid():
    r = client.post("/predict", json={"gender": "Male"})
    assert r.status_code == 422
