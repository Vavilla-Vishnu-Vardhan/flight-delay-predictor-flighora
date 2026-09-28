import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "Predictive Flight Delay Analysis System API"

def test_api_predict():
    payload = {
        "flight_id": "FL-9999",
        "airline": "Delta",
        "origin": "JFK",
        "destination": "LAX",
        "departure_hour": 18,
        "day_of_week": 4,
        "month": 7,
        "distance_miles": 2475.0,
        "temperature_c": 18.0,
        "wind_speed_kmh": 40.0,
        "visibility_km": 2.0,
        "precipitation_mm": 10.0,
        "pressure_hpa": 1005.0,
        "humidity_pct": 90.0,
        "congestion_index": 4.5,
        "airline_delay_rate": 0.22
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "delay_probability" in data
    assert "predicted_delay_minutes" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]

def test_api_metrics():
    response = client.get("/api/v1/metrics")
    assert response.status_code in [200, 404]
