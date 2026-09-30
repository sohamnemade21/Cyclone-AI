import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_liveness_probe():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}

def test_readiness_probe():
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "engine_loaded": True}

def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["models_loaded"] is True
    assert data["model_used"] == "Tuned XGBoost Regression"
    assert data["feature_count"] == 30

def test_get_features():
    response = client.get("/api/features")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert len(data["features"]) == 30
    assert "defaults" in data

def test_get_metrics():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    metrics = data["metrics"]["tuned_xgboost"]
    assert metrics["mae"] == 4.812
    assert metrics["rmse"] == 8.191
    assert metrics["r2"] == 0.7581

def test_get_categories():
    response = client.get("/api/categories")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert len(data["categories"]) == 9

def test_predict_single_valid():
    payload = {
        "latitude": 16.2,
        "longitude": 86.8,
        "season": 2024,
        "forecast_horizon_h": 24,
        "storm_speed_kts": 14.0,
        "storm_direction_deg": 340.0,
        "wmo_wind_kts": 125.0,
        "wmo_pressure_hpa": 925.0,
        "temperature_2m_c": 31.2,
        "relative_humidity_pct": 89.0,
        "precipitation_24h_mm": 240.0,
        "distance_to_land_source": 320.0
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "prediction" in data
    pred = data["prediction"]
    assert pred["model_used"] == "Tuned XGBoost Regression"
    assert "predicted_wind_kts" in pred
    assert "predicted_wind_kmh" in pred
    assert "category" in pred
    assert pred["predicted_wind_kts"] > 0

def test_predict_all_lead_horizons():
    horizons = [3, 6, 9, 12, 18, 24, 36, 48]
    for h in horizons:
        payload = {
            "latitude": 16.2,
            "longitude": 86.8,
            "season": 2024,
            "forecast_horizon_h": h,
            "wmo_wind_kts": 75.0,
            "wmo_pressure_hpa": 970.0
        }
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["prediction"]["forecast_horizon_h"] == h
        assert data["prediction"]["model_used"] == "Tuned XGBoost Regression"
        assert data["prediction"]["predicted_wind_kts"] > 0

def test_predict_single_invalid_latitude():
    # Latitude > 90 should trigger validation error (HTTP 422)
    payload = {
        "latitude": 150.0,
        "longitude": 86.8,
        "season": 2024,
        "forecast_horizon_h": 24
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422

def test_predict_multi_horizon():
    payload = {
        "latitude": 16.2,
        "longitude": 86.8,
        "season": 2024,
        "storm_speed_kts": 14.0,
        "storm_direction_deg": 340.0,
        "wmo_wind_kts": 100.0,
        "wmo_pressure_hpa": 940.0
    }
    response = client.post("/api/predict/multi-horizon", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "forecast" in data
    horizons = data["forecast"]["horizons"]
    assert len(horizons) == 8  # 3, 6, 9, 12, 18, 24, 36, 48
    for pred in horizons:
        assert pred["model_used"] == "Tuned XGBoost Regression"
        assert pred["forecast_horizon_h"] in [3, 6, 9, 12, 18, 24, 36, 48]
        assert "cone_radius_km" in pred

def test_get_historical_storms():
    response = client.get("/api/historical/storms")
    assert response.status_code == 200
    data = response.json()
    assert "storms" in data

def test_get_historical_storm_not_found():
    response = client.get("/api/historical/storm/invalid_id_9999999")
    assert response.status_code == 404
