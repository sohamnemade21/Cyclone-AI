import pytest
from backend.ml_engine import CycloneMLEngine, get_imd_category

def test_imd_category_mapping():
    # LPA
    lpa = get_imd_category(15.0)
    assert lpa["code"] == "LPA"
    
    # Depression
    d = get_imd_category(30.0)
    assert d["code"] == "D"
    
    # Cyclonic Storm
    cs = get_imd_category(55.0)
    assert cs["code"] == "CS"
    
    # Very Severe Cyclonic Storm
    vscs = get_imd_category(100.0)
    assert vscs["code"] == "VSCS"
    
    # Super Cyclonic Storm
    sucs = get_imd_category(170.0)
    assert sucs["code"] == "SuCS"

def test_ml_engine_singleton():
    engine1 = CycloneMLEngine.get_instance()
    engine2 = CycloneMLEngine.get_instance()
    assert engine1 is engine2

def test_ml_engine_predict_bounds():
    engine = CycloneMLEngine.get_instance()
    dummy_input = {
        "latitude": 15.0,
        "longitude": 85.0,
        "forecast_horizon_h": 24,
        "wmo_wind_kts": 60.0
    }
    result = engine.predict_single(dummy_input)
    assert "predicted_wind_kts" in result
    assert 10.0 <= result["predicted_wind_kts"] <= 250.0
