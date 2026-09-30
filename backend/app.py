from pathlib import Path
import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.ml_engine import CycloneMLEngine, get_imd_category

app = FastAPI(
    title="CycloneGuard AI API",
    description="Operational Machine Learning Engine for Cyclone Intensity, Trajectory & Rapid Intensification Forecasting in Bay of Bengal / Indian Ocean",
    version="2.0.0"
)

# Enable CORS for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = CycloneMLEngine.get_instance()

class TelemetryPayload(BaseModel):
    latitude: float = Field(15.5, description="Current Cyclone Latitude (°N)")
    longitude: float = Field(87.5, description="Current Cyclone Longitude (°E)")
    season: int = Field(2024, description="Year / Season")
    forecast_horizon_h: int = Field(24, description="Forecast Horizon in hours (3, 6, 9, 12, 18, 24, 36, 48)")
    
    # Motion
    storm_speed_kts: float = Field(12.0, description="Forward speed in knots")
    storm_direction_deg: float = Field(315.0, description="Movement direction in degrees (0-360°)")
    movement_distance_from_previous_km: float = Field(65.0, description="Displacement in last observation interval")
    wind_change_from_previous_kts: float = Field(5.0, description="Recent 6h wind change")
    pressure_change_from_previous_hpa: float = Field(-4.0, description="Recent central pressure change")
    
    # Wind & Pressure
    wmo_wind_kts: float = Field(55.0, description="Current WMO sustained wind speed (kts)")
    wmo_pressure_hpa: float = Field(988.0, description="Central atmospheric pressure (hPa)")
    imd_newdelhi_wind_kts: Optional[float] = Field(55.0, description="IMD New Delhi estimated wind")
    imd_newdelhi_pressure_hpa: Optional[float] = Field(988.0, description="IMD New Delhi estimated pressure")
    usa_wind_kts: Optional[float] = Field(55.0, description="JTWC / USA estimated wind")
    usa_pressure_hpa: Optional[float] = Field(988.0, description="JTWC / USA estimated pressure")
    
    # Environmental & Atmospheric
    weather_elevation_m: float = Field(0.0, description="Elevation above sea level (m)")
    temperature_2m_c: float = Field(29.2, description="2m Sea/Air temperature (°C)")
    relative_humidity_pct: float = Field(84.0, description="Relative Humidity (%)")
    dew_point_2m_c: float = Field(26.1, description="Dew point temperature (°C)")
    surface_pressure_hpa: float = Field(1004.0, description="Environmental ambient surface pressure (hPa)")
    wind_speed_10m_kmh: float = Field(45.0, description="Ambient 10m wind speed (km/h)")
    wind_direction_10m_deg: float = Field(120.0, description="Ambient wind direction (°)")
    wind_gusts_10m_kmh: float = Field(75.0, description="Max 10m gusts (km/h)")
    
    # Precipitation
    precipitation_1h_mm: float = Field(12.5, description="1h accumulated rain (mm)")
    precipitation_24h_mm: float = Field(110.0, description="24h accumulated rain (mm)")
    precipitation_72h_mm: float = Field(220.0, description="72h accumulated rain (mm)")
    precipitation_168h_mm: float = Field(310.0, description="7-day accumulated rain (mm)")
    
    # Land & Soil
    soil_moisture_0_7cm_m3m3: float = Field(0.35, description="Surface soil moisture (m³/m³)")
    landfall_indicator: int = Field(0, description="Landfall status (0=Ocean, 1=Landfall imminent/active)")
    distance_to_land_source: float = Field(240.0, description="Distance to nearest coastline (km)")

@app.get("/api/health")
def get_health():
    return {
        "status": "online",
        "service": "CycloneGuard AI Engine",
        "model_used": "Tuned XGBoost Regression",
        "model_file": "cycloneguard_xgboost_final.joblib",
        "models_loaded": engine.is_loaded,
        "feature_count": len(engine.features),
        "dataset_loaded": engine.historical_df is not None and not engine.historical_df.empty,
        "total_historical_records": len(engine.historical_df) if engine.historical_df is not None else 0
    }

@app.get("/api/features")
def get_features():
    return {
        "features": engine.features,
        "defaults": engine.feature_defaults,
        "top_features": engine.metadata.get("top_features", [])
    }

@app.get("/api/metrics")
def get_metrics():
    return {
        "training_stats": engine.metadata.get("training_stats", {}),
        "metrics": engine.metadata.get("metrics", {}),
        "top_features": engine.metadata.get("top_features", [])
    }

@app.get("/api/categories")
def get_categories():
    categories = [
        get_imd_category(10),
        get_imd_category(22),
        get_imd_category(30),
        get_imd_category(40),
        get_imd_category(55),
        get_imd_category(75),
        get_imd_category(100),
        get_imd_category(130),
        get_imd_category(170)
    ]
    return {"categories": categories}

@app.post("/api/predict")
def predict_cyclone_intensity(payload: TelemetryPayload):
    try:
        data = payload.model_dump()
        result = engine.predict_single(data, use_horizon_model=True)
        return {"success": True, "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/multi-horizon")
def predict_multi_horizon(payload: TelemetryPayload):
    try:
        data = payload.model_dump()
        result = engine.predict_multi_horizon(data)
        return {"success": True, "forecast": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/historical/storms")
def get_historical_storms():
    storms = engine.get_historical_storms()
    return {"total": len(storms), "storms": storms}

@app.get("/api/historical/storm/{storm_id}")
def get_historical_storm_timeline(storm_id: str):
    timeline = engine.get_storm_timeline(storm_id)
    if not timeline:
        raise HTTPException(status_code=404, detail="Storm ID not found")
    return {"success": True, "timeline": timeline}

# Serve static frontend files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend_index():
        return FileResponse(FRONTEND_DIR / "index.html")
