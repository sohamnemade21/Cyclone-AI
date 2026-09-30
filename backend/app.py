import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.config import settings
from backend.ml_engine import CycloneMLEngine, get_imd_category

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("cycloneguard.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager that validates model loading at application startup."""
    logger.info(f"Initializing {settings.APP_NAME} v{settings.APP_VERSION} ({settings.ENVIRONMENT})...")
    engine = CycloneMLEngine.get_instance()
    if not engine.is_loaded or engine.final_model is None:
        error_msg = "CRITICAL: Final XGBoost model failed to load during startup."
        logger.critical(error_msg)
        raise RuntimeError(error_msg)
    logger.info(f"Production ML Engine successfully verified. Features: {len(engine.features)}")
    yield
    logger.info("Shutting down CycloneGuard AI service...")

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# GZip compression for large responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS Middleware with explicit origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Request timing and logging middleware
@app.middleware("http")
async def add_process_time_and_log(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(process_time_ms)
    
    if not request.url.path.startswith("/static") and request.url.path not in ("/healthz", "/readyz"):
        logger.info(f"{request.method} {request.url.path} -> Status {response.status_code} ({process_time_ms}ms)")
    return response

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "error": "Internal Server Error", "detail": str(exc)}
    )

engine = CycloneMLEngine.get_instance()

# ------------------------------------------------------------------------------
# Pydantic Schemas with Realistic Physical Validation
# ------------------------------------------------------------------------------
class TelemetryPayload(BaseModel):
    latitude: float = Field(15.5, ge=-90.0, le=90.0, description="Cyclone Latitude (°N)")
    longitude: float = Field(87.5, ge=-180.0, le=180.0, description="Cyclone Longitude (°E)")
    season: int = Field(2024, ge=1900, le=2100, description="Year / Season")
    forecast_horizon_h: int = Field(24, ge=1, le=168, description="Forecast Horizon in hours (3, 6, 9, 12, 18, 24, 36, 48)")
    
    # Motion
    storm_speed_kts: float = Field(12.0, ge=0.0, le=100.0, description="Forward speed in knots")
    storm_direction_deg: float = Field(315.0, ge=0.0, le=360.0, description="Movement direction in degrees (0-360°)")
    movement_distance_from_previous_km: float = Field(65.0, ge=0.0, description="Displacement in last interval (km)")
    wind_change_from_previous_kts: float = Field(5.0, description="Recent 6h wind change (kts)")
    pressure_change_from_previous_hpa: float = Field(-4.0, description="Recent central pressure change (hPa)")
    
    # Wind & Pressure
    wmo_wind_kts: float = Field(55.0, ge=0.0, le=250.0, description="Current WMO sustained wind speed (kts)")
    wmo_pressure_hpa: float = Field(988.0, ge=800.0, le=1050.0, description="Central atmospheric pressure (hPa)")
    imd_newdelhi_wind_kts: Optional[float] = Field(55.0, ge=0.0, le=250.0, description="IMD New Delhi estimated wind")
    imd_newdelhi_pressure_hpa: Optional[float] = Field(988.0, ge=800.0, le=1050.0, description="IMD New Delhi estimated pressure")
    usa_wind_kts: Optional[float] = Field(55.0, ge=0.0, le=250.0, description="JTWC / USA estimated wind")
    usa_pressure_hpa: Optional[float] = Field(988.0, ge=800.0, le=1050.0, description="JTWC / USA estimated pressure")
    
    # Environmental & Atmospheric
    weather_elevation_m: float = Field(0.0, description="Elevation above sea level (m)")
    temperature_2m_c: float = Field(29.2, ge=-10.0, le=50.0, description="2m Sea/Air temperature (°C)")
    relative_humidity_pct: float = Field(84.0, ge=0.0, le=100.0, description="Relative Humidity (%)")
    dew_point_2m_c: float = Field(26.1, ge=-20.0, le=40.0, description="Dew point temperature (°C)")
    surface_pressure_hpa: float = Field(1004.0, ge=800.0, le=1050.0, description="Environmental ambient surface pressure (hPa)")
    wind_speed_10m_kmh: float = Field(45.0, ge=0.0, description="Ambient 10m wind speed (km/h)")
    wind_direction_10m_deg: float = Field(120.0, ge=0.0, le=360.0, description="Ambient wind direction (°)")
    wind_gusts_10m_kmh: float = Field(75.0, ge=0.0, description="Max 10m gusts (km/h)")
    
    # Precipitation
    precipitation_1h_mm: float = Field(12.5, ge=0.0, description="1h accumulated rain (mm)")
    precipitation_24h_mm: float = Field(110.0, ge=0.0, description="24h accumulated rain (mm)")
    precipitation_72h_mm: float = Field(220.0, ge=0.0, description="72h accumulated rain (mm)")
    precipitation_168h_mm: float = Field(310.0, ge=0.0, description="7-day accumulated rain (mm)")
    
    # Land & Soil
    soil_moisture_0_7cm_m3m3: float = Field(0.35, ge=0.0, le=1.0, description="Surface soil moisture (m³/m³)")
    landfall_indicator: int = Field(0, ge=0, le=1, description="Landfall status (0=Ocean, 1=Landfall imminent/active)")
    distance_to_land_source: float = Field(240.0, ge=0.0, description="Distance to nearest coastline (km)")

# ------------------------------------------------------------------------------
# System Health & Probes
# ------------------------------------------------------------------------------
@app.get("/healthz", tags=["System Probes"])
def liveness_probe():
    """Liveness probe for Docker / Kubernetes / Render."""
    return {"status": "alive"}

@app.get("/readyz", tags=["System Probes"])
def readiness_probe():
    """Readiness probe checking ML engine state."""
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="ML engine is not ready")
    return {"status": "ready", "engine_loaded": True}

@app.get("/api/health", tags=["System Probes"])
def get_health():
    """Comprehensive service health report."""
    return {
        "status": "online" if engine.is_loaded else "offline",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "model_used": "Tuned XGBoost Regression",
        "models_loaded": engine.is_loaded,
        "feature_count": len(engine.features),
        "dataset_loaded": engine.historical_df is not None and not engine.historical_df.empty,
        "total_historical_records": len(engine.historical_df) if engine.historical_df is not None else 0
    }

# ------------------------------------------------------------------------------
# REST API Endpoints
# ------------------------------------------------------------------------------
@app.get("/api/features", tags=["Metadata"])
def get_features():
    return {
        "features": engine.features,
        "defaults": engine.feature_defaults,
        "top_features": engine.metadata.get("top_features", [])
    }

@app.get("/api/metrics", tags=["Metadata"])
def get_metrics():
    return {
        "training_stats": engine.metadata.get("training_stats", {}),
        "metrics": engine.metadata.get("metrics", {}),
        "top_features": engine.metadata.get("top_features", [])
    }

@app.get("/api/categories", tags=["Metadata"])
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

@app.post("/api/predict", tags=["Inference"])
def predict_cyclone_intensity(payload: TelemetryPayload):
    try:
        data = payload.model_dump()
        result = engine.predict_single(data)
        return {"success": True, "prediction": result}
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/multi-horizon", tags=["Inference"])
def predict_multi_horizon(payload: TelemetryPayload):
    try:
        data = payload.model_dump()
        result = engine.predict_multi_horizon(data)
        return {"success": True, "forecast": result}
    except Exception as e:
        logger.error(f"Multi-horizon forecast error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/historical/storms", tags=["Historical Data"])
def get_historical_storms():
    storms = engine.get_historical_storms()
    return {"total": len(storms), "storms": storms}

@app.get("/api/historical/storm/{storm_id}", tags=["Historical Data"])
def get_historical_storm_timeline(storm_id: str):
    timeline = engine.get_storm_timeline(storm_id)
    if not timeline:
        raise HTTPException(status_code=404, detail=f"Storm ID '{storm_id}' not found")
    return {"success": True, "timeline": timeline}

# Frontend static files mounting
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    css_dir = FRONTEND_DIR / "css"
    js_dir = FRONTEND_DIR / "js"
    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"status": "CycloneGuard AI API is online. Frontend is deployed on Vercel."}, status_code=200)
