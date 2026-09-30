import os
from typing import List
from pathlib import Path

class Settings:
    # App metadata
    APP_NAME: str = "CycloneGuard AI"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "Operational Machine Learning Engine for Cyclone Intensity, Trajectory & Rapid Intensification Forecasting"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production").lower()
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    RELOAD: bool = os.getenv("RELOAD", "false").lower() in ("true", "1", "yes")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info").lower()

    # Directories and Canonical Model Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    BACKEND_DIR: Path = BASE_DIR / "backend"
    MODELS_DIR: Path = BACKEND_DIR / "models"
    DATASET_PATH: Path = BASE_DIR / "CycloneGuard_Bay_of_Bengal_India_Enhanced_Real_Dataset.csv"

    # Resolve Model Path
    _env_model_path = os.getenv("MODEL_PATH")
    if _env_model_path:
        MODEL_PATH: Path = Path(_env_model_path)
    elif (MODELS_DIR / "cycloneguard_xgboost_final.joblib").exists():
        MODEL_PATH: Path = MODELS_DIR / "cycloneguard_xgboost_final.joblib"
    elif (BASE_DIR / "cycloneguard_xgboost_final.joblib").exists():
        MODEL_PATH: Path = BASE_DIR / "cycloneguard_xgboost_final.joblib"
    else:
        MODEL_PATH: Path = MODELS_DIR / "cycloneguard_xgboost_final.joblib"

    # Resolve Metadata Path
    _env_meta_path = os.getenv("METADATA_PATH")
    if _env_meta_path:
        METADATA_PATH: Path = Path(_env_meta_path)
    else:
        METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

    # CORS configuration
    _cors_raw = os.getenv("CORS_ORIGINS", "")
    if _cors_raw.strip():
        CORS_ORIGINS: List[str] = [
            origin.strip() 
            for origin in _cors_raw.split(",") 
            if origin.strip()
        ]
    else:
        CORS_ORIGINS: List[str] = [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8000",
            "http://localhost:10000",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:10000",
            "https://cycloneguard-ai.vercel.app"
        ]

settings = Settings()
