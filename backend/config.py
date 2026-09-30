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

    # CORS configuration
    CORS_ORIGINS: List[str] = [
        origin.strip() 
        for origin in os.getenv("CORS_ORIGINS", "*").split(",") 
        if origin.strip()
    ]

    # Directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    BACKEND_DIR: Path = BASE_DIR / "backend"
    MODELS_DIR: Path = BACKEND_DIR / "models"
    FRONTEND_DIR: Path = BASE_DIR / "frontend"
    DATASET_PATH: Path = BASE_DIR / "CycloneGuard_Bay_of_Bengal_India_Enhanced_Real_Dataset.csv"

settings = Settings()
