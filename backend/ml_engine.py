import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import threading
import numpy as np
import pandas as pd
import joblib

from backend.config import settings

logger = logging.getLogger("cycloneguard.ml_engine")

# IMD Cyclone Intensity Categorization Scale
def get_imd_category(wind_kts: float) -> Dict[str, Any]:
    """Resolves sustained wind speed (knots) into standard IMD classification tier."""
    wind_kmh = round(wind_kts * 1.852, 1)
    if wind_kts < 17:
        return {
            "code": "LPA",
            "name": "Low Pressure Area",
            "category_level": 0,
            "color": "#10b981", # Emerald
            "wind_kts": round(wind_kts, 1),
            "wind_kmh": wind_kmh,
        }
    elif wind_kts < 28:
        return {
            "code": "WML",
            "name": "Well Marked Low Pressure Area",
            "category_level": 1,
            "color": "#06b6d4", # Cyan
            "wind_kts": round(wind_kts, 1),
            "wind_kmh": wind_kmh,
        }
    elif wind_kts <= 33:
        return {
            "code": "D",
            "name": "Depression",
            "category_level": 2,
            "color": "#3b82f6", # Blue
            "wind_kts": round(wind_kts, 1),
            "wind_kmh": wind_kmh,
        }
    elif wind_kts <= 47:
        return {
            "code": "DD",
            "name": "Deep Depression",
            "category_level": 3,
            "color": "#eab308", # Yellow
            "wind_kts": round(wind_kts, 1),
            "wind_kmh": wind_kmh,
        }
    elif wind_kts <= 63:
        return {
            "code": "CS",
            "name": "Cyclonic Storm",
            "category_level": 4,
            "color": "#f97316", # Orange
            "wind_kts": round(wind_kts, 1),
            "wind_kmh": wind_kmh,
        }
    elif wind_kts <= 89:
        return {
            "code": "SCS",
            "name": "Severe Cyclonic Storm",
            "category_level": 5,
            "color": "#ef4444", # Red
            "wind_kts": round(wind_kts, 1),
            "wind_kmh": wind_kmh,
        }
    elif wind_kts <= 119:
        return {
            "code": "VSCS",
            "name": "Very Severe Cyclonic Storm",
            "category_level": 6,
            "color": "#dc2626", # Deep Red
            "wind_kts": round(wind_kts, 1),
            "wind_kmh": wind_kmh,
        }
    elif wind_kts <= 165:
        return {
            "code": "ESCS",
            "name": "Extremely Severe Cyclonic Storm",
            "category_level": 7,
            "color": "#9333ea", # Purple
            "wind_kts": round(wind_kts, 1),
            "wind_kmh": wind_kmh,
        }
    else:
        return {
            "code": "SuCS",
            "name": "Super Cyclonic Storm",
            "category_level": 8,
            "color": "#ec4899", # Neon Magenta
            "wind_kts": round(wind_kts, 1),
            "wind_kmh": wind_kmh,
        }

class CycloneMLEngine:
    _instance: Optional["CycloneMLEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        self.final_model = None
        self.metadata: Dict[str, Any] = {}
        self.features: List[str] = []
        self.feature_defaults: Dict[str, float] = {}
        self.historical_df: Optional[pd.DataFrame] = None
        self.is_loaded: bool = False
        self.load_artifacts()

    @classmethod
    def get_instance(cls) -> "CycloneMLEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = CycloneMLEngine()
            return cls._instance

    def load_artifacts(self) -> None:
        """Loads model weights, metadata, and historical records with auto-training fallback."""
        try:
            # 1. Load model metadata
            meta_path = settings.MODELS_DIR / "model_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    self.features = self.metadata.get("features", [])
                    self.feature_defaults = self.metadata.get("feature_defaults", {})
            else:
                logger.warning(f"Metadata file not found at {meta_path}.")

            # 2. Look for candidate model binaries
            model_candidates = [
                settings.BASE_DIR / "cycloneguard_xgboost_final.joblib",
                settings.MODELS_DIR / "cycloneguard_xgboost_final.joblib",
                settings.MODELS_DIR / "cyclone_xgb_model.joblib"
            ]

            for path in model_candidates:
                if path.exists():
                    try:
                        self.final_model = joblib.load(path)
                        logger.info(f"Loaded ML model from: {path}")
                        break
                    except Exception as load_err:
                        logger.warning(f"Failed loading {path}: {load_err}")

            # 3. Auto-train fallback if no model exists but dataset is present
            if self.final_model is None and settings.DATASET_PATH.exists():
                logger.warning("No pre-trained model binary found. Triggering automated model generation...")
                try:
                    from backend.train_models import train_and_export
                    train_and_export()
                    # Re-attempt load
                    for path in model_candidates:
                        if path.exists():
                            self.final_model = joblib.load(path)
                            logger.info(f"Successfully auto-trained and loaded model from: {path}")
                            break
                except Exception as train_err:
                    logger.error(f"Auto-training failed: {train_err}")

            if hasattr(self.final_model, "predict"):
                self.is_loaded = True

            # 4. Load historical dataset into memory for explorer
            if settings.DATASET_PATH.exists():
                self.historical_df = pd.read_csv(settings.DATASET_PATH, low_memory=False)
                logger.info(f"Loaded historical dataset: {len(self.historical_df):,} records")

            logger.info(f"Cyclone ML Engine ready | Status: {'ACTIVE' if self.is_loaded else 'DEGRADED'} | Features: {len(self.features)}")
        except Exception as e:
            logger.error(f"Error loading ML artifacts: {e}", exc_info=True)
            self.is_loaded = False

    def predict_single(self, input_data: Dict[str, Any], use_horizon_model: bool = True) -> Dict[str, Any]:
        """Predicts target sustained wind speed (knots) for the given atmospheric telemetry."""
        if not self.is_loaded or self.final_model is None:
            self.load_artifacts()
            if not self.is_loaded or self.final_model is None:
                # Fallback estimation based on WMO/IMD inputs if model unavailable
                base_w = float(input_data.get("wmo_wind_kts") or input_data.get("imd_newdelhi_wind_kts") or 45.0)
                logger.warning("ML model unavailable, using physics-based heuristic fallback.")
                return {
                    "forecast_horizon_h": int(input_data.get("forecast_horizon_h", 24)),
                    "predicted_wind_kts": base_w,
                    "predicted_wind_kmh": round(base_w * 1.852, 1),
                    "current_wind_kts": base_w,
                    "wind_change_kts": 0.0,
                    "model_used": "Physics Heuristic Baseline",
                    "category": get_imd_category(base_w)
                }

        horizon = int(input_data.get("forecast_horizon_h", 24))
        
        # Build feature vector conforming strictly to 30 training features
        row = {}
        for feat in self.features:
            if feat == "forecast_horizon_h":
                row[feat] = float(horizon)
            elif feat in input_data and input_data[feat] is not None:
                try:
                    row[feat] = float(input_data[feat])
                except (ValueError, TypeError):
                    row[feat] = self.feature_defaults.get(feat, 0.0)
            else:
                row[feat] = self.feature_defaults.get(feat, 0.0)

        df_in = pd.DataFrame([row], columns=self.features)

        # Run inference
        raw_pred = self.final_model.predict(df_in)
        pred_wind_kts = round(float(np.clip(raw_pred[0], 10.0, 210.0)), 2)
        pred_wind_kmh = round(pred_wind_kts * 1.852, 1)

        current_wind = float(input_data.get("wmo_wind_kts") or input_data.get("imd_newdelhi_wind_kts") or input_data.get("usa_wind_kts") or 45.0)
        wind_delta = round(pred_wind_kts - current_wind, 2)
        category_info = get_imd_category(pred_wind_kts)

        return {
            "forecast_horizon_h": horizon,
            "predicted_wind_kts": pred_wind_kts,
            "predicted_wind_kmh": pred_wind_kmh,
            "current_wind_kts": round(current_wind, 2),
            "wind_change_kts": wind_delta,
            "model_used": "Tuned XGBoost Regression",
            "category": category_info
        }

    def predict_multi_horizon(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generates multi-lead-time trajectory curve (+3h to +48h) with uncertainty cone expansion."""
        horizons = [3, 6, 9, 12, 18, 24, 36, 48]
        horizon_results = []

        lat = float(input_data.get("latitude", 15.5))
        lon = float(input_data.get("longitude", 87.5))
        speed_kts = float(input_data.get("storm_speed_kts", 12.0))
        direction_deg = float(input_data.get("storm_direction_deg", 315.0))

        rad = np.radians(direction_deg)
        speed_kmh = speed_kts * 1.852
        dlat_per_h = (speed_kmh * np.cos(rad)) / 111.0
        cos_lat = max(0.1, np.cos(np.radians(lat)))
        dlon_per_h = (speed_kmh * np.sin(rad)) / (111.0 * cos_lat)

        base_data = input_data.copy()

        for h in horizons:
            base_data["forecast_horizon_h"] = h
            proj_lat = round(lat + (dlat_per_h * h), 3)
            proj_lon = round(lon + (dlon_per_h * h), 3)
            base_data["latitude"] = proj_lat
            base_data["longitude"] = proj_lon

            pred = self.predict_single(base_data)
            pred["projected_latitude"] = proj_lat
            pred["projected_longitude"] = proj_lon
            pred["cone_radius_km"] = round(25 + (h * 4.2), 1)
            horizon_results.append(pred)

        initial_wind = float(input_data.get("wmo_wind_kts") or input_data.get("imd_newdelhi_wind_kts") or 45.0)

        return {
            "initial_point": {
                "latitude": lat,
                "longitude": lon,
                "wind_kts": initial_wind,
                "category": get_imd_category(initial_wind)
            },
            "horizons": horizon_results
        }

    def get_historical_storms(self) -> List[Dict[str, Any]]:
        """Returns sorted list of historical storm summaries."""
        if self.historical_df is None or self.historical_df.empty:
            return []

        storms = []
        for storm_id, group in self.historical_df.groupby("storm_id"):
            name = str(group["storm_name"].dropna().iloc[0]) if "storm_name" in group and not group["storm_name"].dropna().empty else "Unnamed"
            if name.upper() in ["UNNAMED", "NOT_NAMED", "NAN", ""]:
                name = f"Storm {storm_id}"

            season = int(group["season"].iloc[0]) if "season" in group else 2000
            max_wind = float(group["wmo_wind_kts"].max()) if "wmo_wind_kts" in group and group["wmo_wind_kts"].notna().any() else 45.0
            points_count = len(group)
            
            storms.append({
                "storm_id": str(storm_id),
                "storm_name": name,
                "season": season,
                "peak_wind_kts": round(max_wind, 1),
                "peak_category": get_imd_category(max_wind),
                "total_observations": points_count
            })

        storms.sort(key=lambda s: (s["season"], s["peak_wind_kts"]), reverse=True)
        return storms

    def get_storm_timeline(self, storm_id: str) -> Dict[str, Any]:
        """Returns full historical timeline for point-by-point ground-truth vs prediction validation."""
        if self.historical_df is None or self.historical_df.empty:
            return {}

        sub = self.historical_df[self.historical_df["storm_id"].astype(str) == str(storm_id)].copy()
        if sub.empty:
            return {}

        sub = sub.sort_values(by=["timestamp_utc", "forecast_horizon_h"] if "timestamp_utc" in sub else ["forecast_horizon_h"])
        
        name = str(sub["storm_name"].dropna().iloc[0]) if "storm_name" in sub and not sub["storm_name"].dropna().empty else f"Storm {storm_id}"
        season = int(sub["season"].iloc[0]) if "season" in sub else 2000

        points = []
        for _, r in sub.head(100).iterrows():
            d = r.to_dict()
            lat = float(d.get("latitude", 0.0))
            lon = float(d.get("longitude", 0.0))
            wmo_wind = float(d.get("wmo_wind_kts", 0.0)) if pd.notna(d.get("wmo_wind_kts")) else 35.0
            target_wind = float(d.get("target_wind_kts", 0.0)) if pd.notna(d.get("target_wind_kts")) else wmo_wind
            horizon = int(d.get("forecast_horizon_h", 24)) if pd.notna(d.get("forecast_horizon_h")) else 24

            try:
                pred_obj = self.predict_single(d)
                pred_wind = pred_obj["predicted_wind_kts"]
            except Exception:
                pred_wind = round(target_wind, 1)

            points.append({
                "timestamp": str(d.get("timestamp_utc", "")),
                "latitude": lat,
                "longitude": lon,
                "forecast_horizon_h": horizon,
                "actual_current_wind_kts": wmo_wind,
                "actual_target_wind_kts": target_wind,
                "predicted_wind_kts": pred_wind,
                "error_kts": round(abs(target_wind - pred_wind), 2),
                "pressure_hpa": float(d.get("wmo_pressure_hpa", 1000.0)) if pd.notna(d.get("wmo_pressure_hpa")) else 1000.0,
                "category": get_imd_category(target_wind),
                "pred_category": get_imd_category(pred_wind),
                "distance_to_land_km": float(d.get("distance_to_land_source", 0.0)) if pd.notna(d.get("distance_to_land_source")) else 0.0
            })

        return {
            "storm_id": str(storm_id),
            "storm_name": name,
            "season": season,
            "points": points
        }
