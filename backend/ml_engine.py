import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = Path(__file__).parent / "models"
ROOT_DIR = Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "CycloneGuard_Bay_of_Bengal_India_Enhanced_Real_Dataset.csv"

# IMD Cyclone Intensity Categorization Scale
def get_imd_category(wind_kts: float) -> Dict[str, Any]:
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
    _instance = None

    def __init__(self):
        self.final_model = None
        self.metadata = {}
        self.features = []
        self.feature_defaults = {}
        self.historical_df = None
        self.is_loaded = False
        self.load_artifacts()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = CycloneMLEngine()
        return cls._instance

    def load_artifacts(self):
        try:
            # Look for model metadata
            meta_path = MODELS_DIR / "model_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    self.features = self.metadata.get("features", [])
                    self.feature_defaults = self.metadata.get("feature_defaults", {})

            # Load the exact final saved model cycloneguard_xgboost_final.joblib
            model_candidates = [
                ROOT_DIR / "cycloneguard_xgboost_final.joblib",
                MODELS_DIR / "cycloneguard_xgboost_final.joblib",
                MODELS_DIR / "cyclone_xgb_model.joblib"
            ]

            for path in model_candidates:
                if path.exists():
                    self.final_model = joblib.load(path)
                    print(f"Loaded final model from: {path}")
                    break

            # If the loaded object is not a pipeline, check if imputer is needed
            if hasattr(self.final_model, "predict"):
                self.is_loaded = True

            # Load historical dataset
            if DATA_PATH.exists():
                self.historical_df = pd.read_csv(DATA_PATH, low_memory=False)

            print(f"Cyclone ML Engine loaded: {self.is_loaded}, Total Features: {len(self.features)}")
        except Exception as e:
            print(f"Error loading ML artifacts: {e}")
            self.is_loaded = False

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict target_wind_kts for a specific input dictionary using cycloneguard_xgboost_final.joblib"""
        if not self.is_loaded or self.final_model is None:
            self.load_artifacts()
            if not self.is_loaded or self.final_model is None:
                raise RuntimeError("cycloneguard_xgboost_final.joblib is not loaded.")

        horizon = int(input_data.get("forecast_horizon_h", 24))
        
        # Build feature row matching the 30 features
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

        # Predict target_wind_kts directly using the final saved model
        raw_pred = self.final_model.predict(df_in)
        pred_wind_kts = round(float(raw_pred[0]), 2)
        pred_wind_kmh = round(pred_wind_kts * 1.852, 1)

        # Current wind speed comparison
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
        """Generates forecast curve across all 8 horizons (3, 6, 9, 12, 18, 24, 36, 48h) using cycloneguard_xgboost_final.joblib"""
        horizons = [3, 6, 9, 12, 18, 24, 36, 48]
        horizon_results = []

        lat = float(input_data.get("latitude", 15.5))
        lon = float(input_data.get("longitude", 87.5))
        speed_kts = float(input_data.get("storm_speed_kts", 12.0))
        direction_deg = float(input_data.get("storm_direction_deg", 315.0))

        rad = np.radians(direction_deg)
        speed_kmh = speed_kts * 1.852
        dlat_per_h = (speed_kmh * np.cos(rad)) / 111.0
        dlon_per_h = (speed_kmh * np.sin(rad)) / (111.0 * np.cos(np.radians(lat)))

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
        """Returns list of unique storms from the dataset"""
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
        """Returns the full telemetry and ML prediction trajectory for a selected historical storm"""
        if self.historical_df is None:
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
