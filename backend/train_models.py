import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("CycloneGuard_Bay_of_Bengal_India_Enhanced_Real_Dataset.csv")
MODELS_DIR = Path("backend/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_wind_kts"

FEATURES = [
    # Cyclone position
    "latitude",
    "longitude",

    # Time / forecast horizon
    "season",
    "forecast_horizon_h",

    # Cyclone movement
    "storm_speed_kts",
    "storm_direction_deg",
    "movement_distance_from_previous_km",
    "wind_change_from_previous_kts",
    "pressure_change_from_previous_hpa",

    # Current intensity
    "wmo_wind_kts",
    "wmo_pressure_hpa",
    "imd_newdelhi_wind_kts",
    "imd_newdelhi_pressure_hpa",
    "usa_wind_kts",
    "usa_pressure_hpa",

    # Environment
    "weather_elevation_m",
    "temperature_2m_c",
    "relative_humidity_pct",
    "dew_point_2m_c",
    "surface_pressure_hpa",
    "wind_speed_10m_kmh",
    "wind_direction_10m_deg",
    "wind_gusts_10m_kmh",

    # Rainfall
    "precipitation_1h_mm",
    "precipitation_24h_mm",
    "precipitation_72h_mm",
    "precipitation_168h_mm",

    # Land surface
    "soil_moisture_0_7cm_m3m3",

    # Current landfall information
    "landfall_indicator",
    "distance_to_land_source",
]

def train_and_export():
    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    available_features = [f for f in FEATURES if f in df.columns]
    print(f"Available features: {len(available_features)}/{len(FEATURES)}")

    X = df[available_features].copy()
    y = df[TARGET].copy()

    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()
    groups = df.loc[valid_mask, "storm_id"].astype(str)

    # Storm-level Group Shuffle Split (80% train, 20% test)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train storms: {groups.iloc[train_idx].nunique()}, Test storms: {groups.iloc[test_idx].nunique()}")

    # Fit Imputer
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    X_train_proc = imputer.fit_transform(X_train)
    X_test_proc = imputer.transform(X_test)

    # 1. Train Random Forest Baseline
    print("Training Random Forest Baseline...")
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=16,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_proc, y_train)
    y_pred_rf = rf.predict(X_test_proc)

    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_r2 = r2_score(y_test, y_pred_rf)

    # 2. Train Tuned XGBoost General Model
    print("Training Tuned XGBoost Model...")
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=6,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train_proc, y_train)
    y_pred_xgb = xgb.predict(X_test_proc)

    xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    xgb_r2 = r2_score(y_test, y_pred_xgb)

    print(f"Random Forest  -> MAE: {rf_mae:.3f} kts | RMSE: {rf_rmse:.3f} kts | R2: {rf_r2:.4f}")
    print(f"Tuned XGBoost  -> MAE: {xgb_mae:.3f} kts | RMSE: {xgb_rmse:.3f} kts | R2: {xgb_r2:.4f}")

    # Feature importances
    transformed_feature_names = list(imputer.get_feature_names_out(available_features))
    xgb_importances = xgb.feature_importances_
    sorted_feat_indices = np.argsort(xgb_importances)[::-1]
    top_features = [
        {"feature": transformed_feature_names[i], "importance": float(xgb_importances[i])}
        for i in sorted_feat_indices
    ]

    # Error analysis by horizon for general XGBoost
    test_results = df.loc[X_test.index, ["forecast_horizon_h"]].copy()
    test_results["actual"] = y_test.values
    test_results["predicted"] = y_pred_xgb
    test_results["abs_error"] = (test_results["actual"] - test_results["predicted"]).abs()
    test_results["sq_error"] = (test_results["actual"] - test_results["predicted"]) ** 2

    horizon_metrics = {}
    for h, group in test_results.groupby("forecast_horizon_h"):
        horizon_metrics[int(h)] = {
            "samples": int(len(group)),
            "MAE": round(float(group["abs_error"].mean()), 3),
            "RMSE": round(float(np.sqrt(group["sq_error"].mean())), 3)
        }

    # 3. Train Horizon-Specific Models (3h, 6h, 9h, 12h, 18h, 24h, 36h, 48h)
    HORIZONS = [3, 6, 9, 12, 18, 24, 36, 48]
    horizon_models = {}
    horizon_specific_metrics = {}

    for horizon in HORIZONS:
        train_h_mask = (df.loc[X_train.index, "forecast_horizon_h"] == horizon)
        test_h_mask = (df.loc[X_test.index, "forecast_horizon_h"] == horizon)

        X_tr_h = X_train.loc[train_h_mask]
        y_tr_h = y_train.loc[train_h_mask]
        X_te_h = X_test.loc[test_h_mask]
        y_te_h = y_test.loc[test_h_mask]

        if len(X_tr_h) > 20:
            h_imputer = SimpleImputer(strategy="median", add_indicator=True)
            X_tr_h_proc = h_imputer.fit_transform(X_tr_h)
            X_te_h_proc = h_imputer.transform(X_te_h)

            h_model = XGBRegressor(
                n_estimators=250,
                learning_rate=0.035,
                max_depth=5,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )
            h_model.fit(X_tr_h_proc, y_tr_h)
            h_preds = h_model.predict(X_te_h_proc)

            h_mae = mean_absolute_error(y_te_h, h_preds)
            h_rmse = np.sqrt(mean_squared_error(y_te_h, h_preds))
            h_r2 = r2_score(y_te_h, h_preds)

            horizon_models[horizon] = {
                "model": h_model,
                "imputer": h_imputer
            }
            horizon_specific_metrics[int(horizon)] = {
                "samples": int(len(X_te_h)),
                "MAE": round(float(h_mae), 3),
                "RMSE": round(float(h_rmse), 3),
                "R2": round(float(h_r2), 4)
            }
            print(f"Horizon {horizon}h Model -> MAE: {h_mae:.3f} kts | RMSE: {h_rmse:.3f} kts | R2: {h_r2:.4f}")

    # Save artifacts
    print("Saving model artifacts...")
    joblib.dump(imputer, MODELS_DIR / "imputer.joblib")
    joblib.dump(xgb, MODELS_DIR / "cyclone_xgb_model.joblib")
    joblib.dump(rf, MODELS_DIR / "cyclone_rf_model.joblib")
    joblib.dump(horizon_models, MODELS_DIR / "horizon_models.joblib")

    # Save metadata and feature defaults
    # Calculate feature medians and defaults for inference
    feature_defaults = {}
    for col in available_features:
        feature_defaults[col] = float(X[col].median(skipna=True))

    metadata = {
        "target": TARGET,
        "features": available_features,
        "feature_defaults": feature_defaults,
        "metrics": {
            "random_forest": {
                "mae": round(float(rf_mae), 3),
                "rmse": round(float(rf_rmse), 3),
                "r2": round(float(rf_r2), 4)
            },
            "tuned_xgboost": {
                "mae": round(float(xgb_mae), 3),
                "rmse": round(float(xgb_rmse), 3),
                "r2": round(float(xgb_r2), 4)
            },
            "horizon_general_metrics": horizon_metrics,
            "horizon_specific_metrics": horizon_specific_metrics
        },
        "top_features": top_features[:25],
        "training_stats": {
            "total_samples": int(len(X)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "train_storms": int(groups.iloc[train_idx].nunique()),
            "test_storms": int(groups.iloc[test_idx].nunique()),
            "dataset_file": str(DATA_PATH)
        }
    }

    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("All models and metadata successfully trained and exported!")

if __name__ == "__main__":
    train_and_export()
