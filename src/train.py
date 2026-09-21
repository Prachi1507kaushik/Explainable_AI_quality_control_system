"""
PHASE 8 - CHRONOLOGICAL TRAIN/VALIDATION/TEST SPLIT
PHASE 9 - MODEL TRAINING AND COMPARISON

Trains on the 10-station recommended subset from
outputs/metrics/station_coverage.csv (see src/station_coverage_analysis.py
for how/why those 10 were chosen). Training on all 84 stations at once is
left for a later scale-up once this smaller, higher-quality subset proves
the pipeline works end-to-end.

SPLIT METHOD (never randomly shuffled):
  - train      : datetime <  config.TRAIN_END_DATE
  - validation : config.TRAIN_END_DATE <= datetime < config.VALID_END_DATE
  - test       : datetime >= config.VALID_END_DATE
  This is applied globally by calendar date, not per-station, so every
  station contributes to train/valid/test using the SAME real-world time
  boundaries - a model is never validated/tested on a period that overlaps
  its own training window for any station.

MODELS COMPARED:
  1. Persistence baseline - predicts target_pm25_6h = current pm2_5 (i.e.
     "assume no change in the next 6 hours"), using the median-imputed
     pm2_5 value (same imputation used for the other models) so every row
     produces a prediction. No fitting involved; this is the minimum bar
     any real model must beat to be worth using.
  2. Linear Regression
  3. Random Forest
  4. XGBoost

MISSING FEATURE VALUES:
  Lag/rolling features can be NaN (e.g. near the start of a station's
  series, or where the underlying pollutant reading was missing beyond the
  short-gap fill limit in preprocessing.py). A median imputer is fit ONLY
  on the training split and applied unchanged to validation/test - fitting
  on the full dataset would leak validation/test distribution information
  into training.

MODEL SELECTION:
  All 4 models' metrics (MAE, RMSE, R2) are computed on train and
  validation. The best model is selected by validation MAE (lower is
  better) - not by test performance, which is only computed once, after
  selection, for the winner.

NOTHING IN THIS SCRIPT'S METRICS IS FABRICATED - every number printed and
saved comes directly from actually fitting these models on the real,
uploaded CPCB data and evaluating them.
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

COVERAGE_REPORT_PATH = os.path.join(config.OUTPUTS_DIR, "metrics", "station_coverage.csv")
MAX_MISSING_PM25_PCT = 15.0
MIN_CONTINUITY_PCT = 85.0
N_STATIONS = 10

NON_FEATURE_COLS = {
    "station_id", "datetime", "station_name", "city", "state", "status",
    "aqi_bucket", config.TARGET_COLUMN, "target_horizon_valid",
    "negative_value_flag",
}


def get_recommended_stations():
    coverage = pd.read_csv(COVERAGE_REPORT_PATH)
    eligible = coverage[
        (coverage["missing_pm25_pct"] <= MAX_MISSING_PM25_PCT) &
        (coverage["continuity_pct"] >= MIN_CONTINUITY_PCT)
    ].sort_values("usable_training_rows", ascending=False)
    return eligible.head(N_STATIONS)["station_id"].tolist()


def load_modeling_data(station_ids):
    """Reads training_data.csv in chunks, keeping only rows for the
    recommended stations and only rows with a valid target - this keeps
    memory use manageable for the ~2GB source file."""
    chunks = []
    reader = pd.read_csv(config.TRAINING_DATA, chunksize=300_000, low_memory=False,
                         parse_dates=["datetime"])
    for chunk in reader:
        sub = chunk[chunk["station_id"].isin(station_ids)]
        sub = sub[sub[config.TARGET_COLUMN].notna() & (sub["target_horizon_valid"] == True)]
        if len(sub) > 0:
            chunks.append(sub)
    df = pd.concat(chunks, ignore_index=True)
    return df


def chronological_split(df):
    train = df[df["datetime"] < config.TRAIN_END_DATE].copy()
    valid = df[(df["datetime"] >= config.TRAIN_END_DATE) &
               (df["datetime"] < config.VALID_END_DATE)].copy()
    test = df[df["datetime"] >= config.VALID_END_DATE].copy()
    return train, valid, test


def get_feature_columns(df):
    was_filled_cols = {c for c in df.columns if c.endswith("_was_filled")}
    feature_cols = [c for c in df.columns
                     if c not in NON_FEATURE_COLS and c not in was_filled_cols]
    return feature_cols


def compute_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def persistence_baseline_predict(df, feature_cols, X_imputed):
    """Predicts target_pm25_6h = current pm2_5 reading (no-change assumption).
    Uses the SAME median-imputed pm2_5 column as the other models (rather
    than the raw column, which can still contain NaN beyond the short-gap
    fill limit applied in preprocessing.py) so the baseline is evaluable
    on every row and compared on equal footing."""
    pm25_idx = feature_cols.index("pm2_5")
    return X_imputed[:, pm25_idx]


def run():
    print("Step 1: identifying recommended stations...")
    station_ids = get_recommended_stations()
    print(f"Using {len(station_ids)} stations: {station_ids}")

    print("\nStep 2: loading modeling data (chunked read of training_data.csv)...")
    df = load_modeling_data(station_ids)
    print(f"Loaded {len(df):,} rows with a valid target, across {df['station_id'].nunique()} stations")

    print("\nStep 3: chronological split...")
    train_df, valid_df, test_df = chronological_split(df)
    print(f"Train : {len(train_df):,} rows | {train_df['datetime'].min()} to {train_df['datetime'].max()}")
    print(f"Valid : {len(valid_df):,} rows | {valid_df['datetime'].min()} to {valid_df['datetime'].max()}")
    print(f"Test  : {len(test_df):,} rows | {test_df['datetime'].min()} to {test_df['datetime'].max()}")

    if len(train_df) == 0 or len(valid_df) == 0 or len(test_df) == 0:
        raise ValueError("One of the splits is empty - check TRAIN_END_DATE/VALID_END_DATE in config.py")

    feature_cols = get_feature_columns(df)
    print(f"\nUsing {len(feature_cols)} feature columns")

    X_train_raw = train_df[feature_cols]
    X_valid_raw = valid_df[feature_cols]
    X_test_raw = test_df[feature_cols]
    y_train = train_df[config.TARGET_COLUMN].values
    y_valid = valid_df[config.TARGET_COLUMN].values
    y_test = test_df[config.TARGET_COLUMN].values

    print("\nStep 4: imputing missing feature values (median, fit on TRAIN ONLY)...")
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    X_valid = imputer.transform(X_valid_raw)
    X_test = imputer.transform(X_test_raw)

    results = {}
    fitted_models = {}

    # ---- 1. Persistence baseline ----
    print("\nStep 5a: persistence baseline...")
    pred_train = persistence_baseline_predict(train_df, feature_cols, X_train)
    pred_valid = persistence_baseline_predict(valid_df, feature_cols, X_valid)
    results["persistence_baseline"] = {
        "train": compute_metrics(y_train, pred_train),
        "valid": compute_metrics(y_valid, pred_valid),
    }

    # ---- 2. Linear Regression ----
    print("Step 5b: linear regression...")
    t0 = time.time()
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    results["linear_regression"] = {
        "train": compute_metrics(y_train, lin_reg.predict(X_train)),
        "valid": compute_metrics(y_valid, lin_reg.predict(X_valid)),
    }
    fitted_models["linear_regression"] = lin_reg
    print(f"  done in {time.time()-t0:.1f}s")

    # ---- 3. Random Forest ----
    print("Step 5c: random forest...")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        random_state=config.RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results["random_forest"] = {
        "train": compute_metrics(y_train, rf.predict(X_train)),
        "valid": compute_metrics(y_valid, rf.predict(X_valid)),
    }
    fitted_models["random_forest"] = rf
    print(f"  done in {time.time()-t0:.1f}s")

    # ---- 4. XGBoost ----
    print("Step 5d: xgboost...")
    t0 = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=config.RANDOM_SEED, n_jobs=-1,
        early_stopping_rounds=20, eval_metric="mae",
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    results["xgboost"] = {
        "train": compute_metrics(y_train, xgb_model.predict(X_train)),
        "valid": compute_metrics(y_valid, xgb_model.predict(X_valid)),
        "best_iteration": int(xgb_model.best_iteration) if xgb_model.best_iteration is not None else None,
    }
    fitted_models["xgboost"] = xgb_model
    print(f"  done in {time.time()-t0:.1f}s")

    # ---- comparison table ----
    print("\n" + "=" * 78)
    print(f"{'Model':<22}{'Train MAE':>12}{'Valid MAE':>12}{'Valid RMSE':>13}{'Valid R2':>10}")
    print("-" * 78)
    for name, r in results.items():
        print(f"{name:<22}{r['train']['mae']:>12.3f}{r['valid']['mae']:>12.3f}"
              f"{r['valid']['rmse']:>13.3f}{r['valid']['r2']:>10.4f}")
    print("=" * 78)

    # ---- model selection: lowest validation MAE ----
    best_name = min(results.keys(), key=lambda k: results[k]["valid"]["mae"])
    print(f"\nSelected best model (lowest validation MAE): {best_name}")

    # ---- test evaluation - ONLY for the selected model, ONLY now ----
    if best_name == "persistence_baseline":
        pred_test = persistence_baseline_predict(test_df, feature_cols, X_test)
        best_model_object = None
    else:
        best_model_object = fitted_models[best_name]
        pred_test = best_model_object.predict(X_test)

    test_metrics = compute_metrics(y_test, pred_test)
    results[best_name]["test"] = test_metrics
    print(f"\nFINAL TEST METRICS for {best_name}: "
          f"MAE={test_metrics['mae']:.3f}  RMSE={test_metrics['rmse']:.3f}  R2={test_metrics['r2']:.4f}")

    # ---- save artifacts ----
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    if best_model_object is not None:
        bundle = {"imputer": imputer, "model": best_model_object, "model_name": best_name}
        joblib.dump(bundle, config.BEST_MODEL_PATH)
    else:
        bundle = {"imputer": imputer, "model": None, "model_name": best_name,
                  "note": "Persistence baseline has no fitted model object - "
                          "predictions are simply the current pm2_5 value."}
        joblib.dump(bundle, config.BEST_MODEL_PATH)

    joblib.dump(feature_cols, config.FEATURE_COLUMNS_PATH)

    metadata = {
        "target_column": config.TARGET_COLUMN,
        "target_horizon_hours": config.TARGET_HORIZON_HOURS,
        "stations_used": station_ids,
        "n_stations": len(station_ids),
        "split_boundaries": {
            "train_end_date": config.TRAIN_END_DATE,
            "valid_end_date": config.VALID_END_DATE,
        },
        "row_counts": {
            "train": len(train_df), "valid": len(valid_df), "test": len(test_df),
        },
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "missing_value_handling": "SimpleImputer(strategy='median'), fit on TRAIN split only",
        "random_seed": config.RANDOM_SEED,
        "model_comparison": results,
        "selected_model": best_name,
        "selection_criterion": "lowest validation MAE",
        "test_metrics_selected_model_only": test_metrics,
    }
    with open(config.MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nSaved: {config.BEST_MODEL_PATH}")
    print(f"Saved: {config.FEATURE_COLUMNS_PATH}")
    print(f"Saved: {config.MODEL_METADATA_PATH}")

    return metadata


if __name__ == "__main__":
    run()