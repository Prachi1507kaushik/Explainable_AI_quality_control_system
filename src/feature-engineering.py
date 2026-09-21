"""
PHASE 5 - TARGET DEFINITION
PHASE 6 - FEATURE ENGINEERING

Target: target_pm25_6h = PM2.5 value 6 hours after the current timestamp,
for the SAME station.

Leakage prevention rules followed:
  - Data is sorted by (station_id, datetime) before any lag/rolling
    computation.
  - All lag and rolling features use .shift(k) with k >= 1, i.e. only past
    or current-and-past values relative to the row's own timestamp.
  - Rolling windows use `.shift(1).rolling(w)` so the CURRENT row's own
    reading is never included in its own rolling average (that would leak
    the present into a "past average" feature).
  - The target is created with `.shift(-6)` GROUPED BY STATION, so a
    station's data never leaks into another station's target, and the
    target is genuinely the future relative to the feature row.
  - Rows where the exact 6-hour-ahead timestamp does not exist for a
    station (e.g. gaps in reporting) get NaN targets and are dropped only
    at training time (not silently averaged/interpolated).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

POLLUTANT_LAG_COLS = ["pm2_5", "pm10", "no2", "no", "nox", "so2", "co", "o3", "nh3"]
LAGS = [1, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]


def add_time_features(df):
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_and_rolling_features(df):
    """Must be called on data already sorted by (station_id, datetime)."""
    g = df.groupby("station_id")

    for col in POLLUTANT_LAG_COLS:
        if col not in df.columns:
            continue
        for lag in LAGS:
            df[f"{col}_lag_{lag}h"] = g[col].shift(lag)

    # rolling means computed on values shifted by 1 hour, so the current
    # row's own reading never enters its own rolling average
    for col in POLLUTANT_LAG_COLS:
        if col not in df.columns:
            continue
        shifted = g[col].shift(1)
        for w in ROLLING_WINDOWS:
            df[f"{col}_roll_mean_{w}h"] = (
                shifted.groupby(df["station_id"]).rolling(w, min_periods=max(1, w // 2))
                .mean().reset_index(level=0, drop=True)
            )
    return df


def add_target(df, horizon=config.TARGET_HORIZON_HOURS):
    g = df.groupby("station_id")
    df[config.TARGET_COLUMN] = g["pm2_5"].shift(-horizon)

    # verify the target actually corresponds to exactly `horizon` hours
    # ahead in real time (not just the 6th NEXT ROW, which could be wrong
    # if there is a reporting gap)
    future_time = g["datetime"].shift(-horizon)
    time_delta_hours = (future_time - df["datetime"]).dt.total_seconds() / 3600
    valid_horizon = np.isclose(time_delta_hours, horizon)
    df.loc[~valid_horizon, config.TARGET_COLUMN] = np.nan
    df["target_horizon_valid"] = valid_horizon
    return df


def build_feature_table(selected_station_ids):
    hist = pd.read_csv(config.HISTORICAL_CLEAN, low_memory=False)
    hist["datetime"] = pd.to_datetime(hist["datetime"])
    hist = hist[hist["station_id"].isin(selected_station_ids)].copy()
    hist = hist.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    hist = add_time_features(hist)
    hist = add_lag_and_rolling_features(hist)
    hist = add_target(hist)

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    hist.to_csv(config.TRAINING_DATA, index=False)
    print("Feature table shape:", hist.shape)
    print("Rows with valid target:", hist[config.TARGET_COLUMN].notna().sum())

    non_feature_cols = {
        "station_id", "datetime", "station_name", "city", "state", "status",
        "aqi_bucket", config.TARGET_COLUMN, "target_horizon_valid",
        "negative_value_flag",
    }
    was_filled_cols = {c for c in hist.columns if c.endswith("_was_filled")}
    feature_cols = [c for c in hist.columns
                     if c not in non_feature_cols and c not in was_filled_cols]

    print(f"\nTotal engineered feature columns: {len(feature_cols)}")
    for c in feature_cols:
        print(" -", c)
    print(f"\nTarget column: {config.TARGET_COLUMN}")

    return hist


if __name__ == "__main__":
    from src import station_selection
    coverage, selected = station_selection.run()
    build_feature_table(selected["station_id"].tolist())