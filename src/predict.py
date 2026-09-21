"""
PHASE 12 - AIR QUALITY PREDICTION

Loads the already-trained model and generates a 6-hour-ahead PM2.5
prediction for a station.

This module:
    1. Loads the saved model
    2. Loads feature_columns.pkl
    3. Loads processed historical/current station data
    4. Generates the same feature structure used during training
    5. Uses the saved TRAINING imputer
    6. Predicts PM2.5 six hours ahead
    7. Returns:
        - current PM2.5
        - predicted PM2.5
        - prediction horizon
        - station
        - city

IMPORTANT:
    - No model retraining occurs.
    - No new imputer is fitted.
    - The saved training imputer is reused.
    - Feature names/order must match training.
"""

from __future__ import annotations

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src import config


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = config.BEST_MODEL_PATH
FEATURE_COLUMNS_PATH = config.FEATURE_COLUMNS_PATH

HORIZON_HOURS = config.TARGET_HORIZON_HOURS
TARGET_COLUMN = config.TARGET_COLUMN

# Number of recent rows required to calculate lag/rolling features.
# 24 hours is enough for the largest rolling window and lag.
HISTORY_HOURS_REQUIRED = 30


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _safe_float(value):
    """
    Convert numpy/pandas numeric values to normal Python float.
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_column_names(df):
    """
    Standardize column names in the same general style used
    throughout the project.
    """

    df = df.copy()

    df.columns = (
        pd.Index(df.columns)
        .str.strip()
        .str.replace(r"(?<=[a-z0-9])(?=[A-Z])", "_", regex=True)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    return df


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """
    Load the trained model package.

    Expected structure:

        {
            "model": trained_model,
            "imputer": fitted_imputer,
            "model_name": "xgboost"
        }

    Returns
    -------
    model_package : dict
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}\n\n"
            "Run train.py first."
        )

    model_package = joblib.load(MODEL_PATH)

    # The actual trained file is a dictionary.
    if isinstance(model_package, dict):

        if "model" not in model_package:
            raise ValueError(
                "best_model.pkl is a dictionary but does not contain "
                "the 'model' key."
            )

        if "imputer" not in model_package:
            raise ValueError(
                "best_model.pkl does not contain the saved training imputer."
            )

        return model_package

    # Fallback in case a plain estimator was saved.
    warnings.warn(
        "Model was saved as a plain estimator instead of a model package."
    )

    return {
        "model": model_package,
        "imputer": None,
        "model_name": type(model_package).__name__,
    }


# ============================================================
# FEATURE COLUMN LOADING
# ============================================================

def load_feature_columns():
    """
    Load the exact feature columns used during model training.
    """

    if not os.path.exists(FEATURE_COLUMNS_PATH):
        raise FileNotFoundError(
            f"Feature columns file not found:\n{FEATURE_COLUMNS_PATH}"
        )

    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    if isinstance(feature_columns, dict):

        # Handle possible dictionary formats.
        if "feature_columns" in feature_columns:
            feature_columns = feature_columns["feature_columns"]

        elif "features" in feature_columns:
            feature_columns = feature_columns["features"]

    if not isinstance(feature_columns, (list, tuple, np.ndarray)):
        raise TypeError(
            "feature_columns.pkl must contain a list/tuple/array "
            "of feature names."
        )

    feature_columns = list(feature_columns)

    if len(feature_columns) == 0:
        raise ValueError("No feature columns were found.")

    return feature_columns


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

def load_historical_data():
    """
    Load the processed historical dataset.

    This is used to construct lag and rolling features.
    """

    path = config.HISTORICAL_CLEAN

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Historical processed data not found:\n{path}\n\n"
            "Run preprocessing.py first."
        )

    df = pd.read_csv(path, low_memory=False)

    df = _clean_column_names(df)

    if "datetime" not in df.columns:
        raise ValueError(
            "Historical data must contain a 'datetime' column."
        )

    if "station_id" not in df.columns:
        raise ValueError(
            "Historical data must contain a 'station_id' column."
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    df = df[df["datetime"].notna()].copy()

    df = df.sort_values(
        ["station_id", "datetime"]
    ).reset_index(drop=True)

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

POLLUTANT_COLUMNS = [
    "pm2_5",
    "pm10",
    "no",
    "no2",
    "nox",
    "so2",
    "co",
    "o3",
    "nh3",
]

LAGS = [1, 3, 6, 12, 24]

ROLLING_WINDOWS = [3, 6, 12, 24]


def add_time_features(df):
    """
    Generate the same time features used during training.
    """

    df = df.copy()

    df["hour"] = df["datetime"].dt.hour

    df["day_of_week"] = df["datetime"].dt.dayofweek

    df["day_of_year"] = df["datetime"].dt.dayofyear

    df["month"] = df["datetime"].dt.month

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Cyclical hour encoding.
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Cyclical month encoding.
    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    return df


def add_lag_features(df):
    """
    Generate the same lag features used during training.
    """

    df = df.copy()

    df = df.sort_values(
        ["station_id", "datetime"]
    ).reset_index(drop=True)

    grouped = df.groupby(
        "station_id",
        sort=False
    )

    for column in POLLUTANT_COLUMNS:

        if column not in df.columns:
            continue

        for lag in LAGS:

            feature_name = f"{column}_lag_{lag}h"

            df[feature_name] = grouped[column].shift(lag)

    return df


def add_rolling_features(df):
    """
    Generate rolling mean features using only historical values.

    IMPORTANT:
    shift(1) ensures the current reading does not enter
    its own rolling feature.
    """

    df = df.copy()

    df = df.sort_values(
        ["station_id", "datetime"]
    ).reset_index(drop=True)

    grouped = df.groupby(
        "station_id",
        sort=False
    )

    for column in POLLUTANT_COLUMNS:

        if column not in df.columns:
            continue

        shifted = grouped[column].shift(1)

        shifted_grouped = shifted.groupby(
            df["station_id"],
            sort=False
        )

        for window in ROLLING_WINDOWS:

            feature_name = (
                f"{column}_roll_mean_{window}h"
            )

            df[feature_name] = (
                shifted_grouped
                .rolling(
                    window,
                    min_periods=max(1, window // 2)
                )
                .mean()
                .reset_index(
                    level=0,
                    drop=True
                )
            )

    return df


def create_features(df):
    """
    Create the complete feature table required by the model.
    """

    df = df.copy()

    df = add_time_features(df)

    df = add_lag_features(df)

    df = add_rolling_features(df)

    return df


# ============================================================
# STATION SELECTION
# ============================================================

def get_available_stations(df):
    """
    Return available station IDs.
    """

    return sorted(
        df["station_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


# ============================================================
# PREPARE ONE STATION
# ============================================================

def prepare_station_data(
    df,
    station_id,
):
    """
    Extract one station and generate its features.

    Returns the complete station history with engineered features.
    """

    station_id = str(station_id)

    station_df = df[
        df["station_id"].astype(str) == station_id
    ].copy()

    if station_df.empty:
        raise ValueError(
            f"No historical data found for station: {station_id}"
        )

    station_df = station_df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    # We need enough history for lag/rolling features.
    if len(station_df) < HISTORY_HOURS_REQUIRED:
        raise ValueError(
            f"Not enough historical data for station "
            f"{station_id}. "
            f"At least {HISTORY_HOURS_REQUIRED} rows are required."
        )

    station_df = create_features(station_df)

    return station_df


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(
    latest_row,
    feature_columns,
    imputer,
):
    """
    Select exactly the features used during training
    and apply the SAVED training imputer.

    No fitting occurs here.
    """

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in latest_row.columns
    ]

    if missing_features:
        raise ValueError(
            "Required model features are missing:\n"
            + "\n".join(missing_features)
        )

    X = latest_row[
        feature_columns
    ].copy()

    # Ensure numeric values.
    for column in feature_columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    if imputer is not None:

        # IMPORTANT:
        # We only TRANSFORM here.
        # We NEVER fit the imputer during inference.
        X_array = imputer.transform(X)

        X = pd.DataFrame(
            X_array,
            columns=feature_columns,
            index=X.index,
        )

    return X


# ============================================================
# SINGLE STATION PREDICTION
# ============================================================

def predict_station(
    station_id,
    historical_df=None,
):
    """
    Predict PM2.5 six hours ahead for one station.

    Parameters
    ----------
    station_id : str
        Station identifier.

    historical_df : pandas.DataFrame, optional
        Historical processed dataset.

    Returns
    -------
    dict
        Prediction result.
    """

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model_package = load_model()

    model = model_package["model"]

    imputer = model_package.get(
        "imputer",
        None
    )

    model_name = model_package.get(
        "model_name",
        type(model).__name__
    )

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    feature_columns = load_feature_columns()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    if historical_df is None:
        historical_df = load_historical_data()

    # --------------------------------------------------------
    # Prepare station
    # --------------------------------------------------------

    station_df = prepare_station_data(
        historical_df,
        station_id,
    )

    # --------------------------------------------------------
    # Latest available row
    # --------------------------------------------------------

    latest_row = station_df.iloc[[-1]].copy()

    latest_timestamp = latest_row[
        "datetime"
    ].iloc[0]

    # --------------------------------------------------------
    # Current PM2.5
    # --------------------------------------------------------

    if "pm2_5" not in latest_row.columns:
        raise ValueError(
            "pm2_5 column is missing from station data."
        )

    current_pm25 = _safe_float(
        latest_row["pm2_5"].iloc[0]
    )

    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------

    X_latest = prepare_model_input(
        latest_row,
        feature_columns,
        imputer,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        X_latest
    )

    predicted_pm25 = _safe_float(
        np.asarray(prediction).reshape(-1)[0]
    )

    # PM2.5 concentration cannot physically be negative.
    if predicted_pm25 is not None:
        predicted_pm25 = max(
            0.0,
            predicted_pm25
        )

    # --------------------------------------------------------
    # Station metadata
    # --------------------------------------------------------

    station = str(
        latest_row["station_id"].iloc[0]
    )

    if "city" in latest_row.columns:

        city_value = latest_row[
            "city"
        ].iloc[0]

        city = (
            str(city_value)
            if pd.notna(city_value)
            else "Unknown"
        )

    else:
        city = "Unknown"

    # --------------------------------------------------------
    # Prediction timestamp
    # --------------------------------------------------------

    prediction_timestamp = (
        latest_timestamp
        + pd.Timedelta(hours=HORIZON_HOURS)
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "station": station,
        "city": city,
        "current_pm25": current_pm25,
        "predicted_pm25": predicted_pm25,
        "prediction_horizon_hours": HORIZON_HOURS,
        "current_timestamp": str(
            latest_timestamp
        ),
        "prediction_timestamp": str(
            prediction_timestamp
        ),
        "model": str(model_name),
    }

    return result


# ============================================================
# PREDICT FROM CURRENT DATAFRAME
# ============================================================

def predict_from_dataframe(
    station_id,
    df,
):
    """
    Convenience function for Streamlit.

    Example:

        result = predict_from_dataframe(
            "DL034",
            historical_df
        )
    """

    return predict_station(
        station_id=station_id,
        historical_df=df,
    )


# ============================================================
# PREDICT ALL AVAILABLE STATIONS
# ============================================================

def predict_all_stations(
    historical_df=None,
):
    """
    Generate predictions for all available stations.

    Returns a pandas DataFrame.

    This is useful for a Streamlit dashboard showing
    multiple stations on a map/table.
    """

    if historical_df is None:
        historical_df = load_historical_data()

    stations = get_available_stations(
        historical_df
    )

    results = []

    for station_id in stations:

        try:

            result = predict_station(
                station_id,
                historical_df
            )

            results.append(result)

        except Exception as exc:

            warnings.warn(
                f"Could not predict station "
                f"{station_id}: {exc}"
            )

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# ============================================================
# SIMPLE API FUNCTION FOR STREAMLIT
# ============================================================

def get_prediction(
    station_id,
):
    """
    Main public function intended for Streamlit.

    Usage:

        from src.predict import get_prediction

        result = get_prediction("DL034")

        print(result["current_pm25"])
        print(result["predicted_pm25"])
    """

    return predict_station(
        station_id=station_id
    )


# ============================================================
# COMMAND-LINE TEST
# ============================================================

def main():

    print("=" * 60)
    print("PHASE 12 - AIR QUALITY PREDICTION")
    print("=" * 60)

    print("\n1. Loading model...")

    model_package = load_model()

    print(
        "Model:",
        model_package.get(
            "model_name",
            type(model_package["model"]).__name__
        )
    )

    print("\n2. Loading feature columns...")

    feature_columns = load_feature_columns()

    print(
        f"Features: {len(feature_columns)}"
    )

    print("\n3. Loading historical data...")

    historical_df = load_historical_data()

    print(
        f"Rows: {len(historical_df):,}"
    )

    stations = get_available_stations(
        historical_df
    )

    print(
        f"Available stations: {len(stations)}"
    )

    # --------------------------------------------------------
    # Select first station for CLI demonstration.
    # --------------------------------------------------------

    if not stations:
        raise ValueError(
            "No stations available."
        )

    station_id = stations[0]

    print(
        f"\n4. Generating prediction for station: "
        f"{station_id}"
    )

    result = predict_station(
        station_id=station_id,
        historical_df=historical_df,
    )

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(
        f"Station              : {result['station']}"
    )

    print(
        f"City                 : {result['city']}"
    )

    print(
        f"Current PM2.5        : {result['current_pm25']}"
    )

    print(
        f"Predicted PM2.5      : {result['predicted_pm25']}"
    )

    print(
        f"Prediction horizon   : "
        f"{result['prediction_horizon_hours']} hours"
    )

    print(
        f"Current timestamp    : "
        f"{result['current_timestamp']}"
    )

    print(
        f"Prediction timestamp : "
        f"{result['prediction_timestamp']}"
    )

    print(
        f"Model                : "
        f"{result['model']}"
    )

    print("\nPrediction completed successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()