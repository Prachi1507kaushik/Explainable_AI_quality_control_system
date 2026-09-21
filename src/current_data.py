"""
PHASE 4 - CURRENT CPCB SNAPSHOT PROCESSING

Processes cpcb_current_snapshot.csv SEPARATELY from the historical dataset
(station_hour.csv). This file is a single-timestamp snapshot (see the
Stage-1 audit earlier in this project: every row shares the same
`last_update` value), so it is NEVER merged with the historical data and
NEVER used for training - only for current-conditions display/inference.

Raw format of this file: one row per (station, pollutant) - i.e.
`pollutant_id` and `pollutant_avg` are columns, not one column per
pollutant. This script:
  1. Cleans column names
  2. Parses the timestamp
  3. Flags (never silently drops) invalid lat/lon and impossible
     pollutant values
  4. Pivots pollutant_id rows into separate pollutant columns
  5. Preserves state, city, station, timestamp, latitude, longitude,
     and every pollutant's average value

Every decision is logged to data/processed/current_processing_log.json.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data_loader

# Mainland-India bounding box, used only to FLAG suspicious coordinates -
# never to delete rows.
INDIA_LAT_RANGE = (6, 38)
INDIA_LON_RANGE = (68, 98)

ID_COLS = ["country", "state", "city", "station", "timestamp",
           "latitude", "longitude"]


def _to_snake(cols):
    cols = pd.Index(cols).str.strip()
    cols = cols.str.replace(r"(?<=[a-z0-9])(?=[A-Z])", "_", regex=True)
    cols = cols.str.lower()
    cols = cols.str.replace(r"[^a-z0-9]+", "_", regex=True)
    cols = cols.str.strip("_")
    return cols


def process_current_snapshot():
    log = {}

    df = data_loader.load_current_snapshot()
    log["raw_shape"] = list(df.shape)
    log["raw_columns"] = list(df.columns)

    # ---- clean column names ----
    df.columns = _to_snake(df.columns)
    log["standardized_columns"] = list(df.columns)

    # ---- remove exact duplicate rows ----
    before = len(df)
    df = df.drop_duplicates()
    log["exact_duplicates_removed"] = before - len(df)

    # ---- parse timestamp ----
    # observed raw format: DD-MM-YYYY HH:MM:SS
    ts_col = "last_update" if "last_update" in df.columns else "timestamp"
    df[ts_col] = pd.to_datetime(df[ts_col], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    n_bad_ts = int(df[ts_col].isna().sum())
    log["unparseable_timestamps"] = n_bad_ts
    df = df.rename(columns={ts_col: "timestamp"})

    log["is_single_timestamp_snapshot"] = bool(df["timestamp"].nunique() == 1)
    log["timestamp_value"] = str(df["timestamp"].iloc[0]) if len(df) else None

    # ---- numeric conversion ----
    for col in ["pollutant_min", "pollutant_max", "pollutant_avg", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- flag (do not delete) invalid lat/lon ----
    df["lat_lon_flag_invalid"] = (
        (df["latitude"] < INDIA_LAT_RANGE[0]) | (df["latitude"] > INDIA_LAT_RANGE[1]) |
        (df["longitude"] < INDIA_LON_RANGE[0]) | (df["longitude"] > INDIA_LON_RANGE[1]) |
        df["latitude"].isna() | df["longitude"].isna()
    )
    log["rows_flagged_invalid_lat_lon"] = int(df["lat_lon_flag_invalid"].sum())

    # ---- flag (do not delete) impossible pollutant values ----
    df["pollutant_flag_impossible"] = (
        (df["pollutant_min"] < 0) | (df["pollutant_max"] < 0) | (df["pollutant_avg"] < 0) |
        (df["pollutant_min"] > df["pollutant_max"]) |
        (df["pollutant_avg"] < df["pollutant_min"]) |
        (df["pollutant_avg"] > df["pollutant_max"])
    )
    log["rows_flagged_impossible_pollutant_values"] = int(df["pollutant_flag_impossible"].sum())

    log["missing_pollutant_avg_before_pivot"] = int(df["pollutant_avg"].isna().sum())

    # ---- pivot: one row per (station,pollutant) -> one row per station ----
    is_long_format = "pollutant_id" in df.columns and "pollutant_avg" in df.columns
    log["input_was_long_format"] = is_long_format

    if is_long_format:
        missing_id_cols = [c for c in ID_COLS if c not in df.columns]
        if missing_id_cols:
            raise ValueError(f"Cannot pivot: missing expected id columns {missing_id_cols}")

        dup_combo = df.duplicated(subset=ID_COLS + ["pollutant_id"], keep=False)
        log["duplicate_station_pollutant_combinations"] = int(dup_combo.sum())
        df_for_pivot = df.drop_duplicates(subset=ID_COLS + ["pollutant_id"], keep="first") \
            if dup_combo.sum() > 0 else df

        wide = df_for_pivot.pivot_table(
            index=ID_COLS,
            columns="pollutant_id",
            values="pollutant_avg",
            aggfunc="first",
        ).reset_index()
        wide.columns.name = None

        # carry the row-level quality flags forward, aggregated per station
        flags = df_for_pivot.groupby(ID_COLS)[
            ["lat_lon_flag_invalid", "pollutant_flag_impossible"]
        ].any().reset_index()
        wide = wide.merge(flags, on=ID_COLS, how="left")
    else:
        # already wide - use as-is
        wide = df.copy()

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    wide.to_csv(config.CURRENT_PROCESSED, index=False)

    log["final_shape"] = list(wide.shape)
    log["final_columns"] = list(wide.columns)
    log["unique_stations"] = int(wide["station"].nunique()) if "station" in wide.columns else None
    log["unique_cities"] = int(wide["city"].nunique()) if "city" in wide.columns else None

    with open(os.path.join(config.PROCESSED_DIR, "current_processing_log.json"), "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(json.dumps(log, indent=2, default=str))
    return wide


if __name__ == "__main__":
    process_current_snapshot()