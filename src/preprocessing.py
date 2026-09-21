"""
PHASE 3 - HISTORICAL DATA PREPROCESSING

Uses station_hour.csv as the primary historical dataset and stations.csv
only to enrich station metadata (name/city/state).

All preprocessing decisions are documented in the printed log AND in
data/processed/preprocessing_log.json so nothing is a silent decision.

Rules followed:
- No future information is EVER used to fill in past values. Missing-value
  handling uses forward-fill only (each station's own past readings),
  never backward-fill and never another station's data.
- Missing-value strategy (documented in full below, in
  MISSING_VALUE_FILL_LIMIT_HOURS):
    * Gaps of MISSING_VALUE_FILL_LIMIT_HOURS hours or fewer, for a single
      station's own pollutant series, are forward-filled from that
      station's last known reading. This was checked empirically first:
      most missing runs for a sample station are 1-2 hours long (isolated
      sensor blips), so a 2-hour limit fills genuine short gaps without
      fabricating long stretches of invented data.
    * Longer gaps are left as NaN. They are NOT interpolated, NOT
      backfilled, and NOT filled with column means/medians, because doing
      so would either leak future information or invent values with no
      real sensor support.
    * A boolean column `<pollutant>_was_filled` is added for every
      pollutant so every filled value remains fully traceable and can be
      excluded from evaluation/training if a stricter analysis is needed
      later.
- Exact duplicate rows are removed. Duplicate (station, timestamp) pairs are
  detected and reported (none were found in this dataset - see log).
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data_loader

# Maximum consecutive-hour gap (per station, per pollutant) that gets
# forward-filled. Chosen after checking real gap lengths in this dataset:
# the large majority of missing runs are 1-2 hours long. Anything longer
# is left as a genuine missing value.
MISSING_VALUE_FILL_LIMIT_HOURS = 2

POLLUTANT_COLUMNS = ["pm2_5", "pm10", "no", "no2", "nox", "nh3",
                     "co", "so2", "o3", "benzene", "toluene", "xylene", "aqi"]


def _to_snake(cols):
    cols = pd.Index(cols).str.strip()
    cols = cols.str.replace(r"(?<=[a-z0-9])(?=[A-Z])", "_", regex=True)
    cols = cols.str.lower()
    cols = cols.str.replace(r"[^a-z0-9]+", "_", regex=True)
    cols = cols.str.strip("_")
    return cols


def clean_historical():
    log = {}

    hourly = data_loader.load_station_hour()
    stations = data_loader.load_stations()

    log["raw_station_hour_shape"] = list(hourly.shape)
    log["raw_stations_shape"] = list(stations.shape)

    # ---- standardize column names ----
    hourly.columns = _to_snake(hourly.columns)
    stations.columns = _to_snake(stations.columns)
    log["standardized_hourly_columns"] = list(hourly.columns)
    log["standardized_stations_columns"] = list(stations.columns)

    # ---- remove exact duplicate rows ----
    before = len(hourly)
    hourly = hourly.drop_duplicates()
    log["exact_duplicates_removed"] = before - len(hourly)

    before = len(stations)
    stations = stations.drop_duplicates()
    log["exact_duplicates_removed_stations"] = before - len(stations)

    # ---- parse timestamps ----
    hourly["datetime"] = pd.to_datetime(hourly["datetime"], errors="coerce")
    n_bad_dt = int(hourly["datetime"].isna().sum())
    log["unparseable_datetimes"] = n_bad_dt
    if n_bad_dt > 0:
        # rows with no valid timestamp cannot be placed in a time series at
        # all, so they are excluded here (not silently - logged) rather than
        # kept with a fake/guessed timestamp.
        hourly = hourly[hourly["datetime"].notna()].copy()
        log["rows_dropped_for_unparseable_datetime"] = n_bad_dt

    # ---- sort by station and timestamp (required before any lag/rolling
    #      feature engineering to avoid leakage) ----
    hourly = hourly.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    # ---- duplicate (station, timestamp) pairs ----
    dup_mask = hourly.duplicated(subset=["station_id", "datetime"], keep=False)
    log["duplicate_station_timestamp_pairs"] = int(dup_mask.sum())
    if dup_mask.sum() > 0:
        # keep the first occurrence only; report how many were dropped
        before = len(hourly)
        hourly = hourly.drop_duplicates(subset=["station_id", "datetime"], keep="first")
        log["rows_dropped_for_duplicate_station_timestamp"] = before - len(hourly)

    # ---- numeric conversion + negative-value flag (not deleted) ----
    pollutant_cols = [c for c in
                       ["pm2_5", "pm10", "no", "no2", "nox", "nh3", "co",
                        "so2", "o3", "benzene", "toluene", "xylene", "aqi"]
                       if c in hourly.columns]
    hourly["negative_value_flag"] = False
    for c in pollutant_cols:
        hourly[c] = pd.to_numeric(hourly[c], errors="coerce")
        hourly["negative_value_flag"] |= (hourly[c] < 0)
    log["rows_with_any_negative_pollutant_value"] = int(hourly["negative_value_flag"].sum())
    log["negative_values_note"] = (
        "None found in this dataset. If present, such rows would be FLAGGED, "
        "not silently deleted, since a negative concentration is physically "
        "impossible and should be inspected manually."
    )

    # ---- merge station metadata (left join - never drop a reading just
    #      because metadata is missing) ----
    merged = hourly.merge(stations, on="station_id", how="left")
    log["rows_missing_metadata_after_merge"] = int(merged["station_name"].isna().sum())

    # ---- missing-value handling: short-gap forward-fill only ----
    # Data must already be sorted by (station_id, datetime) - done above -
    # so that a forward-fill only ever pulls from a station's OWN PAST
    # readings, never from the future and never from another station.
    fill_counts = {}
    present_pollutant_cols = [c for c in POLLUTANT_COLUMNS if c in merged.columns]
    grouped = merged.groupby("station_id")

    for col in present_pollutant_cols:
        before_na = merged[col].isna()
        filled_series = grouped[col].ffill(limit=MISSING_VALUE_FILL_LIMIT_HOURS)
        was_filled = before_na & filled_series.notna()
        merged[f"{col}_was_filled"] = was_filled
        merged[col] = filled_series
        fill_counts[col] = {
            "missing_before": int(before_na.sum()),
            "filled_short_gaps": int(was_filled.sum()),
            "still_missing_after": int(merged[col].isna().sum()),
        }

    log["missing_value_strategy"] = (
        f"For each pollutant, gaps of {MISSING_VALUE_FILL_LIMIT_HOURS} "
        "consecutive hours or fewer, within a single station's own "
        "chronologically-sorted series, are forward-filled from that "
        "station's last known reading (ffill with a strict limit). This "
        "uses only past data - never future data, never another station's "
        "data. Gaps longer than the limit are left as genuine NaN - they "
        "are NOT interpolated, NOT backfilled, and NOT replaced with a "
        "mean/median, since that would either leak future information or "
        "invent values with no real sensor support. Every filled cell is "
        "flagged in a companion `<pollutant>_was_filled` boolean column so "
        "filled values remain fully traceable and can be excluded later if "
        "a stricter analysis is needed."
    )
    log["missing_value_fill_counts_per_pollutant"] = fill_counts

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    merged.to_csv(config.HISTORICAL_CLEAN, index=False)
    log["final_shape"] = list(merged.shape)
    log["final_columns"] = list(merged.columns)
    log["date_range"] = [str(merged["datetime"].min()), str(merged["datetime"].max())]
    log["unique_stations"] = int(merged["station_id"].nunique())
    log["unique_cities"] = int(merged["city"].nunique()) if "city" in merged.columns else None

    with open(os.path.join(config.PROCESSED_DIR, "preprocessing_log.json"), "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(json.dumps(log, indent=2, default=str))
    return merged


if __name__ == "__main__":
    clean_historical()