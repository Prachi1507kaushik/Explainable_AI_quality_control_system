"""
STATION COVERAGE ANALYSIS (on data/processed/training_data.csv)

This is a DEEPER, second-pass coverage analysis than src/station_selection.py.
station_selection.py filters stations from historical_clean.csv using 3
simple thresholds (PM2.5 coverage, span, record count) to decide which ~84
stations even enter the feature-engineering step.

This script instead analyzes the FINAL, already-feature-engineered
training_data.csv and reports, per station:
  - n_observations            : total hourly rows for that station
  - missing_pm25_pct           : % of those rows with a NaN pm2_5 reading
  - date_range_start/end       : first/last timestamp
  - expected_hourly_slots      : (last - first) in hours + 1, i.e. how many
                                  hourly readings SHOULD exist if reporting
                                  were perfectly continuous
  - continuity_pct             : n_observations / expected_hourly_slots
                                  (100% = no gaps at all in the raw feed;
                                  well below 100% = station drops offline
                                  often, which hurts lag/rolling features)
  - usable_training_rows       : rows where target_pm25_6h is present AND
                                  target_horizon_valid is True (i.e. a real,
                                  leakage-checked 6-hour-ahead label exists)

Selection of the initial 5-10 station subset is NOT arbitrary: stations are
ranked by usable_training_rows (the single number that most directly
determines how much real training signal a station contributes), then
filtered by two hard quality gates (missing_pm25_pct and continuity_pct)
so that a station with a huge row count but poor data quality cannot win
just on volume. See SELECTION CRITERIA below for exact thresholds.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

USECOLS = ["station_id", "station_name", "city", "state", "datetime",
           "pm2_5", "target_pm25_6h", "target_horizon_valid"]

CHUNKSIZE = 300_000

# ------------------------------------------------------------------
# SELECTION CRITERIA (documented, applied after ranking - not arbitrary)
# A station is eligible for the recommended subset only if it clears BOTH
# quality gates below. Among eligible stations, the top by usable rows are
# picked, capped at 10.
# ------------------------------------------------------------------
MAX_MISSING_PM25_PCT = 15.0     # at most 15% of hourly slots missing PM2.5
MIN_CONTINUITY_PCT = 85.0       # at least 85% of expected hourly slots present
TARGET_SUBSET_SIZE_MIN = 5
TARGET_SUBSET_SIZE_MAX = 10


def analyze_coverage():
    agg = {}  # station_id -> running stats

    reader = pd.read_csv(config.TRAINING_DATA, usecols=USECOLS, chunksize=CHUNKSIZE,
                         parse_dates=["datetime"])

    for chunk_num, chunk in enumerate(reader):
        for station_id, g in chunk.groupby("station_id"):
            if station_id not in agg:
                agg[station_id] = {
                    "station_name": g["station_name"].iloc[0],
                    "city": g["city"].iloc[0],
                    "state": g["state"].iloc[0],
                    "n_observations": 0,
                    "n_pm25_present": 0,
                    "n_usable_target_rows": 0,
                    "min_datetime": g["datetime"].min(),
                    "max_datetime": g["datetime"].max(),
                }
            a = agg[station_id]
            a["n_observations"] += len(g)
            a["n_pm25_present"] += int(g["pm2_5"].notna().sum())
            usable = g["target_pm25_6h"].notna() & (g["target_horizon_valid"] == True)
            a["n_usable_target_rows"] += int(usable.sum())
            a["min_datetime"] = min(a["min_datetime"], g["datetime"].min())
            a["max_datetime"] = max(a["max_datetime"], g["datetime"].max())
        print(f"Processed chunk {chunk_num + 1} ({(chunk_num + 1) * CHUNKSIZE:,} rows read so far)")

    rows = []
    for station_id, a in agg.items():
        span_hours = int((a["max_datetime"] - a["min_datetime"]).total_seconds() / 3600) + 1
        continuity_pct = round(100 * a["n_observations"] / span_hours, 2) if span_hours > 0 else 0
        missing_pm25_pct = round(100 * (1 - a["n_pm25_present"] / a["n_observations"]), 2) \
            if a["n_observations"] > 0 else 100.0
        rows.append({
            "station_id": station_id,
            "station_name": a["station_name"],
            "city": a["city"],
            "state": a["state"],
            "n_observations": a["n_observations"],
            "missing_pm25_pct": missing_pm25_pct,
            "date_range_start": a["min_datetime"],
            "date_range_end": a["max_datetime"],
            "expected_hourly_slots": span_hours,
            "continuity_pct": continuity_pct,
            "usable_training_rows": a["n_usable_target_rows"],
        })

    coverage = pd.DataFrame(rows).sort_values("usable_training_rows", ascending=False)
    return coverage


def recommend_subset(coverage):
    eligible = coverage[
        (coverage["missing_pm25_pct"] <= MAX_MISSING_PM25_PCT) &
        (coverage["continuity_pct"] >= MIN_CONTINUITY_PCT)
    ].copy()

    eligible = eligible.sort_values("usable_training_rows", ascending=False)
    n_pick = min(TARGET_SUBSET_SIZE_MAX, max(TARGET_SUBSET_SIZE_MIN, len(eligible)))
    n_pick = min(n_pick, len(eligible))  # never exceed how many actually qualify
    n_pick = min(n_pick, TARGET_SUBSET_SIZE_MAX)
    recommended = eligible.head(n_pick)
    return eligible, recommended


def run():
    coverage = analyze_coverage()

    os.makedirs(os.path.join(config.OUTPUTS_DIR, "metrics"), exist_ok=True)
    out_path = os.path.join(config.OUTPUTS_DIR, "metrics", "station_coverage.csv")
    coverage.to_csv(out_path, index=False)
    print(f"\nFull station coverage report saved to {out_path}")
    print(f"Total stations analyzed: {len(coverage)}")

    eligible, recommended = recommend_subset(coverage)
    print(f"\nStations passing quality gates "
          f"(missing_pm25_pct <= {MAX_MISSING_PM25_PCT}%, "
          f"continuity_pct >= {MIN_CONTINUITY_PCT}%): {len(eligible)}")
    print(f"\nRECOMMENDED INITIAL SUBSET ({len(recommended)} stations):")
    print(recommended[["station_id", "station_name", "city", "state",
                        "n_observations", "missing_pm25_pct",
                        "continuity_pct", "usable_training_rows"]].to_string(index=False))

    return coverage, eligible, recommended


if __name__ == "__main__":
    run()