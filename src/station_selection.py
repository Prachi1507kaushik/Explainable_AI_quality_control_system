"""
PHASE 7 - STATION SELECTION

Selects a manageable, non-arbitrary subset of stations for the initial
prototype, based on measurable coverage criteria defined in config.py:

  - MIN_PM25_COVERAGE : minimum fraction of hourly slots with a PM2.5 value
  - MIN_SPAN_DAYS     : minimum number of calendar days between first and
                        last reading
  - MIN_RECORDS       : minimum total number of PM2.5 readings

A full coverage report (ALL 110 stations with data) is always saved, so the
selection is auditable and the pipeline can later be re-run with different
thresholds to scale up to more stations.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def build_coverage_report(df):
    rows = []
    for station_id, g in df.groupby("station_id"):
        g = g.sort_values("datetime")
        first = g["datetime"].min()
        last = g["datetime"].max()
        span_days = (last - first).days
        n_records = len(g)
        n_pm25_present = g["pm2_5"].notna().sum()
        pm25_coverage = n_pm25_present / n_records if n_records > 0 else 0
        rows.append({
            "station_id": station_id,
            "station_name": g["station_name"].iloc[0],
            "city": g["city"].iloc[0],
            "state": g["state"].iloc[0],
            "first_reading": first,
            "last_reading": last,
            "span_days": span_days,
            "n_records": n_records,
            "n_pm25_present": n_pm25_present,
            "pm25_coverage": round(pm25_coverage, 4),
        })
    coverage = pd.DataFrame(rows).sort_values("pm25_coverage", ascending=False)
    return coverage


def select_stations(coverage):
    mask = (
        (coverage["pm25_coverage"] >= config.MIN_PM25_COVERAGE) &
        (coverage["span_days"] >= config.MIN_SPAN_DAYS) &
        (coverage["n_records"] >= config.MIN_RECORDS)
    )
    selected = coverage[mask].copy()
    return selected


def run():
    df = pd.read_csv(config.HISTORICAL_CLEAN, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"])

    coverage = build_coverage_report(df)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    coverage.to_csv(config.COVERAGE_REPORT, index=False)

    selected = select_stations(coverage)

    print(f"Total stations with any data: {len(coverage)}")
    print(f"Stations meeting selection criteria "
          f"(PM2.5 coverage >= {config.MIN_PM25_COVERAGE*100:.0f}%, "
          f"span >= {config.MIN_SPAN_DAYS} days, "
          f"records >= {config.MIN_RECORDS}): {len(selected)}")
    print(selected[["station_id", "station_name", "city", "state",
                     "span_days", "n_records", "pm25_coverage"]].to_string(index=False))

    return coverage, selected


if __name__ == "__main__":
    run()