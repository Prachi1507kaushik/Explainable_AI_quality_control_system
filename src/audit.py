"""
PHASE 2 - DATA AUDIT
Inspects the three raw files and writes a machine-readable JSON report.
Nothing here modifies the raw data.
"""

import json
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data_loader


def _safe(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (pd.Timestamp,)):
        return str(v)
    return v


def audit_station_hour(df):
    report = {}
    report["n_rows"] = len(df)
    report["n_columns"] = df.shape[1]
    report["columns"] = list(df.columns)
    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    report["missing_values"] = {c: int(df[c].isna().sum()) for c in df.columns}
    report["duplicate_exact_rows"] = int(df.duplicated().sum())

    dt_col = "Datetime" if "Datetime" in df.columns else None
    id_col = "StationId" if "StationId" in df.columns else None

    if id_col:
        report["unique_stations"] = int(df[id_col].nunique())
    if dt_col:
        parsed = pd.to_datetime(df[dt_col], errors="coerce")
        report["date_range"] = {
            "min": _safe(parsed.min()),
            "max": _safe(parsed.max()),
            "unparseable_count": int(parsed.isna().sum()),
        }
        if id_col:
            report["duplicate_station_timestamp_pairs"] = int(
                df.duplicated(subset=[id_col, dt_col], keep=False).sum()
            )

    pollutant_candidates = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
                             "CO", "SO2", "O3", "Benzene", "Toluene",
                             "Xylene", "AQI"]
    present_pollutants = [c for c in pollutant_candidates if c in df.columns]
    report["pollutant_columns_present"] = present_pollutants

    numeric_ranges = {}
    invalid_values = {}
    for c in present_pollutants:
        series = pd.to_numeric(df[c], errors="coerce")
        numeric_ranges[c] = {
            "min": _safe(series.min()),
            "max": _safe(series.max()),
            "mean": _safe(series.mean()),
            "missing": int(series.isna().sum()),
            "missing_pct": _safe(round(series.isna().mean() * 100, 2)),
        }
        invalid_values[c] = int((series < 0).sum())
    report["numerical_ranges"] = numeric_ranges
    report["negative_invalid_value_counts"] = invalid_values

    return report


def audit_stations(df):
    report = {}
    report["n_rows"] = len(df)
    report["n_columns"] = df.shape[1]
    report["columns"] = list(df.columns)
    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    report["missing_values"] = {c: int(df[c].isna().sum()) for c in df.columns}
    report["duplicate_exact_rows"] = int(df.duplicated().sum())
    if "City" in df.columns:
        report["unique_cities"] = int(df["City"].nunique())
    if "State" in df.columns:
        report["unique_states"] = int(df["State"].nunique())
    if "StationId" in df.columns:
        report["unique_station_ids"] = int(df["StationId"].nunique())
    return report


def audit_current_snapshot(df):
    report = {}
    report["n_rows"] = len(df)
    report["n_columns"] = df.shape[1]
    report["columns"] = list(df.columns)
    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    report["missing_values"] = {c: int(df[c].isna().sum()) for c in df.columns}
    report["duplicate_exact_rows"] = int(df.duplicated().sum())
    if "city" in df.columns:
        report["unique_cities"] = int(df["city"].nunique())
    if "station" in df.columns:
        report["unique_stations"] = int(df["station"].nunique())
    if "timestamp" in df.columns:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce")
        report["date_range"] = {"min": _safe(parsed.min()), "max": _safe(parsed.max())}
    return report


def run_audit():
    sh = data_loader.load_station_hour()
    st = data_loader.load_stations()
    cur = data_loader.load_current_snapshot()

    full_report = {
        "station_hour_csv": audit_station_hour(sh),
        "stations_csv": audit_stations(st),
        "cpcb_current_snapshot_csv": audit_current_snapshot(cur),
    }

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(config.DATA_AUDIT_REPORT, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    print(f"Data audit report written to {config.DATA_AUDIT_REPORT}")
    return full_report


if __name__ == "__main__":
    report = run_audit()
    print(json.dumps(report, indent=2, default=str)[:3000])