"""
Central configuration for the AI Air Quality Prediction project.
All paths, constants, and thresholds are defined here.

IMPORTANT: The CPCB National AQI breakpoints below are copied from the
official CPCB document "National Air Quality Index" (2014), which defines
sub-index breakpoints for PM2.5, PM10, NO2, SO2, CO, O3, and NH3.
Source: CPCB, Government of India - https://cpcb.nic.in/National-Air-Quality-Index/
These are NOT invented - they are the same breakpoints used to compute the
official Indian AQI, reused here only for building a transparent, documented
risk-category layer (Phase 12). We do not claim they are the only possible
scheme, and we clearly label our category as MODEL-DERIVED, not an official
government warning.
"""

import os

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")

RAW_STATION_HOUR = os.path.join(RAW_DIR, "station_hour.csv")
RAW_STATIONS = os.path.join(RAW_DIR, "stations.csv")
RAW_CURRENT_SNAPSHOT = os.path.join(RAW_DIR, "cpcb_current_snapshot.csv")

HISTORICAL_CLEAN = os.path.join(PROCESSED_DIR, "historical_clean.csv")
TRAINING_DATA = os.path.join(PROCESSED_DIR, "training_data.csv")
CURRENT_PROCESSED = os.path.join(PROCESSED_DIR, "current_air_quality.csv")

DATA_AUDIT_REPORT = os.path.join(OUTPUTS_DIR, "data_audit_report.json")
COVERAGE_REPORT = os.path.join(OUTPUTS_DIR, "station_coverage_report.csv")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# ------------------------------------------------------------------
# PREDICTION TARGET
# ------------------------------------------------------------------
TARGET_HORIZON_HOURS = 6
TARGET_COLUMN = "target_pm25_6h"

# ------------------------------------------------------------------
# STATION SELECTION CRITERIA (documented, not arbitrary)
# A station qualifies for the modeling subset if it meets ALL of:
#   - PM2.5 non-missing rate >= MIN_PM25_COVERAGE
#   - total span of data >= MIN_SPAN_DAYS
#   - total number of PM2.5 readings >= MIN_RECORDS
# ------------------------------------------------------------------
MIN_PM25_COVERAGE = 0.60      # at least 60% of hourly slots have a PM2.5 reading
MIN_SPAN_DAYS = 365           # at least 1 full year of calendar coverage
MIN_RECORDS = 5000            # at least ~7 months worth of hourly readings

# ------------------------------------------------------------------
# CHRONOLOGICAL SPLIT (documented boundaries, decided after seeing
# the actual coverage of the selected stations - see notebooks/02)
# ------------------------------------------------------------------
TRAIN_END_DATE = "2019-01-01"      # < this date -> train
VALID_END_DATE = "2019-09-01"      # this date to VALID_END_DATE -> validation
                                     # VALID_END_DATE onward -> test

# ------------------------------------------------------------------
# OFFICIAL CPCB NATIONAL AQI BREAKPOINTS (µg/m3, except CO in mg/m3)
# Source: CPCB "National Air Quality Index", 2014
# Sub-index categories: Good, Satisfactory, Moderate, Poor, Very Poor, Severe
# ------------------------------------------------------------------
CPCB_PM25_BREAKPOINTS = [
    (0, 30, "Good"),
    (31, 60, "Satisfactory"),
    (61, 90, "Moderate"),
    (91, 120, "Poor"),
    (121, 250, "Very Poor"),
    (251, 10_000, "Severe"),
]

RANDOM_SEED = 42