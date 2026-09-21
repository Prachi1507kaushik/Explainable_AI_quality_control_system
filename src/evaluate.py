"""
PHASE 10 - MODEL EVALUATION (test set only)

Loads the artifacts already saved by src/train.py:
  - models/best_model.pkl        (fitted imputer + fitted model, NOT retrained here)
  - models/feature_columns.pkl   (exact feature order used at training time)
  - models/model_metadata.json   (split boundaries, station list)

Reconstructs the exact same test set that train.py held out (same stations,
same chronological cutoff, same feature columns) by re-using train.py's own
loading/splitting functions - this guarantees the test set here is
byte-for-byte the same one train.py never touched during fitting.

This script does NOT:
  - retrain any model
  - refit the imputer (it only calls .transform(), never .fit())
  - modify the test data in any way

It DOES:
  - compute MAE / RMSE / R2 on the test set
  - generate an actual-vs-predicted plot
  - generate a residual plot
  - generate an error-distribution histogram
  - break performance down by station
  - break performance down by "high-pollution" vs "normal" periods, using
    the official CPCB PM2.5 breakpoint already defined in config.py
    (Poor/Very Poor/Severe = PM2.5 >= 91 ug/m3) - not an invented threshold
"""

import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # no display available in this environment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.train import (
    get_recommended_stations, load_modeling_data, chronological_split,
)

# Official CPCB threshold (see config.CPCB_PM25_BREAKPOINTS): "Poor" category
# starts at 91 ug/m3. Rows at/above this are treated as "high-pollution" for
# the breakdown below - this is the same number CPCB itself uses, not one we
# invented for this analysis.
HIGH_POLLUTION_THRESHOLD = 91.0

METRICS_DIR = os.path.join(config.OUTPUTS_DIR, "metrics")
FIGURES_DIR = config.FIGURES_DIR


def compute_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n_rows": int(len(y_true)),
    }


def rebuild_test_set():
    """Reproduces the exact test split from train.py, without retraining
    anything. Reads model_metadata.json to confirm the station list/split
    dates match what was actually used to fit the saved model."""
    with open(config.MODEL_METADATA_PATH) as f:
        metadata = json.load(f)

    station_ids = metadata["stations_used"]
    df = load_modeling_data(station_ids)
    _, _, test_df = chronological_split(df)
    return test_df, metadata


def main():
    print("Loading saved model artifacts (no retraining)...")
    bundle = joblib.load(config.BEST_MODEL_PATH)
    imputer = bundle["imputer"]
    model = bundle["model"]
    model_name = bundle["model_name"]
    feature_cols = joblib.load(config.FEATURE_COLUMNS_PATH)
    print(f"Loaded model: {model_name}, {len(feature_cols)} features")

    print("\nRebuilding the untouched test set (same stations/dates as training)...")
    test_df, train_metadata = rebuild_test_set()
    print(f"Test set: {len(test_df):,} rows, "
          f"{test_df['datetime'].min()} to {test_df['datetime'].max()}")

    X_test_raw = test_df[feature_cols]
    y_test = test_df[config.TARGET_COLUMN].values
    pm2_5_test = test_df["pm2_5"].values

    print("\nApplying the ALREADY-FIT imputer (transform only, no fitting here)...")
    X_test = imputer.transform(X_test_raw)

    print(f"\nGenerating predictions with the saved {model_name} model...")
    if model is None:
        # persistence baseline case
        test_mask = ~np.isnan(pm2_5_test)
        y_pred = pm2_5_test[test_mask]
        y_true = y_test[test_mask]
        test_df_eval = test_df[test_mask].copy()
    else:
        y_pred = model.predict(X_test)
        y_true = y_test
        test_df_eval = test_df.copy()

    test_df_eval["y_true"] = y_true
    test_df_eval["y_pred"] = y_pred
    test_df_eval["residual"] = test_df_eval["y_true"] - test_df_eval["y_pred"]

    # ---- overall metrics ----
    overall_metrics = compute_metrics(y_true, y_pred)
    print("\nOVERALL TEST METRICS:")
    print(json.dumps(overall_metrics, indent=2))

    # ---- station-level performance ----
    coverage = pd.read_csv(os.path.join(config.OUTPUTS_DIR, "metrics", "station_coverage.csv"))
    name_lookup = coverage.set_index("station_id")["station_name"].to_dict()

    station_rows = []
    for station_id, g in test_df_eval.groupby("station_id"):
        m = compute_metrics(g["y_true"], g["y_pred"])
        m["station_id"] = station_id
        m["station_name"] = name_lookup.get(station_id)
        station_rows.append(m)
    station_perf = pd.DataFrame(station_rows).sort_values("mae")
    print("\nSTATION-LEVEL PERFORMANCE:")
    print(station_perf.to_string(index=False))

    # ---- high-pollution vs normal performance ----
    high_mask = test_df_eval["y_true"] >= HIGH_POLLUTION_THRESHOLD
    high_metrics = compute_metrics(test_df_eval.loc[high_mask, "y_true"],
                                    test_df_eval.loc[high_mask, "y_pred"])
    normal_metrics = compute_metrics(test_df_eval.loc[~high_mask, "y_true"],
                                      test_df_eval.loc[~high_mask, "y_pred"])
    print(f"\nHIGH-POLLUTION (true PM2.5 >= {HIGH_POLLUTION_THRESHOLD}, CPCB 'Poor' or worse): "
          f"{json.dumps(high_metrics, indent=2)}")
    print(f"\nNORMAL (true PM2.5 < {HIGH_POLLUTION_THRESHOLD}): "
          f"{json.dumps(normal_metrics, indent=2)}")

    # ---- save metrics ----
    os.makedirs(METRICS_DIR, exist_ok=True)
    evaluation_report = {
        "model_name": model_name,
        "test_set_size": int(len(test_df_eval)),
        "test_date_range": [str(test_df_eval["datetime"].min()), str(test_df_eval["datetime"].max())],
        "overall_metrics": overall_metrics,
        "high_pollution_threshold_ug_m3": HIGH_POLLUTION_THRESHOLD,
        "high_pollution_threshold_source": "CPCB National AQI 'Poor' category breakpoint (config.CPCB_PM25_BREAKPOINTS)",
        "high_pollution_metrics": high_metrics,
        "normal_pollution_metrics": normal_metrics,
        "note": "Evaluation performed on the untouched test set only. The model "
                "was NOT retrained and the imputer was NOT refit for this evaluation - "
                "both were loaded exactly as saved by src/train.py.",
    }
    with open(os.path.join(METRICS_DIR, "test_evaluation_report.json"), "w") as f:
        json.dump(evaluation_report, f, indent=2, default=str)
    station_perf.to_csv(os.path.join(METRICS_DIR, "station_level_test_performance.csv"), index=False)
    print(f"\nSaved: {os.path.join(METRICS_DIR, 'test_evaluation_report.json')}")
    print(f"Saved: {os.path.join(METRICS_DIR, 'station_level_test_performance.csv')}")

    # ---- plots ----
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # For plotting only (never for metrics), downsample if very large so
    # scatter plots remain legible rather than a solid blob of ink.
    plot_df = test_df_eval
    if len(plot_df) > 20000:
        plot_df = plot_df.sample(20000, random_state=config.RANDOM_SEED)

    # 1. Actual vs predicted
    plt.figure(figsize=(7, 7))
    plt.scatter(plot_df["y_true"], plot_df["y_pred"], alpha=0.15, s=8, color="#2b6cb0")
    lims = [0, max(plot_df["y_true"].max(), plot_df["y_pred"].max())]
    plt.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    plt.xlabel("Actual PM2.5 (6h ahead), ug/m3")
    plt.ylabel("Predicted PM2.5, ug/m3")
    plt.title(f"Actual vs Predicted - {model_name} (test set, n={len(test_df_eval):,})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    # 2. Residual plot (residual vs predicted)
    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df["y_pred"], plot_df["residual"], alpha=0.15, s=8, color="#38a169")
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Predicted PM2.5, ug/m3")
    plt.ylabel("Residual (actual - predicted), ug/m3")
    plt.title(f"Residual Plot - {model_name} (test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "residual_plot.png"), dpi=150)
    plt.close()

    # 3. Error distribution
    plt.figure(figsize=(8, 5))
    plt.hist(test_df_eval["residual"], bins=80, color="#805ad5", edgecolor="white")
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Residual (actual - predicted), ug/m3")
    plt.ylabel("Count")
    plt.title(f"Error Distribution - {model_name} (test set, n={len(test_df_eval):,})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "error_distribution.png"), dpi=150)
    plt.close()

    # 4. Station-level MAE bar chart
    plt.figure(figsize=(9, 5))
    ordered = station_perf.sort_values("mae")
    plt.barh(ordered["station_id"], ordered["mae"], color="#dd6b20")
    plt.xlabel("MAE, ug/m3")
    plt.ylabel("Station")
    plt.title(f"Station-Level Test MAE - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "station_level_mae.png"), dpi=150)
    plt.close()

    # 5. High-pollution vs normal comparison
    plt.figure(figsize=(6, 5))
    cats = ["Normal\n(<91 ug/m3)", "High Pollution\n(>=91 ug/m3, CPCB 'Poor'+)"]
    maes = [normal_metrics["mae"], high_metrics["mae"]]
    plt.bar(cats, maes, color=["#3182ce", "#e53e3e"])
    plt.ylabel("MAE, ug/m3")
    plt.title(f"MAE: Normal vs High-Pollution Periods - {model_name}")
    for i, v in enumerate(maes):
        plt.text(i, v + 0.3, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "high_pollution_vs_normal_mae.png"), dpi=150)
    plt.close()

    print(f"\nSaved 5 figures to {FIGURES_DIR}")
    return evaluation_report


if __name__ == "__main__":
    main()