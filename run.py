"""
Entry point for the AI Air Quality Prediction pipeline.

CURRENT STAGE: project structure + data verification only.
Model training is intentionally NOT wired in here yet (src/train.py does
not exist yet). Running this file right now will:

  1. Confirm the three raw CSVs are present in data/raw/
  2. Run the data audit
  3. Run historical preprocessing
  4. Process the current CPCB snapshot
  5. Run station selection
  6. Build the feature/target table

It will NOT train a model, NOT load a model, and NOT start Streamlit,
because those pieces haven't been built yet. This file will be extended
once src/train.py, src/evaluate.py, and app/app.py exist.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config


def verify_raw_data():
    required = [config.RAW_STATION_HOUR, config.RAW_STATIONS, config.RAW_CURRENT_SNAPSHOT]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing raw data file(s):\n" + "\n".join(missing) +
            "\nPlace the uploaded CSVs in data/raw/ before running the pipeline."
        )
    print("All 3 raw data files found in data/raw/.")


def main():
    verify_raw_data()

    from src import audit
    audit.run_audit()

    from src import preprocessing
    preprocessing.clean_historical()

    from src import current_data
    current_data.process_current_snapshot()

    from src import station_selection
    coverage, selected = station_selection.run()

    from src import feature_engineering
    feature_engineering.build_feature_table(selected["station_id"].tolist())

    print("\nStructure + preprocessing stage complete.")
    print("Model training is NOT part of this run yet - src/train.py has not been built.")


if __name__ == "__main__":
    main()