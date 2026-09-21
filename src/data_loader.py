
"""
Loads the three raw datasets.
"""

import pandas as pd
from src import config


def load_station_hour():
    return pd.read_csv(config.RAW_STATION_HOUR, low_memory=False)


def load_stations():
    return pd.read_csv(config.RAW_STATIONS)


def load_current_snapshot():
    return pd.read_csv(config.RAW_CURRENT_SNAPSHOT)