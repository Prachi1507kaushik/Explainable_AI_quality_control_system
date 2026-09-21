"""
PHASE 13 - STREAMLIT AI AIR QUALITY DASHBOARD

Run from project root:

    streamlit run app/app.py

Dashboard features:
- Station selection
- Current PM2.5
- 6-hour PM2.5 prediction
- Current vs predicted comparison
- Historical PM2.5 trend
- Prediction explanation using SHAP
- Top 5 features influencing the prediction
- Model information
- Responsible AI notice
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from src.predict import get_prediction


# ============================================================
# PATHS
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

HISTORICAL_FILE = (
    PROJECT_ROOT / "data" / "processed" / "training_data.csv"
)
STATIONS_FILE = RAW_DIR / "stations.csv"
CURRENT_FILE = RAW_DIR / "cpcb_current_snapshot.csv"

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

SHAP_SUMMARY = FIGURES_DIR / "shap_summary.png"
SHAP_GLOBAL = FIGURES_DIR / "shap_global_feature_importance.png"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Air Quality Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
    }

    .metric-title {
        font-size: 15px;
        color: #666;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 700;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🌍 AI Air Quality Prediction System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Explainable AI system for predicting PM2.5 concentration
    6 hours ahead using historical air-quality observations.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

@st.cache_data
def load_historical_data():

    if not HISTORICAL_FILE.exists():
        st.error(
            f"Historical dataset not found:\n{HISTORICAL_FILE}"
        )
        return pd.DataFrame()

    df = pd.read_csv(
        HISTORICAL_FILE,
        low_memory=False,
    )

    # Standardize columns

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
        )

    return df


# ============================================================
# LOAD STATION METADATA
# ============================================================

@st.cache_data
def load_station_metadata():

    if not STATIONS_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        STATIONS_FILE,
        low_memory=False,
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

historical_df = load_historical_data()
stations_df = load_station_metadata()


if historical_df.empty:
    st.stop()


# ============================================================
# FIND STATION COLUMN
# ============================================================

if "station_id" not in historical_df.columns:

    st.error(
        "The historical dataset does not contain station_id."
    )

    st.stop()


# ============================================================
# AVAILABLE STATIONS
# ============================================================

stations = sorted(
    historical_df["station_id"]
    .dropna()
    .astype(str)
    .unique()
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Prediction Settings")

selected_station = st.sidebar.selectbox(
    "Select Monitoring Station",
    stations,
)


st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### Prediction Configuration

    **Target:** PM2.5

    **Forecast Horizon:** 6 hours

    **Model:** XGBoost

    **Approach:** Supervised Machine Learning

    **Explainability:** SHAP
    """
)


# ============================================================
# RUN PREDICTION
# ============================================================

if st.sidebar.button(
    "🔮 Generate Prediction",
    use_container_width=True,
):

    st.session_state["run_prediction"] = True


if "run_prediction" not in st.session_state:

    st.info(
        "Select a monitoring station and click "
        "**Generate Prediction** to begin."
    )

    st.stop()


# ============================================================
# GENERATE PREDICTION
# ============================================================

try:

    with st.spinner(
        "Generating 6-hour PM2.5 prediction..."
    ):

        result = get_prediction(
            selected_station
        )

except Exception as e:

    st.error(
        f"Prediction failed:\n\n{e}"
    )

    st.stop()


# ============================================================
# EXTRACT RESULTS
# ============================================================

station = result.get(
    "station",
    selected_station,
)

city = result.get(
    "city",
    "Unknown",
)

current_pm25 = result.get(
    "current_pm25",
    np.nan,
)

predicted_pm25 = result.get(
    "predicted_pm25",
    np.nan,
)

horizon = result.get(
    "prediction_horizon_hours",
    6,
)

current_timestamp = result.get(
    "current_timestamp",
    None,
)

prediction_timestamp = result.get(
    "prediction_timestamp",
    None,
)

model_name = result.get(
    "model",
    "XGBoost",
)


# ============================================================
# AQI / RISK CATEGORY
# ============================================================

def pm25_category(value):

    if pd.isna(value):
        return "Unknown"

    value = float(value)

    if value <= 30:
        return "Good"

    elif value <= 60:
        return "Satisfactory"

    elif value <= 90:
        return "Moderate"

    elif value <= 120:
        return "Poor"

    elif value <= 250:
        return "Very Poor"

    else:
        return "Severe"


current_category = pm25_category(
    current_pm25
)

predicted_category = pm25_category(
    predicted_pm25
)


# ============================================================
# HEADER INFORMATION
# ============================================================

st.markdown(
    f"### 📍 {city} — {station}"
)

if current_timestamp is not None:

    st.caption(
        f"Current observation: {current_timestamp}"
    )


# ============================================================
# MAIN METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Current PM2.5",
        (
            f"{float(current_pm25):.2f} µg/m³"
            if not pd.isna(current_pm25)
            else "N/A"
        ),
    )


with col2:

    st.metric(
        "Predicted PM2.5",
        (
            f"{float(predicted_pm25):.2f} µg/m³"
            if not pd.isna(predicted_pm25)
            else "N/A"
        ),
    )


with col3:

    st.metric(
        "Forecast Horizon",
        f"{horizon} hours",
    )


with col4:

    st.metric(
        "Model",
        str(model_name).upper(),
    )


# ============================================================
# CURRENT / PREDICTED CATEGORY
# ============================================================

st.markdown("---")

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Current Air Quality"
    )

    st.markdown(
        f"""
        <div class="info-box">

        <b>PM2.5:</b> {current_pm25:.2f} µg/m³<br><br>

        <b>Category:</b> {current_category}

        </div>
        """,
        unsafe_allow_html=True,
    )


with col2:

    st.subheader(
        "6-Hour Forecast"
    )

    st.markdown(
        f"""
        <div class="info-box">

        <b>Predicted PM2.5:</b>
        {predicted_pm25:.2f} µg/m³<br><br>

        <b>Category:</b>
        {predicted_category}

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CHANGE IN PM2.5
# ============================================================

if (
    not pd.isna(current_pm25)
    and not pd.isna(predicted_pm25)
):

    change = (
        predicted_pm25
        - current_pm25
    )

    percentage_change = (
        change / current_pm25 * 100
        if current_pm25 != 0
        else np.nan
    )

    st.subheader(
        "📈 Forecast Change"
    )

    if change > 0:

        st.warning(
            f"Predicted PM2.5 is expected to increase "
            f"by approximately {change:.2f} µg/m³ "
            f"({percentage_change:.1f}%)."
        )

    elif change < 0:

        st.success(
            f"Predicted PM2.5 is expected to decrease "
            f"by approximately {abs(change):.2f} µg/m³ "
            f"({abs(percentage_change):.1f}%)."
        )

    else:

        st.info(
            "Predicted PM2.5 remains approximately unchanged."
        )


# ============================================================
# HISTORICAL TREND
# ============================================================

st.markdown("---")

st.header(
    "📊 Historical PM2.5 Trend"
)


station_data = historical_df[
    historical_df["station_id"].astype(str)
    == str(selected_station)
].copy()


if (
    "pm2_5" in station_data.columns
    and "datetime" in station_data.columns
):

    station_data = (
        station_data
        .dropna(subset=["datetime"])
        .sort_values("datetime")
    )

    # Last 7 days / 168 hours

    recent = station_data.tail(168)

    if not recent.empty:

        fig, ax = plt.subplots(
            figsize=(12, 4)
        )

        ax.plot(
            recent["datetime"],
            recent["pm2_5"],
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "PM2.5 (µg/m³)"
        )

        ax.set_title(
            f"Recent PM2.5 — {selected_station}"
        )

        ax.grid(
            alpha=0.25
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)

    else:

        st.info(
            "Not enough historical data available."
        )


# ============================================================
# PREDICTION TIMESTAMP
# ============================================================

if prediction_timestamp is not None:

    st.info(
        f"🕒 Forecast timestamp: "
        f"**{prediction_timestamp}**"
    )


# ============================================================
# EXPLAINABLE AI
# ============================================================

st.markdown("---")

st.header(
    "🧠 Explainable AI"
)

st.write(
    """
    The prediction is generated by an XGBoost model.
    SHAP is used to explain which input features influenced
    the model prediction.
    """
)


# ============================================================
# LOAD SHAP GLOBAL IMPORTANCE
# ============================================================

if SHAP_GLOBAL.exists():

    st.subheader(
        "Global Feature Importance"
    )

    st.image(
        str(SHAP_GLOBAL),
        use_container_width=True,
    )

else:

    st.info(
        "Global SHAP feature importance has not been generated yet."
    )


# ============================================================
# SHAP SUMMARY
# ============================================================

if SHAP_SUMMARY.exists():

    st.subheader(
        "SHAP Summary"
    )

    st.image(
        str(SHAP_SUMMARY),
        use_container_width=True,
    )


# ============================================================
# TOP FEATURES
# ============================================================

st.subheader(
    "Top Features Influencing the Model Prediction"
)

try:

    from src.explainability import (
        get_top_features_for_prediction,
    )

    top_features = (
        get_top_features_for_prediction(
            selected_station
        )
    )

    if top_features:

        for i, feature in enumerate(
            top_features[:5],
            start=1,
        ):

            if isinstance(feature, dict):

                feature_name = feature.get(
                    "feature",
                    "Unknown",
                )

                shap_value = feature.get(
                    "shap_value",
                    None,
                )

                if shap_value is not None:

                    direction = (
                        "increased"
                        if shap_value > 0
                        else "decreased"
                    )

                    st.write(
                        f"**{i}. {feature_name}** — "
                        f"influenced the model prediction "
                        f"toward a {direction} predicted PM2.5."
                    )

                else:

                    st.write(
                        f"**{i}. {feature_name}**"
                    )

            else:

                st.write(
                    f"**{i}. {feature}**"
                )

    else:

        st.info(
            "Individual SHAP explanation is unavailable."
        )

except Exception as e:

    st.warning(
        "Individual SHAP explanation could not be loaded."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

st.header(
    "🤖 Model Information"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Prediction Target",
        "PM2.5",
    )


with col2:

    st.metric(
        "Prediction Horizon",
        "6 Hours",
    )


with col3:

    st.metric(
        "Algorithm",
        "XGBoost",
    )


# ============================================================
# RESPONSIBLE AI
# ============================================================

st.markdown("---")

st.header(
    "⚖️ Responsible AI Considerations"
)

st.markdown(
    """
    **Transparency**

    The system uses XGBoost and SHAP to provide an explanation
    of features influencing the model prediction.

    **Important distinction**

    SHAP explanations identify **features influencing the model
    prediction**. They do **not** establish causes of pollution.

    **Data limitations**

    Prediction quality depends on the availability and quality
    of historical monitoring data.

    **Privacy**

    The system operates on environmental monitoring data and
    does not require personal user information.

    **Decision support**

    This system is intended as an AI-based decision-support and
    forecasting prototype. It should not replace official
    environmental monitoring or government-issued air-quality
    advisories.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Air Quality Prediction System | "
    "1M1B AI for Sustainability Virtual Internship | "
    "IBM SkillsBuild & AICTE"
)