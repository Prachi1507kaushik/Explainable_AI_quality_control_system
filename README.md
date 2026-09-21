# Explainable AI Air Quality Prediction System

An explainable machine-learning system for predicting **PM2.5 concentration 6 hours ahead** using historical air-quality observations from monitoring stations across India.

The project combines time-series feature engineering, machine-learning prediction, model evaluation, and SHAP-based explainability to provide transparent PM2.5 forecasts.

---

## 1. Project Overview

Air pollution can vary significantly with time, location, weather-related patterns, and pollutant concentrations. PM2.5 is one of the important indicators used to understand air-quality conditions.

This project develops an AI-based system that uses historical station-level air-quality observations to predict the **PM2.5 concentration 6 hours into the future**.

The system focuses on:

* Historical air-quality analysis
* Station-level PM2.5 forecasting
* Time-series feature engineering
* XGBoost-based prediction
* Model evaluation on an untouched chronological test set
* Explainable AI using SHAP
* Current-condition monitoring using a separate CPCB snapshot

---

# 2. Problem Statement

Traditional air-quality monitoring mainly describes current or previously observed conditions.

The objective of this project is:

> **To develop an explainable AI system that predicts PM2.5 concentration 6 hours ahead using historical air-quality observations while providing transparent information about the features influencing the model prediction.**

The system is designed as a decision-support and forecasting tool rather than an official government warning system.

---

# 3. Objectives

The major objectives are:

1. Collect and inspect publicly available air-quality data.
2. Clean and preprocess historical station-level observations.
3. Handle missing values without introducing future-data leakage.
4. Select stations with sufficient historical coverage.
5. Generate time-series features.
6. Predict PM2.5 concentration 6 hours ahead.
7. Compare machine-learning models with baseline approaches.
8. Evaluate the selected model on an untouched chronological test set.
9. Apply SHAP for model explainability.
10. Identify the features influencing individual predictions.
11. Keep current CPCB monitoring data separate from historical training data.

---

# 4. Dataset

The project uses three raw data assets.

### 4.1 Historical Station-Level Data

**File:**

```text
data/raw/station_hour.csv
```

This dataset contains historical hourly air-quality observations used for model development.

The historical data includes pollutant measurements such as:

* PM2.5
* PM10
* NO
* NO2
* NOx
* NH3
* CO
* SO2
* O3
* Benzene
* Toluene
* Xylene
* AQI

The historical dataset is used for:

* preprocessing
* feature engineering
* model training
* validation
* testing
* forecasting

---

### 4.2 Station Metadata

**File:**

```text
data/raw/stations.csv
```

This file provides station-related information used to enrich the historical observations, including information such as:

* Station ID
* Station name
* City
* State

---

### 4.3 Current CPCB Snapshot

**File:**

```text
data/raw/cpcb_current_snapshot.csv
```

This dataset represents current air-quality monitoring information.

It is processed separately from the historical training dataset.

**Important:** The CPCB snapshot is **not used as historical training data**.

Its purpose is to provide current monitoring information and support current-condition display/inference.

---

# 5. System Architecture

The overall pipeline is:

```text
                 ┌─────────────────────────┐
                 │ Historical Air Quality  │
                 │   station_hour.csv      │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Data Preprocessing      │
                 │ Missing values          │
                 │ Data cleaning            │
                 │ Station metadata         │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Station Selection       │
                 │ Coverage-based filtering│
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Feature Engineering     │
                 │ Lags + Rolling Means    │
                 │ Time Features           │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Chronological Split     │
                 │ Train / Validation/Test │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Model Training           │
                 │ XGBoost                  │
                 └────────────┬────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
          ┌─────────────────┐   ┌─────────────────┐
          │ Model Evaluation│   │ SHAP Explainable│
          │ MAE/RMSE/R²     │   │ AI              │
          └─────────────────┘   └─────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ PM2.5 6-Hour Prediction │
                 └─────────────────────────┘


       Separate Monitoring Pipeline

       CPCB Current Snapshot
                │
                ▼
       Current Air Quality
       Monitoring / Inference
```

---

# 6. Prediction Target

The prediction target is:

```text
target_pm25_6h
```

It represents the PM2.5 concentration approximately **6 hours after the current observation** for the same monitoring station.

The target is created using station-level chronological ordering.

A future value is only considered a valid target when the corresponding timestamp is exactly 6 hours ahead.

---

# 7. Data Preprocessing

The historical data preprocessing pipeline performs the following operations:

### Column Standardization

Column names are converted into a consistent format.

### Duplicate Removal

Exact duplicate rows are removed.

Duplicate station-timestamp combinations are detected and handled.

### Timestamp Processing

Historical timestamps are converted into datetime format and the observations are sorted by:

```text
station_id
datetime
```

### Numeric Conversion

Pollutant measurements are converted to numeric values.

### Invalid Values

Negative pollutant values are flagged rather than silently removed.

### Station Metadata

Historical observations are enriched using the station metadata file.

### Missing Values

Short missing-value gaps are handled using station-specific forward filling.

The implemented preprocessing uses a maximum fill limit of:

```text
2 hours
```

Longer gaps remain missing rather than being artificially interpolated.

This approach avoids using future observations to fill historical values.

---

# 8. Station Selection

Stations are selected using measurable historical-data coverage criteria.

The implemented criteria include:

```text
PM2.5 coverage >= 60%
Historical span >= 365 days
PM2.5 records >= 5000
```

A complete station coverage report is generated so that the selection remains auditable.

Output:

```text
outputs/station_coverage_report.csv
```

---

# 9. Feature Engineering

The model uses multiple categories of features.

## Time Features

Examples include:

```text
hour
day_of_week
day_of_year
month
is_weekend
```

Cyclical representations are also generated:

```text
hour_sin
hour_cos
month_sin
month_cos
```

---

## Lag Features

Historical pollutant values are used at different time lags:

```text
1 hour
3 hours
6 hours
12 hours
24 hours
```

For example:

```text
pm2_5_lag_1h
pm2_5_lag_3h
pm2_5_lag_6h
pm2_5_lag_12h
pm2_5_lag_24h
```

Similar lag features are generated for other pollutants.

---

## Rolling Features

Rolling historical averages are generated using:

```text
3 hours
6 hours
12 hours
24 hours
```

Examples:

```text
pm2_5_roll_mean_3h
pm2_5_roll_mean_6h
pm2_5_roll_mean_12h
pm2_5_roll_mean_24h
```

The rolling calculations use previous observations so that the current target does not leak into the features.

---

# 10. Leakage Prevention

Data leakage prevention is an important part of the project.

The following practices are used:

* Historical observations are sorted chronologically.
* Lag features only use previous observations.
* Rolling features exclude the current observation where required.
* The target represents a future PM2.5 value.
* Train, validation, and test sets are separated chronologically.
* Test data is not used for model training.
* Missing-value imputation is fitted using training data only.
* The test set remains untouched during model selection.

---

# 11. Chronological Data Split

The project uses a chronological split rather than a random split.

Configured boundaries:

```text
Training:
Before 2019-01-01

Validation:
2019-01-01 to 2019-08-31

Testing:
2019-09-01 onward
```

This better represents a real forecasting scenario because the model learns from earlier observations and is evaluated on later observations.

---

# 12. Machine Learning Models

The training pipeline compares multiple approaches.

### Persistence Baseline

The current PM2.5 value is used as a simple future estimate.

### Linear Regression

A linear regression model is trained using the engineered features.

### Random Forest

A tree-based ensemble model is evaluated.

### XGBoost

An XGBoost regression model is trained for the final prediction task.

The selected model is determined using validation performance.

The current trained model is:

```text
XGBoost
```

---

# 13. Model Selection

The model selection criterion used in the training pipeline is:

```text
Lowest validation MAE
```

The selected model and associated preprocessing information are saved for later prediction.

Model files include:

```text
models/best_model.pkl
models/feature_columns.pkl
models/model_metadata.json
```

---

# 14. Model Evaluation

The model is evaluated on the previously untouched test set.

The evaluation pipeline calculates:

### Mean Absolute Error

```text
MAE
```

### Root Mean Squared Error

```text
RMSE
```

### Coefficient of Determination

```text
R²
```

Additional analysis includes:

* Actual vs predicted PM2.5
* Residual analysis
* Error distribution
* Station-level performance
* High-pollution performance

Metrics are stored under:

```text
outputs/metrics/
```

Visualizations are stored under:

```text
outputs/figures/
```

---

# 15. Explainable AI

The project uses **SHAP (SHapley Additive exPlanations)** to analyze the trained XGBoost model.

The objective is to understand which input features influence the model prediction.

The system generates:

### Global Feature Importance

Shows the features with the largest overall contribution to model predictions.

### SHAP Summary Plot

Provides a global view of how features influence model outputs across observations.

### Individual Prediction Explanation

For an individual prediction, SHAP values are used to identify the features influencing that specific model output.

The project specifically describes these as:

> **features influencing the model prediction**

It does **not** interpret model features as direct causes of pollution.

---

# 16. Current Air-Quality Monitoring

The current CPCB dataset is handled independently from the historical training data.

The current snapshot is processed from:

```text
data/raw/cpcb_current_snapshot.csv
```

It is transformed into a station-level representation containing pollutant measurements.

The current dataset is used for monitoring current conditions.

It is **not merged into the historical training dataset** and is **not used to retrain the model**.

---

# 17. Prediction Module

The prediction module is:

```text
src/predict.py
```

It loads the saved model and feature configuration without retraining.

The prediction workflow is:

```text
Station Selection
       ↓
Historical Station Data
       ↓
Feature Generation
       ↓
Saved XGBoost Model
       ↓
6-Hour PM2.5 Prediction
```

The prediction result contains information such as:

```text
station
city
current_pm25
predicted_pm25
prediction_horizon_hours
timestamp
prediction_timestamp
model
```

---

# 18. Project Structure

```text
AI-Air_Quality-Prediction/
│
├── data/
│   ├── raw/
│   │   ├── station_hour.csv
│   │   ├── stations.csv
│   │   └── cpcb_current_snapshot.csv
│   │
│   └── processed/
│       ├── historical_clean.csv
│       ├── training_data.csv
│       ├── current_air_quality.csv
│       ├── preprocessing_log.json
│       └── current_processing_log.json
│
├── models/
│   ├── best_model.pkl
│   ├── feature_columns.pkl
│   └── model_metadata.json
│
├── outputs/
│   ├── figures/
│   │   ├── shap_global_feature_importance.png
│   │   ├── shap_summary.png
│   │   └── ...
│   │
│   ├── metrics/
│   │   └── ...
│   │
│   ├── predictions/
│   │   └── ...
│   │
│   ├── data_audit_report.json
│   └── station_coverage_report.csv
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── audit.py
│   ├── preprocessing.py
│   ├── current_data.py
│   ├── station_selection.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explainability.py
│   └── predict.py
│
├── run.py
├── requirements.txt
└── README.md
```

---

# 19. Main Technologies

The project uses:

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib
* Joblib

---

# 20. Installation

Create and activate a Python environment if required.

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

# 21. Running the Pipeline

Place the three raw datasets inside:

```text
data/raw/
```

Then execute the relevant project stages.

### Data Audit

```bash
python src/audit.py
```

### Historical Preprocessing

```bash
python src/preprocessing.py
```

### Current CPCB Processing

```bash
python src/current_data.py
```

### Station Selection

```bash
python src/station_selection.py
```

### Feature Engineering

```bash
python src/feature_engineering.py
```

### Model Training

```bash
python src/train.py
```

### Model Evaluation

```bash
python src/evaluate.py
```

### Explainable AI

```bash
python src/explainability.py
```

### PM2.5 Prediction

```bash
python src/predict.py
```

---

# 22. Example Prediction Output

An example prediction contains:

```text
Station              : AP001
City                 : Amaravati
Current PM2.5        : 22.0
Predicted PM2.5      : 19.72
Prediction horizon   : 6 hours
Prediction timestamp : 2020-07-01 06:00:00
Model                : xgboost
```

The exact prediction changes depending on the selected station and available input data.

---

# 23. Explainability Example

The SHAP analysis identified features such as:

```text
pm2_5_roll_mean_24h
pm2_5
hour_sin
month_cos
hour_cos
hour
day_of_year
pm10_roll_mean_24h
aqi
pm2_5_roll_mean_6h
```

as important features in the analyzed predictions.

These features describe patterns used by the trained model. They should not be interpreted as individually proving the physical cause of pollution.

---

# 24. Responsible AI Considerations

### Transparency

The system provides model evaluation metrics and SHAP-based explanations to make model behavior more interpretable.

### Data Leakage Prevention

Chronological splitting and training-only preprocessing are used to reduce the risk of future information entering the training process.

### Privacy

The project uses environmental monitoring data rather than personally identifiable information.

### Limitations

Predictions depend on:

* historical data quality
* monitoring-station coverage
* pollutant measurement availability
* temporal patterns represented in the training data
* the selected machine-learning model

The model should therefore be treated as a forecasting tool rather than a guaranteed prediction.

### Government Warning Disclaimer

The model's output is **not an official government air-quality warning**.

CPCB data used for current monitoring is kept conceptually separate from the historical data used to train the prediction model.

---

# 25. Limitations of the Current System

The current implementation has several limitations:

1. The prediction model is trained using historical observations.
2. Future environmental conditions may differ from historical patterns.
3. Some pollutant observations may contain missing values.
4. Model performance may vary between monitoring stations.
5. The system predicts PM2.5 concentration rather than directly predicting all dimensions of air quality.
6. Predictions should not be interpreted as official government forecasts.
7. SHAP explanations describe model behavior and should not be interpreted as causal explanations.

---

# 26. Expected Impact

The project demonstrates how machine learning can support air-quality forecasting by transforming historical monitoring data into short-term PM2.5 predictions.

Potential applications include:

* Air-quality monitoring dashboards
* Environmental data analysis
* Research and educational applications
* Decision-support systems
* Short-term pollution forecasting

The system is designed to demonstrate the responsible application of AI to an environmental sustainability problem.

---

# 27. SDG Alignment

### Primary SDG

**SDG 11 — Sustainable Cities and Communities**

The project supports the broader goal of sustainable urban environments by providing an AI-based approach for monitoring and forecasting air-quality conditions.

### Secondary SDG

**SDG 13 — Climate Action**

The project demonstrates the use of data and AI for environmental monitoring and analysis.

---

# 28. Conclusion

The AI Air Quality Prediction System demonstrates an end-to-end machine-learning workflow for **6-hour-ahead PM2.5 prediction**.

The project combines:

```text
Public Air-Quality Data
        ↓
Data Cleaning
        ↓
Station Selection
        ↓
Time-Series Feature Engineering
        ↓
Chronological Model Training
        ↓
XGBoost Prediction
        ↓
Independent Test Evaluation
        ↓
SHAP Explainability
        ↓
6-Hour PM2.5 Forecast
```

The focus of the project is not only prediction accuracy but also **data integrity, leakage prevention, model evaluation, and explainability**.

---

## Author

**Prachi Kaushik**

B.Tech — Computer Science and Engineering
ABES Institute of Technology, Ghaziabad
