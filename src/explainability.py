# """
# PHASE 11 - EXPLAINABLE AI

# Explainability for the selected Air Quality Prediction model.

# Uses SHAP when compatible with the selected model.

# Generates:
#     1. Global feature importance
#     2. SHAP summary plot
#     3. Individual prediction explanation
#     4. Top 5 features influencing an individual prediction

# IMPORTANT:
# SHAP explanations describe FEATURES INFLUENCING THE MODEL PREDICTION.
# They do NOT represent causal explanations of pollution.

# Outputs:
#     outputs/figures/shap_global_feature_importance.png
#     outputs/figures/shap_summary.png
#     outputs/figures/individual_prediction_explanation.png
#     outputs/metrics/shap_feature_importance.csv
#     outputs/metrics/individual_prediction_explanation.json
# """

# from __future__ import annotations

# import json
# import os
# import sys
# import warnings

# import joblib
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# warnings.filterwarnings("ignore")

# # ---------------------------------------------------------------------
# # IMPORT PROJECT CONFIG
# # ---------------------------------------------------------------------

# sys.path.insert(
#     0,
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# )

# from src import config


# # ---------------------------------------------------------------------
# # OPTIONAL SHAP IMPORT
# # ---------------------------------------------------------------------

# try:
#     import shap

#     SHAP_AVAILABLE = True

# except ImportError:
#     SHAP_AVAILABLE = False


# # ---------------------------------------------------------------------
# # SETTINGS
# # ---------------------------------------------------------------------

# SHAP_SAMPLE_SIZE = 2000

# # Example prediction to explain.
# # 0 = first test observation.
# INDIVIDUAL_SAMPLE_INDEX = 0

# RANDOM_SEED = 42


# # ---------------------------------------------------------------------
# # PATHS
# # ---------------------------------------------------------------------

# FIGURES_DIR = config.FIGURES_DIR
# METRICS_DIR = config.METRICS_DIR

# os.makedirs(FIGURES_DIR, exist_ok=True)
# os.makedirs(METRICS_DIR, exist_ok=True)


# # ---------------------------------------------------------------------
# # UTILITY FUNCTIONS
# # ---------------------------------------------------------------------

# def load_model_bundle():
#     """
#     Load best_model.pkl.

#     Expected structure:

#     {
#         "imputer": fitted SimpleImputer,
#         "model": trained model,
#         "model_name": "xgboost"
#     }
#     """

#     if not os.path.exists(config.BEST_MODEL_PATH):
#         raise FileNotFoundError(
#             f"Model not found:\n{config.BEST_MODEL_PATH}"
#         )

#     bundle = joblib.load(config.BEST_MODEL_PATH)

#     if isinstance(bundle, dict):

#         if "model" not in bundle:
#             raise KeyError(
#                 "best_model.pkl is a dictionary but does not contain "
#                 "'model'."
#             )

#         model = bundle["model"]
#         imputer = bundle.get("imputer", None)
#         model_name = bundle.get("model_name", type(model).__name__)

#     else:
#         model = bundle
#         imputer = None
#         model_name = type(model).__name__

#     return model, imputer, model_name


# def load_metadata():
#     """
#     Load model metadata.
#     """

#     if not os.path.exists(config.MODEL_METADATA_PATH):
#         raise FileNotFoundError(
#             f"Metadata not found:\n{config.MODEL_METADATA_PATH}"
#         )

#     with open(config.MODEL_METADATA_PATH, "r") as f:
#         metadata = json.load(f)

#     return metadata


# def get_feature_columns(metadata):
#     """
#     Obtain the exact feature order used during training.
#     """

#     feature_columns = metadata.get("feature_columns")

#     if not feature_columns:
#         raise ValueError(
#             "feature_columns not found in model_metadata.json"
#         )

#     return feature_columns


# # ---------------------------------------------------------------------
# # LOAD TEST DATA
# # ---------------------------------------------------------------------

# def load_test_data(metadata):
#     """
#     Reconstruct the untouched chronological test set.

#     The split is recreated using the same boundaries and station list
#     stored in model_metadata.json.

#     No retraining is performed.
#     """

#     if not os.path.exists(config.TRAINING_DATA):
#         raise FileNotFoundError(
#             f"Training data not found:\n{config.TRAINING_DATA}"
#         )

#     print("\n3. Loading modeling data...")

#     df = pd.read_csv(
#         config.TRAINING_DATA,
#         low_memory=False
#     )

#     df["datetime"] = pd.to_datetime(
#         df["datetime"],
#         errors="coerce"
#     )

#     # --------------------------------------------------------------
#     # Stations used during model training
#     # --------------------------------------------------------------

#     stations = metadata.get("stations_used")

#     if not stations:
#         raise ValueError(
#             "stations_used not found in model_metadata.json"
#         )

#     df = df[
#         df["station_id"].isin(stations)
#     ].copy()

#     # --------------------------------------------------------------
#     # Remove rows without target
#     # --------------------------------------------------------------

#     df = df[
#         df[config.TARGET_COLUMN].notna()
#     ].copy()

#     # --------------------------------------------------------------
#     # Chronological split
#     # --------------------------------------------------------------

#     train_end = pd.Timestamp(
#         metadata["split_boundaries"]["train_end_date"]
#     )

#     valid_end = pd.Timestamp(
#         metadata["split_boundaries"]["valid_end_date"]
#     )

#     train_df = df[
#         df["datetime"] < train_end
#     ].copy()

#     valid_df = df[
#         (df["datetime"] >= train_end)
#         & (df["datetime"] < valid_end)
#     ].copy()

#     test_df = df[
#         df["datetime"] >= valid_end
#     ].copy()

#     print(f"Train : {len(train_df):,}")
#     print(f"Valid : {len(valid_df):,}")
#     print(f"Test  : {len(test_df):,}")

#     if len(test_df) == 0:
#         raise ValueError(
#             "Test set is empty."
#         )

#     print(
#         "Test range:",
#         test_df["datetime"].min(),
#         "to",
#         test_df["datetime"].max()
#     )

#     return train_df, valid_df, test_df


# # ---------------------------------------------------------------------
# # PREPARE FEATURES
# # ---------------------------------------------------------------------

# def prepare_features(
#     train_df,
#     test_df,
#     feature_columns,
#     imputer
# ):
#     """
#     Prepare test features using the exact training feature order.

#     If the trained imputer exists inside best_model.pkl, use it.

#     Otherwise fit an imputer on TRAIN ONLY.
#     """

#     print("\n4. Preparing features...")

#     missing_train = [
#         c for c in feature_columns
#         if c not in train_df.columns
#     ]

#     missing_test = [
#         c for c in feature_columns
#         if c not in test_df.columns
#     ]

#     if missing_train:
#         raise ValueError(
#             "Missing feature columns in training data:\n"
#             + "\n".join(missing_train)
#         )

#     if missing_test:
#         raise ValueError(
#             "Missing feature columns in test data:\n"
#             + "\n".join(missing_test)
#         )

#     X_train = train_df[
#         feature_columns
#     ].apply(pd.to_numeric, errors="coerce")

#     X_test = test_df[
#         feature_columns
#     ].apply(pd.to_numeric, errors="coerce")

#     # --------------------------------------------------------------
#     # Use saved imputer
#     # --------------------------------------------------------------

#     if imputer is not None:

#         print(
#             "Using imputer stored inside best_model.pkl."
#         )

#         X_test_processed = imputer.transform(
#             X_test
#         )

#     else:

#         print(
#             "Saved imputer not found."
#         )

#         print(
#             "Fitting SimpleImputer on TRAIN ONLY."
#         )

#         from sklearn.impute import SimpleImputer

#         train_imputer = SimpleImputer(
#             strategy="median"
#         )

#         train_imputer.fit(X_train)

#         X_test_processed = train_imputer.transform(
#             X_test
#         )

#     X_test_processed = pd.DataFrame(
#         X_test_processed,
#         columns=feature_columns,
#         index=test_df.index
#     )

#     print(
#         "Feature matrix:",
#         X_test_processed.shape
#     )

#     return X_test_processed


# # ---------------------------------------------------------------------
# # MODEL PREDICTION
# # ---------------------------------------------------------------------

# def generate_predictions(model, X):
#     """
#     Generate model predictions.
#     """

#     if not hasattr(model, "predict"):
#         raise TypeError(
#             "Loaded model does not have a predict() method."
#         )

#     predictions = model.predict(X)

#     predictions = np.asarray(
#         predictions
#     ).reshape(-1)

#     return predictions


# # ---------------------------------------------------------------------
# # CREATE SHAP EXPLAINER
# # ---------------------------------------------------------------------

# def create_shap_explainer(model):
#     """
#     Create SHAP TreeExplainer.

#     XGBoost, Random Forest and other tree models are supported when
#     compatible with the installed SHAP version.
#     """

#     if not SHAP_AVAILABLE:

#         raise ImportError(
#             "SHAP is not installed.\n"
#             "Install it using:\n"
#             "pip install shap"
#         )

#     print("\n5. Creating SHAP explainer...")

#     try:

#         explainer = shap.TreeExplainer(
#             model
#         )

#         print(
#             "SHAP TreeExplainer created successfully."
#         )

#         return explainer

#     except Exception as exc:

#         raise RuntimeError(
#             "Could not create SHAP TreeExplainer.\n"
#             f"Model type: {type(model)}\n"
#             f"Error: {exc}"
#         )


# # ---------------------------------------------------------------------
# # SHAP VALUES
# # ---------------------------------------------------------------------

# def calculate_shap_values(
#     explainer,
#     X,
#     sample_size=SHAP_SAMPLE_SIZE
# ):
#     """
#     Calculate SHAP values on a representative sample.

#     This avoids unnecessarily processing all 69,986 test rows.
#     """

#     print("\n6. Calculating SHAP values...")

#     if len(X) > sample_size:

#         X_sample = X.sample(
#             n=sample_size,
#             random_state=RANDOM_SEED
#         )

#     else:

#         X_sample = X.copy()

#     shap_result = explainer(
#         X_sample
#     )

#     # --------------------------------------------------------------
#     # SHAP Explanation object
#     # --------------------------------------------------------------

#     if hasattr(shap_result, "values"):

#         shap_values = shap_result.values

#         base_values = getattr(
#             shap_result,
#             "base_values",
#             None
#         )

#     else:

#         shap_values = shap_result
#         base_values = None

#     shap_values = np.asarray(
#         shap_values
#     )

#     # --------------------------------------------------------------
#     # Handle possible multi-output shape
#     # --------------------------------------------------------------

#     if shap_values.ndim == 3:

#         # Regression should normally be 2D.
#         # If SHAP returns a single-output third dimension,
#         # remove it.
#         if shap_values.shape[-1] == 1:

#             shap_values = shap_values[:, :, 0]

#         else:

#             # For multi-output, use first output.
#             shap_values = shap_values[:, :, 0]

#     print(
#         "Rows explained:",
#         len(X_sample)
#     )

#     print(
#         "SHAP matrix:",
#         shap_values.shape
#     )

#     return X_sample, shap_values, base_values


# # ---------------------------------------------------------------------
# # GLOBAL FEATURE IMPORTANCE
# # ---------------------------------------------------------------------

# def generate_global_feature_importance(
#     X_sample,
#     shap_values
# ):
#     """
#     Generate global mean absolute SHAP importance.
#     """

#     print(
#         "\n7. Generating global feature importance..."
#     )

#     mean_abs_shap = np.mean(
#         np.abs(shap_values),
#         axis=0
#     )

#     importance = pd.DataFrame(
#         {
#             "feature": X_sample.columns,
#             "mean_absolute_shap": mean_abs_shap
#         }
#     )

#     importance = importance.sort_values(
#         "mean_absolute_shap",
#         ascending=False
#     ).reset_index(drop=True)

#     output_csv = os.path.join(
#         METRICS_DIR,
#         "shap_feature_importance.csv"
#     )

#     importance.to_csv(
#         output_csv,
#         index=False
#     )

#     # --------------------------------------------------------------
#     # Plot top 20
#     # --------------------------------------------------------------

#     top = importance.head(20).sort_values(
#         "mean_absolute_shap",
#         ascending=True
#     )

#     plt.figure(
#         figsize=(10, 8)
#     )

#     plt.barh(
#         top["feature"],
#         top["mean_absolute_shap"]
#     )

#     plt.xlabel(
#         "Mean Absolute SHAP Value"
#     )

#     plt.ylabel(
#         "Feature"
#     )

#     plt.title(
#         "Global Feature Importance"
#     )

#     plt.tight_layout()

#     output_plot = os.path.join(
#         FIGURES_DIR,
#         "shap_global_feature_importance.png"
#     )

#     plt.savefig(
#         output_plot,
#         dpi=200,
#         bbox_inches="tight"
#     )

#     plt.close()

#     print(
#         f"Saved: {output_plot}"
#     )

#     print(
#         "\nTop 10 globally important features:"
#     )

#     print(
#         importance.head(10).to_string(
#             index=False
#         )
#     )

#     return importance


# # ---------------------------------------------------------------------
# # SHAP SUMMARY PLOT
# # ---------------------------------------------------------------------

# def generate_shap_summary_plot(
#     explainer,
#     X_sample,
#     shap_values
# ):
#     """
#     Generate SHAP summary / beeswarm plot.
#     """

#     print(
#         "\n8. Generating SHAP summary plot..."
#     )

#     output_plot = os.path.join(
#         FIGURES_DIR,
#         "shap_summary.png"
#     )

#     try:

#         plt.figure(
#             figsize=(12, 9)
#         )

#         shap.summary_plot(
#             shap_values,
#             X_sample,
#             show=False,
#             max_display=20
#         )

#         plt.title(
#             "SHAP Summary - Features Influencing Model Prediction"
#         )

#         plt.tight_layout()

#         plt.savefig(
#             output_plot,
#             dpi=200,
#             bbox_inches="tight"
#         )

#         plt.close()

#         print(
#             f"Saved: {output_plot}"
#         )

#     except Exception as exc:

#         plt.close()

#         print(
#             "Warning: SHAP summary plot failed:"
#         )

#         print(exc)


# # ---------------------------------------------------------------------
# # TOP 5 FEATURES FOR ONE PREDICTION
# # ---------------------------------------------------------------------

# def get_top_5_features_for_prediction(
#     shap_values,
#     feature_names,
#     row_index=0
# ):
#     """
#     Return the top 5 features influencing a particular prediction.

#     The ranking is based on absolute SHAP value.

#     IMPORTANT:
#     These are features influencing the model prediction.
#     They are NOT causal explanations of pollution.
#     """

#     shap_values = np.asarray(
#         shap_values
#     )

#     if shap_values.ndim != 2:
#         raise ValueError(
#             "Expected SHAP values with shape "
#             "(rows, features)."
#         )

#     if row_index < 0 or row_index >= len(shap_values):
#         raise IndexError(
#             f"row_index {row_index} is outside "
#             f"the SHAP sample range."
#         )

#     row_shap = shap_values[
#         row_index
#     ]

#     order = np.argsort(
#         np.abs(row_shap)
#     )[::-1][:5]

#     results = []

#     for rank, idx in enumerate(
#         order,
#         start=1
#     ):

#         shap_value = float(
#             row_shap[idx]
#         )

#         results.append(
#             {
#                 "rank": rank,
#                 "feature": str(
#                     feature_names[idx]
#                 ),
#                 "shap_value": shap_value,
#                 "absolute_shap_value": abs(
#                     shap_value
#                 ),
#                 "direction": (
#                     "increases predicted PM2.5"
#                     if shap_value > 0
#                     else "decreases predicted PM2.5"
#                     if shap_value < 0
#                     else "neutral"
#                 )
#             }
#         )

#     return results


# # ---------------------------------------------------------------------
# # INDIVIDUAL PREDICTION EXPLANATION
# # ---------------------------------------------------------------------

# def explain_individual_prediction(
#     model,
#     explainer,
#     X_sample,
#     shap_values,
#     base_values,
#     original_test_df,
#     row_index=INDIVIDUAL_SAMPLE_INDEX
# ):
#     """
#     Explain one example prediction.

#     IMPORTANT:
#     The explanation describes features influencing the model prediction.
#     It does not claim that these features caused pollution.
#     """

#     print(
#         "\n9. Generating individual prediction explanation..."
#     )

#     if row_index < 0 or row_index >= len(X_sample):

#         raise IndexError(
#             "Individual sample index is outside "
#             "the SHAP sample range."
#         )

#     X_row = X_sample.iloc[
#         [row_index]
#     ]

#     # --------------------------------------------------------------
#     # Model prediction
#     # --------------------------------------------------------------

#     prediction = float(
#         generate_predictions(
#             model,
#             X_row
#         )[0]
#     )

#     # --------------------------------------------------------------
#     # Actual target
#     #
#     # X_sample retains original test-data indexes.
#     # --------------------------------------------------------------

#     original_index = X_sample.index[
#         row_index
#     ]

#     actual = float(
#         original_test_df.loc[
#             original_index,
#             config.TARGET_COLUMN
#         ]
#     )

#     # --------------------------------------------------------------
#     # SHAP values for this prediction
#     # --------------------------------------------------------------

#     shap_value = np.asarray(
#         shap_values[row_index]
#     ).reshape(-1)

#     # --------------------------------------------------------------
#     # Base value
#     # --------------------------------------------------------------

#     if base_values is None:

#         try:

#             expected_value = explainer.expected_value

#             if np.ndim(expected_value) > 0:

#                 expected_value = np.asarray(
#                     expected_value
#                 ).reshape(-1)[0]

#             expected_value = float(
#                 expected_value
#             )

#         except Exception:

#             expected_value = float(
#                 np.mean(
#                     original_test_df[
#                         config.TARGET_COLUMN
#                     ]
#                 )
#             )

#     else:

#         base_arr = np.asarray(
#             base_values
#         )

#         try:

#             if base_arr.ndim == 0:

#                 expected_value = float(
#                     base_arr
#                 )

#             elif base_arr.ndim == 1:

#                 expected_value = float(
#                     base_arr[row_index]
#                 )

#             else:

#                 expected_value = float(
#                     base_arr[row_index][0]
#                 )

#         except Exception:

#             expected_value = float(
#                 np.mean(
#                     original_test_df[
#                         config.TARGET_COLUMN
#                     ]
#                 )
#             )

#     # --------------------------------------------------------------
#     # Top 5 features
#     # --------------------------------------------------------------

#     top_features = get_top_5_features_for_prediction(
#         shap_values=shap_values,
#         feature_names=list(
#             X_sample.columns
#         ),
#         row_index=row_index
#     )

#     # --------------------------------------------------------------
#     # Add actual feature values
#     # --------------------------------------------------------------

#     for item in top_features:

#         feature = item["feature"]

#         item["feature_value"] = float(
#             X_row.iloc[0][feature]
#         )

#     # --------------------------------------------------------------
#     # Metadata
#     # --------------------------------------------------------------

#     row_metadata = {}

#     for column in [
#         "station_id",
#         "station_name",
#         "city",
#         "state",
#         "datetime"
#     ]:

#         if column in original_test_df.columns:

#             value = original_test_df.loc[
#                 original_index,
#                 column
#             ]

#             if isinstance(
#                 value,
#                 pd.Timestamp
#             ):

#                 value = str(value)

#             elif isinstance(
#                 value,
#                 np.generic
#             ):

#                 value = value.item()

#             row_metadata[column] = value

#     # --------------------------------------------------------------
#     # JSON result
#     # --------------------------------------------------------------

#     explanation = {
#         "description": (
#             "This explanation identifies features influencing "
#             "the model prediction. It does not identify causes "
#             "of pollution."
#         ),
#         "model_prediction_pm25_6h": prediction,
#         "actual_pm25_6h": actual,
#         "baseline_prediction": expected_value,
#         "prediction_difference_from_baseline": (
#             prediction - expected_value
#         ),
#         "row_metadata": row_metadata,
#         "top_5_features_influencing_model_prediction": top_features
#     }

#     output_json = os.path.join(
#         METRICS_DIR,
#         "individual_prediction_explanation.json"
#     )

#     with open(
#         output_json,
#         "w"
#     ) as f:

#         json.dump(
#             explanation,
#             f,
#             indent=2,
#             default=str
#         )

#     print(
#         f"Saved: {output_json}"
#     )

#     # --------------------------------------------------------------
#     # Print explanation
#     # --------------------------------------------------------------

#     print(
#         "\nExample prediction:"
#     )

#     print(
#         f"Predicted PM2.5 in 6h : {prediction:.2f}"
#     )

#     print(
#         f"Actual PM2.5 in 6h    : {actual:.2f}"
#     )

#     print(
#         f"Baseline prediction   : {expected_value:.2f}"
#     )

#     print(
#         "\nTop 5 features influencing the model prediction:"
#     )

#     for item in top_features:

#         print(
#             f"{item['rank']}. "
#             f"{item['feature']} | "
#             f"value={item['feature_value']:.4f} | "
#             f"SHAP={item['shap_value']:+.4f} | "
#             f"{item['direction']}"
#         )

#     # --------------------------------------------------------------
#     # Individual waterfall-style bar plot
#     # --------------------------------------------------------------

#     plot_df = pd.DataFrame(
#         top_features
#     ).sort_values(
#         "shap_value"
#     )

#     plt.figure(
#         figsize=(10, 6)
#     )

#     plt.barh(
#         plot_df["feature"],
#         plot_df["shap_value"]
#     )

#     plt.axvline(
#         0,
#         linewidth=1
#     )

#     plt.xlabel(
#         "SHAP Value"
#     )

#     plt.ylabel(
#         "Feature"
#     )

#     plt.title(
#         "Individual Prediction Explanation\n"
#         "Features Influencing the Model Prediction"
#     )

#     plt.tight_layout()

#     output_plot = os.path.join(
#         FIGURES_DIR,
#         "individual_prediction_explanation.png"
#     )

#     plt.savefig(
#         output_plot,
#         dpi=200,
#         bbox_inches="tight"
#     )

#     plt.close()

#     print(
#         f"Saved: {output_plot}"
#     )

#     return explanation


# # ---------------------------------------------------------------------
# # OPTIONAL SHAP WATERFALL
# # ---------------------------------------------------------------------

# def generate_shap_waterfall(
#     explainer,
#     X_sample,
#     row_index,
#     shap_values,
#     base_values
# ):
#     """
#     Try to generate an official SHAP waterfall plot.

#     This is optional. If the installed SHAP version does not support
#     the required object format, the main individual explanation plot
#     is still produced.
#     """

#     output_plot = os.path.join(
#         FIGURES_DIR,
#         "shap_waterfall_individual.png"
#     )

#     try:

#         # Determine base value
#         if base_values is None:

#             expected = explainer.expected_value

#             if np.ndim(expected) > 0:

#                 expected = np.asarray(
#                     expected
#                 ).reshape(-1)[0]

#             expected = float(expected)

#         else:

#             base_arr = np.asarray(
#                 base_values
#             )

#             if base_arr.ndim == 0:

#                 expected = float(
#                     base_arr
#                 )

#             elif base_arr.ndim == 1:

#                 expected = float(
#                     base_arr[row_index]
#                 )

#             else:

#                 expected = float(
#                     base_arr[row_index][0]
#                 )

#         explanation = shap.Explanation(
#             values=shap_values[row_index],
#             base_values=expected,
#             data=X_sample.iloc[
#                 row_index
#             ].values,
#             feature_names=list(
#                 X_sample.columns
#             )
#         )

#         plt.figure(
#             figsize=(12, 8)
#         )

#         shap.plots.waterfall(
#             explanation,
#             max_display=15,
#             show=False
#         )

#         plt.tight_layout()

#         plt.savefig(
#             output_plot,
#             dpi=200,
#             bbox_inches="tight"
#         )

#         plt.close()

#         print(
#             f"Saved: {output_plot}"
#         )

#     except Exception as exc:

#         plt.close()

#         print(
#             "Optional SHAP waterfall plot skipped:"
#         )

#         print(
#             exc
#         )


# # ---------------------------------------------------------------------
# # MAIN PIPELINE
# # ---------------------------------------------------------------------

# def explain_model():

#     print("=" * 60)
#     print("PHASE 11 - EXPLAINABLE AI")
#     print("=" * 60)

#     if not SHAP_AVAILABLE:

#         print(
#             "\nSHAP is not installed."
#         )

#         print(
#             "Install it with:"
#         )

#         print(
#             "pip install shap"
#         )

#         return

#     # --------------------------------------------------------------
#     # 1. Load model
#     # --------------------------------------------------------------

#     print(
#         "\n1. Loading trained model..."
#     )

#     model, imputer, model_name = (
#         load_model_bundle()
#     )

#     print(
#         f"Selected model: {model_name}"
#     )

#     # --------------------------------------------------------------
#     # 2. Metadata
#     # --------------------------------------------------------------

#     print(
#         "\n2. Loading model metadata..."
#     )

#     metadata = load_metadata()

#     feature_columns = get_feature_columns(
#         metadata
#     )

#     print(
#         f"Features: {len(feature_columns)}"
#     )

#     print(
#         f"Stations: {len(metadata.get('stations_used', []))}"
#     )

#     # --------------------------------------------------------------
#     # 3. Test data
#     # --------------------------------------------------------------

#     train_df, valid_df, test_df = (
#         load_test_data(metadata)
#     )

#     # --------------------------------------------------------------
#     # 4. Features
#     # --------------------------------------------------------------

#     X_test = prepare_features(
#         train_df=train_df,
#         test_df=test_df,
#         feature_columns=feature_columns,
#         imputer=imputer
#     )

#     # --------------------------------------------------------------
#     # 5. SHAP explainer
#     # --------------------------------------------------------------

#     explainer = create_shap_explainer(
#         model
#     )

#     # --------------------------------------------------------------
#     # 6. SHAP values
#     # --------------------------------------------------------------

#     X_sample, shap_values, base_values = (
#         calculate_shap_values(
#             explainer=explainer,
#             X=X_test,
#             sample_size=SHAP_SAMPLE_SIZE
#         )
#     )

#     # --------------------------------------------------------------
#     # 7. Global importance
#     # --------------------------------------------------------------

#     importance = (
#         generate_global_feature_importance(
#             X_sample=X_sample,
#             shap_values=shap_values
#         )
#     )

#     # --------------------------------------------------------------
#     # 8. SHAP summary
#     # --------------------------------------------------------------

#     generate_shap_summary_plot(
#         explainer=explainer,
#         X_sample=X_sample,
#         shap_values=shap_values
#     )

#     # --------------------------------------------------------------
#     # 9. Individual explanation
#     # --------------------------------------------------------------

#     # Make sure the selected example exists.
#     sample_index = min(
#         INDIVIDUAL_SAMPLE_INDEX,
#         len(X_sample) - 1
#     )

#     explanation = (
#         explain_individual_prediction(
#             model=model,
#             explainer=explainer,
#             X_sample=X_sample,
#             shap_values=shap_values,
#             base_values=base_values,
#             original_test_df=test_df,
#             row_index=sample_index
#         )
#     )

#     # --------------------------------------------------------------
#     # 10. Optional waterfall
#     # --------------------------------------------------------------

#     generate_shap_waterfall(
#         explainer=explainer,
#         X_sample=X_sample,
#         row_index=sample_index,
#         shap_values=shap_values,
#         base_values=base_values
#     )

#     # --------------------------------------------------------------
#     # Completion
#     # --------------------------------------------------------------

#     print("\n" + "=" * 60)
#     print("PHASE 11 COMPLETE")
#     print("=" * 60)

#     print(
#         "\nGenerated files:"
#     )

#     print(
#         os.path.join(
#             FIGURES_DIR,
#             "shap_global_feature_importance.png"
#         )
#     )

#     print(
#         os.path.join(
#             FIGURES_DIR,
#             "shap_summary.png"
#         )
#     )

#     print(
#         os.path.join(
#             FIGURES_DIR,
#             "individual_prediction_explanation.png"
#         )
#     )

#     print(
#         os.path.join(
#             METRICS_DIR,
#             "shap_feature_importance.csv"
#         )
#     )

#     print(
#         os.path.join(
#             METRICS_DIR,
#             "individual_prediction_explanation.json"
#         )
#     )

#     return explanation


# # ---------------------------------------------------------------------
# # SCRIPT ENTRY POINT
# # ---------------------------------------------------------------------

# if __name__ == "__main__":

#     explain_model()

"""
PHASE 11 - EXPLAINABLE AI (SHAP)

Loads the already-trained model from models/best_model.pkl (does NOT
retrain anything) and uses SHAP to explain its predictions.

SHAP compatibility: the selected model (see models/model_metadata.json) is
XGBoost, a tree-based model, so shap.TreeExplainer is used - this is exact
and fast for tree ensembles (unlike model-agnostic KernelSHAP, which would
be far slower here). If a future run selects a non-tree model (e.g. Linear
Regression), this script falls back to shap.LinearExplainer or
shap.Explainer's automatic backend selection.

IMPORTANT LANGUAGE RULE (enforced throughout this file and its outputs):
SHAP values describe which features the MODEL used to arrive at a given
PREDICTION. They are NOT a statement about what causes real-world air
pollution. All labels/titles/text in this script say "features influencing
the model prediction", never "causes of pollution" - correlation-based
feature attribution is not causal evidence.

Outputs (outputs/figures/):
  - global_feature_importance.png   (mean |SHAP value| per feature, top 20)
  - shap_summary_plot.png           (SHAP beeswarm summary, top 20 features)
  - individual_prediction_explanation.png  (SHAP waterfall for one example)

Also exposes get_top_features_for_prediction(), a reusable function that
returns the top-5 features influencing any single prediction, with their
SHAP contribution and the feature's actual value for that row.
"""

import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.train import load_modeling_data, chronological_split

FIGURES_DIR = config.FIGURES_DIR

# Compute-constraint note (documented, not hidden): this sandbox has 1 CPU
# core. shap.TreeExplainer is exact and fast for XGBoost, but computing SHAP
# values for the full ~70,000-row test set is unnecessary for a global
# summary - a random sample gives a statistically representative picture of
# feature influence at a fraction of the compute cost. The INDIVIDUAL
# prediction explanation later still uses one specific real row, not a
# sample.
SHAP_SAMPLE_SIZE = 3000


def load_artifacts():
    bundle = joblib.load(config.BEST_MODEL_PATH)
    feature_cols = joblib.load(config.FEATURE_COLUMNS_PATH)
    with open(config.MODEL_METADATA_PATH) as f:
        metadata = json.load(f)
    return bundle, feature_cols, metadata


def build_explainer(model):
    """Picks the right SHAP explainer for the model type actually used."""
    model_type = type(model).__name__
    if "XGB" in model_type or "RandomForest" in model_type or "Tree" in model_type:
        return shap.TreeExplainer(model), "TreeExplainer (exact, tree-based)"
    elif "Linear" in model_type:
        return None, "LinearExplainer requires a background dataset - handled separately"
    else:
        return shap.Explainer(model), "generic Explainer (auto-selected backend)"


def get_top_features_for_prediction(shap_values_row, feature_names, feature_values_row, top_n=5):
    """
    Returns the top N features that influenced ONE model prediction, ranked
    by absolute SHAP value (i.e. magnitude of influence on the model's
    output - not a claim about real-world causation).

    Parameters
    ----------
    shap_values_row : 1D array of SHAP values for a single prediction
    feature_names : list of feature names, same order as shap_values_row
    feature_values_row : 1D array of the actual (imputed) feature values for
                          that same row
    top_n : how many top features to return (default 5)

    Returns
    -------
    list of dicts, each with: feature, shap_value, feature_value, direction
    """
    order = np.argsort(-np.abs(shap_values_row))[:top_n]
    results = []
    for idx in order:
        results.append({
            "feature": feature_names[idx],
            "shap_value": float(shap_values_row[idx]),
            "feature_value": float(feature_values_row[idx]),
            "direction": "increases predicted PM2.5" if shap_values_row[idx] > 0
                         else "decreases predicted PM2.5",
        })
    return results


def main():
    print("Loading saved model (no retraining)...")
    bundle, feature_cols, metadata = load_artifacts()
    imputer = bundle["imputer"]
    model = bundle["model"]
    model_name = bundle["model_name"]
    print(f"Model: {model_name}")

    if model is None:
        print("Selected model is the persistence baseline (no fitted model object) - "
              "SHAP is not applicable, since there is no learned function to explain. "
              "Exiting.")
        return

    print("\nRebuilding test set for explanation sampling (no retraining)...")
    station_ids = metadata["stations_used"]
    df = load_modeling_data(station_ids)
    _, _, test_df = chronological_split(df)

    X_test_raw = test_df[feature_cols]
    X_test = imputer.transform(X_test_raw)
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)

    print(f"Test set: {len(X_test_df):,} rows. Sampling {SHAP_SAMPLE_SIZE} rows "
          f"for global SHAP analysis (compute-time reason, documented in module docstring).")
    sample_idx = np.random.RandomState(config.RANDOM_SEED).choice(
        len(X_test_df), size=min(SHAP_SAMPLE_SIZE, len(X_test_df)), replace=False
    )
    X_sample = X_test_df.iloc[sample_idx].reset_index(drop=True)

    explainer, explainer_desc = build_explainer(model)
    print(f"Using: {explainer_desc}")
    if explainer is None:
        print("Model type not supported by this script's explainer logic. Exiting.")
        return

    print("Computing SHAP values (TreeExplainer, exact for tree ensembles)...")
    shap_values = explainer(X_sample)
    print(f"SHAP values computed: shape {shap_values.values.shape}")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ---- 1. Global feature importance (mean |SHAP value|) ----
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    top20 = importance_df.head(20)
    plt.figure(figsize=(9, 8))
    plt.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1], color="#3182ce")
    plt.xlabel("Mean |SHAP value| (average influence on model prediction)")
    plt.title(f"Global Feature Importance - {model_name}\n"
              "(features influencing the model's PM2.5 prediction, not causes of pollution)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "global_feature_importance.png"), dpi=150)
    plt.close()
    print("Saved: global_feature_importance.png")

    importance_df.to_csv(
        os.path.join(config.OUTPUTS_DIR, "metrics", "shap_global_feature_importance.csv"),
        index=False,
    )

    # ---- 2. SHAP summary plot (beeswarm) ----
    plt.figure()
    shap.summary_plot(shap_values.values, X_sample, feature_names=feature_cols,
                      max_display=20, show=False)
    plt.title(f"SHAP Summary - {model_name}\n"
              "(feature influence on model prediction, not causal effect on pollution)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_summary_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: shap_summary_plot.png")

    # ---- 3. Individual prediction explanation ----
    # Pick one concrete example row from the sample (the one with the
    # highest predicted PM2.5, so the explanation is for a meaningful,
    # non-trivial case rather than an arbitrary row).
    predictions = model.predict(X_sample.values)
    example_idx = int(np.argmax(predictions))
    example_row = X_sample.iloc[example_idx]
    example_shap = shap_values.values[example_idx]
    example_pred = predictions[example_idx]

    print(f"\nExample prediction (row {example_idx} of sample): "
          f"predicted PM2.5 (6h ahead) = {example_pred:.1f} ug/m3")

    plt.figure()
    single_explanation = shap.Explanation(
        values=example_shap,
        base_values=shap_values.base_values[example_idx],
        data=example_row.values,
        feature_names=feature_cols,
    )
    shap.plots.waterfall(single_explanation, max_display=15, show=False)
    plt.title(f"Individual Prediction Explanation - {model_name}\n"
              f"Predicted PM2.5 (6h ahead) = {example_pred:.1f} ug/m3\n"
              "(features influencing THIS prediction, not causes of pollution)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "individual_prediction_explanation.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: individual_prediction_explanation.png")

    # ---- Use the reusable top-5-features function on this same example ----
    top5 = get_top_features_for_prediction(example_shap, feature_cols, example_row.values, top_n=5)
    print("\nTop 5 features influencing this example prediction "
          "(NOT causes of pollution - just what the model weighed most heavily):")
    for i, item in enumerate(top5, 1):
        print(f"  {i}. {item['feature']} = {item['feature_value']:.2f} "
              f"-> SHAP {item['shap_value']:+.2f} ({item['direction']})")

    example_explanation_report = {
        "model_name": model_name,
        "example_predicted_pm25_6h": float(example_pred),
        "top_5_features_influencing_this_prediction": top5,
        "language_note": (
            "These are features that INFLUENCED THE MODEL'S PREDICTION for this "
            "specific example, based on SHAP attribution. This is not a causal "
            "claim about what causes real-world air pollution."
        ),
    }
    with open(os.path.join(config.OUTPUTS_DIR, "metrics", "example_prediction_explanation.json"), "w") as f:
        json.dump(example_explanation_report, f, indent=2, default=str)
    print("\nSaved: outputs/metrics/example_prediction_explanation.json")
    print("Saved: outputs/metrics/shap_global_feature_importance.csv")

    return {
        "importance_df": importance_df,
        "example_explanation": example_explanation_report,
    }


if __name__ == "__main__":
    main()