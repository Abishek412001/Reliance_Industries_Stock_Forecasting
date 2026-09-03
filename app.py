"""
Reliance Industries Stock Price — Interactive Forecasting Dashboard
Run locally:
    streamlit run "app.py"
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os
st.set_page_config(page_title="Reliance Industries Forecast", layout="wide")
st.set_page_config(
    page_title="Reliance Industries — Stock Forecast Dashboard",
    layout="wide"
)
# Define paths matching the saved location in Drive
OUTPUT_DIR = "/content/drive/My Drive/DS/Time_series_forecasting"
DATA_PATH = os.path.join(OUTPUT_DIR, "Company_stock_prices_clean.csv")
XGB_MODEL_PATH = os.path.join(OUTPUT_DIR, "xgb_model.pkl")
SARIMA_MODEL_PATH = os.path.join(OUTPUT_DIR, "sarima_model.pkl")
FEATURE_COLS_PATH = os.path.join(OUTPUT_DIR, "feature_cols.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "model_comparison.csv")
# Relative directory resolution for seamless local or cloud deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Company_stock_prices_clean.csv")
XGB_MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
SARIMA_MODEL_PATH = os.path.join(BASE_DIR, "models", "sarima_model.pkl")
ETS_MODEL_PATH = os.path.join(BASE_DIR, "models", "ets_model.pkl")
PROPHET_DEF_PATH = os.path.join(BASE_DIR, "models", "prophet_default_model.pkl")
PROPHET_NO_SEASON_PATH = os.path.join(BASE_DIR, "models", "prophet_no_seasonality_model.pkl")
STATE_SPACE_MODEL_PATH = os.path.join(BASE_DIR, "models", "state_space_model.pkl")
FEATURE_COLS_PATH = os.path.join(BASE_DIR, "models", "feature_cols.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics", "model_comparison.csv")
# ----------------------------------------------------------------------
# Cached Data & Artifact Loaders
