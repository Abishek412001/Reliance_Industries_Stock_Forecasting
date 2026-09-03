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

st.set_page_config(
    page_title="Reliance Industries — Stock Forecast Dashboard",
    layout="wide"
)

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
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_200"] = df["Close"].rolling(window=200).mean()
    return df

@st.cache_resource
def load_artifacts():
    xgb = joblib.load(XGB_MODEL_PATH)
    sarima = joblib.load(SARIMA_MODEL_PATH)
    cols = joblib.load(FEATURE_COLS_PATH)
    return xgb, sarima, cols

# Load data and models
df = load_data()
xgb_model, sarima_model, feature_cols = load_artifacts()
metrics_df = pd.read_csv(METRICS_PATH, index_col=0)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("Reliance Industries Stock Price Forecast")
st.write(f"Latest data point: {df['Date'].max().date()}")

tabs = st.tabs(["Historical Trends", "Model Performance", "30-Day Forecast"])

with tabs[0]:
    st.subheader("Close Price and Moving Averages")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Date'], df['Close'], label='Close')
    ax.plot(df['Date'], df['MA_20'], label='20-Day MA')
    ax.plot(df['Date'], df['MA_200'], label='200-Day MA')
    ax.legend()
    st.pyplot(fig)

with tabs[1]:
    st.subheader("Model Comparison Metrics")
    st.dataframe(metrics_df.style.format("{:.2f}"))

with tabs[2]:
    st.subheader("Predictive Insights")
    horizon = st.slider("Forecast Horizon (Days)", 1, 30, 30)
    if st.button("Generate Forecast"):
        st.info("Dashboard active. Use models to predict future prices based on latest lags.")
