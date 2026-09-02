"""Reliance Industries Stock Price — Forecasting Dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from data import engineer_xgb_features, load_data, trading_day_age
from models import load_models, recursive_xgb_forecast, sarima_forecast
from ui.evaluation import render_evaluation_tab
from ui.forecast import render_forecast_tab
from ui.historical import render_historical_tab
from ui.time_series import render_time_series_tab

DATA_PATH = "Company_stock_prices_clean.csv"
METRICS_PATH = "model_comparison.csv"
MODEL_OPTIONS = ["XGBoost", "SARIMA"]


@st.cache_data
def load_metrics(path: str = METRICS_PATH) -> pd.DataFrame:
    """Load the saved evaluation table without recalculating model metrics."""
    return pd.read_csv(path, index_col=0)


def next_day_prediction(
    model_choice: str,
    data: pd.DataFrame,
    xgb_model: Any,
    sarima_model: Any,
    feature_cols: list[str],
) -> float:
    """Return the active model's one-step forecast from the latest observation."""
    if model_choice == "XGBoost":
        forecast = recursive_xgb_forecast(xgb_model, data, feature_cols, 1)
    else:
        forecast = sarima_forecast(sarima_model, data["Date"].iloc[-1], 1)
    return float(forecast["Forecast_Close"].iloc[0])


# Cached functions are called with stable file paths/dataframes only. No widget
# value is passed to a cached loader, so widget interaction does not reload
# the CSV or pickle artifacts.
df = load_data(DATA_PATH)
feature_df = engineer_xgb_features(df)
xgb_model, sarima_model, feature_cols = load_models()
metrics_df = load_metrics(METRICS_PATH)

with st.sidebar:
    st.header("Dashboard Controls")
    model_choice = st.radio("Active model", MODEL_OPTIONS, index=0)
    horizon = st.slider("Forecast horizon (trading days)", 1, 30, 30)
    date_range = st.date_input(
        "Historical date range",
        value=(df["Date"].min().date(), df["Date"].max().date()),
        min_value=df["Date"].min().date(),
        max_value=df["Date"].max().date(),
    )
    run_forecast = st.button("Run Forecast", type="primary", use_container_width=True)

if not isinstance(date_range, tuple) or len(date_range) != 2:
    date_range = (df["Date"].min().date(), df["Date"].max().date())

latest_close = float(df["Close"].iloc[-1])
previous_close = float(df["Close"].iloc[-2])
delta_pct = (latest_close - previous_close) / previous_close * 100
active_mape = float(metrics_df.loc[model_choice, "MAPE"])
next_prediction = next_day_prediction(
    model_choice, df, xgb_model, sarima_model, feature_cols
)

st.title("Reliance Industries — Stock Price Forecast Dashboard")
summary_cols = st.columns(4)
summary_cols[0].metric("Latest Closing Price", f"₹{latest_close:,.2f}")
summary_cols[1].metric("Next-Day Prediction", f"₹{next_prediction:,.2f}")
summary_cols[2].metric("Active Model MAPE", f"{active_mape:.2f}%")
summary_cols[3].metric("vs Previous Day", f"{delta_pct:+.2f}%", delta=f"{delta_pct:+.2f}%")

last_date = df["Date"].max()
age = trading_day_age(last_date)
st.caption(
    f"Data as of {last_date.date()} · {len(df):,} trading days · "
    f"Active model: {model_choice}"
)
if age > 5:
    st.warning(
        f"Data as of {last_date.date()} is approximately {age} trading days old. "
        "The dashboard may be using stale market data."
    )

# The four existing views remain available as secondary drill-down tabs.
tab_hist, tab_ts, tab_eval, tab_forecast = st.tabs(
    ["Historical Analysis", "Time-Series Analysis", "Model Evaluation", "30-Day Forecast"]
)

with tab_hist:
    render_historical_tab(df, date_range)

with tab_ts:
    render_time_series_tab(df)

with tab_eval:
    render_evaluation_tab(metrics_df)

with tab_forecast:
    render_forecast_tab(
        df,
        xgb_model,
        sarima_model,
        feature_cols,
        feature_df,
        model_choice,
        horizon,
        run_forecast,
    )
