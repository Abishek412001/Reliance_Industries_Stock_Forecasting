"""Cached model loading and inference helpers for the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

LAGS = [1, 2, 3, 5, 10]
XGB_MODEL_PATH = Path("xgb_model.pkl")
SARIMA_MODEL_PATH = Path("sarima_model.pkl")
FEATURE_COLS_PATH = Path("feature_cols.pkl")


@st.cache_resource
def load_models(
    xgb_path: str = str(XGB_MODEL_PATH),
    sarima_path: str = str(SARIMA_MODEL_PATH),
    feature_cols_path: str = str(FEATURE_COLS_PATH),
) -> tuple[Any, Any, list[str]]:
    """Load the existing XGBoost/SARIMA artifacts once per Streamlit process."""
    xgb_model = joblib.load(xgb_path)
    sarima_model = joblib.load(sarima_path)
    feature_cols = list(joblib.load(feature_cols_path))
    return xgb_model, sarima_model, feature_cols


def next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    """Return the next weekday after ``date``; weekends are skipped."""
    nxt = pd.Timestamp(date) + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt


def future_trading_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """Build future weekday dates for the requested forecast horizon."""
    dates: list[pd.Timestamp] = []
    current = pd.Timestamp(last_date)
    for _ in range(horizon):
        current = next_trading_day(current)
        dates.append(current)
    return pd.DatetimeIndex(dates)


def recursive_xgb_forecast(
    model: Any,
    history_df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
) -> pd.DataFrame:
    """Forecast recursively with the existing XGBoost model.

    This preserves the app's existing recursive inference logic and carries
    the last observed volume forward because future volume is unknown.
    """
    history = history_df[["Date", "Close", "Volume"]].copy()
    last_volume = history["Volume"].iloc[-1]
    forecasts: list[dict[str, object]] = []

    for _ in range(horizon):
        next_date = next_trading_day(history["Date"].iloc[-1])
        row: dict[str, float | int] = {"Volume": last_volume}
        for lag in LAGS:
            row[f"Lag_{lag}"] = float(history["Close"].iloc[-lag])
        row["Rolling_Mean_5"] = float(history["Close"].iloc[-5:].mean())
        row["Rolling_Std_5"] = float(history["Close"].iloc[-5:].std())
        row["Rolling_Mean_20"] = float(history["Close"].iloc[-20:].mean())
        row["DayOfWeek"] = int(next_date.dayofweek)
        row["Month"] = int(next_date.month)

        X_next = pd.DataFrame([row])[feature_cols]
        next_close = float(model.predict(X_next)[0])
        forecasts.append({"Date": next_date, "Forecast_Close": next_close})
        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [{"Date": next_date, "Close": next_close, "Volume": last_volume}]
                ),
            ],
            ignore_index=True,
        )
    return pd.DataFrame(forecasts)


def sarima_forecast(model: Any, last_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Return SARIMA forecasts and its native 95% confidence interval."""
    result = model.get_forecast(steps=horizon)
    dates = future_trading_dates(last_date, horizon)
    mean = np.asarray(result.predicted_mean)
    conf_int = result.conf_int(alpha=0.05)
    return pd.DataFrame(
        {
            "Date": dates,
            "Forecast_Close": mean,
            "Lower_CI": np.asarray(conf_int.iloc[:, 0]),
            "Upper_CI": np.asarray(conf_int.iloc[:, 1]),
        }
    )


@st.cache_data
def xgb_residual_std(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    _model: Any,
) -> float:
    """Estimate historical one-step residual standard deviation for XGBoost.

    This is an uncertainty approximation for visualization only. It is not a
    calibrated predictive interval and does not retrain or alter the model.
    """
    predictions = np.asarray(_model.predict(feature_df[feature_cols]))
    residuals = feature_df["Close"].to_numpy() - predictions
    return float(np.std(residuals, ddof=1))


def xgb_forecast_with_band(
    model: Any,
    history_df: pd.DataFrame,
    feature_cols: list[str],
    feature_df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Add a clearly labeled approximate 95% band to recursive XGBoost output."""
    forecast = recursive_xgb_forecast(model, history_df, feature_cols, horizon)
    sigma = xgb_residual_std(feature_df, feature_cols, model)
    margin = 1.96 * sigma
    forecast["Lower_CI"] = forecast["Forecast_Close"] - margin
    forecast["Upper_CI"] = forecast["Forecast_Close"] + margin
    return forecast
