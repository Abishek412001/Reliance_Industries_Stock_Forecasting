"""Data loading and feature engineering for the Reliance dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path("Company_stock_prices_clean.csv")
LAGS = [1, 2, 3, 5, 10]


@st.cache_data
def load_data(path: str = str(DATA_PATH)) -> pd.DataFrame:
    """Load and prepare the display dataset without mutating the source file."""
    data = pd.read_csv(path)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    data["Daily_Return"] = data["Close"].pct_change() * 100
    data["MA_20"] = data["Close"].rolling(window=20).mean()
    data["MA_200"] = data["Close"].rolling(window=200).mean()
    data["Volatility_20"] = data["Daily_Return"].rolling(window=20).std()
    return data


@st.cache_data
def engineer_xgb_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build the notebook's XGBoost features with shifted rolling windows.

    The rolling features use ``shift(1)`` before ``rolling`` so the feature
    values available for a row never contain that row's closing price.
    """
    feature_df = data[["Date", "Close", "Volume"]].copy()
    for lag in LAGS:
        feature_df[f"Lag_{lag}"] = feature_df["Close"].shift(lag)

    shifted_close = feature_df["Close"].shift(1)
    feature_df["Rolling_Mean_5"] = shifted_close.rolling(window=5).mean()
    feature_df["Rolling_Std_5"] = shifted_close.rolling(window=5).std()
    feature_df["Rolling_Mean_20"] = shifted_close.rolling(window=20).mean()
    feature_df["DayOfWeek"] = feature_df["Date"].dt.dayofweek
    feature_df["Month"] = feature_df["Date"].dt.month
    return feature_df.dropna().reset_index(drop=True)


def trading_day_age(last_date: pd.Timestamp, today: pd.Timestamp | None = None) -> int:
    """Return the weekday count from the last data date through today."""
    current = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    last = pd.Timestamp(last_date).normalize()
    if current <= last:
        return 0
    return max(len(pd.bdate_range(last + pd.Timedelta(days=1), current)), 0)
