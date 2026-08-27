"""
Reliance Industries Stock Price — Forecasting Dashboard

Run locally:
    streamlit run app.py

Expects these files in the same directory (produced by export_artifacts.py):
    Company_stock_prices_clean.csv
    xgb_model.pkl
    sarima_model.pkl
    feature_cols.pkl
    model_comparison.csv
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Reliance Industries — Stock Forecast",
    layout="wide"
)

DATA_PATH = "Company_stock_prices_clean.csv"
XGB_MODEL_PATH = "xgb_model.pkl"
SARIMA_MODEL_PATH = "sarima_model.pkl"
FEATURE_COLS_PATH = "feature_cols.pkl"
METRICS_PATH = "model_comparison.csv"

LAGS = [1, 2, 3, 5, 10]


# ----------------------------------------------------------------------
# Loading (cached so the app doesn't redo this on every interaction)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_200"] = df["Close"].rolling(window=200).mean()
    df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()

    return df


@st.cache_resource
def load_models():
    xgb_model = joblib.load(XGB_MODEL_PATH)
    sarima_model = joblib.load(SARIMA_MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return xgb_model, sarima_model, feature_cols


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH, index_col=0)


df = load_data()
xgb_model, sarima_model, feature_cols = load_models()
metrics_df = load_metrics()


# ----------------------------------------------------------------------
# Recursive multi-step forecast for XGBoost
#
# IMPORTANT: the notebook's XGBoost evaluation used the TRUE lagged
# closing prices for every test row (one-step-ahead evaluation). A real
# 30-day-ahead forecast can't do that — day 2's "Lag_1" is day 1's own
# prediction, not a known value. This function forecasts recursively,
# feeding each prediction back in as history for the next step, so the
# error compounds day over day. Expect meaningfully worse accuracy than
# the notebook's reported MAPE, especially further out in the horizon.
# Volume has no known future value either; it's carried forward at its
# last observed level as a simplifying assumption (it was the least
# important feature in training, so this has limited impact).
# ----------------------------------------------------------------------
def next_trading_day(date):
    nxt = date + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:  # skip Sat/Sun
        nxt += pd.Timedelta(days=1)
    return nxt


def recursive_xgb_forecast(model, history_df, feature_cols, horizon):
    history = history_df[["Date", "Close", "Volume"]].copy()
    last_volume = history["Volume"].iloc[-1]

    forecasts = []
    for _ in range(horizon):
        next_date = next_trading_day(history["Date"].iloc[-1])

        row = {"Volume": last_volume}
        for lag in LAGS:
            row[f"Lag_{lag}"] = history["Close"].iloc[-lag]
        row["Rolling_Mean_5"] = history["Close"].iloc[-5:].mean()
        row["Rolling_Std_5"] = history["Close"].iloc[-5:].std()
        row["Rolling_Mean_20"] = history["Close"].iloc[-20:].mean()
        row["DayOfWeek"] = next_date.dayofweek
        row["Month"] = next_date.month

        X_next = pd.DataFrame([row])[feature_cols]
        next_close = float(model.predict(X_next)[0])

        forecasts.append({"Date": next_date, "Forecast_Close": next_close})

        history = pd.concat(
            [history, pd.DataFrame([{
                "Date": next_date, "Close": next_close, "Volume": last_volume
            }])],
            ignore_index=True
        )

    return pd.DataFrame(forecasts)


def sarima_forecast(model, horizon):
    result = model.get_forecast(steps=horizon)
    mean = result.predicted_mean
    conf_int = result.conf_int(alpha=0.05)
    out = pd.DataFrame({
        "Date": mean.index,
        "Forecast_Close": mean.values,
        "Lower_CI": conf_int.iloc[:, 0].values,
        "Upper_CI": conf_int.iloc[:, 1].values,
    })
    return out


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
st.title("Reliance Industries — Stock Price Forecast Dashboard")
st.caption(f"Data through {df['Date'].max().date()} · {len(df)} trading days")

tab_hist, tab_ts, tab_eval, tab_forecast = st.tabs(
    ["Historical Analysis", "Time-Series Analysis", "Model Evaluation", "30-Day Forecast"]
)

# --- Historical Analysis -------------------------------------------------
with tab_hist:
    st.subheader("Price History")
    date_range = st.slider(
        "Date range",
        min_value=df["Date"].min().to_pydatetime(),
        max_value=df["Date"].max().to_pydatetime(),
        value=(df["Date"].min().to_pydatetime(), df["Date"].max().to_pydatetime()),
    )
    mask = (df["Date"] >= date_range[0]) & (df["Date"] <= date_range[1])
    view = df.loc[mask]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(view["Date"], view["Close"], label="Close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Closing Price")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Latest Close", f"{df['Close'].iloc[-1]:.2f}")
    with col2:
        st.metric(
            "1-Day Change",
            f"{df['Daily_Return'].iloc[-1]:.2f}%"
        )

    st.subheader("Trading Volume")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(view["Date"], view["Volume"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.grid(True)
    st.pyplot(fig)

# --- Time-Series Analysis -------------------------------------------------
with tab_ts:
    st.subheader("Trend — 20 & 200 Day Moving Averages")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Close"], label="Close", alpha=0.6)
    ax.plot(df["Date"], df["MA_20"], label="20-Day MA")
    ax.plot(df["Date"], df["MA_200"], label="200-Day MA")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Daily Returns")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Date"], df["Daily_Return"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("20-Day Rolling Volatility")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Date"], df["Volatility_20"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.grid(True)
    st.pyplot(fig)

# --- Model Evaluation -------------------------------------------------
with tab_eval:
    st.subheader("Model Comparison (held-out 2023 test set)")
    st.dataframe(metrics_df.style.format("{:.2f}"))

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_df[["RMSE", "MAE"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Error")
    ax.set_xticklabels(metrics_df.index, rotation=0)
    ax.grid(True, axis="y")
    st.pyplot(fig)

    st.info(
        "These numbers come from one-step-ahead evaluation, where each "
        "prediction used the true prior day's closing price as input. "
        "The 30-Day Forecast tab instead forecasts recursively — real "
        "accuracy there will typically be lower than this table suggests, "
        "especially further into the horizon."
    )

# --- 30-Day Forecast -------------------------------------------------
with tab_forecast:
    st.subheader("Forecast Future Prices")

    horizon = st.slider("Forecast horizon (trading days)", min_value=1, max_value=30, value=30)
    model_choice = st.radio(
        "Model",
        options=["XGBoost", "SARIMA", "Compare both"],
        horizontal=True
    )

    if st.button("Run Forecast"):
        fig, ax = plt.subplots(figsize=(12, 5))
        recent = df.tail(60)
        ax.plot(recent["Date"], recent["Close"], label="Recent Actual", color="black")

        if model_choice in ("XGBoost", "Compare both"):
            xgb_fc = recursive_xgb_forecast(xgb_model, df, feature_cols, horizon)
            ax.plot(xgb_fc["Date"], xgb_fc["Forecast_Close"], label="XGBoost Forecast", marker="o", markersize=3)
            st.write("XGBoost forecast")
            st.dataframe(xgb_fc.set_index("Date").style.format("{:.2f}"))

        if model_choice in ("SARIMA", "Compare both"):
            sar_fc = sarima_forecast(sarima_model, horizon)
            ax.plot(sar_fc["Date"], sar_fc["Forecast_Close"], label="SARIMA Forecast", marker="o", markersize=3)
            ax.fill_between(sar_fc["Date"], sar_fc["Lower_CI"], sar_fc["Upper_CI"], alpha=0.15, label="SARIMA 95% CI")
            st.write("SARIMA forecast")
            st.dataframe(sar_fc.set_index("Date").style.format("{:.2f}"))

        ax.set_xlabel("Date")
        ax.set_ylabel("Closing Price")
        ax.set_title(f"{horizon}-Day Forecast")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.warning(
            "This is a statistical forecast, not investment advice. SARIMA's "
            "selected order collapsed to a random walk in backtesting, meaning "
            "it essentially projects the last known price forward with growing "
            "uncertainty. XGBoost's forecast compounds its own prediction error "
            "day over day the further out it goes — treat both as directional "
            "estimates, not guarantees."
        )
    else:
        st.write("Choose a horizon and model, then click **Run Forecast**.")
