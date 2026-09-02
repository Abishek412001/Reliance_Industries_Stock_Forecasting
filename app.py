"""Reliance Industries Stock Price — Forecasting Dashboard."""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Reliance Industries — Stock Forecast", layout="wide")

DATA_PATH = "Company_stock_prices_clean.csv"
XGB_MODEL_PATH = "xgb_model.pkl"
SARIMA_MODEL_PATH = "sarima_model.pkl"
ETS_MODEL_PATH = "ets_model.pkl"
PROPHET_DEFAULT_MODEL_PATH = "prophet_default_model.pkl"
PROPHET_NO_SEASONALITY_MODEL_PATH = "prophet_no_seasonality_model.pkl"
STATE_SPACE_MODEL_PATH = "state_space_model.pkl"
FEATURE_COLS_PATH = "feature_cols.pkl"
METRICS_PATH = "model_comparison.csv"
LAGS = [1, 2, 3, 5, 10]

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    data["Daily_Return"] = data["Close"].pct_change() * 100
    data["MA_20"] = data["Close"].rolling(20).mean()
    data["MA_200"] = data["Close"].rolling(200).mean()
    data["Volatility_20"] = data["Daily_Return"].rolling(20).std()
    return data

@st.cache_resource
def load_models():
    return (
        joblib.load(XGB_MODEL_PATH),
        joblib.load(SARIMA_MODEL_PATH),
        joblib.load(ETS_MODEL_PATH),
        joblib.load(PROPHET_DEFAULT_MODEL_PATH),
        joblib.load(PROPHET_NO_SEASONALITY_MODEL_PATH),
        joblib.load(STATE_SPACE_MODEL_PATH),
        joblib.load(FEATURE_COLS_PATH),
    )

@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH, index_col=0)

df = load_data()
(
    xgb_model,
    sarima_model,
    ets_model,
    prophet_default_model,
    prophet_no_seasonality_model,
    state_space_model,
    feature_cols,
) = load_models()
metrics_df = load_metrics()

def next_trading_day(date):
    nxt = date + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt

def future_trading_dates(last_date, horizon):
    dates = []
    current = last_date
    for _ in range(horizon):
        current = next_trading_day(current)
        dates.append(current)
    return pd.DatetimeIndex(dates)

def recursive_xgb_forecast(model, history_df, feature_columns, horizon):
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
        X_next = pd.DataFrame([row])[feature_columns]
        next_close = float(model.predict(X_next)[0])
        forecasts.append({"Date": next_date, "Forecast_Close": next_close})
        history = pd.concat(
            [history, pd.DataFrame([{"Date": next_date, "Close": next_close, "Volume": last_volume}])],
            ignore_index=True,
        )
    return pd.DataFrame(forecasts)

def statsmodels_forecast(model, horizon, include_ci=False):
    result = model.get_forecast(steps=horizon)
    dates = future_trading_dates(df["Date"].iloc[-1], horizon)
    out = pd.DataFrame({"Date": dates, "Forecast_Close": np.asarray(result.predicted_mean)})
    if include_ci:
        ci = result.conf_int(alpha=0.05)
        out["Lower_CI"] = np.asarray(ci.iloc[:, 0])
        out["Upper_CI"] = np.asarray(ci.iloc[:, 1])
    return out

def prophet_forecast(model, horizon):
    dates = future_trading_dates(df["Date"].iloc[-1], horizon)
    forecast = model.predict(pd.DataFrame({"ds": dates}))
    return pd.DataFrame({
        "Date": dates,
        "Forecast_Close": forecast["yhat"].to_numpy(),
        "Lower_CI": forecast["yhat_lower"].to_numpy(),
        "Upper_CI": forecast["yhat_upper"].to_numpy(),
    })

st.title("Reliance Industries — Stock Price Forecast Dashboard")
st.caption(f"Data through {df['Date'].max().date()} · {len(df)} trading days")

tab_hist, tab_ts, tab_eval, tab_forecast = st.tabs(
    ["Historical Analysis", "Time-Series Analysis", "Model Evaluation", "30-Day Forecast"]
)

with tab_hist:
    st.subheader("Price History")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Close"], label="Close")
    ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.grid(True); ax.legend()
    st.pyplot(fig)

with tab_ts:
    st.subheader("Trend — 20 & 200 Day Moving Averages")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Close"], label="Close", alpha=0.6)
    ax.plot(df["Date"], df["MA_20"], label="20-Day MA")
    ax.plot(df["Date"], df["MA_200"], label="200-Day MA")
    ax.grid(True); ax.legend(); st.pyplot(fig)
    st.subheader("Daily Returns")
    fig, ax = plt.subplots(figsize=(12, 4)); ax.plot(df["Date"], df["Daily_Return"]); ax.grid(True); st.pyplot(fig)
    st.subheader("20-Day Rolling Volatility")
    fig, ax = plt.subplots(figsize=(12, 4)); ax.plot(df["Date"], df["Volatility_20"]); ax.grid(True); st.pyplot(fig)

with tab_eval:
    st.subheader("Model Comparison (held-out 2023 test set)")
    st.dataframe(metrics_df.style.format("{:.2f}"))
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_df[["RMSE", "MAE"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Error"); ax.grid(True, axis="y"); plt.xticks(rotation=20); st.pyplot(fig)

with tab_forecast:
    st.subheader("Forecast Future Prices")
    horizon = st.slider("Forecast horizon (trading days)", 1, 30, 30)
    model_choice = st.radio(
        "Model",
        ["XGBoost", "SARIMA", "ETS", "Prophet (Default Seasonality)",
         "Prophet (No Seasonality)", "State Space", "Compare all"],
        horizontal=True,
    )

    if st.button("Run Forecast"):
        fig, ax = plt.subplots(figsize=(12, 5))
        recent = df.tail(60)
        ax.plot(recent["Date"], recent["Close"], label="Recent Actual")

        if model_choice in ("XGBoost", "Compare all"):
            fc = recursive_xgb_forecast(xgb_model, df, feature_cols, horizon)
            ax.plot(fc["Date"], fc["Forecast_Close"], label="XGBoost Forecast", marker="o", markersize=3)
            if model_choice != "Compare all": st.dataframe(fc.set_index("Date").style.format("{:.2f}"))

        if model_choice in ("SARIMA", "Compare all"):
            fc = statsmodels_forecast(sarima_model, horizon, include_ci=True)
            ax.plot(fc["Date"], fc["Forecast_Close"], label="SARIMA Forecast", marker="o", markersize=3)
            ax.fill_between(fc["Date"], fc["Lower_CI"], fc["Upper_CI"], alpha=0.15, label="SARIMA 95% CI")
            if model_choice != "Compare all": st.dataframe(fc.set_index("Date").style.format("{:.2f}"))

        if model_choice in ("ETS", "Compare all"):
            fc = statsmodels_forecast(ets_model, horizon)
            ax.plot(fc["Date"], fc["Forecast_Close"], label="ETS Forecast", marker="o", markersize=3)
            if model_choice != "Compare all": st.dataframe(fc.set_index("Date").style.format("{:.2f}"))

        if model_choice in ("Prophet (Default Seasonality)", "Compare all"):
            fc = prophet_forecast(prophet_default_model, horizon)
            ax.plot(fc["Date"], fc["Forecast_Close"], label="Prophet (Default)", marker="o", markersize=3)
            ax.fill_between(fc["Date"], fc["Lower_CI"], fc["Upper_CI"], alpha=0.15, label="Prophet Default 95% CI")
            if model_choice != "Compare all": st.dataframe(fc.set_index("Date").style.format("{:.2f}"))

        if model_choice in ("Prophet (No Seasonality)", "Compare all"):
            fc = prophet_forecast(prophet_no_seasonality_model, horizon)
            ax.plot(fc["Date"], fc["Forecast_Close"], label="Prophet (No Seasonality)", marker="o", markersize=3)
            ax.fill_between(fc["Date"], fc["Lower_CI"], fc["Upper_CI"], alpha=0.15, label="Prophet No-Seasonality 95% CI")
            if model_choice != "Compare all": st.dataframe(fc.set_index("Date").style.format("{:.2f}"))

        if model_choice in ("State Space", "Compare all"):
            fc = statsmodels_forecast(state_space_model, horizon, include_ci=True)
            ax.plot(fc["Date"], fc["Forecast_Close"], label="State Space Forecast", marker="o", markersize=3)
            ax.fill_between(fc["Date"], fc["Lower_CI"], fc["Upper_CI"], alpha=0.15, label="State Space 95% CI")
            if model_choice != "Compare all": st.dataframe(fc.set_index("Date").style.format("{:.2f}"))

        ax.set_xlabel("Date"); ax.set_ylabel("Closing Price"); ax.set_title(f"{horizon}-Day Forecast")
        ax.grid(True); ax.legend(); st.pyplot(fig)
        st.warning("This is a statistical/ML forecast, not investment advice.")
    else:
        st.write("Choose a horizon and model, then click **Run Forecast**.")
