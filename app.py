"""
Reliance Industries Stock Price — Interactive Forecasting Dashboard

Run locally:
    streamlit run app.py
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
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset missing: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_200"] = df["Close"].rolling(window=200).mean()
    df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()

    return df


@st.cache_resource
def load_artifacts():
    xgb = joblib.load(XGB_MODEL_PATH) if os.path.exists(XGB_MODEL_PATH) else None
    sarima = joblib.load(SARIMA_MODEL_PATH) if os.path.exists(SARIMA_MODEL_PATH) else None
    ets = joblib.load(ETS_MODEL_PATH) if os.path.exists(ETS_MODEL_PATH) else None
    prophet_def = joblib.load(PROPHET_DEF_PATH) if os.path.exists(PROPHET_DEF_PATH) else None
    prophet_no_season = joblib.load(PROPHET_NO_SEASON_PATH) if os.path.exists(PROPHET_NO_SEASON_PATH) else None
    state_space = joblib.load(STATE_SPACE_MODEL_PATH) if os.path.exists(STATE_SPACE_MODEL_PATH) else None
    cols = joblib.load(FEATURE_COLS_PATH) if os.path.exists(FEATURE_COLS_PATH) else []
    return xgb, sarima, ets, prophet_def, prophet_no_season, state_space, cols


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        return pd.read_csv(METRICS_PATH, index_col=0)
    return pd.DataFrame()


df = load_data()
xgb_model, sarima_model, ets_model, prophet_def_model, prophet_no_season_model, state_space_model, feature_cols = load_artifacts()
metrics_df = load_metrics()

# ----------------------------------------------------------------------
# Forecast Generators
# ----------------------------------------------------------------------
def next_trading_day(date):
    nxt = date + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:  # Skip weekends
        nxt += pd.Timedelta(days=1)
    return nxt


def recursive_xgb_forecast(model, history_df, cols, horizon):
    history = history_df[["Date", "Close", "Volume"]].copy()
    last_vol = history["Volume"].iloc[-1]
    forecasts = []

    for _ in range(horizon):
        nxt_date = next_trading_day(history["Date"].iloc[-1])
        row = {"Volume": last_vol}
        for lag in [1, 2, 3, 5, 10]:
            if f"Lag_{lag}" in cols:
                row[f"Lag_{lag}"] = history["Close"].iloc[-lag]

        if "Rolling_Mean_5" in cols:
            row["Rolling_Mean_5"] = history["Close"].iloc[-5:].mean()
        if "Rolling_Std_5" in cols:
            row["Rolling_Std_5"] = history["Close"].iloc[-5:].std()
        if "Rolling_Mean_20" in cols:
            row["Rolling_Mean_20"] = history["Close"].iloc[-20:].mean()
        if "DayOfWeek" in cols:
            row["DayOfWeek"] = nxt_date.dayofweek
        if "Month" in cols:
            row["Month"] = nxt_date.month

        X_next = pd.DataFrame([row])[cols]
        pred_close = float(model.predict(X_next)[0])
        forecasts.append({"Date": nxt_date, "Forecast_Close": pred_close})

        history = pd.concat(
            [history, pd.DataFrame([{"Date": nxt_date, "Close": pred_close, "Volume": last_vol}])],
            ignore_index=True
        )

    return pd.DataFrame(forecasts)


def sarima_forecast(model, horizon):
    fc = model.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=0.05)
    dates = [next_trading_day(df["Date"].iloc[-1])]
    for _ in range(1, horizon):
        dates.append(next_trading_day(dates[-1]))
    return pd.DataFrame({
        "Date": dates,
        "Forecast_Close": mean.values,
        "Lower_CI": conf.iloc[:, 0].values,
        "Upper_CI": conf.iloc[:, 1].values
    })


def ets_forecast(model, horizon):
    dates = [next_trading_day(df["Date"].iloc[-1])]
    for _ in range(1, horizon):
        dates.append(next_trading_day(dates[-1]))

    if hasattr(model, 'forecast'):
        fc_vals = model.forecast(steps=horizon).values
    elif isinstance(model, dict):
        last_val = model.get('last_close', df['Close'].iloc[-1])
        slope = model.get('slope', 0)
        fc_vals = [last_val + slope * (i + 1) for i in range(horizon)]
    else:
        fc_vals = [df['Close'].iloc[-1]] * horizon

    return pd.DataFrame({
        "Date": dates,
        "Forecast_Close": fc_vals
    })


def prophet_forecast(model, history_df, horizon):
    last_date = history_df["Date"].iloc[-1]
    dates = [next_trading_day(last_date)]
    for _ in range(1, horizon):
        dates.append(next_trading_day(dates[-1]))

    fut = pd.DataFrame({"ds": dates})
    res = model.predict(fut)
    return pd.DataFrame({
        "Date": dates,
        "Forecast_Close": res["yhat"].values,
        "Lower_CI": res["yhat_lower"].values,
        "Upper_CI": res["yhat_upper"].values
    })


def state_space_forecast(model, horizon):
    fc = model.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=0.05)
    dates = [next_trading_day(df["Date"].iloc[-1])]
    for _ in range(1, horizon):
        dates.append(next_trading_day(dates[-1]))
    return pd.DataFrame({
        "Date": dates,
        "Forecast_Close": mean.values,
        "Lower_CI": conf.iloc[:, 0].values,
        "Upper_CI": conf.iloc[:, 1].values
    })


# ----------------------------------------------------------------------
# Application UI Layout
# ----------------------------------------------------------------------
st.title("Reliance Industries — Stock Price Forecast Dashboard")
st.caption(f"Data Range: {df['Date'].min().date()} to {df['Date'].max().date()} · {len(df)} Trading Sessions")

tab_hist, tab_ts, tab_eval, tab_forecast = st.tabs([
    "Historical Trends", "Time-Series Analysis", "Model Evaluation", "30-Day Forecast"
])

# --- TAB 1: Historical Trends ---
with tab_hist:
    st.subheader("Price & Volume History")
    date_range = st.slider(
        "Select Date Range",
        min_value=df["Date"].min().to_pydatetime(),
        max_value=df["Date"].max().to_pydatetime(),
        value=(df["Date"].min().to_pydatetime(), df["Date"].max().to_pydatetime())
    )
    mask = (df["Date"] >= date_range[0]) & (df["Date"] <= date_range[1])
    sub_df = df.loc[mask]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sub_df["Date"], sub_df["Close"], label="Closing Price", color="#004c99", lw=1.8)
    ax.set_ylabel("Price (INR)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.metric("Latest Closing Price", f"{df['Close'].iloc[-1]:.2f} INR")
    col2.metric("Latest 1-Day Return", f"{df['Daily_Return'].iloc[-1]:.2f}%")

    st.subheader("Volume Profile")
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(sub_df["Date"], sub_df["Volume"], color="gray", alpha=0.6)
    ax.set_ylabel("Trading Volume")
    ax.grid(True)
    st.pyplot(fig)

# --- TAB 2: Time-Series Analysis ---
with tab_ts:
    st.subheader("Short-Term & Long-Term Trend Indicators")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df["Date"], df["Close"], label="Closing Price", alpha=0.5, color="gray")
    ax.plot(df["Date"], df["MA_20"], label="20-Day SMA (Short-term)", color="#ff7f0e", lw=1.5)
    ax.plot(df["Date"], df["MA_200"], label="200-Day SMA (Long-term)", color="#d62728", lw=1.8)
    ax.set_ylabel("Price (INR)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Daily Return Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df["Daily_Return"].dropna(), bins=40, color="purple", alpha=0.7)
        ax.set_xlabel("Return (%)")
        ax.grid(True)
        st.pyplot(fig)
    with col_b:
        st.subheader("20-Day Rolling Volatility")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(df["Date"], df["Volatility_20"], color="black")
        ax.set_ylabel("Std Dev (%)")
        ax.grid(True)
        st.pyplot(fig)

# --- TAB 3: Model Evaluation ---
with tab_eval:
    st.subheader("Model Performance Benchmark (Held-Out 2023 Test Set)")
    if not metrics_df.empty:
        st.dataframe(metrics_df.style.format("{:.2f}"))

        fig, ax = plt.subplots(figsize=(10, 4))
        metrics_df[["RMSE", "MAE"]].plot(kind="bar", ax=ax)
        ax.set_ylabel("Error Metric (INR)")
        ax.set_xticklabels(metrics_df.index, rotation=30, ha="right")
        ax.grid(True, axis="y")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No metrics table found.")

# --- TAB 4: 30-Day Forecast ---
with tab_forecast:
    st.subheader("Multi-Model Future Out-of-Sample Forecasting")
    horizon = st.slider("Forecast Horizon (Business Trading Days)", 1, 30, 30)

    model_options = ["XGBoost", "SARIMA (0,1,0)", "Exponential Smoothing (ETS)", "Prophet (No Seasonality)", "Prophet (Default)", "State Space (Structural)", "Compare All"]
    model_choice = st.radio("Select Model", options=model_options, horizontal=True)

    if st.button("Generate Forecast", type="primary"):
        fig, ax = plt.subplots(figsize=(12, 5))
        recent = df.tail(60)
        ax.plot(recent["Date"], recent["Close"], label="Recent Actual Price", color="black", lw=2)

        if model_choice in ("XGBoost", "Compare All") and xgb_model:
            xgb_fc = recursive_xgb_forecast(xgb_model, df, feature_cols, horizon)
            ax.plot(xgb_fc["Date"], xgb_fc["Forecast_Close"], label="XGBoost", marker="o", ms=4, color="#2ca02c")
            if model_choice == "XGBoost":
                st.dataframe(xgb_fc.set_index("Date").style.format("{:.2f}"))

        if model_choice in ("SARIMA (0,1,0)", "Compare All") and sarima_model:
            sar_fc = sarima_forecast(sarima_model, horizon)
            ax.plot(sar_fc["Date"], sar_fc["Forecast_Close"], label="SARIMA", marker="s", ms=4, color="#ff7f0e")
            if model_choice == "SARIMA (0,1,0)":
                ax.fill_between(sar_fc["Date"], sar_fc["Lower_CI"], sar_fc["Upper_CI"], color="#ff7f0e", alpha=0.15, label="95% CI")
                st.dataframe(sar_fc.set_index("Date").style.format("{:.2f}"))

        if model_choice in ("Exponential Smoothing (ETS)", "Compare All") and ets_model:
            ets_fc = ets_forecast(ets_model, horizon)
            ax.plot(ets_fc["Date"], ets_fc["Forecast_Close"], label="ETS Holt's Linear", marker="^", ms=4, color="#9467bd")
            if model_choice == "Exponential Smoothing (ETS)":
                st.dataframe(ets_fc.set_index("Date").style.format("{:.2f}"))

        if model_choice in ("Prophet (No Seasonality)", "Compare All") and prophet_no_season_model:
            pro_ns_fc = prophet_forecast(prophet_no_season_model, df, horizon)
            ax.plot(pro_ns_fc["Date"], pro_ns_fc["Forecast_Close"], label="Prophet (No Seasonality)", marker="d", ms=4, color="#d62728")
            if model_choice == "Prophet (No Seasonality)":
                ax.fill_between(pro_ns_fc["Date"], pro_ns_fc["Lower_CI"], pro_ns_fc["Upper_CI"], color="#d62728", alpha=0.15, label="95% CI")
                st.dataframe(pro_ns_fc.set_index("Date").style.format("{:.2f}"))

        if model_choice in ("Prophet (Default)", "Compare All") and prophet_def_model:
            pro_def_fc = prophet_forecast(prophet_def_model, df, horizon)
            ax.plot(pro_def_fc["Date"], pro_def_fc["Forecast_Close"], label="Prophet (Default)", marker="v", ms=4, color="#8c564b")
            if model_choice == "Prophet (Default)":
                ax.fill_between(pro_def_fc["Date"], pro_def_fc["Lower_CI"], pro_def_fc["Upper_CI"], color="#8c564b", alpha=0.15, label="95% CI")
                st.dataframe(pro_def_fc.set_index("Date").style.format("{:.2f}"))

        if model_choice in ("State Space (Structural)", "Compare All") and state_space_model:
            ss_fc = state_space_forecast(state_space_model, horizon)
            ax.plot(ss_fc["Date"], ss_fc["Forecast_Close"], label="State Space (UC)", marker="*", ms=5, color="#17becf")
            if model_choice == "State Space (Structural)":
                ax.fill_between(ss_fc["Date"], ss_fc["Lower_CI"], ss_fc["Upper_CI"], color="#17becf", alpha=0.15, label="95% CI")
                st.dataframe(ss_fc.set_index("Date").style.format("{:.2f}"))

        ax.set_ylabel("Price (INR)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
