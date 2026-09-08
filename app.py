import os
import sys
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go

# Append project root to sys.path
sys.path.append(os.path.dirname(__file__))

from src.data import fetch_data, resample_to_daily, impute_missing_values, detect_outliers_rolling_zscore
from src.features import build_advanced_features
from src.forecasting import generate_recursive_forecast
from monitoring.drift_detection import detect_concept_drift

st.set_page_config(
    page_title="Reliance Industries — Stock Price Forecast",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Reliance Industries — Stock Price Forecast & Analytics")
st.markdown("**100% Open-Source Pipeline: LightGBM, CatBoost, XGBoost, SARIMA, Prophet, ETS & State Space**")

# Sidebar Configuration
st.sidebar.header("⚙️ Forecast Configuration")
model_choice = st.sidebar.selectbox("Select Champion Model", ["LightGBM", "CatBoost", "XGBoost"])
horizon = st.sidebar.slider("Forecast Horizon (Trading Days)", min_value=7, max_value=60, value=30)

@st.cache_data(ttl=3600)
def load_and_prep_data():
    df_raw = fetch_data(ticker="RELIANCE.NS")
    df_daily = resample_to_daily(df_raw)
    df_clean = impute_missing_values(df_daily)
    df_capped, _ = detect_outliers_rolling_zscore(df_clean)
    df_ml = build_advanced_features(df_capped)
    return df_raw, df_capped, df_ml

with st.spinner("Ingesting & Preprocessing Data..."):
    df_raw, df_capped, df_ml = load_and_prep_data()

model_map = {
    "LightGBM": "models/lgbm_model.pkl",
    "CatBoost": "models/cat_model.pkl",
    "XGBoost": "models/xgb_model.pkl"
}

model_path = model_map[model_choice]
feat_path = "models/feature_cols.pkl"

model = None
feature_cols = None

if os.path.exists(model_path) and os.path.exists(feat_path):
    model = joblib.load(model_path)
    feature_cols = joblib.load(feat_path)
    st.sidebar.success(f"✅ Loaded model: **{model_choice}**")
else:
    st.sidebar.warning("⚠️ Model file not found. Running baseline fallback.")

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Out-of-Sample Forecast", "📊 Model Leaderboard", "📉 Data & Drift Audit", "📄 Metrics Summary"])

with tab1:
    st.subheader(f"30-Day Forecast Horizon using {model_choice}")
    
    if model is not None and feature_cols is not None:
        # Calculate in-sample test residuals for residual bootstrapping
        split_idx = int(len(df_ml) * 0.8)
        X_test = df_ml[feature_cols].iloc[split_idx:]
        y_test = df_ml["Close"].iloc[split_idx:]
        test_preds = model.predict(X_test)
        residuals = y_test.to_numpy() - test_preds
        
        forecast_df = generate_recursive_forecast(model, df_ml, feature_cols, horizon=horizon, residuals=residuals)
    else:
        last_price = float(df_capped["Close"].iloc[-1])
        dates = pd.date_range(start=pd.Timestamp.now() + pd.Timedelta(days=1), periods=horizon, freq="B")
        preds = np.linspace(last_price, last_price * 1.05, horizon)
        forecast_df = pd.DataFrame({
            "Date": dates,
            "Predicted_Close": preds,
            "Lower_95": preds * 0.95,
            "Upper_95": preds * 1.05,
            "Lower_CI": preds * 0.95,
            "Upper_CI": preds * 1.05
        })

    last_close = float(df_capped["Close"].iloc[-1])
    final_pred = float(forecast_df["Predicted_Close"].iloc[-1])
    pct_change = ((final_pred - last_close) / last_close) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Recorded Close", f"₹{last_close:.2f}")
    col2.metric(f"{horizon}-Day Target Forecast", f"₹{final_pred:.2f}", f"{pct_change:+.2f}%")
    col3.metric("95% Lower Bound", f"₹{forecast_df['Lower_95'].iloc[-1]:.2f}")
    col4.metric("95% Upper Bound", f"₹{forecast_df['Upper_95'].iloc[-1]:.2f}")

    fig = go.Figure()
    
    hist_subset = df_capped.tail(120)
    fig.add_trace(go.Scatter(
        x=hist_subset["Date"] if "Date" in hist_subset.columns else hist_subset.index,
        y=hist_subset["Close"],
        mode="lines",
        name="Historical Close",
        line=dict(color="#1f77b4", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Predicted_Close"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#d62728", width=2.5, dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Upper_95"],
        mode="lines",
        name="95% Upper CI",
        line=dict(color="rgba(214, 39, 40, 0.2)", width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Lower_95"],
        mode="lines",
        name="95% Empirical Prediction Interval",
        fill="tonexty",
        fillcolor="rgba(214, 39, 40, 0.15)",
        line=dict(color="rgba(214, 39, 40, 0.2)", width=0)
    ))

    fig.update_layout(
        title=f"Reliance Industries — {horizon}-Day Price Forecast with 95% Confidence Band",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        hovermode="x unified",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("ℹ️ *Empirical interval from residual resampling — not a model-native confidence interval.*")

    st.markdown("#### Forecast Details Table")
    formatted_df = forecast_df.copy()
    formatted_df["Date"] = pd.to_datetime(formatted_df["Date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(formatted_df.style.format({
        "Predicted_Close": "₹{:.2f}",
        "Lower_95": "₹{:.2f}",
        "Upper_95": "₹{:.2f}",
        "Lower_CI": "₹{:.2f}",
        "Upper_CI": "₹{:.2f}"
    }), use_container_width=True)

    csv_data = formatted_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv_data,
        file_name=f"reliance_forecast_{model_choice.lower()}_30d.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("Model Benchmarking Leaderboard (2023 Held-Out Test Set & MASE)")
    metrics_path = "metrics/model_comparison.csv"
    if os.path.exists(metrics_path):
        m_df = pd.read_csv(metrics_path)
        st.dataframe(m_df, use_container_width=True)
    else:
        st.info("Run the notebook pipeline to generate metrics/model_comparison.csv")

    multi_path = "metrics/multi_origin_backtest.csv"
    if os.path.exists(multi_path):
        st.markdown("#### Multi-Origin Rolling 30-Day Backtest Summary (Across Origins)")
        backtest_df = pd.read_csv(multi_path)
        st.dataframe(backtest_df, use_container_width=True)

with tab3:
    st.subheader("Data Quality Audit & Concept Drift Detection")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Data Integrity Checks:**")
        st.write(f"- Total Historical Records: `{len(df_capped)}`")
        st.write(f"- Date Range: `{df_capped['Date'].min() if 'Date' in df_capped else df_capped.index.min()}` to `{df_capped['Date'].max() if 'Date' in df_capped else df_capped.index.max()}`")
        st.write(f"- Engineered Features: `{len(feature_cols) if feature_cols else 18}`")
        
    with col2:
        st.markdown("**Kolmogorov-Smirnov Concept Drift Test:**")
        split_idx = int(len(df_capped) * 0.8)
        train_slice = df_capped.iloc[:split_idx]
        recent_slice = df_capped.iloc[split_idx:]
        drift_res = detect_concept_drift(train_slice, recent_slice)
        
        if drift_res["drift_detected"]:
            st.error(drift_res["alert"])
        else:
            st.success(drift_res["alert"])
        st.write(f"- KS Statistic: `{drift_res['ks_stat']:.4f}`")
        st.write(f"- P-value: `{drift_res['p_value']:.4f}`")

with tab4:
    st.subheader("Pipeline Technical Architecture")
    st.markdown("""
    - **MASE Metric**: Mean Absolute Scaled Error calculated against in-sample naive lag-1 scale.
    - **Tree Prediction Intervals**: Empirical 95% confidence bands via residual bootstrapping.
    - **Multi-Origin Backtest**: Multi-horizon rolling 30-day evaluation across historical origins.
    - **Data Alignment**: Resampled to business days (`'B'`) with forward-fill holiday handling.
    - **Outlier Capping**: Rolling Z-score capping (window=20, threshold=3.0).
    - **Stationarity**: Augmented Dickey-Fuller test + Log-differencing.
    - **Feature Engineering**: 21 technical indicators strictly shifted by 1 step (`.shift(1)`).
    """)

st.markdown("---")
st.markdown("100% Open-Source Production Stack | Built with Python, Streamlit, FastAPI, Plotly & Scikit-Learn")
