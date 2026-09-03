# ============================================================
# Run this script or final notebook cell to refit all models
# on the full dataset (2020-10-19 to 2023-10-16) and save
# all deployment artifacts for app.py.
#
# Writes into subfolders that app.py expects:
#   data/Company_stock_prices_clean.csv
#   models/*.pkl
#   metrics/model_comparison.csv
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.structural import UnobservedComponents
from prophet import Prophet
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# 1. Data Ingestion
DATA_PATH = os.path.join(DATA_DIR, "Company_stock_prices_clean.csv")
if not os.path.exists(DATA_PATH):
    import yfinance as yf
    df_download = yf.download('RELIANCE.NS', start='2020-10-19', end='2023-10-17')
    if isinstance(df_download.columns, pd.MultiIndex):
        close_val = df_download['Adj Close']['RELIANCE.NS'].values if 'Adj Close' in df_download else df_download['Close']['RELIANCE.NS'].values
        open_val = df_download['Open']['RELIANCE.NS'].values
        high_val = df_download['High']['RELIANCE.NS'].values
        low_val = df_download['Low']['RELIANCE.NS'].values
        vol_val = df_download['Volume']['RELIANCE.NS'].values
        dates_val = df_download.index
    else:
        close_val = df_download['Adj Close'].values if 'Adj Close' in df_download else df_download['Close'].values
        open_val = df_download['Open'].values
        high_val = df_download['High'].values
        low_val = df_download['Low'].values
        vol_val = df_download['Volume'].values
        dates_val = df_download.index

    df = pd.DataFrame({'Date': dates_val, 'Open': open_val, 'High': high_val, 'Low': low_val, 'Close': close_val, 'Volume': vol_val})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

full_close = df.set_index("Date")["Close"]

# 2. Refit SARIMA
sarima_final = SARIMAX(full_close, order=(0, 1, 0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
joblib.dump(sarima_final, os.path.join(MODELS_DIR, "sarima_model.pkl"))

# 3. Refit ETS
ets_final = ExponentialSmoothing(full_close, trend="add", seasonal=None, initialization_method="estimated").fit()
joblib.dump(ets_final, os.path.join(MODELS_DIR, "ets_model.pkl"))

# 4. Refit Prophet Default & No-Seasonality
full_prophet_df = pd.DataFrame({"ds": df["Date"], "y": df["Close"]})
prophet_def = Prophet()
prophet_def.fit(full_prophet_df)
joblib.dump(prophet_def, os.path.join(MODELS_DIR, "prophet_default_model.pkl"))

prophet_no_season = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
prophet_no_season.fit(full_prophet_df)
joblib.dump(prophet_no_season, os.path.join(MODELS_DIR, "prophet_no_seasonality_model.pkl"))

# 5. Refit State Space
ss_final = UnobservedComponents(full_close, level="local linear trend").fit(disp=False)
joblib.dump(ss_final, os.path.join(MODELS_DIR, "state_space_model.pkl"))

# 6. Refit XGBoost
feature_df = df[["Date", "Close", "Volume"]].copy()
for lag in [1, 2, 3, 5, 10]:
    feature_df[f"Lag_{lag}"] = feature_df["Close"].shift(lag)

feature_df["Rolling_Mean_5"] = feature_df["Close"].shift(1).rolling(window=5).mean()
feature_df["Rolling_Std_5"] = feature_df["Close"].shift(1).rolling(window=5).std()
feature_df["Rolling_Mean_20"] = feature_df["Close"].shift(1).rolling(window=20).mean()
feature_df["DayOfWeek"] = feature_df["Date"].dt.dayofweek
feature_df["Month"] = feature_df["Date"].dt.month

feature_df = feature_df.dropna().reset_index(drop=True)
feature_cols = [c for c in feature_df.columns if c not in ["Date", "Close"]]

X_full = feature_df[feature_cols]
y_full = feature_df["Close"]

xgb_final = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_final.fit(X_full, y_full)

joblib.dump(xgb_final, os.path.join(MODELS_DIR, "xgb_model.pkl"))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

# 7. Model Comparison Table
metrics_data = [
    {"Model": "Naive Baseline", "RMSE": 92.81598356820467, "MAE": 79.66570215656567, "MAPE": 19.98513713950862},
    {"Model": "SARIMA (0,1,0)", "RMSE": 92.81598356820467, "MAE": 79.66570215656567, "MAPE": 19.98513713950862},
    {"Model": "Exponential Smoothing (ETS)", "RMSE": 139.10281432420956, "MAE": 121.97941328904576, "MAPE": 30.87715014389279},
    {"Model": "Prophet (Default)", "RMSE": 286.324248, "MAE": 267.390054, "MAPE": 69.504316},
    {"Model": "Prophet (No Seasonality)", "RMSE": 49.73305412852206, "MAE": 39.01912445892552, "MAPE": 10.822365409105436},
    {"Model": "State Space (Structural)", "RMSE": 139.1013251421098, "MAE": 121.9780751280945, "MAPE": 30.876805129084795},
    {"Model": "XGBoost", "RMSE": 16.532353678934328, "MAE": 13.072876000937105, "MAPE": 3.526569606856314}
]
comparison_df = pd.DataFrame(metrics_data).set_index("Model")
comparison_df.to_csv(os.path.join(METRICS_DIR, "model_comparison.csv"))

print("Saved all model artifacts successfully.")
print(f"  Data:    {DATA_PATH}")
print(f"  Models:  {MODELS_DIR}/")
print(f"  Metrics: {os.path.join(METRICS_DIR, 'model_comparison.csv')}")
