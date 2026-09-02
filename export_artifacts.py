# ============================================================
# Run this as a final cell in Stocks_model.ipynb (Colab) AFTER
# all model-building and evaluation cells have already run.
# It refits the selected models on the FULL dataset (train + test)
# so the deployed app forecasts from the most recent data, and
# saves everything app.py needs to disk.
# ============================================================

import joblib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.structural import UnobservedComponents
from xgboost import XGBRegressor
from prophet import Prophet
import os

OUTPUT_DIR = "/content/drive/My Drive/DS/Time_series_forecasting"
os.makedirs(OUTPUT_DIR, exist_ok=True)

full_close = df.set_index("Date")["Close"]

# --- Refit SARIMA on the FULL Close series -------------------------------
sarima_final = SARIMAX(
    full_close,
    order=best_order,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)
joblib.dump(sarima_final, f"{OUTPUT_DIR}/sarima_model.pkl")

# --- Refit selected ETS specification on the FULL Close series -----------
ets_trend = "add" if best_ets_name == "Additive Trend" else "mul"
ets_final = ExponentialSmoothing(
    full_close,
    trend=ets_trend,
    seasonal=None,
    initialization_method="estimated"
).fit()
joblib.dump(ets_final, f"{OUTPUT_DIR}/ets_model.pkl")

# --- Refit both Prophet configurations on the FULL Close series ----------
prophet_full = full_close.reset_index()
prophet_full.columns = ["ds", "y"]

prophet_default_final = Prophet()
prophet_default_final.fit(prophet_full)
joblib.dump(prophet_default_final, f"{OUTPUT_DIR}/prophet_default_model.pkl")

prophet_no_seasonality_final = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False
)
prophet_no_seasonality_final.fit(prophet_full)
joblib.dump(
    prophet_no_seasonality_final,
    f"{OUTPUT_DIR}/prophet_no_seasonality_model.pkl"
)

# --- Refit selected state-space specification on the FULL Close series ----
if best_state_space_name == "Local Level":
    state_space_final = UnobservedComponents(
        full_close,
        level="local level"
    ).fit(disp=False)
else:
    state_space_final = UnobservedComponents(
        full_close,
        level="local linear trend"
    ).fit(disp=False)
joblib.dump(state_space_final, f"{OUTPUT_DIR}/state_space_model.pkl")

# --- Refit XGBoost on the FULL feature set -------------------------------
X_full = feature_df[feature_cols]
y_full = feature_df["Close"]

xgb_final = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_final.fit(X_full, y_full)
joblib.dump(xgb_final, f"{OUTPUT_DIR}/xgb_model.pkl")
joblib.dump(feature_cols, f"{OUTPUT_DIR}/feature_cols.pkl")

comparison_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv")
df.to_csv(f"{OUTPUT_DIR}/Company_stock_prices_clean.csv", index=False)

print("Saved: sarima_model.pkl, ets_model.pkl,")
print("       prophet_default_model.pkl, prophet_no_seasonality_model.pkl,")
print("       state_space_model.pkl, xgb_model.pkl, feature_cols.pkl,")
print("       model_comparison.csv, Company_stock_prices_clean.csv")
print(f"Location: {OUTPUT_DIR}")
