# ============================================================
# Run this as a final cell in Stocks_model.ipynb (Colab) AFTER
# the SARIMA and XGBoost model-building cells have already run.
# It refits both models on the FULL dataset (train + test) so
# the deployed app forecasts from the most recent data, and
# saves everything app.py needs to disk.
# ============================================================

import joblib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

OUTPUT_DIR = "/content/drive/My Drive/DS/Time_series_forecasting"  # adjust if needed

# --- Refit SARIMA on the FULL Close series -------------------------------
full_close = df.set_index("Date")["Close"]

sarima_final = SARIMAX(
    full_close,
    order=best_order,              # reuse the (p, d, q) selected earlier
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

joblib.dump(sarima_final, f"{OUTPUT_DIR}/sarima_model.pkl")

# --- Refit XGBoost on the FULL feature set --------------------------------
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

# --- Save the metrics table so the app doesn't need to recompute it -------
comparison_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv")

# --- Save a clean copy of the raw data the app will load ------------------
df.to_csv(f"{OUTPUT_DIR}/Company_stock_prices_clean.csv", index=False)

print("Saved: sarima_model.pkl, xgb_model.pkl, feature_cols.pkl,")
print("       model_comparison.csv, Company_stock_prices_clean.csv")
print(f"Location: {OUTPUT_DIR}")
