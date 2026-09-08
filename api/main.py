import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Ensure parent directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import fetch_data, resample_to_daily, impute_missing_values, detect_outliers_rolling_zscore
from src.features import build_advanced_features
from src.forecasting import generate_recursive_forecast

app = FastAPI(
    title="Reliance Stock Forecast API",
    description="100% Open-source stock forecasting API for Reliance Industries",
    version="1.0.0"
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm_model.pkl")
FEATURE_PATH = os.getenv("FEATURE_PATH", "models/feature_cols.pkl")

# Load model & feature columns at startup
model = None
feature_cols = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_PATH):
        model = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURE_PATH)
except Exception as e:
    print(f"Warning: Could not load model at startup: {e}")


class ForecastRequest(BaseModel):
    ticker: str = "RELIANCE.NS"
    horizon: int = 30


class ForecastResponse(BaseModel):
    model_type: str
    forecast: list
    dates: list
    lower_ci: list
    upper_ci: list


@app.get("/")
def health_check():
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "healthy",
        "service": "Reliance Stock Forecast API",
        "model_status": model_status,
        "champion_model": "LightGBM"
    }


@app.post("/forecast", response_model=ForecastResponse)
def generate_forecast(request: ForecastRequest):
    try:
        df_raw = fetch_data(ticker=request.ticker)
        if df_raw.empty:
            raise HTTPException(status_code=400, detail="No data available for ticker")

        df_daily = resample_to_daily(df_raw)
        df_clean = impute_missing_values(df_daily)
        df_capped, _ = detect_outliers_rolling_zscore(df_clean)
        df_ml = build_advanced_features(df_capped)

        if model is None or feature_cols is None:
            # Fallback naive forecast if model pickle is absent
            last_price = float(df_capped["Close"].iloc[-1])
            forecast_dates = pd.date_range(
                start=pd.Timestamp.now() + pd.Timedelta(days=1),
                periods=request.horizon,
                freq="B"
            )
            forecast_values = np.linspace(last_price, last_price * 1.05, request.horizon).tolist()
            lower_ci = (np.array(forecast_values) * 0.95).tolist()
            upper_ci = (np.array(forecast_values) * 1.05).tolist()
            dates_str = [d.strftime("%Y-%m-%d") for d in forecast_dates]
            return ForecastResponse(
                model_type="Baseline (Fallback)",
                forecast=forecast_values,
                dates=dates_str,
                lower_ci=lower_ci,
                upper_ci=upper_ci
            )

        forecast_df = generate_recursive_forecast(model, df_ml, feature_cols, horizon=request.horizon)
        
        return ForecastResponse(
            model_type=type(model).__name__,
            forecast=forecast_df["Predicted_Close"].tolist(),
            dates=[d.strftime("%Y-%m-%d") for d in forecast_df["Date"]],
            lower_ci=forecast_df["Lower_95"].tolist(),
            upper_ci=forecast_df["Upper_95"].tolist()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
