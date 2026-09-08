import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import fetch_data, resample_to_daily, impute_missing_values, detect_outliers_rolling_zscore, make_stationary, detect_concept_drift
from src.features import create_lag_features, create_rolling_features, build_advanced_features
from src.models import train_xgboost, train_lightgbm, train_catboost
from src.evaluation import calculate_mase, evaluate_forecast, residual_bootstrap_intervals, rolling_30day_backtest
from src.forecasting import generate_recursive_forecast


@pytest.fixture
def dummy_stock_data():
    dates = pd.date_range(start="2022-01-01", periods=100, freq="B")
    np.random.seed(42)
    close_prices = 1000.0 + np.cumsum(np.random.randn(100) * 5.0)
    df = pd.DataFrame({
        "Date": dates,
        "Open": close_prices - 2.0,
        "High": close_prices + 5.0,
        "Low": close_prices - 5.0,
        "Close": close_prices,
        "Volume": np.random.randint(100000, 5000000, size=100)
    }).set_index("Date")
    return df


def test_mase_calculation():
    y_true = np.array([100.0, 102.0, 101.0, 105.0, 107.0])
    y_pred = np.array([101.0, 101.5, 101.5, 104.0, 106.0])
    y_train = np.array([90.0, 92.0, 95.0, 97.0, 99.0])
    
    mase = calculate_mase(y_true, y_pred, y_train)
    assert not np.isnan(mase)
    assert mase > 0
    
    metrics = evaluate_forecast(y_true, y_pred, "TestModel", y_train=y_train)
    assert metrics["Model"] == "TestModel"
    assert "MASE" in metrics
    assert not np.isnan(metrics["MASE"])


def test_residual_bootstrap_intervals():
    point_forecast = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    residuals = np.array([-1.5, 0.5, -0.2, 1.2, -0.8, 0.1, -1.0, 2.0])
    
    lower, upper = residual_bootstrap_intervals(point_forecast, residuals, n_simulations=500, alpha=0.05)
    assert len(lower) == len(point_forecast)
    assert len(upper) == len(point_forecast)
    assert np.all(lower <= point_forecast)
    assert np.all(upper >= point_forecast)


def test_rolling_30day_backtest(dummy_stock_data):
    series = dummy_stock_data["Close"]
    
    def dummy_forecast_fn(train_slice, horizon):
        last_val = train_slice.iloc[-1]
        return np.full(horizon, last_val)
        
    backtest_df = rolling_30day_backtest(series, dummy_forecast_fn, initial_train_size=40, horizon=10, step=10)
    assert len(backtest_df) >= 2
    assert "MASE" in backtest_df.columns
    assert "Forecast_Origin" in backtest_df.columns


def test_recursive_forecast_with_residuals(dummy_stock_data):
    df_ml = build_advanced_features(dummy_stock_data)
    feature_cols = [c for c in df_ml.columns if c not in ["Date", "Close", "High", "Low", "Volume"]]
    
    X = df_ml[feature_cols]
    y = df_ml["Close"]
    
    model = train_xgboost(X, y)
    residuals = y.to_numpy() - model.predict(X)
    
    forecast_df = generate_recursive_forecast(model, df_ml, feature_cols, horizon=10, residuals=residuals)
    assert len(forecast_df) == 10
    assert "Lower_CI" in forecast_df.columns
    assert "Upper_CI" in forecast_df.columns
