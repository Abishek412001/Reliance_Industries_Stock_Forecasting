"""Notebook-ready cells for the additional forecasting models.

Paste/run the sections in Stocks_model.ipynb after the existing SARIMA/XGBoost
cells and before artifact export. The code deliberately reuses train_close,
test_close, evaluate_forecast, df, feature_df, and feature_cols already
created by the existing notebook.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.structural import UnobservedComponents

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Exponential Smoothing (ETS / Holt-Winters)
# ----------------------------------------------------------------------
# No seasonal component: the EDA/SARIMA analysis did not justify a stable
# repeating seasonal cycle for this stock series.
ets_candidates = {}
for trend_name, trend in [("Additive Trend", "add"), ("Multiplicative Trend", "mul")]:
    model = ExponentialSmoothing(
        train_close,
        trend=trend,
        seasonal=None,
        initialization_method="estimated",
    ).fit()
    ets_candidates[trend_name] = model
    forecast = model.forecast(len(test_close))
    print(f"ETS {trend_name}: AIC = {model.aic:.2f}")
    evaluate_forecast(test_close.values, forecast.values, f"ETS ({trend_name})")

# Training-AIC selection is used for deployment; the final test metrics above
# are retained for both specifications so the alternative is not hidden.
best_ets_name = min(ets_candidates, key=lambda name: ets_candidates[name].aic)
ets_model = ets_candidates[best_ets_name]
ets_forecast = ets_model.forecast(len(test_close))
ets_forecast.index = test_close.index
ets_metrics = evaluate_forecast(test_close.values, ets_forecast.values, "ETS")
print("Selected ETS specification:", best_ets_name)

# ----------------------------------------------------------------------
# Prophet
# ----------------------------------------------------------------------
try:
    from prophet import Prophet
except ImportError as exc:
    raise ImportError(
        "Prophet is required. Install it with `pip install prophet` and rerun this section."
    ) from exc

prophet_train = train_close.reset_index()
prophet_train.columns = ["ds", "y"]
prophet_default = Prophet()
prophet_default.fit(prophet_train)
prophet_no_seasonality = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
prophet_no_seasonality.fit(prophet_train)

prophet_test_future = pd.DataFrame({"ds": test_close.index})
prophet_default_pred = prophet_default.predict(prophet_test_future)["yhat"].to_numpy()
prophet_no_seasonality_pred = prophet_no_seasonality.predict(prophet_test_future)["yhat"].to_numpy()

prophet_default_metrics = evaluate_forecast(
    test_close.values, prophet_default_pred, "Prophet (Default Seasonality)"
)
prophet_no_seasonality_metrics = evaluate_forecast(
    test_close.values, prophet_no_seasonality_pred, "Prophet (No Seasonality)"
)

# ----------------------------------------------------------------------
# State Space Model
# ----------------------------------------------------------------------
state_space_candidates = {
    "Local Level": UnobservedComponents(train_close, level="local level").fit(disp=False),
    "Local Linear Trend": UnobservedComponents(
        train_close, level="local linear trend"
    ).fit(disp=False),
}

for name, model in state_space_candidates.items():
    forecast = model.get_forecast(steps=len(test_close)).predicted_mean
    print(f"State Space {name}: AIC = {model.aic:.2f}")
    evaluate_forecast(test_close.values, forecast.values, f"State Space ({name})")

best_state_space_name = min(
    state_space_candidates, key=lambda name: state_space_candidates[name].aic
)
state_space_model = state_space_candidates[best_state_space_name]
state_space_forecast_result = state_space_model.get_forecast(steps=len(test_close))
state_space_forecast = state_space_forecast_result.predicted_mean
state_space_conf_int = state_space_forecast_result.conf_int(alpha=0.05)
state_space_forecast.index = test_close.index
state_space_conf_int.index = test_close.index
state_space_metrics = evaluate_forecast(
    test_close.values, state_space_forecast.values, "State Space"
)
print("Selected state-space specification:", best_state_space_name)

# ----------------------------------------------------------------------
# Updated comparison table
# ----------------------------------------------------------------------
comparison_df = pd.DataFrame([
    naive_metrics,
    sarima_metrics,
    xgb_metrics,
    {**ets_metrics, "Model": "ETS (AIC-selected)"},
    prophet_default_metrics,
    prophet_no_seasonality_metrics,
    {**state_space_metrics, "Model": "State Space (AIC-selected)"},
]).set_index("Model")

comparison_df.to_csv("model_comparison.csv")
print(comparison_df)
