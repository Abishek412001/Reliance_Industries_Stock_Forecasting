import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_mase(y_true, y_pred, y_train, seasonal_period=1):
    """
    Mean Absolute Scaled Error. Values below 1 indicate the model
    beats the in-sample naive (lag-1) benchmark; values above 1
    mean it's worse than just carrying the last value forward.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if len(y_train) <= seasonal_period:
        return np.nan

    scale = np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period]))
    if scale <= 1e-12:
        return np.nan

    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def evaluate_forecast(y_true, y_pred, model_name, y_train=None):
    """Comprehensive evaluation metrics (RMSE, MAE, MAPE, sMAPE, MASE, Directional Accuracy, R2)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    error = y_true - y_pred
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    
    mask = np.abs(y_true) > 1e-8
    mape = float(np.mean(np.abs(error[mask] / y_true[mask])) * 100) if mask.any() else np.nan
    smape = float(np.mean(2 * np.abs(error) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100)
    
    if len(y_true) > 1:
        actual_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        dir_acc = float(np.mean(actual_dir == pred_dir) * 100)
    else:
        dir_acc = np.nan
        
    mase = calculate_mase(y_true, y_pred, y_train) if y_train is not None else np.nan
    r2 = float(r2_score(y_true, y_pred))
    
    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "sMAPE": smape,
        "MASE": mase,
        "Directional_Accuracy": dir_acc,
        "R2": r2
    }


def residual_bootstrap_intervals(point_forecast, residuals, n_simulations=1000, 
                                   alpha=0.05, random_state=42):
    """
    Empirical prediction intervals from residual resampling.
    NOT a guaranteed probabilistic interval — document this
    clearly wherever it's displayed (notebook markdown AND the
    Streamlit app) so it isn't mistaken for a model-native CI
    like SARIMA's.
    """
    rng = np.random.default_rng(random_state)
    point_forecast = np.asarray(point_forecast, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    simulations = np.empty((n_simulations, len(point_forecast)))
    for i in range(n_simulations):
        sampled = rng.choice(residuals, size=len(point_forecast), replace=True)
        simulations[i] = point_forecast + sampled
    lower = np.quantile(simulations, alpha / 2, axis=0)
    upper = np.quantile(simulations, 1 - alpha / 2, axis=0)
    return lower, upper


def rolling_30day_backtest(series, forecast_function, initial_train_size, 
                             horizon=30, step=30):
    """
    Evaluates a forecast function across multiple historical origins,
    not just one. forecast_function(train_slice, horizon) must return
    exactly `horizon` predictions.
    """
    series = pd.Series(series).dropna()
    records = []
    last_origin = len(series) - horizon
    for origin in range(initial_train_size, last_origin + 1, step):
        train_slice = series.iloc[:origin]
        test_slice = series.iloc[origin:origin + horizon]
        prediction = np.asarray(forecast_function(train_slice, horizon), dtype=float)
        if len(prediction) != len(test_slice):
            raise ValueError(f"Forecast length {len(prediction)} does not match horizon {len(test_slice)}")
        
        origin_label = str(series.index[origin]) if hasattr(series.index[origin], 'strftime') else f"row_{origin}"
        metrics = evaluate_forecast(
            test_slice.to_numpy(), prediction,
            model_name=f"origin_{origin_label}",
            y_train=train_slice.to_numpy(),
        )
        metrics["Forecast_Origin"] = origin_label
        records.append(metrics)
    return pd.DataFrame(records)


def expanding_window_cv(model_class_fn, X, y, n_splits=5):
    """Expanding window time-series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        
        model = model_class_fn()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        
        metrics = evaluate_forecast(y_te.values, preds, f"Fold_{fold+1}", y_train=y_tr.values)
        fold_metrics.append(metrics)
        
    return pd.DataFrame(fold_metrics)
