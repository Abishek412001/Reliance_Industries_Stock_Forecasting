import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.structural import UnobservedComponents
from prophet import Prophet


def train_xgboost(X_train, y_train, params=None):
    """Train XGBoost (open-source gradient boosting)."""
    if params is None:
        params = {
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, params=None):
    """Train LightGBM (Microsoft open-source GBM)."""
    if params is None:
        params = {
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train, params=None):
    """Train CatBoost (Yandex open-source GBM)."""
    if params is None:
        params = {
            'iterations': 300,
            'depth': 4,
            'learning_rate': 0.05,
            'random_seed': 42,
            'verbose': 0
        }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    return model


def train_arima(y_train, order=(0, 1, 0)):
    """Train SARIMA (open-source statistical model)."""
    model = SARIMAX(y_train, order=order, enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)


def train_ets(y_train):
    """Train Exponential Smoothing (ETS, open-source)."""
    model = ExponentialSmoothing(y_train, trend='add', seasonal=None, initialization_method='estimated')
    return model.fit()


def train_prophet(y_train, dates_train):
    """Train Prophet (Meta open-source GAM)."""
    prophet_df = pd.DataFrame({'ds': pd.to_datetime(dates_train), 'y': y_train.values})
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)
    return model


def train_state_space(y_train):
    """Train Unobserved Components State Space model."""
    model = UnobservedComponents(y_train, level='local linear trend')
    return model.fit(disp=False)
