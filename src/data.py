import os
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats


def fetch_data(ticker="RELIANCE.NS", start_date="2020-10-19", local_csv="Company_stock_prices.csv"):
    """Fetch OHLCV data from local CSV or Yahoo Finance (free, open API)."""
    if local_csv and os.path.exists(local_csv):
        raw = pd.read_csv(local_csv)
        raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True)
        raw = raw.set_index("Date").sort_index()
        required = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
        df = raw[required].copy()
    else:
        raw = yf.download(ticker, start=start_date, auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        required = ["Open", "High", "Low", "Close", "Volume"]
        df = raw[required].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)

    df.index.name = "Date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def resample_to_daily(data):
    """Resample to business days ('B') and forward-fill market holidays."""
    data = data.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
        else:
            data.index = pd.to_datetime(data.index)
    
    daily = data.resample('B').first()
    daily = daily.ffill()
    return daily


def impute_missing_values(data, method='ffill', max_gap=5):
    """Impute missing values preserving temporal continuity."""
    data = data.copy()
    if method == 'ffill':
        data = data.ffill(limit=max_gap)
    elif method == 'bfill':
        data = data.bfill(limit=max_gap)
    elif method == 'linear':
        data = data.interpolate(method='linear', limit=max_gap)
    elif method == 'spline':
        data = data.interpolate(method='spline', order=3, limit=max_gap)
    return data.ffill().bfill()


def detect_outliers_rolling_zscore(data, column='Close', window=20, threshold=3.0):
    """Detect outliers using rolling Z-score and cap extreme values."""
    data = data.copy()
    rolling_mean = data[column].rolling(window=window, min_periods=1, center=True).mean()
    rolling_std = data[column].rolling(window=window, min_periods=1, center=True).std().fillna(1e-5)
    
    z_scores = (data[column] - rolling_mean) / (rolling_std + 1e-8)
    outlier_mask = np.abs(z_scores) > threshold
    
    # Cap outliers
    data.loc[outlier_mask, column] = rolling_mean[outlier_mask] + threshold * rolling_std[outlier_mask] * np.sign(z_scores[outlier_mask])
    return data, outlier_mask


def make_stationary(data, column='Close', method='log_diff'):
    """Apply stationarity transformations."""
    data = data.copy()
    if method == 'log_diff':
        data[f'{column}_stationary'] = np.log(data[column] + 1e-8).diff()
    elif method == 'diff':
        data[f'{column}_stationary'] = data[column].diff()
    return data.dropna()


def detect_concept_drift(train_data, new_data, column='Close', window=30):
    """Detect concept drift using Kolmogorov-Smirnov 2-sample test."""
    train_dist = train_data[column].iloc[-window:]
    new_dist = new_data[column].iloc[-window:]
    
    ks_stat, p_value = stats.ks_2samp(train_dist, new_dist)
    drift_detected = p_value < 0.05
    return drift_detected, p_value
