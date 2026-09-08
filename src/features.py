import pandas as pd
import numpy as np
import yfinance as yf


def create_lag_features(data, lags=(1, 2, 3, 5, 10), column='Close'):
    """Create auto-regressive lag features shifted by 1."""
    data = data.copy()
    for lag in lags:
        data[f'Lag_{lag}'] = data[column].shift(lag)
    return data


def create_rolling_features(data, windows=(5, 10, 20), column='Close'):
    """Create rolling window features shifted by 1 to prevent data leakage."""
    data = data.copy()
    shifted = data[column].shift(1)
    for window in windows:
        data[f'Rolling_Mean_{window}'] = shifted.rolling(window).mean()
        data[f'Rolling_Std_{window}'] = shifted.rolling(window).std()
        data[f'Rolling_Min_{window}'] = shifted.rolling(window).min()
        data[f'Rolling_Max_{window}'] = shifted.rolling(window).max()
    return data


def create_calendar_features(data):
    """Extract temporal calendar components."""
    data = data.copy()
    idx = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['Date'])
    data['DayOfWeek'] = idx.dayofweek
    data['Month'] = idx.month
    data['Quarter'] = idx.quarter
    data['IsMonthStart'] = idx.is_month_start.astype(int)
    data['IsMonthEnd'] = idx.is_month_end.astype(int)
    return data


def create_external_features(data, tickers=['^NSEI', 'CL=F', 'INR=X']):
    """Fetch and merge external benchmark macro factors."""
    data = data.copy()
    idx = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['Date'])
    start_date = idx.min().strftime('%Y-%m-%d')
    end_date = idx.max().strftime('%Y-%m-%d')
    
    external_dfs = []
    for ticker in tickers:
        try:
            ext = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(ext.columns, pd.MultiIndex):
                c_val = ext['Adj Close'][ticker].values if 'Adj Close' in ext else ext['Close'][ticker].values
            else:
                c_val = ext['Adj Close'].values if 'Adj Close' in ext else ext['Close'].values
            clean_name = ticker.replace('^', '').replace('=F', '').replace('=X', '')
            df_ext = pd.DataFrame({'Date': pd.to_datetime(ext.index).tz_localize(None).floor('D'), f'{clean_name}_Close': c_val})
            external_dfs.append((clean_name, df_ext))
        except Exception:
            pass

    if isinstance(data.index, pd.DatetimeIndex):
        data_reset = data.reset_index()
    else:
        data_reset = data.copy()
    data_reset['Date'] = pd.to_datetime(data_reset['Date']).astype('datetime64[ns]')

    for clean_name, df_ext in external_dfs:
        df_ext['Date'] = pd.to_datetime(df_ext['Date']).astype('datetime64[ns]')
        data_reset = pd.merge_asof(data_reset.sort_values('Date'), df_ext.sort_values('Date'), on='Date', direction='nearest')

    if isinstance(data.index, pd.DatetimeIndex):
        return data_reset.set_index('Date')
    return data_reset


def build_advanced_features(data):
    """Construct 18 leak-free technical indicators shifted by 1."""
    df_out = data.copy()
    if isinstance(df_out.index, pd.DatetimeIndex):
        df_out = df_out.reset_index()

    # RSI 14
    delta = df_out['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df_out['RSI_14'] = (100 - (100 / (1 + rs))).shift(1)

    # MACD & MACD Signal
    ema12 = df_out['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df_out['Close'].ewm(span=26, adjust=False).mean()
    df_out['MACD'] = (ema12 - ema26).shift(1)
    df_out['MACD_Signal'] = df_out['MACD'].ewm(span=9, adjust=False).mean().shift(1)

    # Bollinger Bands Width
    bb_mean = df_out['Close'].rolling(20).mean()
    bb_std = df_out['Close'].rolling(20).std()
    df_out['Bollinger_Width'] = (((bb_mean + 2 * bb_std) - (bb_mean - 2 * bb_std)) / (bb_mean + 1e-9)).shift(1)

    # ATR 14
    if 'High' in df_out.columns and 'Low' in df_out.columns:
        tr1 = df_out['High'] - df_out['Low']
        tr2 = (df_out['High'] - df_out['Close'].shift(1)).abs()
        tr3 = (df_out['Low'] - df_out['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df_out['ATR_14'] = tr.rolling(14).mean().shift(1)

    # ROC 10
    df_out['ROC_10'] = df_out['Close'].pct_change(10).shift(1) * 100

    # Lags & Rolling Stats
    for lag in [1, 2, 3, 5, 10]:
        df_out[f'Lag_{lag}'] = df_out['Close'].shift(lag)
    for w in [5, 10, 20]:
        df_out[f'Rolling_Mean_{w}'] = df_out['Close'].shift(1).rolling(w).mean()
        df_out[f'Rolling_Std_{w}'] = df_out['Close'].shift(1).rolling(w).std()

    idx = pd.to_datetime(df_out['Date'])
    df_out['DayOfWeek'] = idx.dt.dayofweek
    df_out['Month'] = idx.dt.month
    if 'Volume' in df_out.columns:
        df_out['Volume_Lag_1'] = df_out['Volume'].shift(1)

    df_out = df_out.dropna().reset_index(drop=True)
    return df_out


def create_all_features(data):
    """Complete feature engineering pipeline."""
    return build_advanced_features(data)
