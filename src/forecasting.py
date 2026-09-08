import pandas as pd
import numpy as np
from src.evaluation import residual_bootstrap_intervals


def generate_recursive_forecast(model, df_full, feature_cols, horizon=30, residuals=None, alpha=0.05, random_state=42):
    """
    Generate 30-day out-of-sample recursive stock price predictions.
    If `residuals` is provided, computes empirical prediction intervals via residual bootstrapping.
    """
    df_history = df_full.copy().reset_index(drop=True)
    last_date = pd.to_datetime(df_history['Date'].iloc[-1]) if 'Date' in df_history.columns else df_history.index[-1]
    
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='B')
    predictions = []
    
    current_df = df_history.copy()
    
    for i in range(horizon):
        next_date = future_dates[i]
        
        # Build features for current state
        row = {'Date': next_date}
        close_series = current_df['Close']
        
        row['Lag_1'] = close_series.iloc[-1]
        row['Lag_2'] = close_series.iloc[-2] if len(close_series) >= 2 else row['Lag_1']
        row['Lag_3'] = close_series.iloc[-3] if len(close_series) >= 3 else row['Lag_1']
        row['Lag_5'] = close_series.iloc[-5] if len(close_series) >= 5 else row['Lag_1']
        row['Lag_10'] = close_series.iloc[-10] if len(close_series) >= 10 else row['Lag_1']
        
        for w in [5, 10, 20]:
            sub = close_series.iloc[-w:] if len(close_series) >= w else close_series
            row[f'Rolling_Mean_{w}'] = sub.mean()
            row[f'Rolling_Std_{w}'] = sub.std() if len(sub) > 1 else 0.0
            
        row['DayOfWeek'] = next_date.dayofweek
        row['Month'] = next_date.month
        
        if 'Volume' in current_df.columns:
            row['Volume_Lag_1'] = current_df['Volume'].iloc[-1]
            
        # Copy remaining technical indicator columns from last known row if needed
        last_row = current_df.iloc[-1]
        for col in feature_cols:
            if col not in row:
                row[col] = last_row[col] if col in last_row else 0.0
                
        feat_vector = pd.DataFrame([row])[feature_cols]
        pred_val = float(model.predict(feat_vector)[0])
        predictions.append(pred_val)
        
        # Append predicted row for recursive multi-step forecasting
        new_row = {'Date': next_date, 'Close': pred_val}
        if 'Volume' in current_df.columns:
            new_row['Volume'] = current_df['Volume'].iloc[-1]
        for c in ['Open', 'High', 'Low']:
            if c in current_df.columns:
                new_row[c] = pred_val
                
        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
        
    preds = np.array(predictions)
    
    if residuals is not None and len(residuals) > 0:
        lower_ci, upper_ci = residual_bootstrap_intervals(
            preds, residuals, n_simulations=1000, alpha=alpha, random_state=random_state
        )
    else:
        # Fallback std_err expanding interval
        res = current_df['Close'].iloc[-60:].diff().dropna()
        std_err = np.std(res) if len(res) > 0 else 5.0
        margin = 1.96 * std_err * np.sqrt(np.arange(1, horizon + 1) / 2.0)
        lower_ci = preds - margin
        upper_ci = preds + margin
    
    return pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close": preds,
        "Lower_95": lower_ci,
        "Upper_95": upper_ci,
        "Lower_CI": lower_ci,
        "Upper_CI": upper_ci
    })
