# Reliance Industries — Stock Price Forecasting

An end-to-end time-series project: EDA, preprocessing, model building, evaluation, and a deployed Streamlit dashboard for forecasting Reliance Industries' closing stock price.

## Business Objective

Predict Reliance Industries' stock closing price over a future horizon (up to 30 trading days), using historical OHLC and volume data.

## Data

- Source: daily OHLC + Adjusted Close + Volume for Reliance Industries
- Range used: 2020-10-19 to 2023-10-16 (733 trading days)
- Train/test split is **chronological**, not random: train on data before 2023-01-01 (555 rows), test on 2023 onward (198 rows)

## Project Structure

```
.
├── Stocks_model.ipynb              # Full notebook: EDA → preprocessing → modeling → evaluation
├── export_artifacts.py             # Run as a final notebook cell to save trained models for deployment
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Dependencies
├── Company_stock_prices_clean.csv  # Cleaned data (produced by export_artifacts.py)
├── xgb_model.pkl                   # Trained XGBoost model (produced by export_artifacts.py)
├── sarima_model.pkl                # Trained SARIMA model (produced by export_artifacts.py)
├── feature_cols.pkl                # Feature column list used by the XGBoost model
└── model_comparison.csv            # Saved evaluation metrics table
```

## Methodology

**EDA & preprocessing**
- Verified date range, missing values, duplicate dates, and OHLC internal consistency (High ≥ Open/Close, Low ≤ Open/Close)
- Computed daily returns, 20-day and 200-day moving averages, and 20-day rolling volatility
- Identified and investigated the largest single-day price movements against known external events (COVID-19, corporate actions, etc.)
- Flagged extreme return days (|daily return| > 10%) for audit rather than silently removing them

**Stationarity check**
- Augmented Dickey-Fuller test on the raw closing price (p = 0.79 → non-stationary)
- First-differenced series is stationary (p ≈ 0.0), confirming `d = 1` for SARIMA

**Model 1 — SARIMA**
- Order selected via AIC grid search over p, q ∈ {0, 1, 2} with d = 1
- No seasonal component: closing price is a trending series, not a periodic one — adding seasonality wasn't justified by the data
- Selected order: **(0, 1, 0)** — this is mathematically a random walk, so its forecast is just the last known price carried forward with widening confidence intervals

**Model 2 — XGBoost**
- Feature engineering: lag features (1, 2, 3, 5, 10 days), 5- and 20-day rolling mean/std (shifted to avoid leakage), day-of-week, month, and volume
- Same chronological train/test split as SARIMA

**Baseline**
- Naive forecast (last training price carried forward), included so both models have to prove they beat the simplest possible guess

## Results (2023 held-out test set)

| Model          | RMSE  | MAE   | MAPE   |
|----------------|-------|-------|--------|
| Naive Baseline | 92.82 | 79.67 | 19.99% |
| SARIMA         | 92.82 | 79.67 | 19.99% |
| XGBoost        | 16.53 | 13.07 | 3.53%  |

XGBoost was the clear winner. SARIMA's selected order collapsed to a random walk, so it performed identically to the naive baseline — it added no value over "assume no change" on this data.

**Important caveat:** these XGBoost numbers are from one-step-ahead evaluation, where each prediction used the *true* prior day's closing price as `Lag_1`. A real multi-day-ahead forecast can't do that — it has to use its own prior predictions as inputs, and error compounds with each step. The deployed app's 30-Day Forecast tab does this recursively and will be noticeably less accurate than the table above suggests, especially later in the horizon.

## Deployment

1. In the notebook, after the modeling cells have run, run `export_artifacts.py` as a final cell. It refits both models on the full dataset and saves everything `app.py` needs.
2. Place `app.py`, `requirements.txt`, and the exported artifacts in the same folder.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the dashboard:
   ```
   streamlit run app.py
   ```

The dashboard has four tabs: **Historical Analysis**, **Time-Series Analysis** (trend, returns, volatility), **Model Evaluation** (the table above), and **30-Day Forecast** (choose a horizon and model, XGBoost and/or SARIMA).

## Limitations

- This is a statistical/ML exercise, not investment advice or a guaranteed prediction of future prices
- Future trading volume is unknown; the app carries forward the last observed volume as a simplifying assumption (volume had negligible importance in the trained model, ~0.2%)
- XGBoost's recursive forecast error compounds day over day — treat longer horizons as directional, not precise
- SARIMA's forecast is effectively a flat projection with growing uncertainty bands, given the selected (0,1,0) order

## Tech Stack

Python, pandas, NumPy, statsmodels (SARIMA), XGBoost, scikit-learn (metrics), Matplotlib, Streamlit
