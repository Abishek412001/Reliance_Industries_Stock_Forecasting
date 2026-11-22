# Reliance Industries Stock Analysis & Forecasting

An end-to-end time-series forecasting project for Reliance Industries (`RELIANCE.NS`). It explores data preparation, technical indicators, macro-market features, walk-forward validation, model comparison, explainability, and a Streamlit dashboard.

**Live dashboard:** [Open the Streamlit app](https://relianceindustriesstockforecasting-hmd7omsfbkdel89xyqthzj.streamlit.app/)

> **Disclaimer:** This is an educational analytics project, not investment advice. Historical model performance does not guarantee future results. Forecasts are uncertain and should not be used as the sole basis for financial decisions.

## Project workflow

1. Load and align historical stock and external market data.
2. Handle missing values and inspect outliers.
3. Apply stationarity analysis and transformations where appropriate.
4. Build lagged, technical-indicator, calendar, and macroeconomic features.
5. Use time-ordered validation to evaluate forecasting models.
6. Compare statistical baselines with gradient-boosting models.
7. Inspect model explanations and monitor potential data drift.
8. Serve forecasts through FastAPI and explore results in Streamlit.

## Models and evaluation

The repository benchmarks statistical approaches (including naive and SARIMA baselines, ETS, Prophet, and state-space methods) alongside XGBoost, LightGBM, and CatBoost.

The README's recorded held-out test results are:

| Model | RMSE (INR) | MAE (INR) | MAPE | R² |
|---|---:|---:|---:|---:|
| XGBoost | 14.10 | 11.25 | 2.97% | 0.9134 |
| LightGBM | 16.44 | 13.06 | 3.51% | 0.8821 |
| CatBoost | 16.55 | 13.41 | 3.61% | 0.8806 |
| Prophet | 48.50 | 38.01 | 10.53% | -0.0254 |

These figures are the project's recorded evaluation results for the stated historical test period (2023-01-02 to 2023-10-16), not a statement of current or future accuracy. Re-run the pipeline to reproduce them.

## Key techniques

- Time-series resampling and missing-data handling
- Augmented Dickey-Fuller stationarity testing
- Lagged features and technical indicators (RSI, MACD, Bollinger width, ATR and ROC)
- Expanding-window walk-forward validation with `TimeSeriesSplit`
- MAE, RMSE, MAPE, sMAPE, MASE, directional accuracy and R²
- SHAP-based feature attribution
- Kolmogorov-Smirnov two-sample drift checks
- FastAPI forecasting endpoint and Streamlit visualization

## Technology stack

Python · Pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Statsmodels · Prophet · SHAP · FastAPI · Streamlit · Plotly · Docker · pytest

## Repository structure

```text
src/                  Data, feature, model, evaluation and forecasting modules
api/                  FastAPI application
monitoring/           Drift detection
tests/                Pipeline tests
models/               Serialized model artifacts and feature metadata
data/                 Historical and processed datasets
metrics/              Model comparison results
app.py                Streamlit dashboard
requirements.txt      Python dependencies
Dockerfile            Container build
docker-compose.yml    Local service orchestration
```

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Abishek412001/Reliance_Industries_Stock_Forecasting.git
cd Reliance_Industries_Stock_Forecasting
```

### 2. Create an environment and install dependencies

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

### 3. Run tests

```bash
python -m pytest tests/
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

### 5. Launch the API (optional)

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the API documentation when the service is running.

## Author

**Abishek** · [GitHub profile](https://github.com/Abishek412001)
