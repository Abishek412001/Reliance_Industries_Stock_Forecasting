# 🐧 Reliance Industries — 100% Open-Source Stock Price Analysis & 30-Day Forecasting

An enterprise-grade, 100% open-source machine learning and time-series forecasting system built for **Reliance Industries** (`RELIANCE.NS`). 

Built entirely with open-source technologies (**Python**, **Streamlit**, **FastAPI**, **Scikit-Learn**, **XGBoost**, **LightGBM**, **CatBoost**, **Prophet**, **Statsmodels**, **Plotly**, **Docker**, and **Pytest**).

**Live dashboard:** [Open the Reliance Industries Stock Forecasting app](https://relianceindustriesstockforecasting-hmd7omsfbkdel89xyqthzj.streamlit.app/)

---

## 🌟 Architecture & Key Features

- **Data Quality & Alignment**: Business-day resampling (`'B'`), temporal gap imputation (`ffill`, `spline`), and rolling Z-score outlier capping (window=20, threshold=3.0).
- **Stationarity Transformations**: Log-differencing transformation backed by the Augmented Dickey-Fuller (ADF) test ($p < 0.0001$).
- **Leak-Free Feature Engineering**: Constructs 21 technical indicators (**RSI-14**, **MACD**, **Bollinger Width**, **ATR-14**, **ROC-10**, **Lags 1–10**, **Rolling Stats**, **Calendar components**). Every indicator is strictly shifted by 1 step (`shift(1)`) to ensure **zero data leakage**.
- **External Macro Factors Integration**: Dynamically merges market benchmark data for **Crude Oil** (`CL=F`), **Nifty 50** (`^NSEI`), and **USD/INR** (`INR=X`).
- **Backtesting Strategy**: 5-Fold Expanding Window Walk-Forward Validation (`TimeSeriesSplit`).
- **Comprehensive 8-Model Benchmarking**:
  - **Statistical Baselines**: Naive Baseline, SARIMA (0,1,0), Exponential Smoothing (ETS), Prophet, Unobserved Components (State Space).
  - **Gradient Boosted Trees**: XGBoost, LightGBM, CatBoost.
- **Extended Evaluation Metrics**: Measures **MASE** (Mean Absolute Scaled Error), **sMAPE**, **RMSE**, **MAE**, **MAPE**, **Directional Accuracy**, and $R^2$.
- **Model Explainability**: SHAP (SHapley Additive exPlanations) TreeExplainer global and local feature attribution.
- **Concept Drift Monitoring**: Real-time Kolmogorov-Smirnov 2-sample statistical drift test (`detect_concept_drift`).
- **FastAPI Serving Layer**: REST API (`api/main.py`) exposing `/forecast` and `/` health endpoints.
- **Streamlit Dashboard**: Interactive UI (`app.py`) featuring Plotly charts, 95% confidence bands, model leaderboards, and CSV exports.
- **DevOps & Containerization**: `Dockerfile`, `docker-compose.yml`, GitHub Actions CI/CD (`.github/workflows/ci.yml`), and unit test suite (`tests/test_pipeline.py`).

---

## 📊 Held-Out Model Performance Leaderboard (2023 Test Set)

Models evaluated on unseen held-out test data (2023-01-02 to 2023-10-16):

| Rank | Model | Model Type | RMSE (INR) | MAE (INR) | MAPE (%) | sMAPE (%) | MASE | $R^2$ Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost** | Gradient Boosted Trees | **14.10** | **11.25** | **2.97%** | **2.94%** | **0.871** | **0.9134** |
| 🥈 | **LightGBM** | Gradient Boosted Trees | **16.44** | **13.06** | **3.51%** | **3.46%** | **1.011** | **0.8821** |
| 🥉 | **CatBoost** | Gradient Boosted Trees | **16.55** | **13.41** | **3.61%** | **3.56%** | **1.038** | **0.8806** |
| 4 | **Prophet** | Bayesian Time-Series | 48.50 | 38.01 | 10.53% | 10.15% | 2.943 | -0.0254 |
| 5 | **Naive Baseline** | Baseline | 92.96 | 79.80 | 20.02% | 18.06% | 6.179 | -2.7669 |
| 6 | **SARIMA (0,1,0)** | Statistical TSA | 92.96 | 79.80 | 20.02% | 18.06% | 6.179 | -2.7669 |
| 7 | **State Space (Structural UC)** | Statistical TSA | 139.51 | 122.38 | 30.98% | 26.69% | 9.476 | -7.4836 |
| 8 | **Exponential Smoothing (ETS)** | Statistical TSA | 139.51 | 122.38 | 30.98% | 26.69% | 9.476 | -7.4836 |

---

## 📁 Repository Structure

```
Reliance_stocks_prediction/
│
├── src/                                       # 100% Open-Source Python Package
│   ├── __init__.py                            # Package initialization
│   ├── data.py                                # Resampling, imputation, outlier capping, drift test
│   ├── features.py                            # Technical & calendar feature engineering
│   ├── models.py                              # Open-source model training functions
│   ├── evaluation.py                          # MASE, sMAPE, directional accuracy & CV
│   └── forecasting.py                         # Recursive 30-day forecaster with confidence bands
│
├── api/                                       # Serving Layer
│   └── main.py                                # FastAPI server (GET /, POST /forecast)
│
├── monitoring/                                # Model Monitoring
│   └── drift_detection.py                     # Kolmogorov-Smirnov concept drift detector
│
├── tests/                                     # Automated Unit Test Suite
│   └── test_pipeline.py                       # Pytest assertions
│
├── models/                                    # Serialized Production Artifacts
│   ├── lgbm_model.pkl                         # Champion LightGBM model
│   ├── cat_model.pkl                          # Champion CatBoost model
│   ├── xgb_model.pkl                          # Champion XGBoost model
│   ├── feature_cols.pkl                       # Feature metadata
│   ├── shap_values.pkl                        # Precomputed SHAP feature attributions
│   └── shap_feature_names.pkl                 # Feature names for SHAP plots
│
├── data/                                      # Data directory for UI ingestion
│   └── Company_stock_prices_clean.csv         # Preprocessed stock dataset
│
├── metrics/                                   # Model evaluation metrics
│   └── model_comparison.csv                   # 8-model benchmark results table
│
├── app.py                                     # Interactive Streamlit Dashboard UI
├── Reliance_stocks (1).ipynb                 # Canonical 20-cell end-to-end ML notebook
├── Company_stock_prices.csv                   # Historical raw stock prices
├── Dockerfile                                 # Multi-stage Docker container
├── docker-compose.yml                         # Multi-container orchestration (FastAPI + Streamlit)
├── .github/workflows/ci.yml                   # GitHub Actions CI/CD pipeline
├── requirements.txt                           # Open-source dependencies
└── README.md                                  # Documentation
```

---

## ⚙️ Installation & Local Setup

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Abishek412001/Reliance_Industries_Stock_Forecasting.git
cd Reliance_Industries_Stock_Forecasting

# Create virtual environment
python -m venv venv
# Activate on Windows:
.\venv\Scripts\activate
# Activate on macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pytest Suite
```bash
python -m pytest tests/
```

### 4. Run Notebook Pipeline
```bash
jupyter notebook "Reliance_stocks (1).ipynb"
```

### 5. Launch FastAPI Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 6. Launch Streamlit Dashboard
```bash
streamlit run app.py
```
- Dashboard URL: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Docker Execution

Build and run both the FastAPI serving layer and Streamlit dashboard using Docker Compose:
```bash
docker-compose up --build
```

---

## 🚀 Free Hosting Deployment Options ($0 Cost)

### Option 1: Streamlit Community Cloud (Free)
**Deployed app:** [relianceindustriesstockforecasting-hmd7omsfbkdel89xyqthzj.streamlit.app](https://relianceindustriesstockforecasting-hmd7omsfbkdel89xyqthzj.streamlit.app/)

1. Push repository to **GitHub**:
   ```bash
   git add .
   git commit -m "Deploy open-source pipeline"
   git push origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Select repository and set main file to `app.py`. Click **Deploy**!

### Option 2: Hugging Face Spaces (Free)
1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces) selecting the **Streamlit** SDK.
2. Push your repository files to the Space.

### Option 3: Railway / Render (Free Tier)
1. Connect GitHub repository to Render/Railway.
2. Set build command `pip install -r requirements.txt` and start command `streamlit run app.py --server.port=$PORT`.

---

## 📄 License & Author

- **Author**: Abishek
- **Repository**: [Abishek412001/Reliance_Industries_Stock_Forecasting](https://github.com/Abishek412001/Reliance_Industries_Stock_Forecasting)
- **License**: 100% Open-Source (MIT / Apache 2.0 / BSD)
