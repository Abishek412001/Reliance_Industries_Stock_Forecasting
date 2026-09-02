# Additional forecasting models

The existing chronological split and evaluation function are unchanged: train on observations before `2023-01-01`, evaluate on the 198 observations from 2023 onward, with RMSE, MAE and MAPE.

## Verified results from the supplied dataset

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Naive Baseline | 92.82 | 79.67 | 19.99% |
| SARIMA `(0,1,0)` | 92.82 | 79.67 | 19.99% |
| XGBoost | 16.53 | 13.07 | 3.53% |
| ETS — Additive Trend | 139.10 | 121.98 | 30.88% |
| ETS — Multiplicative Trend | 127.42 | 111.68 | 28.24% |
| State Space — Local Level | 92.82 | 79.67 | 19.99% |
| State Space — Local Linear Trend | 139.10 | 121.98 | 30.88% |

The ETS and state-space variants above were evaluated without using the 2023 test set for fitting. Their training AICs are also used to identify the specification used for deployment. Under that criterion, ETS selects the additive trend and state space selects local level. Both variants remain reported so an underperforming specification is not silently discarded.

## Prophet

Prophet is implemented with two explicit configurations:

1. `Prophet()` — library defaults.
2. `Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)` — no built-in seasonality.

The evaluation code predicts on the exact 2023 trading dates rather than creating synthetic business-day rows. Prophet requires the `prophet` package in `requirements.txt`.

**Important:** Prophet metrics are intentionally not fabricated. The execution environment used to prepare this branch did not have a working Prophet installation, so the checked-in `model_comparison.csv` contains only the independently verified ETS and state-space results. Run the Prophet notebook section after installing the dependency; the notebook code will append both Prophet rows using the same metrics function.

## Deployment artifacts

`export_artifacts.py` now exports:

- `sarima_model.pkl`
- `ets_model.pkl`
- `prophet_default_model.pkl`
- `prophet_no_seasonality_model.pkl`
- `state_space_model.pkl`
- `xgb_model.pkl`
- `feature_cols.pkl`
- `model_comparison.csv`
- `Company_stock_prices_clean.csv`

`app.py` reuses the existing forecast-horizon control and exposes ETS, both Prophet configurations, and State Space alongside the existing XGBoost and SARIMA options.

## Notebook integration

`additional_models.py` contains the notebook-ready model cells. It deliberately reuses the existing `train_close`, `test_close`, `evaluate_forecast`, `feature_df`, and `feature_cols` variables so the original modeling workflow and results are not replaced.
