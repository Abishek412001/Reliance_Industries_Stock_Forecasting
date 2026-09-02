"""Forecast view and Plotly forecast visualization."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models import sarima_forecast, xgb_forecast_with_band
from .common import ACTUAL_COLOR, BAND_COLOR, PREDICTED_COLOR, base_layout


MODEL_OPTIONS = ["XGBoost", "SARIMA", "Compare both"]


def _forecast_figure(
    data: pd.DataFrame,
    forecasts: list[tuple[str, pd.DataFrame, str]],
    horizon: int,
) -> go.Figure:
    """Build the forecast chart with actual OHLC hover details and bands."""
    recent = data.tail(60).copy()
    fig = go.Figure()
    actual_customdata = recent[["Open", "High", "Low", "Close"]].to_numpy()
    fig.add_trace(
        go.Scatter(
            x=recent["Date"],
            y=recent["Close"],
            mode="lines",
            name="Actual Close",
            line={"color": ACTUAL_COLOR},
            customdata=actual_customdata,
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>Actual Close: %{y:.2f}<br>"
                "Open: %{customdata[0]:.2f}<br>High: %{customdata[1]:.2f}<br>"
                "Low: %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )

    for label, forecast, band_label in forecasts:
        fig.add_trace(
            go.Scatter(
                x=forecast["Date"],
                y=forecast["Forecast_Close"],
                mode="lines+markers",
                name=label,
                line={"color": PREDICTED_COLOR},
                marker={"size": 5},
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Predicted: %{y:.2f}<extra></extra>",
            )
        )
        if {"Lower_CI", "Upper_CI"}.issubset(forecast.columns):
            fig.add_trace(
                go.Scatter(
                    x=forecast["Date"],
                    y=forecast["Upper_CI"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast["Date"],
                    y=forecast["Lower_CI"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(245, 158, 11, 0.18)",
                    name=band_label,
                    hoverinfo="skip",
                )
            )

    return base_layout(fig, f"{horizon}-Day Forecast", "Closing Price")


def render_forecast_tab(
    data: pd.DataFrame,
    xgb_model: Any,
    sarima_model: Any,
    feature_cols: list[str],
    feature_df: pd.DataFrame,
    model_choice: str,
    horizon: int,
    run_forecast: bool,
) -> None:
    """Render forecasts using the existing XGBoost and SARIMA artifacts."""
    st.subheader("Forecast Future Prices")
    if not run_forecast:
        st.write("Choose a horizon and model in the sidebar, then click **Run Forecast**.")
        return

    forecasts: list[tuple[str, pd.DataFrame, str]] = []
    if model_choice in ("XGBoost", "Compare both"):
        xgb_fc = xgb_forecast_with_band(
            xgb_model, data, feature_cols, feature_df, horizon
        )
        forecasts.append(
            (
                "XGBoost Forecast",
                xgb_fc,
                "XGBoost ±1.96σ residual approximation (not a predictive interval)",
            )
        )
        st.caption(
            "XGBoost band: approximate ±1.96 historical one-step residual standard "
            "deviations. It is not a calibrated predictive interval."
        )
        st.dataframe(xgb_fc.set_index("Date").style.format("{:.2f}"), use_container_width=True)

    if model_choice in ("SARIMA", "Compare both"):
        sar_fc = sarima_forecast(sarima_model, data["Date"].iloc[-1], horizon)
        forecasts.append(("SARIMA Forecast", sar_fc, "SARIMA 95% CI"))
        st.dataframe(sar_fc.set_index("Date").style.format("{:.2f}"), use_container_width=True)

    st.plotly_chart(
        _forecast_figure(data, forecasts, horizon),
        use_container_width=True,
    )
    st.warning(
        "This is a statistical/ML forecast, not investment advice. SARIMA's "
        "selected order collapsed to a random walk in backtesting, while XGBoost's "
        "recursive forecast compounds its own prediction error as the horizon grows."
    )
