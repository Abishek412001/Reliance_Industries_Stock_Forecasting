"""Model evaluation view."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .common import PREDICTED_COLOR, base_layout


def render_evaluation_tab(metrics: pd.DataFrame) -> None:
    """Render the existing model comparison table and error chart."""
    st.subheader("Model Comparison (held-out 2023 test set)")
    st.dataframe(metrics.style.format("{:.2f}"), use_container_width=True)

    fig = go.Figure()
    for metric in ["RMSE", "MAE"]:
        fig.add_trace(
            go.Bar(x=metrics.index, y=metrics[metric], name=metric)
        )
    fig.update_layout(barmode="group", yaxis_title="Error", xaxis_title="Model")
    st.plotly_chart(base_layout(fig, "Model Error Comparison", "Error"), use_container_width=True)

    st.info(
        "These numbers come from one-step-ahead evaluation, where each "
        "prediction used the true prior day's closing price as input. The "
        "30-Day Forecast view instead forecasts recursively — real accuracy "
        "there will typically be lower than this table suggests, especially "
        "further into the horizon."
    )
