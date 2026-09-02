"""Historical analysis view."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .common import ACTUAL_COLOR, base_layout


def render_historical_tab(data: pd.DataFrame, date_range: tuple[object, object]) -> None:
    """Render the historical closing-price and volume charts."""
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    view = data.loc[data["Date"].between(start, end)].copy()

    st.subheader("Price History")
    fig = go.Figure()
    customdata = view[["Open", "High", "Low", "Close", "Volume"]].to_numpy()
    fig.add_trace(
        go.Scatter(
            x=view["Date"],
            y=view["Close"],
            mode="lines",
            name="Actual Close",
            line={"color": ACTUAL_COLOR},
            customdata=customdata,
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Open: %{customdata[0]:.2f}<br>"
                "High: %{customdata[1]:.2f}<br>"
                "Low: %{customdata[2]:.2f}<br>"
                "Close: %{customdata[3]:.2f}<br>"
                "Volume: %{customdata[4]:,.0f}<extra></extra>"
            ),
        )
    )
    st.plotly_chart(base_layout(fig, "Closing Price", "Price"), use_container_width=True)

    st.subheader("Trading Volume")
    volume_fig = go.Figure(
        go.Scatter(
            x=view["Date"],
            y=view["Volume"],
            mode="lines",
            name="Volume",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Volume: %{y:,.0f}<extra></extra>",
        )
    )
    st.plotly_chart(base_layout(volume_fig, "Trading Volume", "Volume"), use_container_width=True)
