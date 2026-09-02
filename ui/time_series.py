"""Time-series analysis view."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .common import ACTUAL_COLOR, base_layout


def render_time_series_tab(data: pd.DataFrame) -> None:
    """Render moving averages, daily returns, and rolling volatility."""
    st.subheader("Trend — 20 & 200 Day Moving Averages")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Close", line={"color": ACTUAL_COLOR}))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["MA_20"], mode="lines", name="20-Day MA"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["MA_200"], mode="lines", name="200-Day MA"))
    st.plotly_chart(base_layout(fig, "Trend — 20 & 200 Day Moving Averages", "Price"), use_container_width=True)

    st.subheader("Daily Returns")
    returns_fig = go.Figure(
        go.Scatter(
            x=data["Date"],
            y=data["Daily_Return"],
            mode="lines",
            name="Daily Return",
            customdata=data[["Open", "High", "Low", "Close"]].to_numpy(),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>Return: %{y:.2f}%<br>"
                "Open: %{customdata[0]:.2f}<br>High: %{customdata[1]:.2f}<br>"
                "Low: %{customdata[2]:.2f}<br>Close: %{customdata[3]:.2f}<extra></extra>"
            ),
        )
    )
    st.plotly_chart(base_layout(returns_fig, "Daily Returns", "Return (%)"), use_container_width=True)

    st.subheader("20-Day Rolling Volatility")
    volatility_fig = go.Figure(
        go.Scatter(
            x=data["Date"],
            y=data["Volatility_20"],
            mode="lines",
            name="20-Day Volatility",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Volatility: %{y:.4f}<extra></extra>",
        )
    )
    st.plotly_chart(base_layout(volatility_fig, "20-Day Rolling Volatility", "Volatility"), use_container_width=True)
