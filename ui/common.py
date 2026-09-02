"""Shared Plotly styling helpers for dashboard sections."""

from __future__ import annotations

import plotly.graph_objects as go

ACTUAL_COLOR = "#2563EB"
PREDICTED_COLOR = "#F59E0B"
BAND_COLOR = "#F59E0B"


def apply_time_axis(fig: go.Figure) -> go.Figure:
    """Enable Plotly's interactive date range slider and zoom controls."""
    fig.update_xaxes(
        type="date",
        rangeslider_visible=True,
        rangeselector={
            "buttons": [
                {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ]
        },
    )
    fig.update_layout(hovermode="x unified", margin={"l": 20, "r": 20, "t": 50, "b": 20})
    return fig


def base_layout(fig: go.Figure, title: str, y_title: str) -> go.Figure:
    """Apply consistent titles, axes, legend, and interactive controls."""
    fig.update_layout(title=title, yaxis_title=y_title, xaxis_title="Date")
    return apply_time_axis(fig)
