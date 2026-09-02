"""Tests for the dashboard's forecast metrics."""

import numpy as np
import pytest

from models import calculate_mae, calculate_mape, calculate_rmse, evaluate_forecast


def test_rmse_known_example() -> None:
    """RMSE should match a hand-calculated two-observation example."""
    assert calculate_rmse([10, 20], [12, 16]) == pytest.approx(np.sqrt(10))


def test_mae_known_example() -> None:
    """MAE should match the mean absolute error."""
    assert calculate_mae([10, 20], [12, 16]) == pytest.approx(3.0)


def test_mape_known_example() -> None:
    """MAPE should use the project's percentage definition."""
    assert calculate_mape([100, 200], [110, 180]) == pytest.approx(10.0)


def test_evaluate_forecast_returns_expected_schema() -> None:
    """The combined helper should expose the same metric fields as the notebook."""
    result = evaluate_forecast([100, 200], [110, 180], "Example")
    assert result["Model"] == "Example"
    assert result["RMSE"] == pytest.approx(np.sqrt(250))
    assert result["MAE"] == pytest.approx(15.0)
    assert result["MAPE"] == pytest.approx(10.0)
