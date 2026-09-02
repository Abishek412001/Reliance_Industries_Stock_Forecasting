"""Tests for dashboard feature engineering."""

import numpy as np
import pandas as pd

from data import engineer_xgb_features


EXPECTED_COLUMNS = [
    "Date",
    "Close",
    "Volume",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_5",
    "Lag_10",
    "Rolling_Mean_5",
    "Rolling_Std_5",
    "Rolling_Mean_20",
    "DayOfWeek",
    "Month",
]


def make_sample_data(rows: int = 30) -> pd.DataFrame:
    """Create deterministic OHLCV-shaped data for feature tests."""
    return pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-02", periods=rows, freq="D"),
            "Close": np.arange(100.0, 100.0 + rows),
            "Volume": np.full(rows, 1000),
        }
    )


def test_feature_engineering_output_shape_and_columns() -> None:
    """Feature engineering should match the notebook's 13-column output."""
    features = engineer_xgb_features(make_sample_data())
    assert features.shape == (20, 13)
    assert features.columns.tolist() == EXPECTED_COLUMNS
    assert features.isna().sum().sum() == 0


def test_rolling_features_do_not_use_current_row_close() -> None:
    """Changing today's close must not change today's rolling features."""
    original = make_sample_data()
    changed = original.copy()
    changed.loc[19, "Close"] = 999999.0

    original_features = engineer_xgb_features(original)
    changed_features = engineer_xgb_features(changed)

    row_original = original_features.loc[9]
    row_changed = changed_features.loc[9]
    assert row_original["Date"] == original.loc[19, "Date"]
    assert row_original["Rolling_Mean_5"] == row_changed["Rolling_Mean_5"]
    assert row_original["Rolling_Std_5"] == row_changed["Rolling_Std_5"]
    assert row_original["Rolling_Mean_20"] == row_changed["Rolling_Mean_20"]

    # Lag_1 is also intentionally based on the prior day's close.
    assert row_original["Lag_1"] == row_changed["Lag_1"]
