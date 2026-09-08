import numpy as np
import pandas as pd
from scipy import stats


def detect_concept_drift(train_data, new_data, column='Close', window=30):
    """
    Detect concept drift using the Kolmogorov-Smirnov 2-sample statistical test.
    
    Returns:
    - drift_detected (bool): True if p-value < 0.05
    - p_value (float): Kolmogorov-Smirnov test p-value
    - ks_stat (float): Test statistic
    """
    train_dist = train_data[column].iloc[-window:] if isinstance(train_data, pd.DataFrame) else train_data[-window:]
    new_dist = new_data[column].iloc[-window:] if isinstance(new_data, pd.DataFrame) else new_data[-window:]
    
    ks_stat, p_value = stats.ks_2samp(train_dist, new_dist)
    drift_detected = bool(p_value < 0.05)
    
    return {
        "drift_detected": drift_detected,
        "p_value": float(p_value),
        "ks_stat": float(ks_stat),
        "alert": "⚠️ ALERT: Concept drift detected! Retrain model." if drift_detected else "✅ No significant drift detected."
    }
