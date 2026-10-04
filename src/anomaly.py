"""Unsupervised anomaly detection on hourly energy consumption.

Primary: Isolation Forest on behaviour features (level, volatility, shape of
recent usage, time context). Reference: rolling z-score detector.
Both are unsupervised — no labels needed, which matches reality: nobody hands
you a labelled 'weird consumption' column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

TARGET = "Global_active_power"

ANOMALY_FEATURES = [
    TARGET, "rolling_mean_6h", "rolling_std_6h", "delta_1h", "delta_24h",
    "hour_sin", "hour_cos", "is_weekend", "Voltage",
]


def add_anomaly_features(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    out = df.copy()
    shifted = out[target].shift(1)
    out["rolling_mean_6h"] = shifted.rolling(6).mean()
    out["rolling_std_6h"] = shifted.rolling(6).std()
    out["delta_1h"] = out[target].diff()
    out["delta_24h"] = out[target] - out[target].shift(24)
    return out.dropna()


def detect_isolation_forest(df: pd.DataFrame, contamination: float = 0.01,
                            n_estimators: int = 300) -> pd.DataFrame:
    """Flag anomalous hours. contamination = expected share of anomalies (~1%)."""
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    X = df[ANOMALY_FEATURES]
    labels = model.fit_predict(X)          # -1 = anomaly, 1 = normal
    scores = model.decision_function(X)    # lower = more anomalous
    out = df.copy()
    out["anomaly"] = (labels == -1).astype(int)
    out["anomaly_score"] = scores
    return out


def detect_zscore(df: pd.DataFrame, window: int = 168, threshold: float = 3.5,
                  target: str = TARGET) -> pd.Series:
    """Reference detector: |z| vs trailing 7-day (168h) rolling stats."""
    roll = df[target].rolling(window, min_periods=window // 2)
    z = (df[target] - roll.mean()) / roll.std().replace(0, np.nan)
    return (z.abs() > threshold).fillna(False).astype(int)


def summarize_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Top anomalies ranked by score, with excess vs recent typical usage."""
    anomalies = df[df["anomaly"] == 1].copy()
    anomalies["excess_kw_vs_6h_mean"] = (
        anomalies[TARGET] - anomalies["rolling_mean_6h"]
    ).round(3)
    cols = [TARGET, "rolling_mean_6h", "delta_1h", "delta_24h",
            "excess_kw_vs_6h_mean", "anomaly_score"]
    return anomalies.sort_values("anomaly_score")[cols].round(4)
