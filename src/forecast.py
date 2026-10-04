"""24-hour-ahead energy forecasting with a lag-featured XGBoost regressor.

Approach: direct multi-horizon forecasting via recursive 1-step prediction.
Features are lag/rolling statistics of the target plus calendar encodings,
so the same feature builder works at train time and during recursive rollout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

TARGET = "Global_active_power"
LAGS = [1, 2, 3, 24, 48, 168]          # 1h..3h, 1d, 2d, 1w
ROLL_WINDOWS = [6, 24, 168]

FEATURE_COLS = (
    [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
    + ["hour", "dayofweek", "month", "is_weekend",
       "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
)


def add_lag_features(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Add lag + rolling features computed ONLY from past values (no leakage)."""
    out = df.copy()
    for l in LAGS:
        out[f"lag_{l}"] = out[target].shift(l)
    # shift(1) so the current row never sees itself in rolling stats
    for w in ROLL_WINDOWS:
        out[f"roll_mean_{w}"] = out[target].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"] = out[target].shift(1).rolling(w).std()
    return out


def chronological_split(df: pd.DataFrame, test_days: int = 90):
    """Split by time — never shuffle time series."""
    cutoff = df.index.max() - pd.Timedelta(days=test_days)
    return df[df.index <= cutoff].copy(), df[df.index > cutoff].copy()


def train_forecaster(train: pd.DataFrame) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",      # fast histogram method — CPU friendly
        n_jobs=-1,
        random_state=42,
    )
    model.fit(train[FEATURE_COLS], train[TARGET])
    return model


def recursive_forecast(model: XGBRegressor, history: pd.Series,
                       horizon: int = 24) -> pd.Series:
    """Predict `horizon` hours ahead, feeding each prediction back as a lag."""
    hist = list(history.values)
    start = history.index[-1] + pd.Timedelta(hours=1)
    future_idx = pd.date_range(start, periods=horizon, freq="h")
    preds = []
    for i, ts in enumerate(future_idx):
        row = {}
        for l in LAGS:
            row[f"lag_{l}"] = hist[-l] if len(hist) >= l else hist[0]
        for w in ROLL_WINDOWS:
            window = hist[-w:] if len(hist) >= w else hist
            row[f"roll_mean_{w}"] = float(np.mean(window))
            row[f"roll_std_{w}"] = float(np.std(window))
        hour, dow, month = ts.hour, ts.dayofweek, ts.month
        row.update({
            "hour": hour, "dayofweek": dow, "month": month,
            "is_weekend": int(dow >= 5),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
        })
        X = pd.DataFrame([row])[FEATURE_COLS]
        pred = float(model.predict(X)[0])
        preds.append(pred)
        hist.append(pred)
    return pd.Series(preds, index=future_idx, name="forecast")


def evaluate_rolling_origin(model: XGBRegressor, test: pd.DataFrame,
                            train_tail: pd.Series, horizon: int = 24,
                            n_origins: int = 30) -> tuple[pd.DataFrame, pd.Series]:
    """Honest evaluation: rolling-origin backtest over the last 30 days of test.

    At each origin, the model only sees data up to that point and forecasts
    the next 24h recursively. Compares against a seasonal-naive baseline
    (prediction = value 24h ago) — the bar any real forecaster must beat.
    """
    origins = pd.date_range(
        test.index.max() - pd.Timedelta(days=30),
        test.index.max() - pd.Timedelta(hours=horizon),
        periods=n_origins,
    )
    records, all_preds = [], []
    history_full = pd.concat([train_tail, test[TARGET]])
    for origin in origins:
        history = history_full[history_full.index <= origin]
        actual = test.loc[origin + pd.Timedelta(hours=1):
                          origin + pd.Timedelta(hours=horizon), TARGET]
        if len(actual) < horizon:
            continue
        fc = recursive_forecast(model, history, horizon)
        naive = history_full.shift(24).loc[actual.index]
        records.append({
            "origin": origin,
            "mae_model": mean_absolute_error(actual, fc),
            "mae_naive": mean_absolute_error(actual, naive),
        })
        all_preds.append(pd.DataFrame({"actual": actual, "forecast": fc.values,
                                       "naive": naive.values, "origin": origin}))
    bt = pd.DataFrame(records)
    pred_df = pd.concat(all_preds)
    return bt, pred_df


def overall_metrics(pred_df: pd.DataFrame) -> dict:
    """Aggregate backtest metrics for model and seasonal-naive baseline."""
    y = pred_df["actual"]
    out = {}
    for col in ["forecast", "naive"]:
        p = pred_df[col]
        out[col] = {
            "MAE_kW": round(mean_absolute_error(y, p), 4),
            "RMSE_kW": round(mean_squared_error(y, p) ** 0.5, 4),
            "R2": round(r2_score(y, p), 4),
            "MAPE_%": round(float(np.mean(np.abs((y - p) / np.clip(y, 0.1, None))) * 100), 2),
        }
    return out
