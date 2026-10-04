"""End-to-end training & evaluation pipeline.

Run:  python src/train.py
Produces: models/*.joblib, outputs/figures/*.png, outputs/*.csv, outputs/report.md
"""

from __future__ import annotations

import json
import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import eda
from anomaly import add_anomaly_features, detect_isolation_forest, detect_zscore, summarize_anomalies
from forecast import (
    TARGET,
    add_lag_features,
    chronological_split,
    evaluate_rolling_origin,
    overall_metrics,
    train_forecaster,
)
from preprocess import PROCESSED_HOURLY, build_dataset

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)


def main() -> None:
    t0 = time.time()

    # ---- 1. Data ----
    if os.path.exists(PROCESSED_HOURLY):
        hourly = pd.read_parquet(PROCESSED_HOURLY)
        print(f"[data] loaded cached hourly data: {hourly.shape}")
    else:
        hourly = build_dataset()
        print(f"[data] built hourly dataset: {hourly.shape}")

    # ---- 2. EDA figures ----
    for p in eda.run_eda(hourly):
        print(f"[eda] {p}")

    # ---- 3. Forecasting: train on ~3.5y, test on last 90 days ----
    feat = add_lag_features(hourly).dropna()
    train, test = chronological_split(feat, test_days=90)
    print(f"[split] train={len(train):,}h ({train.index.min().date()}..{train.index.max().date()}) "
          f"test={len(test):,}h ({test.index.min().date()}..{test.index.max().date()})")

    t = time.time()
    model = train_forecaster(train)
    print(f"[forecast] trained XGBoost in {time.time()-t:.1f}s")

    bt, pred_df = evaluate_rolling_origin(model, test, train[TARGET], horizon=24, n_origins=30)
    metrics = overall_metrics(pred_df)
    metrics["train_rows"] = int(len(train))
    metrics["test_rows"] = int(len(test))
    metrics["backtest_origins"] = int(len(bt))
    metrics["mae_improvement_vs_naive_%"] = round(
        100 * (1 - pred_df.groupby_origin_mae[0] if False else
               (metrics["forecast"]["MAE_kW"] / metrics["naive"]["MAE_kW"])), 2)
    print(f"[forecast] metrics: {json.dumps(metrics, indent=2)}")

    with open("outputs/metrics_forecast.json", "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(model, "models/xgb_forecaster.joblib")

    # Representative forecast plot: final 24h of the backtest
    last = pred_df["origin"].max()
    pdf = pred_df[pred_df["origin"] == last]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(pdf.index, pdf["actual"], label="Actual", lw=2)
    ax.plot(pdf.index, pdf["forecast"], label="XGBoost forecast", lw=2, ls="--")
    ax.plot(pdf.index, pdf["naive"], label="Seasonal-naive (t-24h)", alpha=0.6)
    ax.set(ylabel="Active power (kW)",
           title=f"24-hour-ahead forecast (origin {last:%Y-%m-%d %H:%M})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig("outputs/figures/forecast_vs_actual.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Feature importance
    imp = pd.Series(model.feature_importances_,
                    index=model.feature_names_in_).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot.barh(ax=ax, color="#4C78A8")
    ax.set_title("XGBoost feature importance")
    fig.savefig("outputs/figures/feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    imp.to_csv("outputs/feature_importance.csv")

    # ---- 4. Anomaly detection on the full series ----
    t = time.time()
    ano = detect_isolation_forest(add_anomaly_features(hourly))
    ano["anomaly_zscore"] = detect_zscore(ano)
    agree = int(((ano["anomaly"] == 1) & (ano["anomaly_zscore"] == 1)).sum())
    top = summarize_anomalies(ano)
    top.to_csv("outputs/top_anomalies.csv")
    ano[["anomaly", "anomaly_score", "anomaly_zscore"]].to_csv("outputs/anomaly_flags.csv")
    joblib.dump({"features": " — fitted full-series; see src/anomaly.py"}, "models/_note.joblib")

    n_anom = int(ano["anomaly"].sum())
    print(f"[anomaly] IsolationForest flagged {n_anom:,} hours "
          f"({100*n_anom/len(ano):.2f}%) in {time.time()-t:.1f}s; "
          f"{agree} also flagged by z-score")

    # Anomaly plot: the densest anomaly week
    counts = ano["anomaly"].resample("W").sum()
    hot_week_start = counts.idxmax() - pd.Timedelta(days=3)
    wk = ano.loc[hot_week_start: hot_week_start + pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(wk.index, wk[TARGET], lw=1, label="Consumption")
    hits = wk[wk["anomaly"] == 1]
    ax.scatter(hits.index, hits[TARGET], color="red", zorder=5, label="Flagged anomaly")
    ax.set(ylabel="Active power (kW)", title=f"Detected anomalies — week of {hot_week_start:%Y-%m-%d}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig("outputs/figures/anomalies.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- 5. Report ----
    with open("outputs/run_summary.json", "w") as f:
        json.dump({"rows_hourly": int(len(hourly)), "anomalies": n_anom,
                   "both_detectors": agree, "runtime_s": round(time.time() - t0, 1),
                   "forecast": metrics}, f, indent=2)
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
