import ast
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Energy Forecast & Anomaly Detection", layout="wide")

st.title("⚡ Smart Energy Forecasting & Anomaly Detection")
st.markdown(
    "This dashboard demonstrates a lightweight, full-stack data science pipeline that forecasts "
    "household power consumption and flags unusual usage, running entirely on CPU."
)

@st.cache_data
def load_data():
    # Load processed data and metrics if available
    hourly = pd.DataFrame()
    anomalies = pd.DataFrame()
    metrics = {}
    
    if os.path.exists("data/processed/hourly.parquet"):
        hourly = pd.read_parquet("data/processed/hourly.parquet")
    
    if os.path.exists("outputs/anomaly_flags.csv"):
        anomalies = pd.read_csv("outputs/anomaly_flags.csv", index_col=0, parse_dates=True)
        if not hourly.empty:
            hourly = hourly.join(anomalies)

    if os.path.exists("outputs/run_summary.json"):
        with open("outputs/run_summary.json", "r") as f:
            metrics = json.load(f)
            
    return hourly, metrics

hourly, metrics = load_data()

if hourly.empty:
    st.warning("Processed data not found. Please run `python src/train.py` first.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Dataset Rows", f"{len(hourly):,}")
if metrics:
    col2.metric("XGBoost MAE (kW)", metrics["forecast"]["forecast"]["MAE_kW"])
    col3.metric("Anomalies Flagged", f"{metrics['anomalies']:,}")

st.divider()

st.subheader("Last 30 Days: Consumption & Anomalies")
mask = hourly.index >= (hourly.index.max() - pd.Timedelta(days=30))
recent = hourly[mask]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(recent.index, recent["Global_active_power"], label="Active Power (kW)", lw=1, alpha=0.8)

if "anomaly" in recent.columns:
    anoms = recent[recent["anomaly"] == 1]
    ax.scatter(anoms.index, anoms["Global_active_power"], color="red", label="Anomaly", zorder=5)

ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

st.divider()

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Model Pipeline Figures")
    if os.path.exists("outputs/figures/forecast_vs_actual.png"):
        st.image("outputs/figures/forecast_vs_actual.png", caption="24-Hour XGBoost Forecast Backtest")
    else:
        st.info("Run `python src/train.py` to generate forecast plots.")

with col_b:
    st.subheader("Feature Importance")
    if os.path.exists("outputs/figures/feature_importance.png"):
        st.image("outputs/figures/feature_importance.png", caption="Top Predictors (Lags & Rolling Means)")
