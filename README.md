# Smart Energy Forecasting & Anomaly Detection ⚡

A comprehensive, lightweight data science project demonstrating end-to-end machine learning on time-series data. Designed to be highly efficient, running completely on CPU without heavy GPU requirements (perfect for Apple Silicon or standard cloud VMs).

## Overview
This project ingests a 2M+ row raw dataset (UCI Household Electric Power Consumption), processes it down to hourly windows, and trains two models:
1. **XGBoost Regressor:** Predicts energy consumption 24 hours into the future using lag features, rolling statistics, and cyclical time encodings. Validated using a rolling-origin time-series backtest.
2. **Isolation Forest:** Unsupervised anomaly detection that identifies unusual energy consumption patterns based on volatility, time context, and recent trends (cross-verified with rolling Z-scores).

## Project Structure
```text
energy-forecast/
├── data/
│   ├── raw/             # Raw UCI datasets 
│   └── processed/       # Cleaned, resampled parquet files
├── notebooks/           # Jupyter notebooks for sandbox exploration
├── src/                 # Modularized pipeline scripts
│   ├── preprocess.py    # Cleaning, resampling, and feature engineering
│   ├── eda.py           # Exploratory data visualization
│   ├── forecast.py      # XGBoost forecasting & evaluation
│   └── anomaly.py       # Unsupervised anomaly detection
├── outputs/             # Generated figures and metrics
├── models/              # Serialized joblib models
├── app.py               # Streamlit interactive dashboard
└── requirements.txt
```

## Quick Start

1. **Install Dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   *(Note: macOS users might need `brew install libomp` for XGBoost).*

2. **Run the Full Pipeline (Data -> EDA -> Train -> Evaluate):**
   ```bash
   python src/train.py
   ```
   This generates all figures in `outputs/figures/`, serialized models in `models/`, and CSV/JSON metrics in `outputs/`.

3. **Launch the Web Dashboard:**
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` to view your data, detected anomalies, and model evaluations interactively.

## Technical Highlights
* **Time-Series Integrity**: Avoids data leakage by using strict chronological splits and recursive multi-horizon forecasting (predicting $t+1$, feeding it back to predict $t+2$).
* **Baseline Benchmarking**: Defends model performance by benchmarking against a "Seasonal Naive" model (predicting the same usage as 24 hours ago).
* **Efficiency**: No deep learning overhead. Uses XGBoost's Fast Histogram method (`tree_method="hist"`) which is incredibly fast and memory-efficient on CPU arrays.
