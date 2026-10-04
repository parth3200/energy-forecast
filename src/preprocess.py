"""Data loading and preprocessing for household energy consumption."""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/household_power_consumption.txt"
PROCESSED_HOURLY = "data/processed/hourly.parquet"

# Columns kept from the raw minute-level file.
NUMERIC_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw UCI minute-level file into a clean datetime-indexed DataFrame.

    The raw file is semicolon-separated, uses '?' for missing values (~1.25%
    of rows), and stores date/time as two DD/MM/YYYY + HH:MM:SS columns.
    """
    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?"],
        dtype={c: "float32" for c in NUMERIC_COLS},
        low_memory=False,
    )
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample minute-level power to hourly means.

    Minutes with missing measurements produce NaN hourly buckets; gaps are
    interpolated only when <= 6 consecutive hours (short outages). Longer
    gaps are left as NaN so downstream code can drop them explicitly rather
    than training on fabricated data.
    """
    hourly = df.resample("h").mean()
    hourly = hourly.interpolate(method="time", limit=6, limit_area="inside")
    hourly = hourly.dropna()
    return hourly


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar + cyclical encodings and the engineered meter gap feature."""
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    # Cyclical encoding so hour 23 and hour 0 are neighbours.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    # Un-metered load: total active power not captured by the three sub-meters.
    # Sub-metering is in watt-hours; Global_active_power is kW -> *1000/60 per
    # minute, but after hourly resampling 1 kW-h mean == 1 kWh, so *1000 applies.
    out["other_load_wh"] = (
        out["Global_active_power"] * 1000
        - out["Sub_metering_1"]
        - out["Sub_metering_2"]
        - out["Sub_metering_3"]
    )
    return out


def build_dataset(raw_path: str = RAW_PATH) -> pd.DataFrame:
    """Full cleaning pipeline: raw txt -> hourly, gap-handled, feature-enriched."""
    df = load_raw(raw_path)
    hourly = resample_hourly(df)
    hourly = add_time_features(hourly)
    hourly.to_parquet(PROCESSED_HOURLY, index=True)
    return hourly


if __name__ == "__main__":
    out = build_dataset()
    print(f"Processed hourly dataset: {out.shape[0]:,} rows x {out.shape[1]} cols")
    print(f"Range: {out.index.min()} -> {out.index.max()}")
    print(out.describe().round(3).to_string())
