"""Exploratory data analysis: generates all figures into outputs/figures/."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = "outputs/figures"
TARGET = "Global_active_power"


def _save(fig, name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_daily_profile(h: pd.DataFrame) -> str:
    """Average consumption by hour of day, weekday vs weekend."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, mask in [("Weekday", h["is_weekend"] == 0), ("Weekend", h["is_weekend"] == 1)]:
        prof = h[mask].groupby("hour")[TARGET].mean()
        ax.plot(prof.index, prof.values, marker="o", ms=3, label=label)
    ax.set(xlabel="Hour of day", ylabel="Mean active power (kW)",
           title="Daily load profile — weekday vs weekend")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "daily_profile.png")


def plot_monthly_seasonality(h: pd.DataFrame) -> str:
    """Monthly mean consumption showing winter heating peak."""
    monthly = h.groupby("month")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(monthly.index, monthly.values, color="#4C78A8")
    ax.set(xlabel="Month", ylabel="Mean active power (kW)",
           title="Monthly seasonality (2006-12 to 2010-11)", xticks=range(1, 13))
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "monthly_seasonality.png")


def plot_rolling(h: pd.DataFrame) -> str:
    """Daily consumption with 7-day rolling mean for one representative year."""
    daily = h[TARGET].resample("D").mean()
    year = daily.loc["2008"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(year.index, year.values, alpha=0.35, label="Daily mean")
    ax.plot(year.index, year.rolling(7).mean(), color="#E45756",
            lw=2, label="7-day rolling mean")
    ax.set(ylabel="Active power (kW)", title="Daily consumption and rolling trend — 2008")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "rolling_trend.png")


def plot_correlation(h: pd.DataFrame) -> str:
    """Correlation heatmap of measurements and engineered load."""
    cols = [TARGET, "Global_reactive_power", "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "other_load_wh"]
    corr = h[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, shrink=0.8)
    ax.set_title("Feature correlation heatmap")
    return _save(fig, "correlation_heatmap.png")


def run_eda(h: pd.DataFrame) -> list[str]:
    return [
        plot_daily_profile(h),
        plot_monthly_seasonality(h),
        plot_rolling(h),
        plot_correlation(h),
    ]


if __name__ == "__main__":
    hourly = pd.read_parquet("data/processed/hourly.parquet")
    for p in run_eda(hourly):
        print("wrote", p)
