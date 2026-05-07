"""
feature_engineering.py
Creates all required features for the forecasting models:
  - Lag features (t-1, t-7, t-30 in days → t-1, t-4 in weeks for weekly data)
  - Rolling mean / std
  - Calendar features: week-of-year, month, quarter, day-of-week
  - Indian public holiday flag
  - Train / validation split without leakage
"""
import pandas as pd
import numpy as np
import holidays
from loguru import logger
from typing import Tuple


# ── Indian national holidays ──────────────────────────────────────────────────
IN_HOLIDAYS = holidays.India(years=range(2019, 2027))


def flag_holiday(date: pd.Timestamp) -> int:
    """Return 1 if the week contains a major Indian public holiday."""
    for day_offset in range(7):
        if (date + pd.Timedelta(days=day_offset)) in IN_HOLIDAYS:
            return 1
    return 0


# ── Missing-value handler ─────────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing sales values per state using:
      1. Linear interpolation (preserves trend)
      2. Forward-fill for leading/trailing NaNs
    """
    df = df.copy()
    logger.info(f"Missing values before imputation: {df['sales'].isna().sum()}")

    df = df.sort_values(["state", "week_start_date"])
    df["sales"] = (
        df.groupby("state")["sales"]
        .transform(lambda s: s.interpolate(method="linear").ffill().bfill())
    )

    logger.info(f"Missing values after imputation: {df['sales'].isna().sum()}")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all time-series features.  Must be called AFTER handle_missing().
    """
    df = df.copy().sort_values(["state", "week_start_date"]).reset_index(drop=True)

    g = df.groupby("state")["sales"]

    # ── Lag features (weekly cadence) ─────────────────────────────────────────
    # t-1  ≈ previous week
    # t-4  ≈ ~1 month ago  (closest to t-30 days in weekly data)
    # t-7  ≈ ~7 weeks ago  (kept as-is per assignment – "t-7" means 7 periods)
    # t-52 ≈ same week last year
    df["lag_1"]  = g.shift(1)
    df["lag_4"]  = g.shift(4)
    df["lag_7"]  = g.shift(7)
    df["lag_52"] = g.shift(52)

    # ── Rolling statistics ────────────────────────────────────────────────────
    df["rolling_mean_4"]  = g.shift(1).transform(lambda s: s.rolling(4,  min_periods=1).mean())
    df["rolling_mean_12"] = g.shift(1).transform(lambda s: s.rolling(12, min_periods=1).mean())
    df["rolling_std_4"]   = g.shift(1).transform(lambda s: s.rolling(4,  min_periods=1).std().fillna(0))
    df["rolling_std_12"]  = g.shift(1).transform(lambda s: s.rolling(12, min_periods=1).std().fillna(0))

    # ── Calendar features ─────────────────────────────────────────────────────
    df["week_of_year"] = df["week_start_date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["week_start_date"].dt.month
    df["quarter"]      = df["week_start_date"].dt.quarter
    df["year"]         = df["week_start_date"].dt.year
    df["day_of_week"]  = df["week_start_date"].dt.dayofweek   # 0 = Monday

    # ── Holiday flag ──────────────────────────────────────────────────────────
    df["is_holiday"] = df["week_start_date"].apply(flag_holiday)

    # ── Cyclical encoding for week_of_year & month ────────────────────────────
    df["woy_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["woy_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["mon_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["mon_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    logger.info(f"Feature matrix shape: {df.shape}")
    return df


# ── Train / Validation split (no leakage) ────────────────────────────────────
def time_series_split(
    df: pd.DataFrame,
    val_weeks: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split per state – last `val_weeks` weeks = validation.
    No shuffling; no data from the future leaks into training.
    """
    train_parts, val_parts = [], []

    for state, grp in df.groupby("state"):
        grp = grp.sort_values("week_start_date")
        split_idx = len(grp) - val_weeks
        train_parts.append(grp.iloc[:split_idx])
        val_parts.append(grp.iloc[split_idx:])

    train = pd.concat(train_parts).reset_index(drop=True)
    val   = pd.concat(val_parts).reset_index(drop=True)

    logger.info(
        f"Train: {train['week_start_date'].min()} → {train['week_start_date'].max()} "
        f"({len(train)} rows)"
    )
    logger.info(
        f"Val  : {val['week_start_date'].min()} → {val['week_start_date'].max()} "
        f"({len(val)} rows)"
    )
    return train, val


# ── Feature columns for ML models ────────────────────────────────────────────
ML_FEATURES = [
    "lag_1", "lag_4", "lag_7", "lag_52",
    "rolling_mean_4", "rolling_mean_12",
    "rolling_std_4",  "rolling_std_12",
    "week_of_year", "month", "quarter", "year",
    "day_of_week", "is_holiday",
    "woy_sin", "woy_cos", "mon_sin", "mon_cos",
]
TARGET = "sales"
