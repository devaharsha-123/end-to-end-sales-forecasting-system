"""
data_generator.py
Generates synthetic sales data matching the assignment dataset structure.
In production, replace with actual Excel loading logic.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sales_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic weekly sales data per state.
    Mimics the structure of the provided Excel dataset.
    """
    np.random.seed(seed)

    states = [
        "Andhra Pradesh", "Telangana", "Maharashtra", "Karnataka",
        "Tamil Nadu", "Gujarat", "Rajasthan", "Uttar Pradesh",
        "West Bengal", "Kerala"
    ]

    start_date = datetime(2021, 1, 4)   # First Monday of 2021
    end_date   = datetime(2024, 12, 30) # Last week of 2024

    weeks = pd.date_range(start=start_date, end=end_date, freq="W-MON")

    records = []
    for state in states:
        # State-level base sales (heterogeneous across states)
        base     = np.random.uniform(50_000, 300_000)
        trend    = np.random.uniform(100, 800)          # weekly upward trend
        seasonal_amp = np.random.uniform(0.05, 0.20)   # 5-20% seasonality

        for i, week in enumerate(weeks):
            # Trend component
            trend_val = base + trend * i

            # Yearly seasonality (peaks around Diwali / year-end)
            week_of_year = week.isocalendar()[1]
            seasonal_val = trend_val * seasonal_amp * np.sin(
                2 * np.pi * week_of_year / 52 - np.pi / 3
            )

            # Diwali bump (week ~44-46)
            diwali_bump = 0
            if 42 <= week_of_year <= 47:
                diwali_bump = trend_val * 0.25

            # Random noise
            noise = np.random.normal(0, trend_val * 0.04)

            # Occasional missing values (2% chance)
            sales = trend_val + seasonal_val + diwali_bump + noise
            if np.random.rand() < 0.02:
                sales = np.nan

            records.append({
                "week_start_date": week,
                "state":           state,
                "sales":           max(sales, 0) if not np.isnan(sales) else np.nan
            })

    df = pd.DataFrame(records)
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    return df


if __name__ == "__main__":
    df = generate_sales_data()
    print(df.head(20))
    print(f"\nShape: {df.shape}")
    print(f"States: {df['state'].nunique()}")
    print(f"Date range: {df['week_start_date'].min()} → {df['week_start_date'].max()}")
    print(f"Missing values: {df['sales'].isna().sum()}")
