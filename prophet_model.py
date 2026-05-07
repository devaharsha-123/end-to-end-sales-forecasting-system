"""
prophet_model.py
Facebook Prophet forecaster with Indian holiday effects.
"""
import numpy as np
import pandas as pd
import holidays
from prophet import Prophet
from loguru import logger

from models.base_model import BaseForecaster


# Build an Indian holidays DataFrame for Prophet
def _get_india_holidays() -> pd.DataFrame:
    rows = []
    for year in range(2019, 2028):
        for date, name in holidays.India(years=year).items():
            rows.append({"ds": pd.Timestamp(date), "holiday": name})
    return pd.DataFrame(rows)


INDIA_HOLIDAYS = _get_india_holidays()


class ProphetForecaster(BaseForecaster):
    """
    Facebook Prophet with:
      - Weekly & yearly seasonality enabled
      - Indian public holiday regressors
      - Multiplicative seasonality (better for growing sales)
    """

    name = "Prophet"

    def __init__(self):
        self._models: dict = {}     # state → fitted Prophet
        self._last_dates: dict = {} # state → last training date

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, train: pd.DataFrame, state: str) -> None:
        df = (
            train[train["state"] == state][["week_start_date", "sales"]]
            .rename(columns={"week_start_date": "ds", "sales": "y"})
            .dropna()
        )

        mdl = Prophet(
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=INDIA_HOLIDAYS,
            interval_width=0.80,
        )

        mdl.fit(df, iter=300)
        self._models[state]      = mdl
        self._last_dates[state]  = df["ds"].max()
        logger.info(f"[Prophet] {state}: trained on {len(df)} observations.")

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, horizon: int, last_known: pd.DataFrame, state: str = None) -> np.ndarray:
        if state is None:
            raise ValueError("Prophet.predict() requires `state` keyword argument.")

        mdl = self._models[state]
        future = mdl.make_future_dataframe(periods=horizon, freq="W")
        forecast = mdl.predict(future)

        # Return only the future `horizon` rows
        fc_values = forecast["yhat"].tail(horizon).values
        return np.maximum(fc_values, 0)

    def get_components(self, state: str) -> pd.DataFrame:
        """Return the full Prophet forecast DataFrame including components."""
        mdl = self._models[state]
        future = mdl.make_future_dataframe(periods=8, freq="W")
        return mdl.predict(future)
