"""
sarima_model.py
SARIMA forecasting model with automatic seasonal configuration.
"""

import warnings
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ✅ FIXED IMPORT
from base_model import BaseForecaster

warnings.filterwarnings("ignore")


class SARIMAForecaster(BaseForecaster):

    def __init__(
        self,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
    ):

        self.order = order

        self.seasonal_order = seasonal_order

        self.models = {}

    # ── Train ───────────────────────────────────────────────────────────────
    def fit(
        self,
        df: pd.DataFrame,
        state: str,
    ):

        state_df = (
            df[df["state"] == state]
            .sort_values("week_start_date")
        )

        series = state_df["sales"].values

        model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted_model = model.fit(disp=False)

        self.models[state] = fitted_model

    # ── Predict ────────────────────────────────────────────────────────────
    def predict(
        self,
        horizon: int,
        last_known: pd.DataFrame,
        state: str,
    ):

        model = self.models.get(state)

        if model is None:
            raise ValueError(
                f"No trained SARIMA model found for {state}"
            )

        forecast = model.forecast(steps=horizon)

        return np.array(forecast)