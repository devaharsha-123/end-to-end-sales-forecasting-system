"""
prophet_model.py
Facebook Prophet forecasting model.
"""

import warnings
import numpy as np
import pandas as pd

from prophet import Prophet

from base_model import BaseForecaster

warnings.filterwarnings("ignore")


class ProphetForecaster(BaseForecaster):

    def __init__(self):

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

        prophet_df = pd.DataFrame({
            "ds": state_df["week_start_date"],
            "y": state_df["sales"],
        })

        model = Prophet()

        model.fit(prophet_df)

        self.models[state] = model

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
                f"No Prophet model found for {state}"
            )

        future = model.make_future_dataframe(
            periods=horizon,
            freq="W-MON",
        )

        forecast = model.predict(future)

        preds = forecast["yhat"].tail(horizon).values

        return np.array(preds)