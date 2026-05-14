"""
xgboost_model.py
XGBoost forecasting model.
"""

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from base_model import BaseForecaster

from feature_engineering import (
    ML_FEATURES,
    build_features,
)

warnings.filterwarnings("ignore")


class XGBoostForecaster(BaseForecaster):

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

        state_df = build_features(state_df)

        state_df = state_df.dropna()

        X = state_df[ML_FEATURES]

        y = state_df["sales"]

        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        )

        model.fit(X, y)

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
                f"No XGBoost model found for {state}"
            )

        state_df = build_features(last_known.copy())

        state_df = state_df.dropna()

        X_latest = state_df[ML_FEATURES].tail(1)

        preds = []

        for _ in range(horizon):

            pred = model.predict(X_latest)[0]

            preds.append(pred)

        return np.array(preds)