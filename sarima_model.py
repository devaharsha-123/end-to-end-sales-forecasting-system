"""
sarima_model.py
SARIMA forecaster with automatic order selection via AIC grid search.
"""
import warnings
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from loguru import logger

from models.base_model import BaseForecaster

warnings.filterwarnings("ignore")


class SARIMAForecaster(BaseForecaster):
    """
    Seasonal ARIMA model.
    - Auto-selects (p,d,q)(P,D,Q,s=52) via AIC on the training set.
    - s=52 because data is weekly (52 weeks per year).
    """

    name = "SARIMA"

    # Reduced grid for speed; expand for production hyperparameter search
    P_RANGE = range(0, 2)
    Q_RANGE = range(0, 2)
    D       = 1
    S       = 52       # weekly seasonality

    def __init__(self):
        self._models: dict = {}        # state → fitted SARIMAXResults
        self._orders: dict = {}        # state → (p,d,q,P,D,Q)

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, train: pd.DataFrame, state: str) -> None:
        series = (
            train[train["state"] == state]
            .set_index("week_start_date")["sales"]
            .asfreq("W-MON")
        )

        best_aic, best_order, best_model = np.inf, None, None

        for p, q in itertools.product(self.P_RANGE, self.Q_RANGE):
            for P, Q in itertools.product(self.P_RANGE, self.Q_RANGE):
                try:
                    mdl = SARIMAX(
                        series,
                        order=(p, self.D, q),
                        seasonal_order=(P, 1, Q, self.S),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)

                    if mdl.aic < best_aic:
                        best_aic   = mdl.aic
                        best_order = (p, self.D, q, P, 1, Q)
                        best_model = mdl
                except Exception:
                    continue

        if best_model is None:
            # Fallback: simple ARIMA(1,1,1)
            best_model = SARIMAX(series, order=(1, 1, 1)).fit(disp=False)
            best_order = (1, 1, 1, 0, 0, 0)

        self._models[state] = best_model
        self._orders[state] = best_order
        logger.info(f"[SARIMA] {state}: best order {best_order}, AIC={best_aic:.1f}")

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, horizon: int, last_known: pd.DataFrame, state: str = None) -> np.ndarray:
        if state is None:
            raise ValueError("SARIMA.predict() requires `state` keyword argument.")
        mdl = self._models[state]
        fc  = mdl.get_forecast(steps=horizon)
        return np.maximum(fc.predicted_mean.values, 0)

    def get_order(self, state: str) -> tuple:
        return self._orders.get(state, None)
