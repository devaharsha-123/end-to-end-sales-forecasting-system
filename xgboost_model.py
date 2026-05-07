"""
xgboost_model.py
XGBoost forecaster with lag / rolling / calendar features.
Uses recursive multi-step forecasting to generate `horizon` predictions.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from loguru import logger

from models.base_model import BaseForecaster
from data.feature_engineering import ML_FEATURES, build_features, flag_holiday


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost with:
      - Lag features (1, 4, 7, 52 weeks)
      - Rolling mean / std (4 and 12 weeks)
      - Calendar & holiday features
      - Recursive multi-step prediction (auto-regressive rollout)
    """

    name = "XGBoost"

    PARAMS = {
        "n_estimators":     500,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "random_state":     42,
        "n_jobs":           -1,
    }

    def __init__(self):
        self._models:    dict = {}   # state → XGBRegressor
        self._scalers:   dict = {}   # state → StandardScaler (for target)
        self._histories: dict = {}   # state → last N rows for recursive pred

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, train: pd.DataFrame, state: str) -> None:
        df = train[train["state"] == state].copy()

        # Drop rows with NaN features (warm-up period for lags)
        feature_df = df[ML_FEATURES + ["sales"]].dropna()

        X = feature_df[ML_FEATURES].values
        y = feature_df["sales"].values

        # Scale target → helps XGBoost convergence
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

        mdl = xgb.XGBRegressor(**self.PARAMS)
        mdl.fit(X, y_scaled, eval_set=[(X, y_scaled)], verbose=False)

        self._models[state]    = mdl
        self._scalers[state]   = scaler
        # Keep last 52 rows for recursive step feature building
        self._histories[state] = df.tail(52).copy()

        logger.info(f"[XGBoost] {state}: trained on {len(feature_df)} samples.")

    # ── Recursive multi-step prediction ───────────────────────────────────────
    def predict(self, horizon: int, last_known: pd.DataFrame = None, state: str = None) -> np.ndarray:
        if state is None:
            raise ValueError("XGBoost.predict() requires `state` keyword argument.")

        mdl    = self._models[state]
        scaler = self._scalers[state]
        hist   = self._histories[state].copy()

        # Last known date → generate future dates
        last_date = hist["week_start_date"].max()
        preds = []

        for step in range(horizon):
            next_date = last_date + pd.Timedelta(weeks=step + 1)

            # Build a single-row feature vector from history
            row = self._build_row(hist, next_date, state)
            X_new = np.array([[row[f] for f in ML_FEATURES]])

            y_scaled = mdl.predict(X_new)[0]
            y_hat    = float(scaler.inverse_transform([[y_scaled]])[0][0])
            y_hat    = max(y_hat, 0)
            preds.append(y_hat)

            # Append prediction to history so next step can use it as a lag
            new_row = pd.DataFrame([{
                "week_start_date": next_date,
                "state":           state,
                "sales":           y_hat,
                **row,
            }])
            hist = pd.concat([hist, new_row], ignore_index=True)

        return np.array(preds)

    # ── Feature-row builder ───────────────────────────────────────────────────
    def _build_row(self, hist: pd.DataFrame, date: pd.Timestamp, state: str) -> dict:
        sales = hist["sales"].values

        def lag(k):
            return sales[-k] if len(sales) >= k else sales[0]

        return {
            "lag_1":          lag(1),
            "lag_4":          lag(4),
            "lag_7":          lag(7),
            "lag_52":         lag(52) if len(sales) >= 52 else sales[0],
            "rolling_mean_4": np.mean(sales[-4:])  if len(sales) >= 4  else np.mean(sales),
            "rolling_mean_12":np.mean(sales[-12:]) if len(sales) >= 12 else np.mean(sales),
            "rolling_std_4":  np.std(sales[-4:])   if len(sales) >= 4  else 0.0,
            "rolling_std_12": np.std(sales[-12:])  if len(sales) >= 12 else 0.0,
            "week_of_year":   date.isocalendar()[1],
            "month":          date.month,
            "quarter":        (date.month - 1) // 3 + 1,
            "year":           date.year,
            "day_of_week":    date.dayofweek,
            "is_holiday":     flag_holiday(date),
            "woy_sin":        np.sin(2 * np.pi * date.isocalendar()[1] / 52),
            "woy_cos":        np.cos(2 * np.pi * date.isocalendar()[1] / 52),
            "mon_sin":        np.sin(2 * np.pi * date.month / 12),
            "mon_cos":        np.cos(2 * np.pi * date.month / 12),
        }
