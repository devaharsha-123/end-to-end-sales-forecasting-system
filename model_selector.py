"""
model_selector.py
Trains all forecasting models, evaluates them,
and selects the best model per state using MAPE.
"""

import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from loguru import logger
from typing import Dict, Any

# ── Fixed imports for flat folder structure ────────────────────────────────
from base_model import BaseForecaster
from sarima_model import SARIMAForecaster
from prophet_model import ProphetForecaster
from xgboost_model import XGBoostForecaster
from lstm_model import LSTMForecaster

from feature_engineering import (
    build_features,
    time_series_split,
    handle_missing,
)

# ── Artifacts directory ────────────────────────────────────────────────────
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


class ModelSelector:
    """
    Full forecasting pipeline:
      1. Preprocess data
      2. Feature engineering
      3. Train all models
      4. Evaluate models
      5. Select best model per state
      6. Save forecasts + artifacts
    """

    HORIZON = 8
    VAL_WEEKS = 12

    def __init__(self):

        self.forecasters: Dict[str, BaseForecaster] = {
            "SARIMA": SARIMAForecaster(),
            "Prophet": ProphetForecaster(),
            "XGBoost": XGBoostForecaster(),
            "LSTM": LSTMForecaster(epochs=50),
        }

        self.leaderboard: pd.DataFrame = None

        self.best_models: Dict[str, str] = {}

        self.all_metrics: Dict[str, Any] = {}

        self.final_forecasts: Dict[str, pd.DataFrame] = {}

    # ── Main training pipeline ─────────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("=" * 60)
        logger.info("Starting End-to-End Forecasting Pipeline")
        logger.info("=" * 60)

        # ── Step 1: Preprocess ─────────────────────────────────────────────
        df = handle_missing(df)

        df = build_features(df)

        # ── Step 2: Train/validation split ────────────────────────────────
        train, val = time_series_split(
            df,
            val_weeks=self.VAL_WEEKS
        )

        states = df["state"].unique().tolist()

        rows = []

        # ── Step 3: Train models per state ────────────────────────────────
        for state in states:

            logger.info(f"\n{'─'*50}")
            logger.info(f"Processing State: {state}")
            logger.info(f"{'─'*50}")

            state_metrics = {}

            for model_name, forecaster in self.forecasters.items():

                try:

                    logger.info(f"Training {model_name}...")

                    # ── Train ─────────────────────────────────────────────
                    forecaster.fit(train, state)

                    # ── Validation data ─────────────────────────────────
                    val_state = (
                        val[val["state"] == state]
                        .sort_values("week_start_date")
                    )

                    y_true = val_state["sales"].values

                    # ── Predict ─────────────────────────────────────────
                    y_pred = forecaster.predict(
                        horizon=len(val_state),
                        last_known=train[train["state"] == state],
                        state=state,
                    )

                    # ── Align prediction lengths ───────────────────────
                    min_len = min(len(y_true), len(y_pred))

                    metrics = BaseForecaster.evaluate(
                        y_true[:min_len],
                        y_pred[:min_len]
                    )

                    forecaster.log_metrics(state, metrics)

                    state_metrics[model_name] = metrics

                    rows.append({
                        "state": state,
                        "model": model_name,
                        **metrics,
                    })

                except Exception as e:

                    logger.error(
                        f"[{model_name}] {state} failed: {e}"
                    )

                    state_metrics[model_name] = {
                        "MAE": 1e9,
                        "RMSE": 1e9,
                        "MAPE": 1e9,
                    }

            # ── Save metrics ────────────────────────────────────────────
            self.all_metrics[state] = state_metrics

            # ── Select best model ───────────────────────────────────────
            best_model = min(
                state_metrics,
                key=lambda m: state_metrics[m]["MAPE"]
            )

            self.best_models[state] = best_model

            logger.info(
                f"✅ BEST MODEL for {state}: "
                f"{best_model} "
                f"(MAPE={state_metrics[best_model]['MAPE']:.2f}%)"
            )

        # ── Leaderboard ───────────────────────────────────────────────────
        self.leaderboard = pd.DataFrame(rows)

        self._save_leaderboard()

        # ── Final forecasts ──────────────────────────────────────────────
        self._generate_final_forecasts(df)

        return self.leaderboard

    # ── Generate future forecasts ─────────────────────────────────────────
    def _generate_final_forecasts(
        self,
        df_full: pd.DataFrame
    ) -> None:

        logger.info("\nGenerating future forecasts...")

        for state, model_name in self.best_models.items():

            forecaster = self.forecasters[model_name]

            last_date = (
                df_full[df_full["state"] == state]
                ["week_start_date"]
                .max()
            )

            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(weeks=1),
                periods=self.HORIZON,
                freq="W-MON",
            )

            try:

                forecast = forecaster.predict(
                    horizon=self.HORIZON,
                    last_known=df_full[df_full["state"] == state],
                    state=state,
                )

                self.final_forecasts[state] = pd.DataFrame({
                    "week_start_date": future_dates,
                    "state": state,
                    "forecast_sales": np.round(forecast, 2),
                    "model_used": model_name,
                })

                logger.info(
                    f"{state}: {model_name} forecast generated"
                )

            except Exception as e:

                logger.error(
                    f"Forecast failed for {state}: {e}"
                )

    # ── Save leaderboard ──────────────────────────────────────────────────
    def _save_leaderboard(self) -> None:

        path = ARTIFACTS_DIR / "leaderboard.csv"

        self.leaderboard.to_csv(path, index=False)

        logger.info(f"Leaderboard saved → {path}")

    # ── Save models/artifacts ─────────────────────────────────────────────
    def save_artifacts(self) -> None:

        joblib.dump(
            self.forecasters,
            ARTIFACTS_DIR / "forecasters.pkl"
        )

        joblib.dump(
            self.best_models,
            ARTIFACTS_DIR / "best_models.pkl"
        )

        joblib.dump(
            self.final_forecasts,
            ARTIFACTS_DIR / "final_forecasts.pkl"
        )

        summary = {
            state: {
                "best_model": model,
                "metrics": self.all_metrics[state][model],
            }
            for state, model in self.best_models.items()
        }

        with open(
            ARTIFACTS_DIR / "model_summary.json",
            "w"
        ) as f:

            json.dump(summary, f, indent=2)

        logger.info(
            f"All artifacts saved to {ARTIFACTS_DIR}/"
        )

    # ── Forecast getter ───────────────────────────────────────────────────
    def get_forecast(self, state: str) -> pd.DataFrame:

        return self.final_forecasts.get(
            state,
            pd.DataFrame()
        )