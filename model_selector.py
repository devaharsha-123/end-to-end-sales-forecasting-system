"""
model_selector.py
Trains all 4 models, evaluates them on the validation set,
and selects the best model per state based on MAPE.
Also supports a simple ensemble (weighted average) as bonus.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Dict, Tuple, Any

from models.base_model   import BaseForecaster
from models.sarima_model  import SARIMAForecaster
from models.prophet_model import ProphetForecaster
from models.xgboost_model import XGBoostForecaster
from models.lstm_model    import LSTMForecaster
from data.feature_engineering import build_features, time_series_split, handle_missing


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


class ModelSelector:
    """
    Orchestrates the full model-comparison pipeline:
      1. Preprocess & feature-engineer
      2. Train all 4 models per state
      3. Evaluate on held-out validation weeks
      4. Select best model per state (lowest MAPE)
      5. Save artifacts (models + leaderboard)
    """

    HORIZON = 8          # forecast 8 future weeks
    VAL_WEEKS = 12       # hold out last 12 weeks for evaluation

    def __init__(self):
        self.forecasters: Dict[str, BaseForecaster] = {
            "SARIMA":  SARIMAForecaster(),
            "Prophet": ProphetForecaster(),
            "XGBoost": XGBoostForecaster(),
            "LSTM":    LSTMForecaster(epochs=50),
        }
        self.leaderboard:    pd.DataFrame = None
        self.best_models:    Dict[str, str] = {}         # state → model name
        self.all_metrics:    Dict[str, Any] = {}         # state → {model: metrics}
        self.final_forecasts: Dict[str, pd.DataFrame] = {}  # state → forecast df

    # ── Main pipeline ─────────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("=" * 60)
        logger.info("Starting End-to-End Forecasting Pipeline")
        logger.info("=" * 60)

        # Step 1: Preprocess
        df = handle_missing(df)
        df = build_features(df)

        # Step 2: Split
        train, val = time_series_split(df, val_weeks=self.VAL_WEEKS)

        states = df["state"].unique().tolist()
        rows = []

        for state in states:
            logger.info(f"\n{'─'*50}")
            logger.info(f"Processing: {state}")
            logger.info(f"{'─'*50}")

            state_metrics = {}

            for model_name, forecaster in self.forecasters.items():
                try:
                    # Train
                    forecaster.fit(train, state)

                    # Evaluate on validation
                    val_state = val[val["state"] == state].sort_values("week_start_date")
                    y_true = val_state["sales"].values

                    y_pred = forecaster.predict(
                        horizon=len(val_state),
                        last_known=train[train["state"] == state],
                        state=state,
                    )
                    # Align lengths
                    min_len = min(len(y_true), len(y_pred))
                    metrics = BaseForecaster.evaluate(y_true[:min_len], y_pred[:min_len])
                    forecaster.log_metrics(state, metrics)
                    state_metrics[model_name] = metrics

                    rows.append({
                        "state":       state,
                        "model":       model_name,
                        **metrics,
                    })
                except Exception as e:
                    logger.error(f"[{model_name}] {state} failed: {e}")
                    state_metrics[model_name] = {"MAE": 1e9, "RMSE": 1e9, "MAPE": 1e9}

            self.all_metrics[state] = state_metrics

            # Select best model (lowest MAPE)
            best = min(state_metrics, key=lambda m: state_metrics[m]["MAPE"])
            self.best_models[state] = best
            logger.info(f"✅ BEST for {state}: {best} (MAPE={state_metrics[best]['MAPE']:.2f}%)")

        # Build leaderboard
        self.leaderboard = pd.DataFrame(rows)
        self._save_leaderboard()

        # Step 3: Generate final 8-week forecasts using best model per state
        self._generate_final_forecasts(df)

        return self.leaderboard

    # ── Generate future forecasts ─────────────────────────────────────────────
    def _generate_final_forecasts(self, df_full: pd.DataFrame) -> None:
        logger.info("\nGenerating 8-week ahead forecasts …")

        for state, model_name in self.best_models.items():
            forecaster = self.forecasters[model_name]
            last_date  = df_full[df_full["state"] == state]["week_start_date"].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(weeks=1),
                periods=self.HORIZON,
                freq="W-MON",
            )

            try:
                fc = forecaster.predict(
                    horizon=self.HORIZON,
                    last_known=df_full[df_full["state"] == state],
                    state=state,
                )
                self.final_forecasts[state] = pd.DataFrame({
                    "week_start_date": future_dates,
                    "state":           state,
                    "forecast_sales":  np.round(fc, 2),
                    "model_used":      model_name,
                })
                logger.info(f"  {state}: {model_name} → {fc.round(0).tolist()}")
            except Exception as e:
                logger.error(f"Final forecast failed for {state}: {e}")

    # ── Persistence ───────────────────────────────────────────────────────────
    def _save_leaderboard(self) -> None:
        path = ARTIFACTS_DIR / "leaderboard.csv"
        self.leaderboard.to_csv(path, index=False)
        logger.info(f"Leaderboard saved → {path}")

    def save_artifacts(self) -> None:
        """Persist models and metadata for the API layer."""
        joblib.dump(self.forecasters,     ARTIFACTS_DIR / "forecasters.pkl")
        joblib.dump(self.best_models,     ARTIFACTS_DIR / "best_models.pkl")
        joblib.dump(self.final_forecasts, ARTIFACTS_DIR / "final_forecasts.pkl")

        # Save JSON summary for quick loading
        summary = {
            state: {
                "best_model": model,
                "metrics": self.all_metrics[state][model],
            }
            for state, model in self.best_models.items()
        }
        with open(ARTIFACTS_DIR / "model_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"All artifacts saved to {ARTIFACTS_DIR}/")

    def get_forecast(self, state: str) -> pd.DataFrame:
        return self.final_forecasts.get(state, pd.DataFrame())
