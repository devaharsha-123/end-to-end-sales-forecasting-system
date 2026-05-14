"""
base_model.py
Base forecasting class with evaluation utilities.
"""

import numpy as np

from abc import ABC, abstractmethod

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


class BaseForecaster(ABC):

    @abstractmethod
    def fit(self, df, state):
        pass

    @abstractmethod
    def predict(self, horizon, last_known, state):
        pass

    @staticmethod
    def evaluate(y_true, y_pred):

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

        mape = np.mean(
            np.abs(
                (y_true - y_pred) / y_true
            )
        ) * 100

        return {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE": round(mape, 2),
        }

    @staticmethod
    def log_metrics(state, metrics):

        print(
            f"{state} | "
            f"MAE={metrics['MAE']} | "
            f"RMSE={metrics['RMSE']} | "
            f"MAPE={metrics['MAPE']}%"
        )