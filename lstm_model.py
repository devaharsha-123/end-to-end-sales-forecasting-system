"""
lstm_model.py
LSTM (Long Short-Term Memory) forecaster using TensorFlow/Keras.
Uses a sliding-window approach: the past `lookback` weeks predict the next week.
Multi-step forecasting is done recursively.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from loguru import logger

from models.base_model import BaseForecaster

# Lazy-import TensorFlow to avoid startup cost when not needed
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available – LSTM model will use simple baseline.")


class LSTMForecaster(BaseForecaster):
    """
    LSTM with:
      - Sliding-window input (lookback = 26 weeks ~ 6 months)
      - 2-layer stacked LSTM with dropout
      - Single-step output (recursive multi-step prediction)
      - MinMax scaling per state
    """

    name = "LSTM"
    LOOKBACK = 26   # 6 months of weekly data

    def __init__(self, epochs: int = 50, batch_size: int = 16):
        self.epochs     = epochs
        self.batch_size = batch_size
        self._models:   dict = {}
        self._scalers:  dict = {}
        self._histories: dict = {}

    # ── Build Keras model ─────────────────────────────────────────────────────
    @staticmethod
    def _build_model(lookback: int) -> "keras.Model":
        if not TF_AVAILABLE:
            return None

        model = keras.Sequential([
            keras.layers.Input(shape=(lookback, 1)),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss="mse")
        return model

    # ── Sliding-window dataset creator ───────────────────────────────────────
    @staticmethod
    def _make_sequences(series: np.ndarray, lookback: int):
        X, y = [], []
        for i in range(len(series) - lookback):
            X.append(series[i : i + lookback])
            y.append(series[i + lookback])
        return np.array(X), np.array(y)

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, train: pd.DataFrame, state: str) -> None:
        series = (
            train[train["state"] == state]
            .sort_values("week_start_date")["sales"]
            .values.reshape(-1, 1)
        )

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series)

        X, y = self._make_sequences(scaled, self.LOOKBACK)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        self._scalers[state] = scaler

        if not TF_AVAILABLE:
            # Fallback: store last lookback values for naive prediction
            self._models[state]    = None
            self._histories[state] = scaled[-self.LOOKBACK:].ravel()
            logger.warning(f"[LSTM] {state}: TF unavailable, using naive fallback.")
            return

        # Suppress TF logging
        tf.get_logger().setLevel("ERROR")

        model = self._build_model(self.LOOKBACK)
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0),
        ]
        model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0,
        )

        self._models[state]    = model
        self._histories[state] = scaled[-self.LOOKBACK:].ravel()
        logger.info(f"[LSTM] {state}: trained {self.epochs} epochs, lookback={self.LOOKBACK}.")

    # ── Recursive multi-step predict ──────────────────────────────────────────
    def predict(self, horizon: int, last_known: pd.DataFrame = None, state: str = None) -> np.ndarray:
        if state is None:
            raise ValueError("LSTM.predict() requires `state` keyword argument.")

        scaler  = self._scalers[state]
        model   = self._models[state]
        window  = list(self._histories[state])   # scaled values

        preds_scaled = []
        for _ in range(horizon):
            x_in = np.array(window[-self.LOOKBACK:]).reshape(1, self.LOOKBACK, 1)

            if model is not None:
                y_hat_s = float(model.predict(x_in, verbose=0)[0, 0])
            else:
                # Naive fallback: repeat last value
                y_hat_s = window[-1]

            preds_scaled.append(y_hat_s)
            window.append(y_hat_s)

        preds = scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).ravel()
        return np.maximum(preds, 0)
