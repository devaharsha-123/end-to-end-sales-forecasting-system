"""
lstm_model.py
LSTM forecasting model.
"""

import warnings
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
)

from sklearn.preprocessing import MinMaxScaler

from base_model import BaseForecaster

warnings.filterwarnings("ignore")


class LSTMForecaster(BaseForecaster):

    def __init__(
        self,
        epochs=10,
        lookback=12,
    ):

        self.epochs = epochs

        self.lookback = lookback

        self.models = {}

        self.scalers = {}

    # ── Create sequences ───────────────────────────────────────────────────
    def _create_sequences(
        self,
        data,
    ):

        X, y = [], []

        for i in range(
            self.lookback,
            len(data)
        ):

            X.append(
                data[i-self.lookback:i]
            )

            y.append(data[i])

        return np.array(X), np.array(y)

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

        values = state_df["sales"].values.reshape(-1, 1)

        scaler = MinMaxScaler()

        scaled = scaler.fit_transform(values)

        X, y = self._create_sequences(scaled)

        X = X.reshape(
            (X.shape[0], X.shape[1], 1)
        )

        model = Sequential()

        model.add(
            LSTM(
                64,
                activation="relu",
                input_shape=(X.shape[1], 1),
            )
        )

        model.add(Dense(1))

        model.compile(
            optimizer="adam",
            loss="mse",
        )

        model.fit(
            X,
            y,
            epochs=self.epochs,
            verbose=0,
        )

        self.models[state] = model

        self.scalers[state] = scaler

    # ── Predict ────────────────────────────────────────────────────────────
    def predict(
        self,
        horizon: int,
        last_known: pd.DataFrame,
        state: str,
    ):

        model = self.models.get(state)

        scaler = self.scalers.get(state)

        if model is None:
            raise ValueError(
                f"No LSTM model found for {state}"
            )

        values = (
            last_known["sales"]
            .values
            .reshape(-1, 1)
        )

        scaled = scaler.transform(values)

        seq = scaled[-self.lookback:]

        preds = []

        for _ in range(horizon):

            X = seq.reshape(
                (1, self.lookback, 1)
            )

            pred = model.predict(
                X,
                verbose=0,
            )[0][0]

            preds.append(pred)

            seq = np.append(
                seq[1:],
                [[pred]],
                axis=0,
            )

        preds = scaler.inverse_transform(
            np.array(preds).reshape(-1, 1)
        )

        return preds.flatten()