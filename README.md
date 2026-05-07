# 📈 End-to-End Sales Forecasting System

A **production-ready time series forecasting system** that trains multiple ML models, automatically selects the best per state, and exposes predictions through a REST API.

---

## 🏗️ Architecture Overview

```
forecasting_system/
│
├── data/
│   ├── data_generator.py      # Synthetic data (replace with real Excel loader)
│   └── feature_engineering.py # Lag, rolling, calendar, holiday features
│
├── models/
│   ├── base_model.py          # Abstract base class (fit/predict/evaluate)
│   ├── sarima_model.py        # SARIMA with auto AIC order selection
│   ├── prophet_model.py       # Facebook Prophet + Indian holidays
│   ├── xgboost_model.py       # XGBoost with recursive multi-step forecast
│   ├── lstm_model.py          # Stacked LSTM (TensorFlow/Keras)
│   └── model_selector.py      # Trains all 4, picks best per state
│
├── api/
│   └── main.py                # FastAPI REST API (8 endpoints)
│
├── artifacts/                  # Saved models & metadata (auto-created)
├── logs/                       # Training logs (auto-created)
├── train.py                    # 🚀 Main training entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models

**With synthetic data (demo):**
```bash
python train.py
```

**With your actual Excel file:**
```bash
python train.py --excel path/to/your/sales_data.xlsx
```

### 3. Start the API
```bash
cd api
uvicorn main:app --reload --port 8000
```

### 4. Open API Docs
```
http://localhost:8000/docs
```

---

## 📊 Models Implemented

| Model | Description | Strengths |
|-------|-------------|-----------|
| **SARIMA** | Seasonal ARIMA with auto AIC selection | Strong on stationary series, interpretable |
| **Prophet** | Facebook's decomposition model | Handles holidays, missing data, trend changes |
| **XGBoost** | Gradient boosting with lag features | Captures complex patterns, fast |
| **LSTM** | 2-layer stacked Long Short-Term Memory | Long-range temporal dependencies |

### Model Selection Criterion
- **Metric**: MAPE (Mean Absolute Percentage Error)
- **Strategy**: Best model per state chosen independently
- **Validation**: Chronological split — no data leakage

---

## 🔧 Feature Engineering

| Feature Category | Features Created |
|-----------------|-----------------|
| **Lag features** | `lag_1` (1 week), `lag_4` (~1 month), `lag_7` (7 weeks), `lag_52` (1 year) |
| **Rolling stats** | `rolling_mean_4`, `rolling_mean_12`, `rolling_std_4`, `rolling_std_12` |
| **Calendar** | `week_of_year`, `month`, `quarter`, `year`, `day_of_week` |
| **Cyclical encoding** | `woy_sin/cos`, `mon_sin/cos` (avoids discontinuity at boundaries) |
| **Holidays** | `is_holiday` — Indian public holidays via the `holidays` library |

### Train / Validation Split
- Last **12 weeks** held out per state for evaluation
- Remaining weeks used for training
- **No shuffling** — preserves temporal order
- Feature computation uses only past data (no look-ahead bias)

---

## 🌐 API Reference

Base URL: `http://localhost:8000`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/states` | List all available states |
| `GET` | `/forecast/{state}` | 8-week forecast for a state |
| `GET` | `/forecast/{state}?weeks=N` | Custom N-week forecast |
| `GET` | `/forecast` | Forecasts for ALL states |
| `GET` | `/models/leaderboard` | Full model comparison table |
| `GET` | `/models/best` | Best model per state |
| `GET` | `/models/summary` | Metrics summary |
| `POST` | `/retrain` | Trigger retraining (background) |

### Example: Forecast for Telangana
```bash
curl http://localhost:8000/forecast/Telangana
```

**Response:**
```json
{
  "state": "Telangana",
  "model_used": "Prophet",
  "horizon": 8,
  "generated_at": "2025-01-15T10:30:00Z",
  "forecasts": [
    {
      "week_start_date": "2025-01-06",
      "forecast_sales": 182450.50,
      "lower_bound": 164205.45,
      "upper_bound": 200695.55
    },
    ...
  ],
  "model_metrics": {
    "MAE": 8230.12,
    "RMSE": 11450.88,
    "MAPE": 4.23
  }
}
```

### Example: All States Forecast
```bash
curl http://localhost:8000/forecast?weeks=8
```

---

## 📈 Missing Value Handling

1. **Detection**: Identify gaps in weekly time series per state
2. **Interpolation**: Linear interpolation preserves trend (works best for < 3 consecutive gaps)
3. **Forward/Backward fill**: Used for leading/trailing gaps
4. **No synthetic noise injection**: Missing values filled deterministically

---

## 🧪 Model Evaluation

| Metric | Formula | Description |
|--------|---------|-------------|
| **MAE** | `mean(|y - ŷ|)` | Average absolute error |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Penalises large errors |
| **MAPE** | `mean(|y - ŷ| / y) × 100` | Scale-independent, our selection criterion |

---

## 🔄 Retraining

The API supports on-demand retraining via:
```bash
curl -X POST http://localhost:8000/retrain
```
This runs training in a background thread and reloads artifacts when complete.

---

## 📦 Production Considerations

| Concern | Implementation |
|---------|---------------|
| **Model versioning** | Artifacts saved to `artifacts/` with JSON metadata |
| **Logging** | `loguru` with file rotation (10 MB, 7-day retention) |
| **Input validation** | Pydantic schemas on all API inputs/outputs |
| **Error handling** | Per-model try/catch — one model failing doesn't block others |
| **CORS** | Enabled for all origins (restrict in production) |
| **Background tasks** | FastAPI `BackgroundTasks` for async retraining |
| **Horizontal scaling** | Artifacts are serialised (joblib) — load-balance behind Nginx |

---

## 📋 Dataset Requirements

Your Excel file should have these columns (names are flexible — the loader auto-maps common variants):

| Required Column | Common Variants |
|----------------|----------------|
| `week_start_date` | `date`, `week`, `week_date` |
| `state` | `state_name`, `region` |
| `sales` | `revenue`, `amount`, `sale_amount` |

---

## 👤 Author Notes

- **SARIMA** uses `s=52` (weekly seasonality period) — change if data is daily/monthly
- **Prophet** includes all Indian national holidays from 2019–2027
- **XGBoost** uses recursive multi-step prediction — each step's prediction feeds into the next
- **LSTM** uses a 26-week lookback window (≈ 6 months)
- Best model is selected independently per state — different states may use different models
