"""
main.py  –  FastAPI REST API for the Sales Forecasting System
─────────────────────────────────────────────────────────────
Endpoints:
  GET  /                           Health check
  GET  /states                     List available states
  GET  /forecast/{state}           8-week forecast for a state
  GET  /forecast/{state}?weeks=N   Custom horizon
  GET  /models/leaderboard         Model comparison table
  GET  /models/best                Best model per state
  POST /retrain                    Trigger model retraining
  GET  /forecast/all               Forecasts for all states
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from loguru import logger

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sales Forecasting API",
    description=(
        "Production-ready end-to-end time series forecasting system. "
        "Trains SARIMA, Prophet, XGBoost, and LSTM; auto-selects best model per state."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = Path("artifacts")

# ── Pydantic response schemas ─────────────────────────────────────────────────
class WeekForecast(BaseModel):
    week_start_date: str
    forecast_sales:  float
    lower_bound:     Optional[float] = None
    upper_bound:     Optional[float] = None


class ForecastResponse(BaseModel):
    state:       str
    model_used:  str
    horizon:     int
    generated_at: str
    forecasts:   List[WeekForecast]
    model_metrics: dict


class LeaderboardRow(BaseModel):
    state: str
    model: str
    MAE:   float
    RMSE:  float
    MAPE:  float


class BestModelResponse(BaseModel):
    state:      str
    best_model: str
    metrics:    dict


# ── Load artifacts ─────────────────────────────────────────────────────────────
def load_artifacts():
    """Load pre-trained models and forecasts from disk."""
    global forecasters, best_models, final_forecasts, model_summary, leaderboard_df

    try:
        forecasters     = joblib.load(ARTIFACTS_DIR / "forecasters.pkl")
        best_models     = joblib.load(ARTIFACTS_DIR / "best_models.pkl")
        final_forecasts = joblib.load(ARTIFACTS_DIR / "final_forecasts.pkl")
        leaderboard_df  = pd.read_csv(ARTIFACTS_DIR / "leaderboard.csv")
        with open(ARTIFACTS_DIR / "model_summary.json") as f:
            model_summary = json.load(f)
        logger.info("✅ Artifacts loaded successfully.")
    except FileNotFoundError:
        logger.warning("⚠️  No artifacts found – run training first via /retrain.")
        forecasters     = {}
        best_models     = {}
        final_forecasts = {}
        model_summary   = {}
        leaderboard_df  = pd.DataFrame()


forecasters     = {}
best_models     = {}
final_forecasts = {}
model_summary   = {}
leaderboard_df  = pd.DataFrame()


@app.on_event("startup")
def startup_event():
    load_artifacts()


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Sales Forecasting API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
        "states_available": list(best_models.keys()),
    }


@app.get("/states", tags=["Data"])
def list_states():
    """Return all states for which models have been trained."""
    if not best_models:
        raise HTTPException(503, "No models trained yet. POST /retrain first.")
    return {"states": sorted(best_models.keys()), "count": len(best_models)}


@app.get("/forecast/{state}", response_model=ForecastResponse, tags=["Forecast"])
def get_forecast(
    state:  str,
    weeks:  int = Query(default=8, ge=1, le=26, description="Forecast horizon (1–26 weeks)"),
):
    """
    Get sales forecast for a given state.
    - Default horizon: 8 weeks
    - Max horizon: 26 weeks
    """
    # Normalize state name
    state_match = next(
        (s for s in final_forecasts if s.lower() == state.lower()), None
    )
    if state_match is None:
        available = sorted(best_models.keys())
        raise HTTPException(
            404,
            f"State '{state}' not found. Available: {available}"
        )

    fc_df = final_forecasts[state_match].head(weeks)
    model_name = best_models[state_match]
    metrics    = model_summary.get(state_match, {}).get("metrics", {})

    # ±10% confidence interval (simple heuristic; replace with model-specific intervals)
    forecast_rows = []
    for _, row in fc_df.iterrows():
        fc_sales = float(row["forecast_sales"])
        forecast_rows.append(WeekForecast(
            week_start_date=str(row["week_start_date"])[:10],
            forecast_sales=round(fc_sales, 2),
            lower_bound=round(fc_sales * 0.90, 2),
            upper_bound=round(fc_sales * 1.10, 2),
        ))

    return ForecastResponse(
        state=state_match,
        model_used=model_name,
        horizon=len(forecast_rows),
        generated_at=datetime.utcnow().isoformat() + "Z",
        forecasts=forecast_rows,
        model_metrics=metrics,
    )


@app.get("/forecast", tags=["Forecast"])
def get_all_forecasts(weeks: int = Query(default=8, ge=1, le=26)):
    """Return 8-week forecasts for ALL states."""
    if not final_forecasts:
        raise HTTPException(503, "No forecasts available. POST /retrain first.")

    result = {}
    for state, fc_df in final_forecasts.items():
        rows = fc_df.head(weeks)
        result[state] = {
            "model_used": best_models[state],
            "forecasts": [
                {
                    "week": str(r["week_start_date"])[:10],
                    "sales": round(float(r["forecast_sales"]), 2),
                }
                for _, r in rows.iterrows()
            ],
        }
    return result


@app.get("/models/leaderboard", tags=["Models"])
def get_leaderboard():
    """Full model comparison leaderboard across all states and algorithms."""
    if leaderboard_df.empty:
        raise HTTPException(503, "No leaderboard data. POST /retrain first.")
    return leaderboard_df.to_dict(orient="records")


@app.get("/models/best", tags=["Models"])
def get_best_models():
    """Return the best-performing model for each state."""
    if not best_models:
        raise HTTPException(503, "No models trained. POST /retrain first.")
    return [
        BestModelResponse(
            state=state,
            best_model=model,
            metrics=model_summary.get(state, {}).get("metrics", {}),
        )
        for state, model in sorted(best_models.items())
    ]


@app.post("/retrain", tags=["Admin"])
def retrain(background_tasks: BackgroundTasks):
    """
    Trigger model retraining in the background.
    Returns immediately; check /states after a few minutes.
    """
    def _run_training():
        import sys
        sys.path.insert(0, ".")
        from data.data_generator import generate_sales_data
        from models.model_selector import ModelSelector

        logger.info("🔄 Retraining triggered …")
        df = generate_sales_data()
        selector = ModelSelector()
        selector.run(df)
        selector.save_artifacts()
        load_artifacts()
        logger.info("✅ Retraining complete.")

    background_tasks.add_task(_run_training)
    return {
        "status": "Retraining started in background.",
        "note":   "Check /states in a few minutes to confirm completion.",
    }


@app.get("/models/summary", tags=["Models"])
def get_summary():
    """High-level summary: best model + metrics per state."""
    if not model_summary:
        raise HTTPException(503, "No models trained. POST /retrain first.")
    return model_summary
