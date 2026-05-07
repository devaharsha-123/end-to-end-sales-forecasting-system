"""
train.py  –  Main training entry point
────────────────────────────────────────
Run:  python train.py
      python train.py --excel path/to/data.xlsx
"""
import sys
import argparse
import pandas as pd
from pathlib import Path
from loguru import logger

# ── Logger setup ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add("logs/training.log", rotation="10 MB", retention="7 days")

Path("logs").mkdir(exist_ok=True)


def load_excel(path: str) -> pd.DataFrame:
    """
    Load the actual assignment Excel file.
    Expected columns: week_start_date, state, sales
    Adjust column names below to match your actual file.
    """
    df = pd.read_excel(path)
    logger.info(f"Loaded Excel: {df.shape}, columns: {df.columns.tolist()}")

    # ── Normalise column names (case-insensitive) ─────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Common column-name variants
    col_map = {
        "date":         "week_start_date",
        "week":         "week_start_date",
        "week_date":    "week_start_date",
        "start_date":   "week_start_date",
        "revenue":      "sales",
        "amount":       "sales",
        "value":        "sales",
        "sale_amount":  "sales",
        "state_name":   "state",
        "region":       "state",
    }
    df = df.rename(columns=col_map)

    required = {"week_start_date", "state", "sales"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Excel file missing columns: {missing}. "
            f"Found: {df.columns.tolist()}"
        )

    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    df["sales"]           = pd.to_numeric(df["sales"], errors="coerce")
    return df[["week_start_date", "state", "sales"]]


def main():
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument("--excel", type=str, default=None,
                        help="Path to Excel data file")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.excel:
        logger.info(f"Loading data from Excel: {args.excel}")
        df = load_excel(args.excel)
    else:
        logger.info("No Excel file provided – using synthetic data generator.")
        from data.data_generator import generate_sales_data
        df = generate_sales_data()

    logger.info(f"Dataset: {df.shape[0]} rows, {df['state'].nunique()} states, "
                f"{df['week_start_date'].min()} → {df['week_start_date'].max()}")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    from models.model_selector import ModelSelector
    selector = ModelSelector()
    leaderboard = selector.run(df)

    # ── Save artifacts ────────────────────────────────────────────────────────
    selector.save_artifacts()

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE – LEADERBOARD SUMMARY")
    logger.info("=" * 60)

    summary = (
        leaderboard
        .sort_values(["state", "MAPE"])
        .groupby("state")
        .first()
        .reset_index()[["state", "model", "MAE", "RMSE", "MAPE"]]
    )
    logger.info("\n" + summary.to_string(index=False))

    logger.info("\n" + "=" * 60)
    logger.info("BEST MODEL PER STATE")
    logger.info("=" * 60)
    for state, model in selector.best_models.items():
        metrics = selector.all_metrics[state][model]
        logger.info(
            f"  {state:25s} → {model:10s}  MAPE={metrics['MAPE']:.2f}%"
        )

    logger.info("\n✅ Artifacts saved. Start the API with:")
    logger.info("   cd api && uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()
