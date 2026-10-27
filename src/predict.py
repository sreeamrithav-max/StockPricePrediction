"""
src/predict.py
--------------
Inference pipeline for Stock Price Prediction.

Loads saved artifacts (LSTM model, RF, LR, scalers, metadata),
downloads fresh data, re-applies the EXACT same preprocessing pipeline
used during training, and predicts the next trading day's Close.

Per SPEC §9, the inference pipeline:
1. Downloads fresh historical data for the ticker.
2. Recalculates features in the same order.
3. Validates sufficient history exists.
4. Applies saved fitted transformations without refitting.
5. Creates the correct 60-day input sequence.
6. Predicts next trading day's Close.
7. Inverse-transforms correctly.
8. Handles errors gracefully.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOOK_BACK = 60
N_LAG_CLOSE = 5


def load_artifacts(artifact_dir: str, ticker: str = "AAPL") -> Optional[Dict]:
    """
    Load all saved artifacts for inference.

    Parameters
    ----------
    artifact_dir : str
        Directory containing saved models and scalers.
    ticker : str
        Ticker symbol.

    Returns
    -------
    dict with keys: lstm_model, rf_model, lr_model,
                    feature_scaler, close_scaler,
                    feature_cols, metadata
    or None on failure.
    """
    import joblib
    from tensorflow import keras

    artifacts = {}
    try:
        # Scalers
        artifacts["feature_scaler"] = joblib.load(
            os.path.join(artifact_dir, "feature_scaler.joblib")
        )
        artifacts["close_scaler"] = joblib.load(
            os.path.join(artifact_dir, "close_scaler.joblib")
        )

        # Models
        lstm_path = os.path.join(artifact_dir, f"{ticker}_lstm.keras")
        if os.path.exists(lstm_path):
            artifacts["lstm_model"] = keras.models.load_model(lstm_path)
        else:
            logger.warning("LSTM model not found at %s", lstm_path)
            artifacts["lstm_model"] = None

        rf_path = os.path.join(artifact_dir, f"{ticker}_random_forest.joblib")
        if os.path.exists(rf_path):
            artifacts["rf_model"] = joblib.load(rf_path)
        else:
            logger.warning("RF model not found at %s", rf_path)
            artifacts["rf_model"] = None

        lr_path = os.path.join(artifact_dir, f"{ticker}_linear_regression.joblib")
        if os.path.exists(lr_path):
            artifacts["lr_model"] = joblib.load(lr_path)
        else:
            logger.warning("LR model not found at %s", lr_path)
            artifacts["lr_model"] = None

        # Metadata (feature columns, training info)
        meta_path = os.path.join(artifact_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            artifacts["feature_cols"] = metadata.get("feature_cols", [])
            artifacts["metadata"] = metadata
        else:
            logger.warning("metadata.json not found — using default feature columns.")
            from src.feature_engineering import get_feature_columns
            artifacts["feature_cols"] = get_feature_columns()
            artifacts["metadata"] = {}

    except Exception as exc:
        logger.error("Failed to load artifacts: %s", exc)
        return None

    return artifacts


def predict_next_day(
    ticker: str,
    artifact_dir: str,
    model_name: str = "lstm",
) -> Optional[Dict]:
    """
    Full end-to-end inference: download fresh data → features → predict.

    Parameters
    ----------
    ticker : str
        Stock ticker.
    artifact_dir : str
        Directory with saved artifacts.
    model_name : str
        'lstm', 'rf', or 'lr'.

    Returns
    -------
    dict with keys:
        ticker, last_close, last_date, predicted_close,
        predicted_date, change_abs, change_pct, retrieved_at
    or None on failure.
    """
    # Load artifacts
    artifacts = load_artifacts(artifact_dir, ticker)
    if artifacts is None:
        logger.error("Could not load artifacts from %s", artifact_dir)
        return None

    feature_scaler = artifacts["feature_scaler"]
    close_scaler = artifacts["close_scaler"]
    feature_cols = artifacts["feature_cols"]

    # Download fresh data
    from src.data_loader import download_stock_data
    from src.feature_engineering import add_technical_indicators

    df_raw = download_stock_data(ticker, years=5, save_dir=os.path.join(artifact_dir, "..", "..", "data", "raw"))
    if df_raw is None:
        return None

    df_feat = add_technical_indicators(df_raw)

    # Validate sufficient history
    min_required = LOOK_BACK + 5
    if len(df_feat) < min_required:
        logger.error(
            "Insufficient history for inference: need %d rows, got %d.",
            min_required, len(df_feat),
        )
        return None

    last_close = float(df_feat["Close"].iloc[-1])
    last_date = df_feat.index[-1]

    # Estimate next trading day (skip weekends)
    next_date = last_date + timedelta(days=1)
    while next_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_date += timedelta(days=1)

    # --- LSTM inference ---
    if model_name.lower() == "lstm" and artifacts.get("lstm_model") is not None:
        from src.sequence_builder import build_inference_sequence
        X_infer = build_inference_sequence(df_feat, feature_scaler, feature_cols, LOOK_BACK)
        scaled_pred = artifacts["lstm_model"].predict(X_infer, verbose=0)
        predicted_close = float(close_scaler.inverse_transform(scaled_pred)[0][0])

    # --- RF inference ---
    elif model_name.lower() == "rf" and artifacts.get("rf_model") is not None:
        from src.sequence_builder import build_tabular_inference_row
        X_tab = build_tabular_inference_row(df_feat, feature_scaler, feature_cols, N_LAG_CLOSE)
        predicted_close = float(artifacts["rf_model"].predict(X_tab)[0])

    # --- LR inference ---
    elif model_name.lower() == "lr" and artifacts.get("lr_model") is not None:
        from src.sequence_builder import build_tabular_inference_row
        X_tab = build_tabular_inference_row(df_feat, feature_scaler, feature_cols, N_LAG_CLOSE)
        predicted_close = float(artifacts["lr_model"].predict(X_tab)[0])

    else:
        logger.error("Model '%s' not available in artifacts.", model_name)
        return None

    change_abs = predicted_close - last_close
    change_pct = (change_abs / last_close) * 100

    result = {
        "ticker": ticker,
        "last_close": round(last_close, 4),
        "last_date": str(last_date.date()),
        "predicted_close": round(predicted_close, 4),
        "predicted_date": str(next_date.date()),
        "change_abs": round(change_abs, 4),
        "change_pct": round(change_pct, 4),
        "model": model_name.upper(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "%s | Last Close: $%.2f (%s) | Predicted: $%.2f (%s) | Change: %+.2f (%+.2f%%)",
        ticker, last_close, last_date.date(),
        predicted_close, next_date.date(),
        change_abs, change_pct,
    )
    return result


if __name__ == "__main__":
    result = predict_next_day("AAPL", artifact_dir="artifacts/AAPL", model_name="lstm")
    if result:
        import pprint
        pprint.pprint(result)
