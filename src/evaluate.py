"""
src/evaluate.py
---------------
Model evaluation utilities for Stock Price Prediction.

Calculates RMSE, MAE, and R² on the original price scale.
All predictions must be inverse-transformed to dollar prices
before calling these functions.

Per SPEC §8: metrics are calculated from actual model outputs,
never hardcoded. Reference report values are NOT used here.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> Dict[str, float]:
    """
    Compute RMSE, MAE, and R² on the original price scale.

    Parameters
    ----------
    y_true : np.ndarray
        Actual Close prices (dollar scale).
    y_pred : np.ndarray
        Predicted Close prices (dollar scale).
    model_name : str
        Label for logging.

    Returns
    -------
    dict with keys 'RMSE', 'MAE', 'R2'
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    label = f"[{model_name}]" if model_name else ""
    logger.info("%s RMSE=%.4f  MAE=%.4f  R²=%.4f", label, rmse, mae, r2)

    return {"Model": model_name, "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}


def build_metrics_table(results: List[Dict]) -> pd.DataFrame:
    """Convert a list of metric dicts to a formatted DataFrame."""
    df = pd.DataFrame(results).set_index("Model")
    df.columns = ["RMSE ($)", "MAE ($)", "R²"]
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    dates: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    ticker: str = "AAPL",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot actual vs. predicted closing prices for all models.

    Parameters
    ----------
    dates : array-like
        Date values for x-axis.
    y_true : np.ndarray
        Actual Close prices.
    predictions : dict
        {model_name: predicted_array}
    ticker : str
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, y_true, label="Actual", color="black", linewidth=2)

    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    for (name, preds), color in zip(predictions.items(), colors):
        ax.plot(dates, preds, label=name, linestyle="--", color=color, linewidth=1.5)

    ax.set_title(f"{ticker} — Actual vs. Predicted Closing Prices (Test Period)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Actual vs predicted chart saved → %s", save_path)

    return fig


def plot_lstm_loss(
    history,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot LSTM training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history["loss"], label="Train Loss", color="#2196F3")
    ax.plot(history.history["val_loss"], label="Val Loss", color="#FF5722")
    ax.set_title("LSTM Training vs. Validation Loss", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("LSTM loss curve saved → %s", save_path)

    return fig


def plot_feature_importance(
    importance_dict: Dict[str, float],
    top_n: int = 15,
    ticker: str = "AAPL",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Random Forest feature importances (top N features)."""
    items = list(importance_dict.items())[:top_n]
    features = [x[0] for x in items]
    importances = [x[1] for x in items]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features[::-1], importances[::-1], color="#4CAF50", alpha=0.85)
    ax.set_title(f"{ticker} — Random Forest Feature Importance (Top {top_n})", fontsize=14)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Feature importance chart saved → %s", save_path)

    return fig


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    ticker: str = "AAPL",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing RMSE across models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    bars = ax.bar(metrics_df.index, metrics_df["RMSE ($)"], color=colors[:len(metrics_df)])
    ax.set_title(f"{ticker} — Model Comparison by RMSE", fontsize=14)
    ax.set_ylabel("RMSE ($)")
    ax.set_xlabel("Model")

    for bar, val in zip(bars, metrics_df["RMSE ($)"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"${val:.2f}",
            ha="center", va="bottom", fontsize=11,
        )

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Model comparison chart saved → %s", save_path)

    return fig


def plot_residuals(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "LSTM",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot residuals (actual - predicted) over time."""
    residuals = np.array(y_true).flatten() - np.array(y_pred).flatten()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time series residuals
    axes[0].plot(dates, residuals, color="#9C27B0", linewidth=1)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_title(f"{model_name} Residuals Over Time")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Residual ($)")
    axes[0].grid(alpha=0.3)

    # Residual histogram
    axes[1].hist(residuals, bins=30, color="#9C27B0", alpha=0.75, edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_title(f"{model_name} Residual Distribution")
    axes[1].set_xlabel("Residual ($)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Residual plot saved → %s", save_path)

    return fig


def save_metrics_csv(metrics_df: pd.DataFrame, save_path: str) -> None:
    """Persist evaluation metrics to a CSV file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics_df.to_csv(save_path)
    logger.info("Metrics CSV saved → %s", save_path)
