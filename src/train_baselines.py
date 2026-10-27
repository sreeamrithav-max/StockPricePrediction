"""
src/train_baselines.py
-----------------------
Random Forest Regressor and Linear Regression training for Stock Price Prediction.

Both models use the same tabular lag-feature representation:
  - Last N_LAG_CLOSE raw Close values
  - Yesterday's full scaled feature vector

This ensures a fair, leakage-safe comparison against LSTM on the same
held-out test dates and same next-day Close target.

See DECISIONS.md D11, D12.
"""

import logging
import os
from typing import Tuple, Dict, Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

RF_N_ESTIMATORS = 200
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = RF_N_ESTIMATORS,
    random_state: int = RANDOM_SEED,
) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (tabular lag features).
    y_train : np.ndarray
        Training targets — raw Close prices.
    n_estimators : int
        Number of trees (default 200 per SPEC).
    random_state : int
        For reproducibility.

    Returns
    -------
    RandomForestRegressor
        Fitted model.
    """
    logger.info(
        "Training Random Forest: n_estimators=%d, train_samples=%d",
        n_estimators, len(X_train),
    )
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    logger.info("Random Forest training complete.")
    return rf


def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LinearRegression:
    """
    Train a Linear Regression baseline model.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (same tabular lag features as RF).
    y_train : np.ndarray
        Training targets — raw Close prices.

    Returns
    -------
    LinearRegression
        Fitted model.
    """
    logger.info(
        "Training Linear Regression: train_samples=%d", len(X_train)
    )
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    logger.info("Linear Regression training complete.")
    return lr


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(
    rf_model: RandomForestRegressor,
    feature_names: list,
) -> dict:
    """
    Extract feature importances from a trained Random Forest.

    Parameters
    ----------
    rf_model : RandomForestRegressor
    feature_names : list of str

    Returns
    -------
    dict
        Mapping feature_name -> importance (sorted descending).
    """
    importances = rf_model.feature_importances_
    importance_dict = dict(zip(feature_names, importances))
    sorted_dict = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )
    return sorted_dict


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_baseline_models(
    rf_model: RandomForestRegressor,
    lr_model: LinearRegression,
    save_dir: str,
    ticker: str = "AAPL",
) -> None:
    """Save both baseline models using joblib."""
    os.makedirs(save_dir, exist_ok=True)
    rf_path = os.path.join(save_dir, f"{ticker}_random_forest.joblib")
    lr_path = os.path.join(save_dir, f"{ticker}_linear_regression.joblib")
    joblib.dump(rf_model, rf_path)
    joblib.dump(lr_model, lr_path)
    logger.info("Random Forest saved → %s", rf_path)
    logger.info("Linear Regression saved → %s", lr_path)


def load_baseline_models(
    save_dir: str,
    ticker: str = "AAPL",
) -> Tuple[RandomForestRegressor, LinearRegression]:
    """Load saved baseline models."""
    rf_path = os.path.join(save_dir, f"{ticker}_random_forest.joblib")
    lr_path = os.path.join(save_dir, f"{ticker}_linear_regression.joblib")
    rf_model = joblib.load(rf_path)
    lr_model = joblib.load(lr_path)
    logger.info("Models loaded from %s", save_dir)
    return rf_model, lr_model
