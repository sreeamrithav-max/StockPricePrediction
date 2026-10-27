"""
src/preprocessing.py
--------------------
Leakage-safe chronological data splitting and scaling for Stock Price Prediction.

Design decisions (DECISIONS.md):
  D7  - 70/10/20 chronological split (train/val/test)
  D8  - MinMaxScaler fitted on train only; dedicated Close scaler
  D9  - 60-day look-back window
  D11 - Tabular lag features for RF/LR (last 5 Close values + last feature row)
"""

import logging
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.feature_engineering import get_feature_columns

logger = logging.getLogger(__name__)

LOOK_BACK = 60          # LSTM sequence length (trading days)
N_LAG_CLOSE = 5        # Number of lagged Close values for RF/LR
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
# TEST_RATIO = 0.20 (remainder)


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered DataFrame into train / validation / test subsets
    without shuffling.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame sorted by date (ascending).
    train_ratio : float
        Fraction for training set.
    val_ratio : float
        Fraction for validation set.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        "Chronological split: train=%d (%s–%s), val=%d (%s–%s), test=%d (%s–%s)",
        len(train_df), train_df.index[0].date(), train_df.index[-1].date(),
        len(val_df), val_df.index[0].date(), val_df.index[-1].date(),
        len(test_df), test_df.index[0].date(), test_df.index[-1].date(),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_scalers(
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[MinMaxScaler, MinMaxScaler]:
    """
    Fit a feature scaler and a dedicated Close scaler on TRAINING data only.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training partition.
    feature_cols : list of str
        Ordered list of feature column names.

    Returns
    -------
    (feature_scaler, close_scaler)
        feature_scaler: fitted MinMaxScaler for all features
        close_scaler  : fitted MinMaxScaler for Close only (for LSTM target)
    """
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(train_df[feature_cols].values)

    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(train_df[["Close"]].values)

    logger.info("Scalers fitted on training data only (%d rows).", len(train_df))
    return feature_scaler, close_scaler


def scale_data(
    df: pd.DataFrame,
    feature_scaler: MinMaxScaler,
    feature_cols: List[str],
) -> np.ndarray:
    """
    Apply a FITTED feature scaler to a DataFrame partition.
    Never refit — only transform.
    """
    return feature_scaler.transform(df[feature_cols].values)


# ---------------------------------------------------------------------------
# LSTM sequence builder
# ---------------------------------------------------------------------------

def build_lstm_sequences(
    scaled_features: np.ndarray,
    close_scaler: MinMaxScaler,
    raw_close: np.ndarray,
    look_back: int = LOOK_BACK,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping input sequences for LSTM.

    For each index i (starting at look_back), the input is
    scaled_features[i-look_back : i]  (shape: look_back x n_features)
    and the target is the SCALED Close price at index i.

    Parameters
    ----------
    scaled_features : np.ndarray
        Shape (n_samples, n_features) — already scaled.
    close_scaler : MinMaxScaler
        Fitted scaler for Close column only (used to scale the target).
    raw_close : np.ndarray
        Raw Close prices aligned with scaled_features rows.
    look_back : int
        Sequence length (default 60).

    Returns
    -------
    (X, y)
        X shape: (n_samples - look_back, look_back, n_features)
        y shape: (n_samples - look_back,)  — scaled Close values
    """
    X, y = [], []
    scaled_close = close_scaler.transform(raw_close.reshape(-1, 1)).flatten()

    for i in range(look_back, len(scaled_features)):
        X.append(scaled_features[i - look_back : i])   # past look_back days
        y.append(scaled_close[i])                        # next day's Close (scaled)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Tabular lag features for RF / LR
# ---------------------------------------------------------------------------

def build_tabular_features(
    df: pd.DataFrame,
    feature_scaler: MinMaxScaler,
    feature_cols: List[str],
    n_lag_close: int = N_LAG_CLOSE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build tabular (non-sequential) feature matrix for Random Forest / Linear Regression.

    For each row i (starting at n_lag_close), features are:
      - Last n_lag_close Close values (unscaled, raw)
      - Scaled feature vector at row i-1 (yesterday's features)

    Target: Close at row i (raw price — we predict on original scale).

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (any partition).
    feature_scaler : MinMaxScaler
        Fitted scaler (transform only, never fit here).
    feature_cols : list of str
        Ordered feature columns.
    n_lag_close : int
        Number of lagged Close values.

    Returns
    -------
    (X_tab, y_tab)
        X_tab : np.ndarray, shape (n - n_lag_close, n_lag_close + n_features)
        y_tab : np.ndarray, shape (n - n_lag_close,)  — raw Close prices
    """
    scaled = feature_scaler.transform(df[feature_cols].values)
    close_raw = df["Close"].values

    X_list, y_list = [], []
    for i in range(n_lag_close, len(df)):
        # Lag Close values: t-n_lag_close, ..., t-1
        lag_closes = close_raw[i - n_lag_close : i]
        # Yesterday's full scaled feature row
        prev_features = scaled[i - 1]
        row = np.concatenate([lag_closes, prev_features])
        X_list.append(row)
        y_list.append(close_raw[i])  # raw next-day Close

    X_tab = np.array(X_list, dtype=np.float32)
    y_tab = np.array(y_list, dtype=np.float32)
    return X_tab, y_tab


def get_tabular_feature_names(feature_cols: List[str], n_lag_close: int = N_LAG_CLOSE) -> List[str]:
    """Return human-readable column names for the tabular feature matrix."""
    lag_names = [f"Close_lag_{n_lag_close - i}" for i in range(n_lag_close)]
    prev_names = [f"prev_{c}" for c in feature_cols]
    return lag_names + prev_names


# ---------------------------------------------------------------------------
# Save / load scalers
# ---------------------------------------------------------------------------

def save_scalers(
    feature_scaler: MinMaxScaler,
    close_scaler: MinMaxScaler,
    save_dir: str,
) -> None:
    """Persist fitted scalers to disk."""
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(feature_scaler, os.path.join(save_dir, "feature_scaler.joblib"))
    joblib.dump(close_scaler, os.path.join(save_dir, "close_scaler.joblib"))
    logger.info("Scalers saved to %s", save_dir)


def load_scalers(
    save_dir: str,
) -> Tuple[MinMaxScaler, MinMaxScaler]:
    """Load previously saved scalers from disk."""
    feature_scaler = joblib.load(os.path.join(save_dir, "feature_scaler.joblib"))
    close_scaler = joblib.load(os.path.join(save_dir, "close_scaler.joblib"))
    logger.info("Scalers loaded from %s", save_dir)
    return feature_scaler, close_scaler
