"""
src/sequence_builder.py
-----------------------
Thin wrapper around the sequence-building logic in preprocessing.py.
Kept separate so it can be imported independently (e.g., during inference).

Also contains a helper to build the final single-sample prediction sequence.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

from src.preprocessing import build_lstm_sequences, LOOK_BACK


def build_sequences(
    scaled_features: np.ndarray,
    close_scaler: MinMaxScaler,
    raw_close: np.ndarray,
    look_back: int = LOOK_BACK,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Public API for building LSTM sequences.
    Delegates to preprocessing.build_lstm_sequences.
    """
    return build_lstm_sequences(scaled_features, close_scaler, raw_close, look_back)


def build_inference_sequence(
    df: pd.DataFrame,
    feature_scaler: MinMaxScaler,
    feature_cols: List[str],
    look_back: int = LOOK_BACK,
) -> np.ndarray:
    """
    Build the single 60-day input sequence for next-day inference.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with at least `look_back` rows.
        Must be sorted ascending by date.
    feature_scaler : MinMaxScaler
        Fitted scaler (transform only).
    feature_cols : list of str
        Ordered feature column names.
    look_back : int
        Sequence length.

    Returns
    -------
    np.ndarray
        Shape (1, look_back, n_features) — ready for LSTM.predict().

    Raises
    ------
    ValueError
        If the DataFrame has fewer than `look_back` rows.
    """
    if len(df) < look_back:
        raise ValueError(
            f"Need at least {look_back} rows for inference, got {len(df)}."
        )

    # Take the last `look_back` rows
    tail = df[feature_cols].iloc[-look_back:].values
    scaled_tail = feature_scaler.transform(tail)
    # Shape: (1, look_back, n_features)
    return scaled_tail[np.newaxis, :, :].astype(np.float32)


def build_tabular_inference_row(
    df: pd.DataFrame,
    feature_scaler: MinMaxScaler,
    feature_cols: List[str],
    n_lag_close: int = 5,
) -> np.ndarray:
    """
    Build a single tabular feature row for RF / LR inference.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with at least `n_lag_close` rows.
    feature_scaler : MinMaxScaler
        Fitted scaler.
    feature_cols : list of str
        Ordered feature columns.
    n_lag_close : int
        Number of lagged Close values to include.

    Returns
    -------
    np.ndarray
        Shape (1, n_lag_close + n_features)
    """
    if len(df) < n_lag_close:
        raise ValueError(
            f"Need at least {n_lag_close} rows for tabular inference, got {len(df)}."
        )

    close_raw = df["Close"].values
    lag_closes = close_raw[-n_lag_close:]

    last_row = df[feature_cols].iloc[-1:].values
    scaled_last = feature_scaler.transform(last_row).flatten()

    row = np.concatenate([lag_closes, scaled_last]).astype(np.float32)
    return row.reshape(1, -1)
