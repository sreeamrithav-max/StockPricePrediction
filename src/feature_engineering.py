"""
src/feature_engineering.py
---------------------------
Technical indicator calculation for Stock Price Prediction.

Features calculated (all use only current and past data):
  - MA5   : 5-day Simple Moving Average of Close
  - MA20  : 20-day Simple Moving Average of Close
  - RSI   : 14-period RSI using Wilder's smoothing (EMA-based)
  - BB_Upper / BB_Lower : Bollinger Bands (20-day window, 2 std)
  - Daily_Return        : Percentage change in Close
  - Price_to_Volume_Ratio : Close / (Volume + 1)  [guards zero volume]

Warm-up rows where rolling features are undefined are DROPPED,
not filled, per DECISIONS.md D4.

See DECISIONS.md for full rationale.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ----- Constants -----
MA_SHORT = 5
MA_LONG = 20
RSI_PERIOD = 14
BB_WINDOW = 20
BB_STD = 2

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA5",
    "MA20",
    "RSI",
    "BB_Upper",
    "BB_Lower",
    "Daily_Return",
    "Price_to_Volume_Ratio",
]


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Compute the Relative Strength Index using Wilder's exponential smoothing.

    Parameters
    ----------
    series : pd.Series
        Closing price series.
    period : int
        Lookback period (default 14).

    Returns
    -------
    pd.Series
        RSI values in [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger_bands(
    series: pd.Series, window: int = BB_WINDOW, num_std: int = BB_STD
):
    """
    Compute upper and lower Bollinger Bands.

    Parameters
    ----------
    series : pd.Series
        Closing price series.
    window : int
        Rolling window (default 20).
    num_std : int
        Number of standard deviations (default 2).

    Returns
    -------
    (pd.Series, pd.Series)
        (BB_Upper, BB_Lower)
    """
    rolling_mean = series.rolling(window=window)
    sma = rolling_mean.mean()
    std = rolling_mean.std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all required technical indicators to an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Open, High, Low, Close, Volume.
        Must be sorted in ascending date order.

    Returns
    -------
    pd.DataFrame
        DataFrame with new feature columns appended.
        Rows where any rolling feature is NaN (warm-up period) are DROPPED.
    """
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]

    # Moving averages
    df["MA5"] = close.rolling(window=MA_SHORT).mean()
    df["MA20"] = close.rolling(window=MA_LONG).mean()

    # RSI (14-period)
    df["RSI"] = compute_rsi(close, period=RSI_PERIOD)

    # Bollinger Bands (20-day, 2-std)
    df["BB_Upper"], df["BB_Lower"] = compute_bollinger_bands(
        close, window=BB_WINDOW, num_std=BB_STD
    )

    # Daily Return (percentage change)
    df["Daily_Return"] = close.pct_change() * 100  # in percent

    # Price-to-Volume Ratio (guard against zero volume)
    df["Price_to_Volume_Ratio"] = close / (volume + 1)

    # Drop warm-up NaN rows (MA20 / BB needs 20 rows; RSI needs 14)
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)
    dropped = rows_before - rows_after
    if dropped > 0:
        logger.info(
            "Dropped %d warm-up row(s) with NaN features (rolling window initialisation).",
            dropped,
        )

    # Verify all feature columns are present
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing after engineering: {missing}")

    return df


def get_feature_columns() -> List[str]:
    """Return the canonical ordered list of feature columns."""
    return list(FEATURE_COLUMNS)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import download_stock_data

    raw = download_stock_data("AAPL")
    if raw is not None:
        features = add_technical_indicators(raw)
        print(features[FEATURE_COLUMNS].tail(5))
        print("Feature columns:", get_feature_columns())
