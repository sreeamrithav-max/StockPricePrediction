"""
src/data_loader.py
------------------
Reusable data download and validation module for Stock Price Prediction.
Uses yfinance to download historical daily OHLCV data.

Key design decisions (see DECISIONS.md):
- Dynamic 5-year lookback from today's date (D2)
- Uses adjusted close prices (D3)
- No fabricated rows for weekends/holidays (D4)
- Saves raw CSV for reproducibility
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
DEFAULT_YEARS = 5


def download_stock_data(
    ticker: str,
    years: int = DEFAULT_YEARS,
    save_dir: str = "data/raw",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Download historical OHLCV data for a given ticker using yfinance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL').
    years : int
        Number of years of history to retrieve.
    save_dir : str
        Directory where the raw CSV will be saved.
    interval : str
        Data interval. Default '1d' for daily data.

    Returns
    -------
    pd.DataFrame or None
        Cleaned DataFrame with Date index and OHLCV columns,
        or None if download fails.
    """
    ticker = ticker.strip().upper()
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365 + 10)  # slight buffer

    logger.info(
        "Downloading %s | %s to %s | interval=%s",
        ticker,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        interval,
    )

    try:
        raw = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,   # gives split/dividend-adjusted prices
            progress=False,
        )
    except Exception as exc:
        logger.error("Download failed for %s: %s", ticker, exc)
        return None

    if raw is None or raw.empty:
        logger.error(
            "No data returned for '%s'. Check ticker symbol and internet connection.",
            ticker,
        )
        return None

    # --- Handle MultiIndex columns (some yfinance versions return them) ---
    if isinstance(raw.columns, pd.MultiIndex):
        # Drop the ticker level so we get plain OHLCV names
        raw.columns = raw.columns.get_level_values(0)

    # --- Normalise column names to title-case ---
    raw.columns = [str(c).strip().title() for c in raw.columns]

    # --- Ensure Date is in the index ---
    if "Date" in raw.columns:
        raw = raw.set_index("Date")
    raw.index.name = "Date"

    # --- Check required columns exist ---
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        logger.error("Missing required columns after download: %s", missing)
        return None

    # --- Keep only required columns ---
    df = raw[REQUIRED_COLUMNS].copy()

    # --- Sort by date ascending ---
    df.sort_index(inplace=True)

    # --- Remove duplicate dates (keep last occurrence) ---
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        logger.warning("Removing %d duplicate date(s).", n_dupes)
        df = df[~df.index.duplicated(keep="last")]

    # --- Convert to numeric, coerce errors to NaN ---
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Report missing values ---
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        logger.warning(
            "Found %d missing value(s) after download. Dropping affected rows.",
            n_missing,
        )
        df.dropna(inplace=True)

    # --- Validate sufficient history ---
    min_rows = 100  # need at least enough for warm-up + a few sequences
    if len(df) < min_rows:
        logger.error(
            "Insufficient data for '%s': only %d rows after cleaning (min %d).",
            ticker,
            len(df),
            min_rows,
        )
        return None

    # --- Save raw CSV ---
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{ticker}_daily.csv")
    df.to_csv(csv_path)
    logger.info("Raw data saved → %s  (%d rows)", csv_path, len(df))
    logger.info("Actual data range: %s to %s", df.index[0].date(), df.index[-1].date())

    return df


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate a DataFrame that is supposed to contain OHLCV stock data.

    Returns
    -------
    (bool, str)
        (True, 'OK') on success, or (False, reason) on failure.
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None."

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"

    non_numeric = [
        c for c in REQUIRED_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        return False, f"Non-numeric columns: {non_numeric}"

    if df.index.duplicated().any():
        return False, "DataFrame contains duplicate dates."

    if not df.index.is_monotonic_increasing:
        return False, "DataFrame is not sorted in ascending date order."

    if df.isnull().any().any():
        return False, "DataFrame contains NaN values."

    return True, "OK"


def load_raw_csv(ticker: str, data_dir: str = "data/raw") -> Optional[pd.DataFrame]:
    """Load a previously saved raw CSV for a given ticker."""
    path = os.path.join(data_dir, f"{ticker.upper()}_daily.csv")
    if not os.path.exists(path):
        logger.error("Raw CSV not found: %s", path)
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df.sort_index(inplace=True)
    return df


if __name__ == "__main__":
    df = download_stock_data("AAPL")
    if df is not None:
        print(df.tail())
