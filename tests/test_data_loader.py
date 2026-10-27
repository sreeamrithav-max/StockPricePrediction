"""
tests/test_data_loader.py
--------------------------
Tests for src/data_loader.py

These tests use synthetic data (no network calls) to verify:
- valid ticker handling (with mocked download)
- empty data detection
- column normalisation
- date ordering and duplicate removal
- validation function
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.data_loader import validate_dataframe, REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# Helpers to build synthetic DataFrames
# ---------------------------------------------------------------------------

def make_ohlcv_df(n_rows=100, start="2020-01-01"):
    """Return a clean synthetic OHLCV DataFrame."""
    dates = pd.date_range(start=start, periods=n_rows, freq="B")  # Business days
    np.random.seed(42)
    close = 150 + np.cumsum(np.random.randn(n_rows))
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n_rows).astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# validate_dataframe tests
# ---------------------------------------------------------------------------

class TestValidateDataframe:
    def test_valid_df_returns_true(self):
        df = make_ohlcv_df()
        ok, msg = validate_dataframe(df)
        assert ok, f"Expected OK, got: {msg}"

    def test_none_returns_false(self):
        ok, msg = validate_dataframe(None)
        assert not ok

    def test_empty_df_returns_false(self):
        ok, msg = validate_dataframe(pd.DataFrame())
        assert not ok

    def test_missing_column_detected(self):
        df = make_ohlcv_df()
        df = df.drop(columns=["Volume"])
        ok, msg = validate_dataframe(df)
        assert not ok
        assert "Volume" in msg

    def test_duplicate_dates_detected(self):
        df = make_ohlcv_df(n_rows=10)
        df = pd.concat([df, df.iloc[:2]])  # add duplicates
        ok, msg = validate_dataframe(df)
        assert not ok
        assert "duplicate" in msg.lower()

    def test_unsorted_dates_detected(self):
        df = make_ohlcv_df(n_rows=10)
        df = df.iloc[::-1]  # reverse order
        ok, msg = validate_dataframe(df)
        assert not ok
        assert "sorted" in msg.lower() or "ascending" in msg.lower()

    def test_nan_values_detected(self):
        df = make_ohlcv_df(n_rows=10)
        df.loc[df.index[2], "Close"] = np.nan
        ok, msg = validate_dataframe(df)
        assert not ok
        assert "nan" in msg.lower() or "missing" in msg.lower()

    def test_non_numeric_column_detected(self):
        df = make_ohlcv_df(n_rows=10)
        df["Close"] = "not_a_number"
        ok, msg = validate_dataframe(df)
        assert not ok


# ---------------------------------------------------------------------------
# download_stock_data tests (mocked)
# ---------------------------------------------------------------------------

class TestDownloadStockData:

    @patch("yfinance.download")
    def test_successful_download(self, mock_download, tmp_path):
        """Mock a successful yfinance download."""
        from src.data_loader import download_stock_data
        df_mock = make_ohlcv_df(n_rows=300)
        mock_download.return_value = df_mock

        result = download_stock_data("AAPL", save_dir=str(tmp_path))
        assert result is not None
        assert len(result) > 0
        for col in REQUIRED_COLUMNS:
            assert col in result.columns

    @patch("yfinance.download")
    def test_empty_download_returns_none(self, mock_download, tmp_path):
        """Empty response should return None."""
        from src.data_loader import download_stock_data
        mock_download.return_value = pd.DataFrame()

        result = download_stock_data("INVALID_XYZ", save_dir=str(tmp_path))
        assert result is None

    @patch("yfinance.download")
    def test_insufficient_data_returns_none(self, mock_download, tmp_path):
        """Less than 100 rows should return None."""
        from src.data_loader import download_stock_data
        df_mock = make_ohlcv_df(n_rows=50)
        mock_download.return_value = df_mock

        result = download_stock_data("TINY", save_dir=str(tmp_path))
        assert result is None

    @patch("yfinance.download")
    def test_duplicate_dates_removed(self, mock_download, tmp_path):
        """Duplicate dates in download result should be removed."""
        from src.data_loader import download_stock_data
        df_mock = make_ohlcv_df(n_rows=200)
        # Inject 5 duplicate rows
        df_with_dupes = pd.concat([df_mock, df_mock.iloc[:5]])
        mock_download.return_value = df_with_dupes

        result = download_stock_data("AAPL", save_dir=str(tmp_path))
        assert result is not None
        assert not result.index.duplicated().any()

    @patch("yfinance.download")
    def test_data_sorted_ascending(self, mock_download, tmp_path):
        """Result should always be sorted ascending by date."""
        from src.data_loader import download_stock_data
        df_mock = make_ohlcv_df(n_rows=200)
        mock_download.return_value = df_mock.iloc[::-1]  # reversed order

        result = download_stock_data("AAPL", save_dir=str(tmp_path))
        assert result is not None
        assert result.index.is_monotonic_increasing

    @patch("yfinance.download")
    def test_network_error_returns_none(self, mock_download, tmp_path):
        """Network exceptions should be handled gracefully."""
        from src.data_loader import download_stock_data
        mock_download.side_effect = ConnectionError("Network error")

        result = download_stock_data("AAPL", save_dir=str(tmp_path))
        assert result is None

    @patch("yfinance.download")
    def test_raw_csv_saved(self, mock_download, tmp_path):
        """Raw CSV should be saved to the specified directory."""
        from src.data_loader import download_stock_data
        df_mock = make_ohlcv_df(n_rows=200)
        mock_download.return_value = df_mock

        download_stock_data("AAPL", save_dir=str(tmp_path))
        assert (tmp_path / "AAPL_daily.csv").exists()
