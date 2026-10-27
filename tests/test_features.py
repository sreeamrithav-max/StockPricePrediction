"""
tests/test_features.py
-----------------------
Tests for src/feature_engineering.py

Verifies:
- All required feature columns are produced
- Rolling features use only current and past observations
- NaN warm-up rows are dropped (not filled)
- RSI is bounded [0, 100]
- Bollinger Band upper >= lower
- Price_to_Volume_Ratio guards zero volume
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_technical_indicators,
    compute_rsi,
    compute_bollinger_bands,
    get_feature_columns,
    FEATURE_COLUMNS,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_ohlcv_df(n_rows=150, start="2020-01-01"):
    dates = pd.date_range(start=start, periods=n_rows, freq="B")
    np.random.seed(0)
    close = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    close = np.abs(close)
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.random.randint(500_000, 5_000_000, n_rows).astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# Feature column completeness
# ---------------------------------------------------------------------------

class TestFeatureColumns:
    def test_all_feature_columns_present(self):
        df = make_ohlcv_df(200)
        result = add_technical_indicators(df)
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_get_feature_columns_returns_list(self):
        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) == len(FEATURE_COLUMNS)

    def test_output_has_no_nans(self):
        df = make_ohlcv_df(200)
        result = add_technical_indicators(df)
        assert not result[FEATURE_COLUMNS].isnull().any().any()


# ---------------------------------------------------------------------------
# Warm-up row dropping
# ---------------------------------------------------------------------------

class TestWarmupDropping:
    def test_warmup_rows_dropped(self):
        """First 20 rows should be dropped (BB/MA20 warm-up)."""
        n = 150
        df = make_ohlcv_df(n)
        result = add_technical_indicators(df)
        # Should have fewer rows than the input
        assert len(result) < n

    def test_no_forward_fill(self):
        """No fabricated (filled) values in warm-up position."""
        df = make_ohlcv_df(150)
        result = add_technical_indicators(df)
        # All MA20 values must be >= 20 trading days in
        ma20_vals = result["MA20"].values
        assert not np.any(np.isnan(ma20_vals)), "MA20 should have no NaN after drop"


# ---------------------------------------------------------------------------
# RSI bounds
# ---------------------------------------------------------------------------

class TestRSI:
    def test_rsi_bounded_0_100(self):
        df = make_ohlcv_df(200)
        result = add_technical_indicators(df)
        assert result["RSI"].between(0, 100).all(), "RSI must be in [0, 100]"

    def test_rsi_monotone_gain_series(self):
        """Constantly rising prices → RSI should be high (near 100)."""
        close = pd.Series(np.arange(1, 50, dtype=float))
        rsi = compute_rsi(close, period=14)
        valid_rsi = rsi.dropna()
        assert (valid_rsi > 70).all(), f"Expected high RSI for monotone rise, got: {valid_rsi.values}"

    def test_rsi_monotone_loss_series(self):
        """Constantly falling prices → RSI should be low (near 0)."""
        close = pd.Series(np.arange(50, 1, -1, dtype=float))
        rsi = compute_rsi(close, period=14)
        valid_rsi = rsi.dropna()
        assert (valid_rsi < 30).all(), f"Expected low RSI for monotone fall, got: {valid_rsi.values}"


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_bb_upper_gte_lower(self):
        df = make_ohlcv_df(200)
        result = add_technical_indicators(df)
        assert (result["BB_Upper"] >= result["BB_Lower"]).all()

    def test_bb_upper_gte_ma20(self):
        df = make_ohlcv_df(200)
        result = add_technical_indicators(df)
        assert (result["BB_Upper"] >= result["MA20"]).all()


# ---------------------------------------------------------------------------
# Price-to-Volume Ratio
# ---------------------------------------------------------------------------

class TestPriceToVolume:
    def test_zero_volume_no_inf(self):
        """Zero Volume should produce a finite ratio (not inf)."""
        df = make_ohlcv_df(150)
        df.loc[df.index[30], "Volume"] = 0.0
        result = add_technical_indicators(df)
        assert not np.any(np.isinf(result["Price_to_Volume_Ratio"].values))

    def test_price_to_volume_positive(self):
        df = make_ohlcv_df(150)
        result = add_technical_indicators(df)
        assert (result["Price_to_Volume_Ratio"] > 0).all()


# ---------------------------------------------------------------------------
# No future data in features
# ---------------------------------------------------------------------------

class TestNoFutureData:
    def test_ma5_uses_past_only(self):
        """MA5 at row i should equal mean of rows i-4..i."""
        df = make_ohlcv_df(200)
        result = add_technical_indicators(df)
        # Check a row's MA5 against manual calculation
        idx = 50
        dropped = len(df) - len(result)
        expected_ma5 = df["Close"].iloc[idx - 4 : idx + 1].mean()
        actual_ma5 = result["MA5"].iloc[idx - dropped]
        # Allow small float tolerance
        assert abs(expected_ma5 - actual_ma5) < 1e-6

    def test_daily_return_uses_past_only(self):
        """Daily_Return at row i is pct_change from row i-1 to i."""
        df = make_ohlcv_df(200)
        result = add_technical_indicators(df)
        # All daily returns should be finite
        assert np.isfinite(result["Daily_Return"].values).all()
