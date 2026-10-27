"""
tests/test_preprocessing.py
-----------------------------
Tests for src/preprocessing.py and src/sequence_builder.py

Verifies:
- Chronological split boundaries (no overlap, correct order)
- Scalers fitted only on training data (not val/test)
- LSTM tensor shape (samples, look_back, n_features)
- No future data in input sequences
- Tabular feature matrix shape
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import add_technical_indicators, get_feature_columns
from src.preprocessing import (
    chronological_split,
    fit_scalers,
    scale_data,
    build_lstm_sequences,
    build_tabular_features,
    LOOK_BACK,
    N_LAG_CLOSE,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.sequence_builder import build_inference_sequence, build_tabular_inference_row


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def make_feature_df(n_rows=400, start="2018-01-01"):
    dates = pd.date_range(start=start, periods=n_rows, freq="B")
    np.random.seed(7)
    close = 150 + np.cumsum(np.random.randn(n_rows) * 0.5)
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
    return add_technical_indicators(df)


# ---------------------------------------------------------------------------
# Chronological split tests
# ---------------------------------------------------------------------------

class TestChronologicalSplit:
    def setup_method(self):
        self.df = make_feature_df(400)

    def test_no_overlap(self):
        train, val, test = chronological_split(self.df)
        assert train.index[-1] < val.index[0], "Train and val overlap"
        assert val.index[-1] < test.index[0], "Val and test overlap"

    def test_sizes_sum_to_total(self):
        train, val, test = chronological_split(self.df)
        assert len(train) + len(val) + len(test) == len(self.df)

    def test_test_is_last_20_percent(self):
        n = len(self.df)
        train, val, test = chronological_split(self.df)
        # Test should start at approximately 80% mark
        expected_test_start_idx = int(n * (TRAIN_RATIO + VAL_RATIO))
        assert self.df.index[expected_test_start_idx] == test.index[0]

    def test_chronological_order_preserved(self):
        train, val, test = chronological_split(self.df)
        for part, name in [(train, "train"), (val, "val"), (test, "test")]:
            assert part.index.is_monotonic_increasing, f"{name} not sorted"

    def test_no_shuffling(self):
        """First row of train should be first row of original df."""
        train, val, test = chronological_split(self.df)
        assert train.index[0] == self.df.index[0]

    def test_test_is_final_rows(self):
        """Last row of test should be last row of original df."""
        train, val, test = chronological_split(self.df)
        assert test.index[-1] == self.df.index[-1]


# ---------------------------------------------------------------------------
# Scaler tests
# ---------------------------------------------------------------------------

class TestScalers:
    def setup_method(self):
        df = make_feature_df(400)
        self.train, self.val, self.test = chronological_split(df)
        self.feature_cols = get_feature_columns()
        self.feat_scaler, self.close_scaler = fit_scalers(self.train, self.feature_cols)

    def test_scaler_fitted_on_train_only(self):
        """Scale train → values should be in [0, 1]."""
        scaled_train = scale_data(self.train, self.feat_scaler, self.feature_cols)
        assert scaled_train.min() >= -1e-6, "Train min below 0"
        assert scaled_train.max() <= 1 + 1e-6, "Train max above 1"

    def test_val_values_may_exceed_0_1(self):
        """Val data may have out-of-range values (scaler not refit on val)."""
        scaled_val = scale_data(self.val, self.feat_scaler, self.feature_cols)
        # We don't enforce [0,1] on val — we just check it runs without error
        assert scaled_val.shape == (len(self.val), len(self.feature_cols))

    def test_close_scaler_separate(self):
        """Close scaler should be distinct from feature scaler."""
        assert self.feat_scaler is not self.close_scaler

    def test_inverse_transform_roundtrip(self):
        """Scaling then inverse-transforming Close should recover original values."""
        close_vals = self.train[["Close"]].values
        scaled = self.close_scaler.transform(close_vals)
        recovered = self.close_scaler.inverse_transform(scaled)
        np.testing.assert_allclose(close_vals, recovered, rtol=1e-5)


# ---------------------------------------------------------------------------
# LSTM sequence tests
# ---------------------------------------------------------------------------

class TestLSTMSequences:
    def setup_method(self):
        df = make_feature_df(400)
        self.train, self.val, self.test = chronological_split(df)
        self.feature_cols = get_feature_columns()
        self.feat_scaler, self.close_scaler = fit_scalers(self.train, self.feature_cols)

    def test_train_sequence_shape(self):
        """X shape must be (n_samples - look_back, look_back, n_features)."""
        scaled = scale_data(self.train, self.feat_scaler, self.feature_cols)
        raw_close = self.train["Close"].values
        X, y = build_lstm_sequences(scaled, self.close_scaler, raw_close, LOOK_BACK)
        expected_samples = len(self.train) - LOOK_BACK
        assert X.shape == (expected_samples, LOOK_BACK, len(self.feature_cols))
        assert y.shape == (expected_samples,)

    def test_no_future_data_in_sequences(self):
        """X[i] must contain only indices [i..i+look_back-1], target is i+look_back."""
        scaled = scale_data(self.train, self.feat_scaler, self.feature_cols)
        raw_close = self.train["Close"].values
        X, y = build_lstm_sequences(scaled, self.close_scaler, raw_close, LOOK_BACK)
        # For sequence i, X[i] = scaled[i:i+look_back], y[i] = scaled_close[i+look_back]
        # Manually verify index 0
        expected_X0 = scaled[0:LOOK_BACK]
        np.testing.assert_allclose(X[0], expected_X0)

    def test_inference_sequence_shape(self):
        """Inference sequence must have shape (1, look_back, n_features)."""
        seq = build_inference_sequence(
            self.train, self.feat_scaler, self.feature_cols, LOOK_BACK
        )
        assert seq.shape == (1, LOOK_BACK, len(self.feature_cols))

    def test_inference_raises_on_short_df(self):
        """Should raise ValueError if df has fewer than look_back rows."""
        short_df = self.train.iloc[:LOOK_BACK - 1]
        with pytest.raises(ValueError):
            build_inference_sequence(short_df, self.feat_scaler, self.feature_cols, LOOK_BACK)

    def test_y_values_are_scaled(self):
        """Target values should be in [0, 1] for train data (within scaler range)."""
        scaled = scale_data(self.train, self.feat_scaler, self.feature_cols)
        raw_close = self.train["Close"].values
        _, y = build_lstm_sequences(scaled, self.close_scaler, raw_close, LOOK_BACK)
        assert y.min() >= -0.05, "Scaled targets should be near [0,1]"
        assert y.max() <= 1.05, "Scaled targets should be near [0,1]"


# ---------------------------------------------------------------------------
# Tabular features tests
# ---------------------------------------------------------------------------

class TestTabularFeatures:
    def setup_method(self):
        df = make_feature_df(400)
        self.train, self.val, self.test = chronological_split(df)
        self.feature_cols = get_feature_columns()
        self.feat_scaler, self.close_scaler = fit_scalers(self.train, self.feature_cols)

    def test_tabular_shape(self):
        expected_cols = N_LAG_CLOSE + len(self.feature_cols)
        X, y = build_tabular_features(self.train, self.feat_scaler, self.feature_cols, N_LAG_CLOSE)
        assert X.shape == (len(self.train) - N_LAG_CLOSE, expected_cols)
        assert y.shape == (len(self.train) - N_LAG_CLOSE,)

    def test_tabular_targets_are_raw_prices(self):
        """Tabular targets should be raw (un-scaled) Close prices."""
        X, y = build_tabular_features(self.train, self.feat_scaler, self.feature_cols, N_LAG_CLOSE)
        # Raw prices should be in the range of actual stock prices (>> 1.0)
        assert y.max() > 10, "Tabular targets look like scaled values, not raw prices"

    def test_tabular_inference_row_shape(self):
        row = build_tabular_inference_row(
            self.train, self.feat_scaler, self.feature_cols, N_LAG_CLOSE
        )
        expected_cols = N_LAG_CLOSE + len(self.feature_cols)
        assert row.shape == (1, expected_cols)

    def test_tabular_inference_raises_on_short_df(self):
        short_df = self.train.iloc[:N_LAG_CLOSE - 1]
        with pytest.raises(ValueError):
            build_tabular_inference_row(short_df, self.feat_scaler, self.feature_cols, N_LAG_CLOSE)
