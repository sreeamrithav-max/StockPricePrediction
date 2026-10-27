"""
tests/test_prediction.py
-------------------------
Tests for the inference pipeline (src/predict.py, model save/load).

Tests:
- Correct inverse transformation to price units
- Inference with insufficient history raises/returns None
- Model saving and loading (RF, LR via joblib)
- Tabular inference row produces a valid prediction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
import tempfile

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from src.feature_engineering import add_technical_indicators, get_feature_columns
from src.preprocessing import (
    chronological_split,
    fit_scalers,
    scale_data,
    build_lstm_sequences,
    build_tabular_features,
    save_scalers,
    load_scalers,
    LOOK_BACK,
    N_LAG_CLOSE,
)
from src.sequence_builder import build_tabular_inference_row
from src.train_baselines import (
    train_random_forest,
    train_linear_regression,
    save_baseline_models,
    load_baseline_models,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_feature_df(n_rows=400, start="2018-01-01"):
    dates = pd.date_range(start=start, periods=n_rows, freq="B")
    np.random.seed(99)
    close = 200 + np.cumsum(np.random.randn(n_rows) * 0.3)
    close = np.abs(close)
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
    return add_technical_indicators(df)


# ---------------------------------------------------------------------------
# Inverse transformation
# ---------------------------------------------------------------------------

class TestInverseTransform:
    def test_close_scaler_roundtrip(self):
        """Scaling and inverse-transforming should recover original price."""
        close_scaler = MinMaxScaler()
        original = np.array([[150.0], [160.0], [155.0]])
        close_scaler.fit(original)

        scaled = close_scaler.transform(original)
        recovered = close_scaler.inverse_transform(scaled)
        np.testing.assert_allclose(original, recovered, rtol=1e-5)

    def test_inverse_transform_on_prediction(self):
        """Simulated model output (scaled) → inverse transform → dollar price."""
        df = make_feature_df(400)
        train, val, test = chronological_split(df)
        feat_scaler, close_scaler = fit_scalers(train, get_feature_columns())

        # Simulate a model predicting 0.5 (mid-range scaled)
        scaled_pred = np.array([[0.5]])
        price_pred = close_scaler.inverse_transform(scaled_pred)[0][0]
        # Should be a real-looking price, not in [0,1]
        assert price_pred > 1.0, f"Inverse-transformed price looks wrong: {price_pred}"


# ---------------------------------------------------------------------------
# RF / LR save and load
# ---------------------------------------------------------------------------

class TestModelSaveLoad:
    def setup_method(self):
        df = make_feature_df(400)
        self.train, self.val, self.test = chronological_split(df)
        self.feature_cols = get_feature_columns()
        self.feat_scaler, self.close_scaler = fit_scalers(self.train, self.feature_cols)

        X_train, y_train = build_tabular_features(
            self.train, self.feat_scaler, self.feature_cols, N_LAG_CLOSE
        )
        self.rf = train_random_forest(X_train, y_train, n_estimators=10)
        self.lr = train_linear_regression(X_train, y_train)

    def test_rf_save_and_load(self, tmp_path):
        save_baseline_models(self.rf, self.lr, save_dir=str(tmp_path), ticker="TEST")
        rf_loaded, lr_loaded = load_baseline_models(str(tmp_path), ticker="TEST")

        X_test, y_test = build_tabular_features(
            self.test, self.feat_scaler, self.feature_cols, N_LAG_CLOSE
        )
        preds_original = self.rf.predict(X_test)
        preds_loaded = rf_loaded.predict(X_test)
        np.testing.assert_allclose(preds_original, preds_loaded, rtol=1e-5)

    def test_lr_save_and_load(self, tmp_path):
        save_baseline_models(self.rf, self.lr, save_dir=str(tmp_path), ticker="TEST")
        rf_loaded, lr_loaded = load_baseline_models(str(tmp_path), ticker="TEST")

        X_test, y_test = build_tabular_features(
            self.test, self.feat_scaler, self.feature_cols, N_LAG_CLOSE
        )
        preds_original = self.lr.predict(X_test)
        preds_loaded = lr_loaded.predict(X_test)
        np.testing.assert_allclose(preds_original, preds_loaded, rtol=1e-5)

    def test_scaler_save_and_load(self, tmp_path):
        save_scalers(self.feat_scaler, self.close_scaler, str(tmp_path))
        feat_loaded, close_loaded = load_scalers(str(tmp_path))
        X_test = self.test[self.feature_cols].values
        scaled_orig = self.feat_scaler.transform(X_test)
        scaled_loaded = feat_loaded.transform(X_test)
        np.testing.assert_allclose(scaled_orig, scaled_loaded, rtol=1e-5)

    def test_lstm_model_structure_and_save_load(self, tmp_path):
        from src.train_lstm import build_lstm_model, save_lstm_model, load_lstm_model
        model = build_lstm_model(look_back=LOOK_BACK, n_features=len(self.feature_cols))
        assert len(model.layers) == 6
        save_path = save_lstm_model(model, str(tmp_path), ticker="TEST")
        assert os.path.exists(save_path)
        loaded = load_lstm_model(str(tmp_path), ticker="TEST")
        assert loaded is not None
        assert loaded.output_shape == (None, 1)


# ---------------------------------------------------------------------------
# Inference with insufficient history
# ---------------------------------------------------------------------------

class TestInsufficientHistory:
    def test_inference_sequence_raises_on_short_data(self):
        """build_inference_sequence must raise ValueError for short df."""
        df = make_feature_df(400)
        train, val, test = chronological_split(df)
        feature_cols = get_feature_columns()
        feat_scaler, close_scaler = fit_scalers(train, feature_cols)

        short_df = test.iloc[: LOOK_BACK - 1]
        from src.sequence_builder import build_inference_sequence
        with pytest.raises(ValueError, match=str(LOOK_BACK)):
            build_inference_sequence(short_df, feat_scaler, feature_cols, LOOK_BACK)

    def test_tabular_inference_raises_on_short_data(self):
        df = make_feature_df(400)
        train, val, test = chronological_split(df)
        feature_cols = get_feature_columns()
        feat_scaler, close_scaler = fit_scalers(train, feature_cols)

        short_df = test.iloc[: N_LAG_CLOSE - 1]
        with pytest.raises(ValueError):
            build_tabular_inference_row(short_df, feat_scaler, feature_cols, N_LAG_CLOSE)


# ---------------------------------------------------------------------------
# RF / LR predictions are in plausible price range
# ---------------------------------------------------------------------------

class TestPredictionRange:
    def test_rf_predictions_plausible(self):
        df = make_feature_df(400)
        train, val, test = chronological_split(df)
        feature_cols = get_feature_columns()
        feat_scaler, close_scaler = fit_scalers(train, feature_cols)

        X_train, y_train = build_tabular_features(train, feat_scaler, feature_cols, N_LAG_CLOSE)
        X_test, y_test = build_tabular_features(test, feat_scaler, feature_cols, N_LAG_CLOSE)

        rf = train_random_forest(X_train, y_train, n_estimators=10)
        preds = rf.predict(X_test)

        # Predictions should be in a similar range as actual prices
        assert preds.min() > 0, "Negative price predictions"
        # Should be in the ballpark (within 2x) of true values
        ratio = preds.mean() / y_test.mean()
        assert 0.5 < ratio < 2.0, f"RF predictions seem wildly off: ratio={ratio}"

    def test_lr_predictions_plausible(self):
        df = make_feature_df(400)
        train, val, test = chronological_split(df)
        feature_cols = get_feature_columns()
        feat_scaler, close_scaler = fit_scalers(train, feature_cols)

        X_train, y_train = build_tabular_features(train, feat_scaler, feature_cols, N_LAG_CLOSE)
        X_test, y_test = build_tabular_features(test, feat_scaler, feature_cols, N_LAG_CLOSE)

        lr = train_linear_regression(X_train, y_train)
        preds = lr.predict(X_test)

        ratio = preds.mean() / y_test.mean()
        assert 0.1 < ratio < 10.0, f"LR predictions seem wildly off: ratio={ratio}"
