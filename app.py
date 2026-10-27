"""
app.py — Stock Price Prediction Streamlit Dashboard
=====================================================

Loads saved model artifacts from artifacts/<TICKER>/ and presents:
  - Latest observed closing price
  - LSTM, Random Forest, and Linear Regression next-day predictions
  - Historical price chart
  - Actual vs predicted chart (test period)
  - Model comparison metrics table
  - Feature importance chart (Random Forest)
  - Training loss curve (LSTM)
  - Clear error messages and financial disclaimer

Run locally:
    streamlit run app.py

Requirements: see requirements.txt
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
TRAINED_TICKER = "AAPL"  # The ticker used during training
ARTIFACT_DIR = os.path.join("artifacts", TRAINED_TICKER)
LOOK_BACK = 60
N_LAG_CLOSE = 5

# --------------------------------------------------------------------------
# Custom CSS
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }
    .metric-label {
        font-size: 13px;
        opacity: 0.85;
        margin-bottom: 5px;
    }
    .positive { color: #4CAF50; }
    .negative { color: #F44336; }
    .disclaimer {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_all_artifacts(artifact_dir: str, ticker: str):
    """Load saved models, scalers, and metadata. Cached across reruns."""
    artifacts = {}
    errors = []

    try:
        import joblib
        feat_path = os.path.join(artifact_dir, "feature_scaler.joblib")
        close_path = os.path.join(artifact_dir, "close_scaler.joblib")
        if os.path.exists(feat_path) and os.path.exists(close_path):
            artifacts["feature_scaler"] = joblib.load(feat_path)
            artifacts["close_scaler"] = joblib.load(close_path)
        else:
            errors.append("Scalers not found. Run the Colab notebook first.")
    except Exception as e:
        errors.append(f"Scaler load error: {e}")

    try:
        from tensorflow import keras
        lstm_path = os.path.join(artifact_dir, f"{ticker}_lstm.keras")
        if os.path.exists(lstm_path):
            artifacts["lstm_model"] = keras.models.load_model(lstm_path)
        else:
            artifacts["lstm_model"] = None
            errors.append("LSTM model not found.")
    except Exception as e:
        artifacts["lstm_model"] = None
        errors.append(f"LSTM load error: {e}")

    try:
        import joblib
        rf_path = os.path.join(artifact_dir, f"{ticker}_random_forest.joblib")
        lr_path = os.path.join(artifact_dir, f"{ticker}_linear_regression.joblib")
        artifacts["rf_model"] = joblib.load(rf_path) if os.path.exists(rf_path) else None
        artifacts["lr_model"] = joblib.load(lr_path) if os.path.exists(lr_path) else None
    except Exception as e:
        artifacts["rf_model"] = None
        artifacts["lr_model"] = None
        errors.append(f"Baseline model load error: {e}")

    try:
        meta_path = os.path.join(artifact_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                artifacts["metadata"] = json.load(f)
            artifacts["feature_cols"] = artifacts["metadata"].get("feature_cols", [])
        else:
            from src.feature_engineering import get_feature_columns
            artifacts["feature_cols"] = get_feature_columns()
            artifacts["metadata"] = {}
            errors.append("metadata.json not found — using defaults.")
    except Exception as e:
        errors.append(f"Metadata load error: {e}")

    # Load saved metrics
    try:
        metrics_path = os.path.join(artifact_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            artifacts["metrics_df"] = pd.read_csv(metrics_path, index_col=0)
        else:
            artifacts["metrics_df"] = None
    except Exception:
        artifacts["metrics_df"] = None

    # Load test predictions CSV (saved during training)
    try:
        preds_path = os.path.join(artifact_dir, "test_predictions.csv")
        if os.path.exists(preds_path):
            artifacts["test_preds_df"] = pd.read_csv(preds_path, index_col=0, parse_dates=True)
        else:
            artifacts["test_preds_df"] = None
    except Exception:
        artifacts["test_preds_df"] = None

    return artifacts, errors


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str, years: int = 5):
    """Download fresh stock data (cached for 1 hour)."""
    try:
        sys.path.insert(0, ".")
        from src.data_loader import download_stock_data
        from src.feature_engineering import add_technical_indicators
        df_raw = download_stock_data(ticker, years=years, save_dir=os.path.join("data", "raw"))
        if df_raw is None:
            return None, None, f"Could not download data for '{ticker}'. Check ticker symbol."
        df_feat = add_technical_indicators(df_raw)
        return df_raw, df_feat, None
    except Exception as e:
        return None, None, str(e)


def make_prediction(artifacts: dict, df_feat: pd.DataFrame, model_name: str) -> Optional[float]:
    """Run inference for one model. Returns predicted price or None."""
    try:
        sys.path.insert(0, ".")
        feature_cols = artifacts.get("feature_cols", [])
        feat_scaler = artifacts.get("feature_scaler")
        close_scaler = artifacts.get("close_scaler")

        if not feature_cols or feat_scaler is None:
            return None

        if model_name == "lstm" and artifacts.get("lstm_model") is not None:
            from src.sequence_builder import build_inference_sequence
            X = build_inference_sequence(df_feat, feat_scaler, feature_cols, LOOK_BACK)
            scaled_pred = artifacts["lstm_model"].predict(X, verbose=0)
            return float(close_scaler.inverse_transform(scaled_pred)[0][0])

        elif model_name == "rf" and artifacts.get("rf_model") is not None:
            from src.sequence_builder import build_tabular_inference_row
            X = build_tabular_inference_row(df_feat, feat_scaler, feature_cols, N_LAG_CLOSE)
            return float(artifacts["rf_model"].predict(X)[0])

        elif model_name == "lr" and artifacts.get("lr_model") is not None:
            from src.sequence_builder import build_tabular_inference_row
            X = build_tabular_inference_row(df_feat, feat_scaler, feature_cols, N_LAG_CLOSE)
            return float(artifacts["lr_model"].predict(X)[0])

    except Exception as e:
        logger.error("Prediction error (%s): %s", model_name, e)

    return None


def next_trading_day(last_date: pd.Timestamp) -> str:
    next_d = last_date + timedelta(days=1)
    while next_d.weekday() >= 5:
        next_d += timedelta(days=1)
    return next_d.strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Stockmarket.svg/320px-Stockmarket.svg.png", width=80)
    st.title("📈 Stock Predictor")
    st.markdown("---")

    ticker_input = st.text_input(
        "Ticker Symbol",
        value=TRAINED_TICKER,
        max_chars=10,
        help="Enter a valid Yahoo Finance ticker (e.g. AAPL, MSFT, TSLA).",
        key="ticker_input",
    ).strip().upper()

    predict_button = st.button("🔮 Run Prediction", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("**⚠️ Disclaimer**")
    st.markdown(
        "This tool is for **educational purposes only**. Predictions are not "
        "investment advice. Market prices are inherently uncertain.",
        help="Past performance does not guarantee future results.",
    )
    st.markdown("---")
    st.markdown(f"**Trained on:** `{TRAINED_TICKER}`")
    st.markdown(f"**Models:** LSTM · RF · LR")
    st.caption("Data: Yahoo Finance (delayed)")


# --------------------------------------------------------------------------
# Main content
# --------------------------------------------------------------------------
st.title("📊 Stock Price Prediction Dashboard")
st.markdown(
    "Uses LSTM, Random Forest, and Linear Regression models to forecast the "
    "**next trading day's closing price**."
)

# Disclaimer banner
st.markdown(
    '<div class="disclaimer">'
    "⚠️ <strong>Educational Project:</strong> Market predictions are inherently uncertain. "
    "This is not financial advice and should not be used to make investment decisions. "
    "Yahoo Finance data may be delayed."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# --------------------------------------------------------------------------
# Load artifacts (always load, regardless of ticker)
# --------------------------------------------------------------------------
artifacts, artifact_errors = load_all_artifacts(ARTIFACT_DIR, TRAINED_TICKER)

if artifact_errors:
    for err in artifact_errors:
        if "not found" in err.lower():
            st.warning(f"⚠️ {err}")
        else:
            st.error(f"❌ {err}")

artifacts_ok = (
    artifacts.get("feature_scaler") is not None
    and artifacts.get("feature_cols")
)

# --------------------------------------------------------------------------
# Ticker compatibility check
# --------------------------------------------------------------------------
if ticker_input != TRAINED_TICKER:
    st.warning(
        f"⚠️ **Ticker mismatch**: Models were trained on **{TRAINED_TICKER}** data. "
        f"Applying them to **{ticker_input}** may produce unreliable predictions. "
        "The same preprocessing pipeline (scaler) will be used, which was fitted "
        f"on {TRAINED_TICKER} prices. Proceed with caution."
    )

# --------------------------------------------------------------------------
# Data download
# --------------------------------------------------------------------------
if predict_button or "df_feat" not in st.session_state:
    with st.spinner(f"Downloading {ticker_input} data from Yahoo Finance..."):
        df_raw, df_feat, data_error = fetch_stock_data(ticker_input)
    if data_error:
        st.error(f"❌ Data error: {data_error}")
        st.stop()
    st.session_state["df_raw"] = df_raw
    st.session_state["df_feat"] = df_feat
    st.session_state["ticker"] = ticker_input
else:
    df_raw = st.session_state.get("df_raw")
    df_feat = st.session_state.get("df_feat")

if df_raw is None or df_feat is None:
    st.info("👈 Click **Run Prediction** in the sidebar to start.")
    st.stop()

last_close = float(df_feat["Close"].iloc[-1])
last_date = df_feat.index[-1]
pred_date_str = next_trading_day(last_date)

# --------------------------------------------------------------------------
# Run predictions
# --------------------------------------------------------------------------
lstm_pred, rf_pred, lr_pred = None, None, None
if artifacts_ok:
    with st.spinner("Running model predictions..."):
        lstm_pred = make_prediction(artifacts, df_feat, "lstm")
        rf_pred = make_prediction(artifacts, df_feat, "rf")
        lr_pred = make_prediction(artifacts, df_feat, "lr")

# --------------------------------------------------------------------------
# Metric cards
# --------------------------------------------------------------------------
st.subheader(f"📌 {ticker_input} — Predictions for {pred_date_str}")
st.caption(f"Last observed close: **${last_close:,.2f}** on {last_date.strftime('%Y-%m-%d')}")

col_last, col_lstm, col_rf, col_lr = st.columns(4)

with col_last:
    st.metric("Last Close", f"${last_close:,.2f}", delta=None)

def metric_col(col, label, pred):
    with col:
        if pred is not None:
            delta = pred - last_close
            delta_pct = (delta / last_close) * 100
            st.metric(
                label,
                f"${pred:,.2f}",
                delta=f"{delta:+.2f} ({delta_pct:+.2f}%)",
            )
        else:
            st.metric(label, "N/A", delta="Model not loaded")

metric_col(col_lstm, "🧠 LSTM Prediction", lstm_pred)
metric_col(col_rf, "🌲 Random Forest", rf_pred)
metric_col(col_lr, "📐 Linear Regression", lr_pred)

st.markdown("---")

# --------------------------------------------------------------------------
# Historical Price Chart
# --------------------------------------------------------------------------
st.subheader("📈 Historical Closing Price")

fig_hist = go.Figure()
fig_hist.add_trace(
    go.Scatter(
        x=df_raw.index,
        y=df_raw["Close"],
        mode="lines",
        name="Close Price",
        line=dict(color="#2196F3", width=1.5),
    )
)
# Add MA20 if available
if "MA20" in df_feat.columns:
    fig_hist.add_trace(
        go.Scatter(
            x=df_feat.index,
            y=df_feat["MA20"],
            mode="lines",
            name="MA20",
            line=dict(color="#FF9800", width=1, dash="dash"),
        )
    )
fig_hist.update_layout(
    title=f"{ticker_input} — 5-Year Daily Closing Price",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400,
)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------
# Actual vs Predicted (Test Period) — from saved CSV
# --------------------------------------------------------------------------
st.subheader("📊 Actual vs. Predicted Prices (Test Period)")

test_preds_df = artifacts.get("test_preds_df")
if test_preds_df is not None:
    fig_pred = go.Figure()
    fig_pred.add_trace(
        go.Scatter(x=test_preds_df.index, y=test_preds_df["Actual"],
                   name="Actual", line=dict(color="black", width=2))
    )
    model_colors = {"LSTM": "#2196F3", "Random Forest": "#4CAF50", "Linear Regression": "#FF5722"}
    for col, color in model_colors.items():
        if col in test_preds_df.columns:
            fig_pred.add_trace(
                go.Scatter(x=test_preds_df.index, y=test_preds_df[col],
                           name=col, line=dict(color=color, width=1.5, dash="dash"))
            )
    fig_pred.update_layout(
        title=f"{TRAINED_TICKER} — Test Period: Actual vs. Predicted",
        xaxis_title="Date", yaxis_title="Price ($)",
        hovermode="x unified", height=400,
    )
    st.plotly_chart(fig_pred, use_container_width=True)
else:
    st.info("Test-period prediction data not available. Run the notebook to generate it.")

st.markdown("---")

# --------------------------------------------------------------------------
# Model Metrics Table
# --------------------------------------------------------------------------
st.subheader("📋 Model Evaluation Metrics (Test Period)")

metrics_df = artifacts.get("metrics_df")
if metrics_df is not None:
    st.dataframe(
        metrics_df.style.format("{:.4f}").highlight_min(axis=0, color="#c8e6c9"),
        use_container_width=True,
    )
    # Bar chart
    if "RMSE ($)" in metrics_df.columns or "RMSE" in metrics_df.columns:
        rmse_col = "RMSE ($)" if "RMSE ($)" in metrics_df.columns else "RMSE"
        fig_rmse = px.bar(
            metrics_df.reset_index(),
            x="Model" if "Model" in metrics_df.reset_index().columns else metrics_df.index.name,
            y=rmse_col,
            color=rmse_col,
            color_continuous_scale="Blues_r",
            title="Model Comparison — RMSE ($)",
        )
        fig_rmse.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_rmse, use_container_width=True)
else:
    st.info("Metrics not available. Run the notebook to generate them.")

st.markdown("---")

# --------------------------------------------------------------------------
# Feature Importance (RF)
# --------------------------------------------------------------------------
st.subheader("🌲 Random Forest Feature Importance")

feat_importance_path = os.path.join(ARTIFACT_DIR, "feature_importance.json")
if os.path.exists(feat_importance_path):
    with open(feat_importance_path) as f:
        feat_imp = json.load(f)
    feat_names = list(feat_imp.keys())[:15]
    feat_vals = list(feat_imp.values())[:15]

    fig_fi = go.Figure(
        go.Bar(
            x=feat_vals[::-1], y=feat_names[::-1],
            orientation="h",
            marker_color="#4CAF50",
        )
    )
    fig_fi.update_layout(
        title=f"{TRAINED_TICKER} — Random Forest Feature Importance",
        xaxis_title="Importance",
        height=400,
    )
    st.plotly_chart(fig_fi, use_container_width=True)
else:
    st.info("Feature importance not available. Run the notebook to generate it.")

st.markdown("---")

# --------------------------------------------------------------------------
# Technical Indicators Chart
# --------------------------------------------------------------------------
st.subheader("📉 Technical Indicators")

tab_rsi, tab_bb, tab_ma = st.tabs(["RSI", "Bollinger Bands", "Moving Averages"])

with tab_rsi:
    if "RSI" in df_feat.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_feat.index, y=df_feat["RSI"], name="RSI", line=dict(color="#9C27B0")))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.update_layout(title="14-Period RSI", height=300)
        st.plotly_chart(fig_rsi, use_container_width=True)

with tab_bb:
    if "BB_Upper" in df_feat.columns:
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df_feat.index, y=df_feat["Close"], name="Close", line=dict(color="#2196F3")))
        fig_bb.add_trace(go.Scatter(x=df_feat.index, y=df_feat["BB_Upper"], name="BB Upper", line=dict(color="red", dash="dash")))
        fig_bb.add_trace(go.Scatter(x=df_feat.index, y=df_feat["BB_Lower"], name="BB Lower", line=dict(color="green", dash="dash")))
        fig_bb.update_layout(title="Bollinger Bands (20-day, 2σ)", height=350)
        st.plotly_chart(fig_bb, use_container_width=True)

with tab_ma:
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df_feat.index, y=df_feat["Close"], name="Close", line=dict(color="#2196F3")))
    if "MA5" in df_feat.columns:
        fig_ma.add_trace(go.Scatter(x=df_feat.index, y=df_feat["MA5"], name="MA5", line=dict(color="#FF9800", dash="dot")))
    if "MA20" in df_feat.columns:
        fig_ma.add_trace(go.Scatter(x=df_feat.index, y=df_feat["MA20"], name="MA20", line=dict(color="#F44336", dash="dash")))
    fig_ma.update_layout(title="Moving Averages (5-day & 20-day)", height=350)
    st.plotly_chart(fig_ma, use_container_width=True)

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"Data retrieved from Yahoo Finance · Last data point: {last_date.strftime('%Y-%m-%d')} · "
    f"Artifact trained on: {TRAINED_TICKER} · "
    f"App generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
)
st.caption(
    "⚠️ **Not financial advice.** This is an educational machine learning demonstration. "
    "Market prices are highly uncertain and past patterns do not guarantee future results."
)
