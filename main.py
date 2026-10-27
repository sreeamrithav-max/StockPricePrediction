"""
main.py
-------
Command-line runner for the Stock Price Prediction pipeline.

Runs the complete workflow:
  1. Data download via yfinance (or cached raw CSV)
  2. Feature engineering (technical indicators)
  3. Chronological train/val/test split & MinMaxScaler fitting
  4. Sequence creation (LSTM) and tabular feature creation (RF / LR)
  5. Training LSTM, Random Forest, and Linear Regression
  6. Evaluation on held-out test set with metrics and plots
  7. Model and artifact persistence to artifacts/<ticker>/
  8. Next-day inference, direction prediction, and trading signals

Usage:
    python main.py --ticker AAPL --epochs 25 --batch-size 32
"""

import argparse
import json
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stock_price_prediction")

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators, get_feature_columns
from src.preprocessing import (
    chronological_split,
    fit_scalers,
    scale_data,
    build_lstm_sequences,
    build_tabular_features,
    get_tabular_feature_names,
    save_scalers,
    LOOK_BACK,
    N_LAG_CLOSE,
)
from src.sequence_builder import build_inference_sequence, build_tabular_inference_row
from src.evaluate import (
    compute_metrics,
    build_metrics_table,
    plot_actual_vs_predicted,
    plot_lstm_loss,
    plot_feature_importance,
    plot_model_comparison,
    plot_residuals,
    save_metrics_csv,
)
from src.train_baselines import (
    train_random_forest,
    train_linear_regression,
    get_feature_importance,
    save_baseline_models,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Stock Price Prediction pipeline end-to-end."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (default: AAPL).",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="5y",
        help="Data history period (default: 5y).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Max training epochs for LSTM (default: 25).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for LSTM training (default: 32).",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip LSTM training (train only Random Forest and Linear Regression).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper().strip()
    logger.info("=" * 70)
    logger.info("STARTING STOCK PRICE PREDICTION PIPELINE: %s", ticker)
    logger.info("=" * 70)

    # 1. Download stock data
    logger.info("[Step 1/8] Fetching stock data for %s (%s)...", ticker, args.period)
    df_raw = download_stock_data(ticker=ticker, years=5)
    if df_raw is None or len(df_raw) < 100:
        logger.error("Failed to fetch sufficient data for %s. Aborting.", ticker)
        sys.exit(1)
    logger.info(
        "Downloaded %d trading days (%s to %s).",
        len(df_raw),
        df_raw.index[0].date(),
        df_raw.index[-1].date(),
    )

    # 2. Feature Engineering
    logger.info("[Step 2/8] Engineering technical indicators...")
    df_feat = add_technical_indicators(df_raw)
    feature_cols = get_feature_columns()
    logger.info(
        "Features engineered. %d rows remaining after warm-up drop.", len(df_feat)
    )

    # 3. Train / Val / Test Split
    logger.info("[Step 3/8] Performing chronological 70/10/20 train/val/test split...")
    df_train, df_val, df_test = chronological_split(df_feat)
    logger.info(
        "Split sizes: Train=%d, Val=%d, Test=%d",
        len(df_train),
        len(df_val),
        len(df_test),
    )

    # 4. Scalers (fitted strictly on training set)
    logger.info("[Step 4/8] Fitting MinMaxScaler on training data only...")
    feat_scaler, close_scaler = fit_scalers(df_train, feature_cols)
    scaled_train = scale_data(df_train, feat_scaler, feature_cols)
    scaled_val = scale_data(df_val, feat_scaler, feature_cols)
    scaled_test = scale_data(df_test, feat_scaler, feature_cols)

    # Save scalers
    artifacts_dir = os.path.join("artifacts", ticker)
    os.makedirs(artifacts_dir, exist_ok=True)
    save_scalers(feat_scaler, close_scaler, artifacts_dir)

    # 5. Build Sequences & Tabular Datasets
    logger.info("[Step 5/8] Building LSTM sequences & tabular features...")
    X_train_lstm, y_train_lstm = build_lstm_sequences(
        scaled_train, close_scaler, df_train["Close"].values, LOOK_BACK
    )
    X_val_lstm, y_val_lstm = build_lstm_sequences(
        scaled_val, close_scaler, df_val["Close"].values, LOOK_BACK
    )
    X_test_lstm, y_test_lstm = build_lstm_sequences(
        scaled_test, close_scaler, df_test["Close"].values, LOOK_BACK
    )

    X_train_tab, y_train_tab = build_tabular_features(
        df_train, feat_scaler, feature_cols, N_LAG_CLOSE
    )
    X_val_tab, y_val_tab = build_tabular_features(
        df_val, feat_scaler, feature_cols, N_LAG_CLOSE
    )
    X_test_tab, y_test_tab = build_tabular_features(
        df_test, feat_scaler, feature_cols, N_LAG_CLOSE
    )
    tab_names = get_tabular_feature_names(feature_cols, N_LAG_CLOSE)

    # Align test predictions across models
    tab_offset = LOOK_BACK - N_LAG_CLOSE
    X_test_tab_aligned = X_test_tab[tab_offset:]
    actual_test_close = df_test["Close"].iloc[LOOK_BACK:].values
    test_dates = df_test.index[LOOK_BACK:].to_numpy()

    # 6. Model Training
    models = {}
    test_predictions = {}
    history = None

    # LSTM
    if not args.skip_lstm:
        try:
            from src.train_lstm import (
                build_lstm_model,
                train_lstm,
                save_lstm_model,
            )

            logger.info("[Step 6a/8] Building and training 2-layer LSTM...")
            lstm_model = build_lstm_model(
                look_back=LOOK_BACK,
                n_features=len(feature_cols),
            )
            history = train_lstm(
                lstm_model,
                X_train_lstm,
                y_train_lstm,
                X_val_lstm,
                y_val_lstm,
                batch_size=args.batch_size,
                max_epochs=args.epochs,
            )
            save_lstm_model(lstm_model, artifacts_dir, ticker)
            models["LSTM"] = lstm_model

            # Predict on test set
            pred_lstm_scaled = lstm_model.predict(X_test_lstm, verbose=0)
            pred_lstm = close_scaler.inverse_transform(pred_lstm_scaled).flatten()
            test_predictions["LSTM"] = pred_lstm
        except Exception as e:
            logger.warning(
                "Could not train LSTM (%s). Proceeding with baseline models.", e
            )
    else:
        logger.info("[Step 6a/8] Skipping LSTM as requested (--skip-lstm).")

    # Random Forest
    logger.info("[Step 6b/8] Training Random Forest Regressor (200 trees)...")
    rf_model = train_random_forest(X_train_tab, y_train_tab)
    models["Random Forest"] = rf_model
    pred_rf = rf_model.predict(X_test_tab_aligned)
    test_predictions["Random Forest"] = pred_rf

    # Linear Regression
    logger.info("[Step 6c/8] Training Linear Regression baseline...")
    lr_model = train_linear_regression(X_train_tab, y_train_tab)
    models["Linear Regression"] = lr_model
    pred_lr = lr_model.predict(X_test_tab_aligned)
    test_predictions["Linear Regression"] = pred_lr

    # Save baselines
    save_baseline_models(rf_model, lr_model, artifacts_dir, ticker)

    # 7. Evaluation & Metrics
    logger.info("[Step 7/8] Evaluating models on held-out test set...")
    metrics_list = []
    for name, preds in test_predictions.items():
        m = compute_metrics(actual_test_close, preds, model_name=name)
        metrics_list.append(m)

    # Print formatted metrics table
    metrics_df = build_metrics_table(metrics_list)
    print("\n" + "=" * 55)
    print(f" MODEL EVALUATION METRICS (TEST SET: {len(actual_test_close)} TRADING DAYS)")
    print("=" * 55)
    print(metrics_df.to_string())
    print("=" * 55 + "\n")

    # Generate and save report plots
    figures_dir = os.path.join("reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_actual_vs_predicted(
        dates=test_dates,
        y_true=actual_test_close,
        predictions=test_predictions,
        ticker=ticker,
        save_path=os.path.join(figures_dir, f"{ticker}_predictions_comparison.png"),
    )

    plot_model_comparison(
        metrics_df=metrics_df,
        ticker=ticker,
        save_path=os.path.join(figures_dir, f"{ticker}_model_comparison.png"),
    )

    # Primary model residuals
    primary_model = "LSTM" if "LSTM" in test_predictions else "Random Forest"
    plot_residuals(
        dates=test_dates,
        y_true=actual_test_close,
        y_pred=test_predictions[primary_model],
        model_name=primary_model,
        save_path=os.path.join(figures_dir, f"{ticker}_residuals.png"),
    )

    if history is not None:
        plot_lstm_loss(
            history,
            save_path=os.path.join(figures_dir, f"{ticker}_learning_curves.png"),
        )

    rf_importance = get_feature_importance(rf_model, tab_names)
    plot_feature_importance(
        rf_importance,
        top_n=15,
        ticker=ticker,
        save_path=os.path.join(figures_dir, f"{ticker}_feature_importance.png"),
    )

    save_metrics_csv(
        metrics_df, os.path.join(figures_dir, f"{ticker}_metrics.csv")
    )
    # Also save metrics.csv and test_predictions.csv to artifacts_dir for Streamlit dashboard
    metrics_df.to_csv(os.path.join(artifacts_dir, "metrics.csv"))
    test_preds_df = pd.DataFrame(
        {"Actual": actual_test_close, **test_predictions},
        index=test_dates,
    )
    test_preds_df.index.name = "Date"
    test_preds_df.to_csv(os.path.join(artifacts_dir, "test_predictions.csv"))
    logger.info("Saved all evaluation figures to %s", figures_dir)

    # Save metadata.json
    metadata = {
        "ticker": ticker,
        "feature_cols": feature_cols,
        "look_back": LOOK_BACK,
        "n_lag_close": N_LAG_CLOSE,
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "test_rows": len(df_test),
        "metrics": metrics_list,
    }
    with open(os.path.join(artifacts_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", os.path.join(artifacts_dir, "metadata.json"))

    # 8. Next-Day Prediction & Trading Signal
    logger.info("[Step 8/8] Computing next-day inference & trading signals...")
    last_actual_close = float(df_feat["Close"].iloc[-1])
    last_date = df_feat.index[-1].strftime("%Y-%m-%d")

    print("\n" + "=" * 75)
    print(
        f" NEXT-DAY PREDICTION FOR {ticker} (Last Date: {last_date}, Last Close: ${last_actual_close:.2f})"
    )
    print("=" * 75)

    for name in test_predictions.keys():
        if name == "LSTM" and "LSTM" in models:
            X_infer = build_inference_sequence(
                df_feat, feat_scaler, feature_cols, LOOK_BACK
            )
            scaled_next = models["LSTM"].predict(X_infer, verbose=0)
            pred_price = float(close_scaler.inverse_transform(scaled_next)[0][0])
        else:
            model = models[name]
            X_tab_infer = build_tabular_inference_row(
                df_feat, feat_scaler, feature_cols, N_LAG_CLOSE
            )
            pred_price = float(model.predict(X_tab_infer)[0])

        pct_change = ((pred_price - last_actual_close) / last_actual_close) * 100
        direction = "UP" if pct_change > 0 else "DOWN"
        if abs(pct_change) > 1.0:
            signal = "BUY" if direction == "UP" else "SELL"
            conf = "HIGH"
        elif abs(pct_change) > 0.3:
            signal = "BUY" if direction == "UP" else "SELL"
            conf = "MODERATE"
        else:
            signal = "HOLD"
            conf = "LOW"

        print(
            f"[{name:18s}] Predicted: ${pred_price:.2f} | "
            f"Expected Change: {pct_change:+.2f}% | "
            f"Direction: {direction:4s} | "
            f"Signal: {signal:4s} | "
            f"Confidence: {conf}"
        )

    print("=" * 75)
    print(f"Pipeline complete! Artifacts saved to: {artifacts_dir}")
    print(f"Reports saved to: {figures_dir}")
    print("Run the dashboard: streamlit run app.py")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
