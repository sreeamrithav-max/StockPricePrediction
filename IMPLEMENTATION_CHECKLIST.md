# IMPLEMENTATION_CHECKLIST.md

## Stock Price Prediction — Implementation Checklist

Tracking all requirements from SPEC.md against implementation files.

---

## Phase 1 — Workspace & Data

| Requirement | File | Status |
|---|---|---|
| Project folder structure created | `data/`, `src/`, `artifacts/`, `reports/`, `notebooks/`, `tests/` | ✅ |
| `requirements.txt` | `requirements.txt` | ✅ |
| `.gitignore` | `.gitignore` | ✅ |
| `DECISIONS.md` | `DECISIONS.md` | ✅ |
| Data download function (yfinance, dynamic 5-year range) | `src/data_loader.py`, notebook §2 | ✅ |
| Handle MultiIndex columns in yfinance | `src/data_loader.py` | ✅ |
| Normalize column names, sort dates ascending | `src/data_loader.py` | ✅ |
| Remove duplicate dates | `src/data_loader.py` | ✅ |
| Validate required columns, numeric types | `src/data_loader.py` | ✅ |
| Handle invalid ticker / network error | `src/data_loader.py` | ✅ |
| Save raw CSV to `data/raw/<TICKER>_daily.csv` | `src/data_loader.py` | ✅ |
| Never fabricate weekend/holiday rows | `src/data_loader.py` | ✅ |
| Test: valid/invalid ticker | `tests/test_data_loader.py` | ✅ |
| Test: empty/missing/malformed data | `tests/test_data_loader.py` | ✅ |
| Test: date ordering and duplicate removal | `tests/test_data_loader.py` | ✅ |

## Phase 2 — Feature Engineering & Preprocessing

| Requirement | File | Status |
|---|---|---|
| MA5 — 5-day moving average of Close | `src/feature_engineering.py`, notebook §5 | ✅ |
| MA20 — 20-day moving average of Close | `src/feature_engineering.py`, notebook §5 | ✅ |
| RSI — 14-period RSI | `src/feature_engineering.py`, notebook §5 | ✅ |
| BB_Upper, BB_Lower — 20-day Bollinger Bands | `src/feature_engineering.py`, notebook §5 | ✅ |
| Daily_Return — pct change of Close | `src/feature_engineering.py`, notebook §5 | ✅ |
| Price_to_Volume_Ratio — Close/(Volume+1) | `src/feature_engineering.py`, notebook §5 | ✅ |
| Drop warm-up NaN rows (not fill) | `src/feature_engineering.py` | ✅ |
| No future data in features | `src/feature_engineering.py` | ✅ |
| Save processed CSV to `data/processed/` | notebook §5 | ✅ |
| Chronological 70/10/20 split | `src/preprocessing.py`, notebook §6 | ✅ |
| MinMaxScaler fitted on train only | `src/preprocessing.py`, notebook §6 | ✅ |
| Dedicated Close scaler for LSTM target | `src/preprocessing.py`, notebook §6 | ✅ |
| No shuffle of time series | `src/preprocessing.py` | ✅ |
| 60-day LSTM sequence builder | `src/sequence_builder.py`, notebook §6 | ✅ |
| RF/LR tabular lag features | `src/preprocessing.py`, notebook §8-9 | ✅ |
| Test: feature calculations | `tests/test_features.py` | ✅ |
| Test: no future data in sequences | `tests/test_preprocessing.py` | ✅ |
| Test: chronological split boundaries | `tests/test_preprocessing.py` | ✅ |
| Test: scaler fitted only on training data | `tests/test_preprocessing.py` | ✅ |
| Test: LSTM tensor shape (samples, 60, features) | `tests/test_preprocessing.py` | ✅ |

## Phase 3 — Models

| Requirement | File | Status |
|---|---|---|
| LSTM: TF/Keras, 60-day window, specified arch | `src/train_lstm.py`, notebook §7 | ✅ |
| LSTM: Adam optimizer, MSE loss | `src/train_lstm.py` | ✅ |
| LSTM: batch=32, max 50 epochs, early stopping (patience=10) | `src/train_lstm.py` | ✅ |
| LSTM: validation from pre-test period | `src/train_lstm.py` | ✅ |
| LSTM: fixed random seeds | `src/train_lstm.py` | ✅ |
| Random Forest: sklearn, n_estimators=200 | `src/train_baselines.py`, notebook §8 | ✅ |
| Random Forest: lag features, trained on train only | `src/train_baselines.py` | ✅ |
| RF: feature importance generated | `src/train_baselines.py` | ✅ |
| Linear Regression: sklearn, same lag features | `src/train_baselines.py`, notebook §9 | ✅ |
| LR: baseline predicts same next-day Close target | `src/train_baselines.py` | ✅ |
| Save LSTM as .keras | `src/train_lstm.py`, notebook §12 | ✅ |
| Save RF and LR as .joblib | `src/train_baselines.py`, notebook §12 | ✅ |
| Save scalers as .joblib | notebook §12 | ✅ |
| Save feature column metadata as JSON | notebook §12 | ✅ |
| Test: model saving and loading | `tests/test_prediction.py` | ✅ |

## Phase 4 — Evaluation

| Requirement | File | Status |
|---|---|---|
| RMSE on original price scale | `src/evaluate.py`, notebook §10 | ✅ |
| MAE on original price scale | `src/evaluate.py`, notebook §10 | ✅ |
| R² on original price scale | `src/evaluate.py`, notebook §10 | ✅ |
| All models evaluated on same test dates | `src/evaluate.py` | ✅ |
| Actual vs. predicted chart | `src/evaluate.py`, notebook §11 | ✅ |
| Model comparison table and bar chart | `src/evaluate.py`, notebook §10 | ✅ |
| LSTM training/validation loss curve | notebook §7 | ✅ |
| RF feature importance chart | notebook §11 | ✅ |
| Residual plot | notebook §11 | ✅ |
| Save metrics CSV | notebook §12 | ✅ |
| Save all plots | notebook §12 | ✅ |
| No hardcoded reference report metrics | `src/evaluate.py` | ✅ |

## Phase 5 — Inference

| Requirement | File | Status |
|---|---|---|
| Download fresh data for inference | `src/predict.py`, notebook §13 | ✅ |
| Recalculate same features in same order | `src/predict.py` | ✅ |
| Validate sufficient history exists | `src/predict.py` | ✅ |
| Apply saved scalers without refitting | `src/predict.py` | ✅ |
| Build correct 60-day input sequence | `src/predict.py` | ✅ |
| Predict next eligible trading day Close | `src/predict.py`, notebook §13 | ✅ |
| Inverse-transform target correctly | `src/predict.py` | ✅ |
| Show last observed close + forecast | notebook §13 | ✅ |
| Handle errors gracefully | `src/predict.py` | ✅ |
| Test: correct inverse transformation | `tests/test_prediction.py` | ✅ |
| Test: inference with insufficient history | `tests/test_prediction.py` | ✅ |

## Phase 6 — Streamlit App

| Requirement | File | Status |
|---|---|---|
| Ticker input | `app.py` | ✅ |
| Latest Close displayed | `app.py` | ✅ |
| Predicted next-day Close displayed | `app.py` | ✅ |
| Predicted change (abs + %) | `app.py` | ✅ |
| Historical price chart | `app.py` | ✅ |
| Actual vs. predicted chart | `app.py` | ✅ |
| Model comparison metrics table | `app.py` | ✅ |
| Training loss chart | `app.py` | ✅ |
| Feature importance chart | `app.py` | ✅ |
| Error messages and financial disclaimer | `app.py` | ✅ |
| Loads saved artifacts, no mock predictions | `app.py` | ✅ |
| Ticker not in training: shows limitation message | `app.py` | ✅ |

## Phase 7 — Documentation & Final

| Requirement | File | Status |
|---|---|---|
| README.md with workflow, limitations, commands | `README.md` | ✅ |
| Colab setup and run instructions | `README.md`, notebook §1 | ✅ |
| Streamlit launch commands | `README.md` | ✅ |
| .gitignore | `.gitignore` | ✅ |
| All tests passing locally | `tests/` | ✅ (run report in README) |
| Notebook self-contained, top-to-bottom runnable | notebook | ✅ |
| No TODO placeholders in required functionality | all files | ✅ |
