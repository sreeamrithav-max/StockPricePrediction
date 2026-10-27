# DECISIONS.md — Technical Decisions and Assumptions

## Project: Stock Price Prediction Using Machine Learning

---

## D1 — Primary Execution Environment
**Decision:** Google Colab with all code self-contained in the notebook.  
**Rationale:** Spec mandates Colab. The notebook imports from inline-defined helper functions so that no file uploads are required. Supporting `src/` modules replicate the same logic for local use.

## D2 — Data Source and Date Range
**Decision:** `yfinance` with a dynamic 5-year lookback ending at `datetime.today()`.  
**Rationale:** Spec section 4.1 explicitly states "retrieve the most recent available five-year period rather than hardcoding an expired end date."

## D3 — Price Column — Adjusted vs. Unadjusted
**Decision:** Use the raw `Close` column as returned by yfinance (auto-adjusted=True by default in recent yfinance versions), which provides split/dividend-adjusted prices.  
**Rationale:** Spec section 4.2 says "adjusted-price convention consistently." We document the choice and apply it uniformly.

## D4 — Missing Value Strategy
**Decision:** Drop NaN rows that result from rolling-window warm-up periods (MA20 requires 20 rows, RSI requires 14, BB requires 20). No forward-fill fabrication.  
**Rationale:** Spec section 4.3 prohibits forward/backward fill to fabricate observations. Warm-up rows are dropped and documented.

## D5 — Feature Set
**Decision:** Features used for all models: `['Open','High','Low','Close','Volume','MA5','MA20','RSI','BB_Upper','BB_Lower','Daily_Return','Price_to_Volume_Ratio']`  
**Rationale:** Exactly the features listed in Spec section 5.

## D6 — Price_to_Volume_Ratio Definition
**Decision:** `Price_to_Volume_Ratio = Close / (Volume + 1)` — adding 1 avoids division by zero.  
**Rationale:** Spec section 5 requires "numerically safe, documented definition; guard against zero volume."

## D7 — Train / Validation / Test Split
**Decision:** Chronological split — 70% train, 10% validation, 20% test.  
**Rationale:** Spec section 6 mandates "final 20% as held-out test" and "validation from pre-test period." 70/10/20 satisfies both constraints.

## D8 — Scaling Strategy
**Decision:** `MinMaxScaler` fitted on training features only. A **separate** `MinMaxScaler` is fitted on the training `Close` column alone for LSTM target inverse-transformation.  
**Rationale:** Spec section 6 and section 7.1 require a "dedicated Close scaler so inverse transformation is correct and simple."

## D9 — LSTM Look-Back Window
**Decision:** 60 trading days as specified.  
**Rationale:** Spec section 6 states "60-trading-day look-back is the default."

## D10 — LSTM Architecture
**Decision:** LSTM(128, return_sequences=True) -> Dropout(0.2) -> LSTM(64) -> Dropout(0.2) -> Dense(32, relu) -> Dense(1)  
**Rationale:** Exact architecture from Spec section 7.1.

## D11 — Random Forest Features
**Decision:** Tabular representation using lag features (last 5 days of Close + last observation's full feature vector).  
**Rationale:** Spec section 7.2 requires "engineered lag features / a defined tabular representation."

## D12 — Linear Regression Baseline
**Decision:** Same tabular lag-feature representation as Random Forest.  
**Rationale:** Spec section 7.3 requires "lagged Close and selected past-only features." Using same features as RF ensures a fair comparison.

## D13 — Model Serialization
**Decision:** LSTM saved as `.keras` (TF SavedModel format). RF and LR saved as `.joblib` files. Scalers as `.joblib`.  
**Rationale:** Most robust formats for each library. `.keras` is the recommended TF2 format; `joblib` is the scikit-learn standard.

## D14 — Evaluation Scale
**Decision:** All metrics (RMSE, MAE, R squared) calculated on the original dollar price scale after inverse-transforming predictions.  
**Rationale:** Spec section 8 requires "original price scale."

## D15 — Streamlit App Inference
**Decision:** `app.py` loads saved artifacts from `artifacts/AAPL/` and re-uses the exact same preprocessing pipeline defined in `src/`.  
**Rationale:** Spec section 10 requires "actual saved models and same preprocessing pipeline."

## D16 — Google Drive (Colab)
**Decision:** Optional Drive mounting at the start of the notebook. If Drive is not mounted, all outputs go to Colab's local `/content/` tree.  
**Rationale:** Spec section 7 states "notebook must also work with Colab's temporary storage."

## D17 — Seed
**Decision:** `RANDOM_SEED = 42` applied to numpy, random, and TensorFlow.  
**Rationale:** Spec section 7.1 requires "fixed random seeds where practical."

## D18 — RSI Period
**Decision:** 14-period RSI using Wilder's smoothing (EMA-based).  
**Rationale:** Spec section 5 explicitly documents "14-period RSI."

## D19 — Bollinger Bands
**Decision:** 20-day window, 2 standard deviations.  
**Rationale:** Industry standard; warm-up aligns with MA20.

## D20 — Tests
**Decision:** Tests in `tests/` run via `pytest`. They use small synthetic data frames so they can run locally without network access.  
**Rationale:** Spec section 12 lists required test cases. Tests must be runnable offline.
