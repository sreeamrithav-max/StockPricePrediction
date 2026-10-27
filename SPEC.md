Stock Price Prediction Using Machine Learning
Master Specification for Google Antigravity
Project type: Machine Learning Project-Based Learning (PBL)
Primary environment: Google Colab
Application: Streamlit
Primary data source: Yahoo Finance via yfinance
Reference: User-provided report, Stock Price Prediction Using
Machine Learning
1. Project objective
Build a complete, reproducible machine learning project that predicts
the next trading day's closing price for a user-selected stock using
historical market data. Compare an LSTM neural network against Linear
Regression and Random Forest Regressor. Present predictions, model
metrics, and charts in an interactive Streamlit application.

This is an educational forecasting project, not a financial-advice or
trading system. Never promise accuracy or profitability.
2. Scope and required outputs
The finished project must include:
1. Historical market-data collection and validation.
2. Data cleaning and feature engineering.
3. Leakage-safe chronological train/validation/test splitting.
4. Three trained regression models:
   - LSTM
   - Random Forest Regressor
   - Linear Regression
5. Consistent evaluation on the same held-out test period.
6. Saved model artifacts and preprocessing objects.
7. Visualizations: historical prices, predicted vs. actual prices,
   training/validation loss, and Random Forest feature importance where
   applicable.
8. Streamlit application accepting a ticker and displaying the latest
   available price, predicted next-trading-day close, charts, and
   evaluation information.
9. Automated tests, requirements file, README, and clear Colab
   instructions.
10. No fake predictions, fabricated metrics, or unfinished placeholder
    code.
3. Recommended execution approach
Develop and train in Google Colab. Store persistent project files in a
dedicated Google Drive folder. The notebook should be runnable
top-to-bottom from a fresh Colab runtime.

The Streamlit application must be developed as regular Python files in
the project folder. Test it from Colab only if a supported tunnel setup
is available; a tunnel is temporary and is not permanent hosting. Also
provide instructions for running the same application locally or
deploying it separately.

Use free tools and libraries where practical. Do not require paid
market-data APIs.
4. Data specification
4.1 Source and initial ticker
- Source: Yahoo Finance through yfinance.
- Default ticker: AAPL.
- Initial historical period: approximately five years of daily
  observations, consistent with the reference report's 2019--2024
  example.
- Support other valid tickers, including examples such as MSFT and
  INFY, subject to data-source availability.
- Use daily historical data. Do not describe daily closing-price
  forecasting as tick-level or intraday prediction.
Because the current date changes, implement dates dynamically: retrieve
the most recent available five-year period rather than hardcoding an
expired end date. Record the actual start and end dates returned.
4.2 Raw fields
Required fields: - Date - Open - High - Low - Close - Volume

Use the adjusted-price convention consistently. The default project
target is Close, as specified in the report. If using adjusted close
or adjusted OHLC data, document that choice and apply it consistently
throughout training and inference.
4.3 Data collection requirements
Create a reusable data-download function that: - accepts ticker,
start/end or period, and interval; - downloads daily historical data
with yfinance; - handles the MultiIndex columns that some yfinance
versions return; - normalizes column names; - sorts dates ascending; -
removes duplicate dates; - checks that required columns exist and
contain numeric values; - detects empty responses, invalid tickers,
network errors, and insufficient history; - saves a raw CSV for
reproducibility; - records ticker, retrieval timestamp, requested
period, and actual data range; - never fills weekends or exchange
holidays with artificial trading rows.

Do not use forward-fill/backward-fill to fabricate missing market
observations. Drop unusable rows or handle feature warm-up periods
appropriately, and document the choice.
4.4 Storage layout
Save data and artifacts in the project folder or a configurable Google
Drive project directory:
- data/raw/<TICKER>_daily.csv
- data/processed/<TICKER>_features.csv
- artifacts/<TICKER>/
- reports/figures/
Never commit credentials or private tokens. This project should not need
API credentials for yfinance.
5. Feature engineering
Begin with the raw OHLCV fields and calculate these report-specified
features:
- MA5: 5-trading-day moving average of Close.
- MA20: 20-trading-day moving average of Close.
- RSI: 14-period RSI, with its period explicitly documented.
- BB_Upper and BB_Lower: Bollinger Bands, using documented window
  and standard-deviation settings.
- Daily_Return: percentage change in Close.
- Price_to_Volume_Ratio: implement a numerically safe, documented
  definition; guard against zero volume and explain that it is only a
  proxy.
All rolling features must be calculated using current and past
observations only. Remove initial rows where rolling features are
undefined rather than fabricating values. Verify features do not use
future data.
6. Leakage-safe preprocessing and time splits
This section is mandatory.
1. Sort observations by date before processing.
2. Create chronological training, validation, and test partitions. The
   overall test partition should be the final 20% of eligible
   observations, consistent with the report. Reserve validation
   observations from the pre-test period for model selection and early
   stopping.
3. Fit every scaler and learned preprocessing transformation using
   training observations only.
4. Apply the fitted transformation unchanged to validation and test
   observations.
5. Do not shuffle time-series observations.
6. Generate sequence windows without allowing future target values into
   inputs.
7. Make sure the LSTM, Linear Regression, and Random Forest are
   evaluated against the same test dates and target definition.
8. Fit preprocessing objects separately for different target tickers if
   required; do not silently apply an incompatible scaler to a
   different stock.
The 60-trading-day look-back is the default. Each LSTM input must
contain the previous 60 trading days of available features and predict
the next eligible trading day's Close.

For the LSTM target, use a clearly documented target-scaling approach.
Prefer a dedicated Close scaler so inverse transformation of the
predicted target is correct and simple. Do not use the Close feature's
index from an arbitrary multivariate scaler unless the inverse
transformation is mathematically correct.
7. Model requirements
7.1 LSTM regressor
Use TensorFlow/Keras. A reasonable starting architecture, based on the
report, is:
- LSTM(128, return_sequences=True)
- Dropout(0.2)
- LSTM(64)
- Dropout(0.2)
- Dense(32, activation="relu")
- Dense(1)
Training defaults: - Adam optimizer - Mean squared error loss - Batch
size 32 - Maximum 50 epochs - Early stopping based on validation loss,
patience 10, restore best weights - Fixed random seeds where practical

These are initial settings, not guaranteed optimal values. Ensure
validation data is chronologically appropriate and is not drawn from the
held-out test set.
7.2 Random Forest Regressor
- Use scikit-learn RandomForestRegressor.
- Start with n_estimators=200 unless runtime constraints justify a
  documented adjustment.
- Use engineered lag features / a defined tabular representation of
  historical observations.
- Train only on training data and evaluate on the same held-out test
  dates.
- Generate feature importance where supported.
7.3 Linear Regression
- Use scikit-learn LinearRegression.
- Define an explicit tabular baseline using lagged Close and selected
  past-only features, or the last observation's engineered features.
- Ensure the baseline predicts the same next-day Close target and is
  evaluated on the same test dates.
Do not compare models whose target definitions, dates, or evaluation
preprocessing differ.
8. Evaluation requirements
Calculate on the original price scale: - RMSE - MAE - R²

Save a comparison CSV and display a table/bar chart. Include a
predicted-versus-actual line chart for the test period and a residual
plot if practical. For LSTM, save the training/validation loss curve.

Do not hardcode performance values from the report. Report metrics
produced by the actual run. Explain that a low price RMSE alone does not
prove a profitable strategy. If the report's example metrics are used
for discussion, label them as reference-report values, not results from
the newly trained implementation.

Optional but recommended: walk-forward validation or TimeSeriesSplit,
implemented without leakage. Avoid claiming a trading backtest or
profitability unless a real, clearly defined backtest has been
implemented.
9. Inference and saved artifacts
Save the trained model(s), feature-column order, scalers, model
configuration, and metadata. Use a robust serialization format
compatible with the selected libraries.

The inference pipeline must: 1. Download fresh historical data for the
requested ticker. 2. Recalculate the exact same features in the same
order. 3. Validate that enough valid observations exist. 4. Apply the
saved fitted transformations without refitting them. 5. Create the
correct 60-day input sequence. 6. Predict the next eligible trading
day's Close. 7. Inverse-transform the target correctly. 8. Show the last
observed close, forecast value, forecast date convention, and data
update timestamp. 9. Handle errors gracefully.

If the user enters a ticker that was not represented in training, do not
assume a single-stock model generalizes reliably to it. Either restrict
inference to the trained ticker or explicitly design and validate a
multi-ticker approach. The UI must explain the limitation.
10. Streamlit application requirements
Create a clean, responsive dashboard with:
- Project title and brief disclaimer.
- Ticker input and prediction button.
- Data-source / latest-data timestamp.
- Latest available Close and predicted next-trading-day Close.
- Predicted change in currency and percentage, calculated from the
  latest Close.
- Historical Close chart.
- Actual-vs-predicted test chart.
- Model comparison table with RMSE, MAE and R².
- Training-loss chart and feature-importance chart where relevant.
- Clear loading indicators and validation/error messages.
- A note that market predictions are uncertain and are not investment
  advice.
Do not claim the application provides exchange-real-time data. Yahoo
Finance data availability and delay depend on the instrument and source.
11. Proposed repository structure
stock-price-prediction/
├── notebooks/
│   └── stock_price_prediction_colab.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   └── AAPL/
├── reports/
│   └── figures/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── sequence_builder.py
│   ├── train_lstm.py
│   ├── train_baselines.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   └── test_prediction.py
├── app.py
├── requirements.txt
├── README.md
├── SPEC.md
└── .gitignore
Adjust this structure only when needed, and document any changes. Keep
notebook code aligned with reusable modules; avoid maintaining two
contradictory implementations.
12. Testing requirements
At minimum, test: - valid and invalid ticker handling; - empty / missing
/ malformed data; - date ordering and duplicate removal; - feature
calculations and rolling warm-up rows; - no future data in input
sequences; - chronological split boundaries; - scalers fitted only on
training data; - expected LSTM tensor shape
(samples, look_back, features); - model saving and loading; - correct
inverse transformation to price units; - inference with insufficient
history; - Streamlit input validation.

Run the tests and report the actual pass/fail status. If a test cannot
be run in the current environment, say so.
13. Acceptance criteria
The project is complete only when:
1. A fresh Colab runtime can execute the notebook from top to bottom
   using documented steps.
2. The data pipeline downloads and validates actual historical data.
3. Engineered features are calculated correctly and consistently.
4. Data leakage controls are implemented and documented.
5. All three regression models train successfully or any
   environment-specific blocker is documented.
6. All models are evaluated on identical held-out dates and the same
   price-scale target.
7. Evaluation metrics are calculated by code, not typed in manually.
8. Model artifacts and preprocessing objects can be saved and loaded.
9. A next-day prediction can be generated through the same pipeline
   used during training.
10. The Streamlit app handles valid and invalid input without
    placeholder outputs.
11. Tests and setup instructions are included.
12. The README explains the workflow, limitations, and how to reproduce
    results.
14. Implementation phases for Antigravity
Build in this order and verify each phase before continuing:

Phase 1 --- Workspace and data - Create folders, requirements, data
downloader, and data validation tests. - Download AAPL daily data and
save raw CSV. - Inspect date range, row count, missing values, and
columns.

Phase 2 --- Features and preprocessing - Implement technical
indicators. - Create chronological splits and leakage-safe scalers. -
Test features and sequence generation.

Phase 3 --- Models - Train LSTM, Random Forest, and Linear
Regression. - Save artifacts and training metadata. - Keep the final
test period untouched during model selection.

Phase 4 --- Evaluation - Calculate RMSE, MAE and R² on the original
price scale. - Generate comparison tables and plots. - Save the actual
metrics to files.

Phase 5 --- Inference - Implement one reusable prediction
pipeline. - Verify saved model/scaler loading and inverse scaling.

Phase 6 --- Streamlit - Build the dashboard using the actual
inference pipeline. - Test valid and invalid tickers and display
limitations clearly.

Phase 7 --- Verification and handoff - Run tests, fix issues, update
README, and provide exact launch commands. - Report any uncompleted
requirement honestly.
15. Instructions to the coding agent
Read the full specification before writing code. First inspect the
workspace and produce a checklist mapping each requirement to a file and
test. Then implement one phase at a time.

Do not generate the entire application as one unverified code dump. Do
not fabricate metrics, fake a trained model, leave TODO placeholders
in required functionality, or state that commands succeeded without
executing them. Ask before deleting or overwriting user data. Prefer
modular, testable code and explain any material deviations from this
specification.

At completion, provide: - a list of created/modified files; - exact
Colab setup and run steps; - exact training and Streamlit launch
commands; - actual test results; - actual evaluation metrics; - known
limitations and any incomplete work.