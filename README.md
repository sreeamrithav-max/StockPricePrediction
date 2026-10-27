# Stock Price Prediction Using Machine Learning

> **Educational Project** — Predicts the next trading day's closing price using LSTM, Random Forest, and Linear Regression. Not financial advice.

---

## 🎯 Project Overview

This project builds a complete, reproducible ML pipeline to forecast daily stock closing prices using:
- **LSTM** neural network (TensorFlow/Keras)
- **Random Forest Regressor** (scikit-learn)
- **Linear Regression** baseline (scikit-learn)

**Primary entry point:** `notebooks/stock_price_prediction_colab.ipynb` — run top-to-bottom in Google Colab.

**Default ticker:** AAPL (Apple Inc.) — approximately 5 years of daily data via Yahoo Finance.

---

## 📁 Repository Structure

```
stock-price-prediction/
├── notebooks/
│   └── stock_price_prediction_colab.ipynb   ← PRIMARY DELIVERABLE
├── src/
│   ├── data_loader.py         ← yfinance download & validation
│   ├── feature_engineering.py ← MA5, MA20, RSI, BB, Daily Return, PVR
│   ├── preprocessing.py       ← Chronological split, MinMaxScaler, sequences
│   ├── sequence_builder.py    ← LSTM sequence & tabular feature builders
│   ├── train_lstm.py          ← LSTM model architecture & training
│   ├── train_baselines.py     ← Random Forest & Linear Regression
│   ├── evaluate.py            ← RMSE, MAE, R², plots
│   └── predict.py             ← End-to-end inference pipeline
├── data/
│   ├── raw/                   ← Downloaded CSVs (generated)
│   └── processed/             ← Feature-engineered CSVs (generated)
├── artifacts/
│   └── AAPL/                  ← Saved models, scalers, metadata (generated)
├── reports/
│   └── figures/               ← Generated plots (generated)
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   └── test_prediction.py
├── app.py                     ← Streamlit dashboard
├── requirements.txt
├── DECISIONS.md               ← Technical decisions & assumptions
├── IMPLEMENTATION_CHECKLIST.md
└── README.md
```

---

## 🚀 Quick Start — Google Colab (Recommended)

### Step 1: Upload or clone the repository

**Option A — Upload:** Upload the entire project folder to your Google Drive, then open the notebook from Drive.

**Option B — GitHub:** Push the repo to GitHub, then open Colab and clone:
```
!git clone https://github.com/YOUR_USERNAME/stock-price-prediction.git
%cd stock-price-prediction
```

### Step 2: Open the notebook

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. File → Open notebook → Upload → select `notebooks/stock_price_prediction_colab.ipynb`
3. Set runtime: Runtime → Change runtime type → **GPU** (recommended for LSTM)

### Step 3: Run all cells

Runtime → **Run all** (Ctrl+F9)

The notebook will:
1. Install all dependencies automatically (`pip install`)
2. Download AAPL historical data via yfinance
3. Engineer features (MA5, MA20, RSI, Bollinger Bands, etc.)
4. Split data chronologically (70% train / 10% val / 20% test)
5. Train all three models (LSTM, Random Forest, Linear Regression)
6. Evaluate on the held-out test set → real RMSE, MAE, R²
7. Generate all charts
8. Save models, scalers, metrics to `artifacts/AAPL/`
9. Predict the next trading day's closing price

**Expected runtime:** ~10–20 minutes (GPU) or ~25–40 minutes (CPU).

### Optional: Save to Google Drive (persist across sessions)

The notebook includes an optional section to mount Google Drive and redirect
all outputs there. Run the Drive mounting cell before training.

---

## 💻 Local Execution

### Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/stock-price-prediction.git
cd stock-price-prediction

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run tests

```bash
pytest tests/ -v
```

### Run full pipeline locally (CLI)

```bash
# Run full pipeline with all models (LSTM + Random Forest + Linear Regression)
python main.py --ticker AAPL --epochs 25 --batch-size 32

# Or run quickly with baseline models only:
python main.py --ticker AAPL --skip-lstm

# Run standalone inference using saved artifacts:
python src/predict.py
```

### Launch the Streamlit app

> ⚠️ You must run the Colab notebook first to generate artifacts in `artifacts/AAPL/`.

```bash
streamlit run app.py
```

Open your browser at [http://localhost:8501](http://localhost:8501).

---

## 🧠 Models

### LSTM (Long Short-Term Memory)
- Input: 60-day rolling window of 12 scaled features
- Architecture: LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)
- Optimizer: Adam | Loss: MSE | Early stopping (patience=10)
- Target: next-day Close (scaled, then inverse-transformed)

### Random Forest Regressor
- 200 estimators, trained on tabular lag features
- Features: last 5 Close values + previous day's full feature vector
- Target: next-day raw Close price

### Linear Regression
- Same tabular features as RF
- Interpretable baseline model

---

## 📊 Evaluation

All models evaluated on the **same held-out test period** (final 20% of data).

Metrics computed on the **original dollar price scale**:
| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error — penalises large errors |
| MAE | Mean Absolute Error — average absolute deviation |
| R² | Coefficient of determination — fraction of variance explained |

> ⚠️ A low RMSE does not imply a profitable strategy. This project does not
> implement or claim to implement a trading backtest.

---

## 🛡️ Data Leakage Prevention

This project implements strict leakage controls:
- All data sorted chronologically before any processing
- Scalers fitted **only on training data**, applied unchanged to val/test
- No shuffling of time series observations
- LSTM sequences use only past observations as inputs
- Validation data drawn from the **pre-test period**
- All three models evaluated on **identical held-out dates**

---

## 🔧 Feature Engineering

| Feature | Definition |
|---------|-----------|
| MA5 | 5-day simple moving average of Close |
| MA20 | 20-day simple moving average of Close |
| RSI | 14-period Relative Strength Index (Wilder's EMA) |
| BB_Upper | 20-day Bollinger Band upper (SMA + 2σ) |
| BB_Lower | 20-day Bollinger Band lower (SMA − 2σ) |
| Daily_Return | Percentage change in Close |
| Price_to_Volume_Ratio | Close / (Volume + 1) — proxy metric |

Warm-up rows where rolling features are NaN are **dropped**, not filled.

---

## ⚙️ Configuration

Key settings are at the top of each source file. To change the default ticker:
1. Edit `TICKER = "AAPL"` in the notebook's config cell.
2. Re-run the notebook from the top.
3. New artifacts will be saved to `artifacts/<TICKER>/`.

---

## 📋 Tests

Run with pytest:
```bash
pytest tests/ -v --tb=short
```

Tests cover:
- Valid and invalid ticker handling
- Empty / missing / malformed data
- Date ordering and duplicate removal
- Feature calculations and warm-up row dropping
- No future data in input sequences
- Chronological split boundaries
- Scalers fitted only on training data
- LSTM tensor shape `(samples, 60, n_features)`
- Model saving and loading
- Correct inverse transformation to price units
- Inference with insufficient history

---

## ⚠️ Known Limitations

1. **Single-stock model:** Models are trained on one ticker (AAPL by default). Applying them to a different ticker uses an incompatible scaler — the Streamlit app warns users about this.
2. **No trading backtest:** Evaluation shows prediction accuracy metrics, not trading profitability.
3. **Data delay:** Yahoo Finance data may be delayed by 15–20 minutes or more depending on the instrument.
4. **Colab session loss:** Model artifacts are lost when a Colab runtime disconnects unless saved to Google Drive.
5. **Market regime shifts:** Models trained on historical patterns may not generalise to structural market changes.

---

## 📝 License

MIT License — Educational use only.

---

## 🙏 Acknowledgements

- [yfinance](https://github.com/ranaroussi/yfinance) for free Yahoo Finance data access.
- [TensorFlow/Keras](https://www.tensorflow.org/) for the LSTM implementation.
- [scikit-learn](https://scikit-learn.org/) for Random Forest and Linear Regression.
- [Streamlit](https://streamlit.io/) for the dashboard framework.
