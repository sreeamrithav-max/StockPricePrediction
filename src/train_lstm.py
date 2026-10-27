"""
src/train_lstm.py
-----------------
LSTM model construction and training for Stock Price Prediction.

Architecture (DECISIONS.md D10):
    LSTM(128, return_sequences=True)
    Dropout(0.2)
    LSTM(64)
    Dropout(0.2)
    Dense(32, activation='relu')
    Dense(1)

Training config (SPEC.md §7.1):
    Optimizer  : Adam
    Loss       : Mean Squared Error
    Batch size : 32
    Max epochs : 50
    Early stopping: patience=10, restore_best_weights=True
    Random seed: 42
"""

import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 50
PATIENCE = 10


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_lstm_model(
    look_back: int,
    n_features: int,
    lstm_units_1: int = 128,
    lstm_units_2: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Construct the LSTM regression model.

    Parameters
    ----------
    look_back : int
        Sequence length (time steps per input).
    n_features : int
        Number of input features per time step.
    lstm_units_1 : int
        Units in first LSTM layer.
    lstm_units_2 : int
        Units in second LSTM layer.
    dense_units : int
        Units in Dense hidden layer.
    dropout_rate : float
        Dropout fraction after each LSTM layer.
    learning_rate : float
        Adam learning rate.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(look_back, n_features)),
            layers.LSTM(lstm_units_1, return_sequences=True),
            layers.Dropout(dropout_rate),
            layers.LSTM(lstm_units_2, return_sequences=False),
            layers.Dropout(dropout_rate),
            layers.Dense(dense_units, activation="relu"),
            layers.Dense(1),
        ],
        name="stock_lstm",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )

    logger.info("LSTM model built: input shape = (%s, %s)", look_back, n_features)
    model.summary(print_fn=logger.info)
    return model


def train_lstm(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> keras.callbacks.History:
    """
    Train the LSTM model with early stopping on validation loss.

    Parameters
    ----------
    model : keras.Model
        Compiled LSTM model.
    X_train, y_train : np.ndarray
        Training sequences and targets.
    X_val, y_val : np.ndarray
        Validation sequences and targets.
    batch_size : int
    max_epochs : int
    patience : int
        Early stopping patience.

    Returns
    -------
    keras.callbacks.History
        Training history object.
    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    logger.info(
        "Training LSTM: %d train samples, %d val samples, batch=%d, max_epochs=%d",
        len(X_train), len(X_val), batch_size, max_epochs,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=max_epochs,
        callbacks=[early_stopping],
        verbose=1,
    )

    stopped_epoch = early_stopping.stopped_epoch
    if stopped_epoch > 0:
        logger.info("Early stopping at epoch %d.", stopped_epoch + 1)
    else:
        logger.info("Training completed all %d epochs.", max_epochs)

    return history


def save_lstm_model(model: keras.Model, save_dir: str, ticker: str = "AAPL") -> str:
    """Save trained LSTM model in .keras format."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{ticker}_lstm.keras")
    model.save(path)
    logger.info("LSTM model saved → %s", path)
    return path


def load_lstm_model(save_dir: str, ticker: str = "AAPL") -> keras.Model:
    """Load a saved LSTM model."""
    path = os.path.join(save_dir, f"{ticker}_lstm.keras")
    model = keras.models.load_model(path)
    logger.info("LSTM model loaded from %s", path)
    return model
