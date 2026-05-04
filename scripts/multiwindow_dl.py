"""DL multi-window backtest with fixed (sensible) architectures.

Per-window re-tuning with KerasTuner would take hours for K=6 x 4
architectures, so we use fixed reasonable architectures consistent with
common LSTM/GRU/CNN configurations for univariate price prediction.
Training uses EarlyStopping; the test window is predicted in batch from
the scaled sliding windows just like the notebook does.
"""
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, MaxPooling1D, Flatten, Dropout

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

from pathlib import Path as _Path
CSV = str(_Path(__file__).resolve().parent.parent / "BNBL_price_report_All.csv")
K = 6
H = 30
SEQ = 60
EPOCHS = 60
BATCH = 32
VAL_SIZE = 60   # held out from end of training portion as validation


def load_daily():
    df = pd.read_csv(CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby(df["Date"].dt.date)["Close Price"].last().reset_index()
    d.rename(columns={"Date": "Date", "Close Price": "Close"}, inplace=True)
    d["Date"] = pd.to_datetime(d["Date"])
    d.set_index("Date", inplace=True)
    return d


def create_sequences(scaled, seq):
    X, y = [], []
    for i in range(seq, len(scaled)):
        X.append(scaled[i - seq:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X).reshape(-1, seq, 1)
    return X, np.array(y)


def build_lstm():
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m


def build_gru():
    m = Sequential([
        GRU(64, return_sequences=True, input_shape=(SEQ, 1)),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m


def build_cnn():
    m = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", input_shape=(SEQ, 1)),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m


def build_cnn_lstm():
    m = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", input_shape=(SEQ, 1)),
        MaxPooling1D(2),
        LSTM(64),
        Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m


BUILDERS = {"LSTM": build_lstm, "GRU": build_gru, "CNN": build_cnn, "CNN_LSTM": build_cnn_lstm}


def metrics(y_true, y_pred, prev_actual):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    prev = np.asarray(prev_actual, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nz = y_true != 0
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100
    actual_move = y_true - prev
    pred_move = y_pred - prev
    mask = np.abs(actual_move) > 1e-9
    da = float("nan") if mask.sum() == 0 else (np.sign(actual_move[mask]) == np.sign(pred_move[mask])).mean() * 100
    return mae, rmse, mape, da


def window_slices(n, k, h):
    out = []
    end = n
    for _ in range(k):
        start = end - h
        if start - SEQ - VAL_SIZE < 30:
            break
        out.append((start, end))
        end = start
    return list(reversed(out))


def run_window(daily, test_start, test_end):
    """Train all 4 DL models on data before `test_start`, evaluate on the
    test window. Returns dict of model -> (mae, rmse, mape, da)."""
    series = daily["Close"].astype(float).values.reshape(-1, 1)

    train_end = test_start
    val_start = train_end - VAL_SIZE
    train_data = series[:val_start]
    val_data = series[val_start - SEQ:train_end]      # +seq overlap
    test_data = series[train_end - SEQ:test_end]      # +seq overlap

    scaler = MinMaxScaler()
    scaler.fit(train_data)
    tr = scaler.transform(train_data)
    va = scaler.transform(val_data)
    te = scaler.transform(test_data)

    X_tr, y_tr = create_sequences(tr, SEQ)
    X_va, y_va = create_sequences(va, SEQ)
    X_te, y_te = create_sequences(te, SEQ)

    actual_prices = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
    prev_actual = np.concatenate(
        [[float(daily["Close"].iloc[test_start - 1])], actual_prices[:-1]]
    )

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    out = {}
    for name, build in BUILDERS.items():
        tf.keras.utils.set_random_seed(SEED)
        m = build()
        m.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=EPOCHS, batch_size=BATCH, callbacks=[es], verbose=0)
        pred_scaled = m.predict(X_te, verbose=0).flatten()
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        out[name] = metrics(actual_prices, pred, prev_actual)
        del m
        tf.keras.backend.clear_session()
    return out


def main():
    daily = load_daily()
    n = len(daily)
    windows = window_slices(n, K, H)
    print(f"DL multi-window backtest. K={K}, H={H}, SEQ={SEQ}, VAL={VAL_SIZE}")
    for i, (a, b) in enumerate(windows):
        print(f"  window {i+1}: {daily.index[a].date()} .. {daily.index[b-1].date()}")

    rows = []
    for wi, (a, b) in enumerate(windows, start=1):
        print(f"\n=== DL window {wi}/{len(windows)} : {daily.index[a].date()} -> {daily.index[b-1].date()} ===", flush=True)
        out = run_window(daily, a, b)
        for name, (mae, rmse, mape, da) in out.items():
            rows.append((name, wi, mae, rmse, mape, da))
            print(f"  {name:<10} MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}% DA={da:.2f}%", flush=True)

    res = pd.DataFrame(rows, columns=["model", "window", "MAE", "RMSE", "MAPE", "DA"])
    res.to_csv("/tmp/multiwindow_dl_per_window.csv", index=False)

    agg = res.groupby("model").agg(
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAPE_mean=("MAPE", "mean"), MAPE_std=("MAPE", "std"),
        DA_mean=("DA", "mean"), DA_std=("DA", "std"),
    ).round(4)
    agg.to_csv("/tmp/multiwindow_dl_aggregate.csv")
    print("\n=== DL Mean ± std across windows ===")
    print(agg.to_string())
    print("\nSaved: /tmp/multiwindow_dl_per_window.csv  /tmp/multiwindow_dl_aggregate.csv")


if __name__ == "__main__":
    main()
