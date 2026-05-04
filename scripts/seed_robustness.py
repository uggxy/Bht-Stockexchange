"""Seed-robustness check.

For each model that contains randomness (RF/XGB/LGBM in ML; LSTM/GRU/
CNN/CNN-LSTM in DL), we re-run the K=6 expanding-window backtest with
multiple random seeds. For the deterministic models (LR, SVR, ARIMA,
Holt-Winters) we run once -- their seed-std is 0 by construction.

For each (model, seed) combination we compute the mean MAE across the
6 windows, then report mean ± std of those per-seed means. This
quantifies how much the reported numbers depend on a single random
initialization.

Seeds used:
  ML stochastic (RF, XGB, LGBM)        : 5 seeds [0,1,2,3,4]
  DL (LSTM, GRU, CNN, CNN-LSTM)        : 3 seeds [0,1,2]   (each DL run
                                                              takes ~10 min;
                                                              3 seeds keeps
                                                              total runtime
                                                              ~35-50 min)
"""
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
import random
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, MaxPooling1D, Flatten, Dropout

warnings.filterwarnings("ignore")

from pathlib import Path as _Path
CSV = str(_Path(__file__).resolve().parent.parent / "BNBL_price_report_All.csv")
K = 6
H = 30
SEQ = 60
VAL_SIZE = 60
ML_SEEDS = [0, 1, 2, 3, 4]
DL_SEEDS = [0, 1, 2]


def load_daily():
    df = pd.read_csv(CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby(df["Date"].dt.date)["Close Price"].last().reset_index()
    d.rename(columns={"Date": "Date", "Close Price": "Close"}, inplace=True)
    d["Date"] = pd.to_datetime(d["Date"])
    d.set_index("Date", inplace=True)
    return d


def create_features(data, lags=7, rolling_windows=(3, 5, 10, 20)):
    df = pd.DataFrame(index=data.index)
    df["Close"] = data["Close"]
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = data["Close"].shift(lag)
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = data["Close"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = data["Close"].shift(1).rolling(w).std()
    df["return"] = data["Close"].pct_change().shift(1)
    df["volatility"] = data["Close"].pct_change().rolling(5).std().shift(1)
    df.dropna(inplace=True)
    return df


def windows(n, k=K, h=H):
    out, end = [], n
    for _ in range(k):
        start = end - h
        if start - SEQ - VAL_SIZE < 30:
            break
        out.append((start, end))
        end = start
    return list(reversed(out))


# ---------------------------------------------------------- ML
def ml_models_with_seed(seed):
    return {
        "Linear Regression": (
            Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            {"model__fit_intercept": [True, False]},
        ),
        "SVR": (
            Pipeline([("scaler", StandardScaler()), ("model", SVR())]),
            {"model__kernel": ["rbf", "poly"], "model__C": [1, 10], "model__gamma": ["scale"]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=seed),
            {"n_estimators": [100, 200], "max_depth": [3, 5, None], "min_samples_split": [2, 5]},
        ),
        "XGBoost": (
            XGBRegressor(objective="reg:squarederror", random_state=seed, verbosity=0),
            {"n_estimators": [200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        ),
        "LightGBM": (
            LGBMRegressor(random_state=seed, verbose=-1),
            {"n_estimators": [200], "max_depth": [5, 10],
             "learning_rate": [0.05, 0.1], "num_leaves": [31, 63]},
        ),
    }


def run_ml_for_seed(df_feat, daily, wins, seed):
    """Return {model: per-window list of MAE} for this seed."""
    X = df_feat.drop("Close", axis=1)
    y = df_feat["Close"]
    tscv = TimeSeriesSplit(n_splits=3)
    out = {name: [] for name in ml_models_with_seed(seed)}
    for a, b in wins:
        sd = daily.index[a]; ed = daily.index[b - 1]
        X_tr, X_te = X[X.index < sd], X[(X.index >= sd) & (X.index <= ed)]
        y_tr, y_te = y[y.index < sd], y[(y.index >= sd) & (y.index <= ed)]
        for name, (model, params) in ml_models_with_seed(seed).items():
            gs = GridSearchCV(model, params, cv=tscv,
                              scoring="neg_mean_squared_error", n_jobs=-1)
            gs.fit(X_tr, y_tr)
            pred = gs.best_estimator_.predict(X_te)
            mae = mean_absolute_error(y_te, pred)
            out[name].append(float(mae))
    return out


# ---------------------------------------------------------- DL
def _build_lstm():
    m = Sequential([LSTM(64, return_sequences=True, input_shape=(SEQ, 1)), Dropout(0.2),
                    LSTM(32), Dropout(0.2), Dense(1)])
    m.compile(optimizer="adam", loss="mse"); return m
def _build_gru():
    m = Sequential([GRU(64, return_sequences=True, input_shape=(SEQ, 1)), Dropout(0.2),
                    GRU(32), Dropout(0.2), Dense(1)])
    m.compile(optimizer="adam", loss="mse"); return m
def _build_cnn():
    m = Sequential([Conv1D(64, kernel_size=3, activation="relu", input_shape=(SEQ, 1)),
                    MaxPooling1D(2), Flatten(), Dense(64, activation="relu"), Dense(1)])
    m.compile(optimizer="adam", loss="mse"); return m
def _build_cnn_lstm():
    m = Sequential([Conv1D(64, kernel_size=3, activation="relu", input_shape=(SEQ, 1)),
                    MaxPooling1D(2), LSTM(64), Dense(1)])
    m.compile(optimizer="adam", loss="mse"); return m

DL_BUILDERS = {"LSTM": _build_lstm, "GRU": _build_gru, "CNN": _build_cnn, "CNN_LSTM": _build_cnn_lstm}


def run_dl_for_seed(daily, wins, seed):
    series_arr = daily["Close"].astype(float).values.reshape(-1, 1)
    out = {n: [] for n in DL_BUILDERS}
    for a, b in wins:
        val_start = a - VAL_SIZE
        train_data = series_arr[:val_start]
        val_data = series_arr[val_start - SEQ:a]
        test_data = series_arr[a - SEQ:b]
        sc = MinMaxScaler(); sc.fit(train_data)
        tr = sc.transform(train_data); va = sc.transform(val_data); te = sc.transform(test_data)
        def _seq(arr, sl=SEQ):
            X, y = [], []
            for i in range(sl, len(arr)):
                X.append(arr[i - sl:i, 0]); y.append(arr[i, 0])
            return np.array(X).reshape(-1, sl, 1), np.array(y)
        X_tr, y_tr = _seq(tr); X_va, y_va = _seq(va); X_te, y_te = _seq(te)
        actuals = sc.inverse_transform(y_te.reshape(-1, 1)).flatten()
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        for name, build in DL_BUILDERS.items():
            random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
            tf.keras.utils.set_random_seed(seed)
            m = build()
            m.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=60, batch_size=32,
                  callbacks=[es], verbose=0)
            pred = sc.inverse_transform(m.predict(X_te, verbose=0)).flatten()
            mae = mean_absolute_error(actuals, pred)
            out[name].append(float(mae))
            tf.keras.backend.clear_session()
    return out


# ---------------------------------------------------------- Stat (deterministic)
def run_stat(daily, wins):
    series = daily["Close"].astype(float)
    out = {"ARIMA": [], "Holt-Winters": []}
    for a, b in wins:
        train = series.iloc[:a]; test = series.iloc[a:b]
        # ARIMA
        history = train.copy(); preds = []
        for t_idx in range(len(test)):
            f = ARIMA(history, order=(1, 1, 1)).fit()
            preds.append(float(f.forecast(steps=1).iloc[0]))
            history = pd.concat([history, pd.Series([test.iloc[t_idx]], index=[test.index[t_idx]])])
        out["ARIMA"].append(mean_absolute_error(test.values, preds))
        # Holt-Winters
        history = train.copy(); preds = []
        for t_idx in range(len(test)):
            f = ExponentialSmoothing(history, trend='add', seasonal=None,
                                     initialization_method='estimated').fit()
            preds.append(float(f.forecast(steps=1).iloc[0]))
            history = pd.concat([history, pd.Series([test.iloc[t_idx]], index=[test.index[t_idx]])])
        out["Holt-Winters"].append(mean_absolute_error(test.values, preds))
    return out


# ---------------------------------------------------------- main
def main():
    t0 = time.time()
    daily = load_daily()
    df_feat = create_features(daily)
    n = len(daily)
    wins = windows(n)
    print(f"K={len(wins)} windows, H={H}, ML seeds={ML_SEEDS}, DL seeds={DL_SEEDS}")

    # Stat once (deterministic)
    print(f"\n[Stat] running once (deterministic)...", flush=True)
    stat_out = run_stat(daily, wins)
    stat_per_seed = {n: [stat_out[n]] for n in stat_out}  # one "seed"
    print(f"  ARIMA per-window MAE: {[f'{v:.4f}' for v in stat_out['ARIMA']]}")
    print(f"  HW    per-window MAE: {[f'{v:.4f}' for v in stat_out['Holt-Winters']]}")

    # ML across seeds
    print(f"\n[ML] running 5 seeds...", flush=True)
    ml_per_seed = {name: [] for name in ml_models_with_seed(0)}
    for s in ML_SEEDS:
        print(f"  seed={s}", flush=True)
        out_s = run_ml_for_seed(df_feat, daily, wins, s)
        for name, lst in out_s.items():
            ml_per_seed[name].append(lst)

    # DL across seeds
    print(f"\n[DL] running {len(DL_SEEDS)} seeds...", flush=True)
    dl_per_seed = {name: [] for name in DL_BUILDERS}
    for s in DL_SEEDS:
        print(f"  seed={s}", flush=True)
        out_s = run_dl_for_seed(daily, wins, s)
        for name, lst in out_s.items():
            dl_per_seed[name].append(lst)

    # Aggregate: for each model, per-seed mean MAE across windows; then
    # mean ± std across seeds.
    rows = []
    def _agg(per_seed_dict, label):
        for name, seeds in per_seed_dict.items():
            seed_means = [float(np.mean(lst)) for lst in seeds]
            mean = float(np.mean(seed_means))
            std = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
            rows.append((name, len(seeds), mean, std, label))
            print(f"  {name:<18} mean MAE = {mean:.4f} ± {std:.4f}  (n_seeds={len(seeds)})")

    print("\n=== SEED-ROBUSTNESS SUMMARY ===")
    print("Statistical:")
    _agg(stat_per_seed, "stat")
    print("Machine learning:")
    _agg(ml_per_seed, "ml")
    print("Deep learning:")
    _agg(dl_per_seed, "dl")

    df = pd.DataFrame(rows, columns=["model", "n_seeds", "MAE_mean", "MAE_std", "category"])
    df.to_csv("/tmp/seed_robustness.csv", index=False)
    print(f"\nSaved /tmp/seed_robustness.csv")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
