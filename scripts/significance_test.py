"""Statistical significance testing for the forecast comparison.

Steps:
  1. Re-run the K=6 expanding-window backtest for the statistical and ML
     models, saving every per-day prediction (N=180 forecasts per model).
  2. Re-load the DL per-day predictions (computed by /tmp/multiwindow_dl.py
     -- here we re-run a streamlined version saving predictions).
  3. Run the Diebold-Mariano test pairwise on the 11 models with the
     Harvey-Leybourne-Newbold (HLN) small-sample correction.
  4. Apply Holm-Bonferroni multiple-testing correction across the matrix.
  5. Produce: pairwise DM stat matrix, p-value matrix, Holm-adjusted p-values,
     and a "model confidence set"-style table by elimination.
"""
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
import random
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
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
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

from pathlib import Path as _Path
CSV = str(_Path(__file__).resolve().parent.parent / "BNBL_price_report_All.csv")
K = 6
H = 30
SEQ = 60
VAL_SIZE = 60


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


def stat_predictions(daily, wins):
    series = daily["Close"].astype(float)
    arima_pred, hw_pred, actuals, dates = [], [], [], []
    for a, b in wins:
        train = series.iloc[:a]; test = series.iloc[a:b]
        # ARIMA
        history = train.copy()
        for t_idx in range(len(test)):
            f = ARIMA(history, order=(1, 1, 1)).fit()
            arima_pred.append(float(f.forecast(steps=1).iloc[0]))
            history = pd.concat([history, pd.Series([test.iloc[t_idx]], index=[test.index[t_idx]])])
        # Holt-Winters
        history = train.copy()
        for t_idx in range(len(test)):
            f = ExponentialSmoothing(
                history, trend='add', seasonal=None, initialization_method='estimated'
            ).fit()
            hw_pred.append(float(f.forecast(steps=1).iloc[0]))
            history = pd.concat([history, pd.Series([test.iloc[t_idx]], index=[test.index[t_idx]])])
        actuals.extend(list(test.values.astype(float)))
        dates.extend(list(test.index))
    return {
        "Date": pd.Index(dates),
        "Actual": np.array(actuals, dtype=float),
        "ARIMA": np.array(arima_pred, dtype=float),
        "Holt-Winters": np.array(hw_pred, dtype=float),
    }


def ml_models():
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
            RandomForestRegressor(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [3, 5, None], "min_samples_split": [2, 5]},
        ),
        "XGBoost": (
            XGBRegressor(objective="reg:squarederror", random_state=42, verbosity=0),
            {"n_estimators": [200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        ),
        "LightGBM": (
            LGBMRegressor(random_state=42, verbose=-1),
            {"n_estimators": [200], "max_depth": [5, 10], "learning_rate": [0.05, 0.1], "num_leaves": [31, 63]},
        ),
    }


def ml_predictions(df_feat, daily, wins):
    """Slice df_feat by DATE so it aligns with `series` despite the
    feature-engineering dropna trimming the head of df_feat."""
    X = df_feat.drop("Close", axis=1)
    y = df_feat["Close"]
    pred_dict = {name: [] for name in ml_models()}
    for a, b in wins:
        start_date = daily.index[a]
        end_date = daily.index[b - 1]
        train_mask = X.index < start_date
        test_mask = (X.index >= start_date) & (X.index <= end_date)
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        tscv = TimeSeriesSplit(n_splits=3)
        for name, (model, params) in ml_models().items():
            gs = GridSearchCV(model, params, cv=tscv,
                              scoring="neg_mean_squared_error", n_jobs=-1)
            gs.fit(X_tr, y_tr)
            pred = gs.best_estimator_.predict(X_te)
            pred_dict[name].extend(list(pred))
    return {n: np.array(v, dtype=float) for n, v in pred_dict.items()}


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


def dl_predictions(daily, wins):
    series_arr = daily["Close"].astype(float).values.reshape(-1, 1)
    pred_dict = {n: [] for n in DL_BUILDERS}
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
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        for name, build in DL_BUILDERS.items():
            tf.keras.utils.set_random_seed(SEED)
            m = build()
            m.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=60, batch_size=32,
                  callbacks=[es], verbose=0)
            p = sc.inverse_transform(m.predict(X_te, verbose=0)).flatten()
            pred_dict[name].extend(list(p))
            tf.keras.backend.clear_session()
    return {n: np.array(v, dtype=float) for n, v in pred_dict.items()}


# ----------------------------------------------------------------------
# Diebold-Mariano test (Harvey-Leybourne-Newbold small-sample correction)
# ----------------------------------------------------------------------
def dm_test(e1, e2, h=1, loss="abs"):
    """One-sided test of H0: equal predictive accuracy vs H1: e1 has
    LOWER expected loss than e2 (i.e. model-1 better than model-2).
    Returns (DM_HLN statistic, two-sided p-value, one-sided p-value
    that model-1 has lower loss)."""
    e1 = np.asarray(e1, dtype=float); e2 = np.asarray(e2, dtype=float)
    if loss == "abs":
        d = np.abs(e1) - np.abs(e2)
    elif loss == "sqr":
        d = e1 ** 2 - e2 ** 2
    else:
        raise ValueError(loss)
    T = len(d)
    d_bar = np.mean(d)
    # Long-run variance of d_bar via Newey-West with lag h-1.
    gamma_0 = np.var(d, ddof=0)
    var = gamma_0
    for k in range(1, h):
        cov_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        var += 2 * cov_k
    if var <= 0:
        var = gamma_0
    dm_stat = d_bar / np.sqrt(var / T)
    # HLN small-sample correction
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = correction * dm_stat
    df = T - 1
    # Two-sided p-value
    p_two = 2 * (1 - student_t.cdf(np.abs(dm_hln), df=df))
    # One-sided p-value (model-1 better == d_bar < 0)
    p_one_better = student_t.cdf(dm_hln, df=df)
    return dm_hln, p_two, p_one_better


def holm_correction(pvals_flat):
    """Holm-Bonferroni step-down adjustment. Input: 1D array of raw p-values.
    Output: array of adjusted p-values, same order."""
    p = np.asarray(pvals_flat, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    for k, idx in enumerate(order):
        adj[idx] = min((m - k) * p[idx], 1.0)
    # enforce monotonicity
    sorted_idx = np.argsort(p)
    sorted_adj = adj[sorted_idx]
    for i in range(1, len(sorted_adj)):
        if sorted_adj[i] < sorted_adj[i - 1]:
            sorted_adj[i] = sorted_adj[i - 1]
    adj_out = np.empty_like(adj)
    adj_out[sorted_idx] = sorted_adj
    return adj_out


def main():
    daily = load_daily()
    df_feat = create_features(daily)
    n = len(daily)
    wins = windows(n)
    print(f"K={len(wins)} windows, H={H}; total test days = {sum(b-a for a,b in wins)}", flush=True)

    print("\n[1/3] Statistical predictions ...", flush=True)
    stat = stat_predictions(daily, wins)
    actuals = stat["Actual"]; dates = stat["Date"]

    print("[2/3] ML predictions ...", flush=True)
    ml = ml_predictions(df_feat, daily, wins)
    # Align ML predictions with stat dates: same windows, same order, same length
    assert all(len(v) == len(actuals) for v in ml.values()), "length mismatch"

    print("[3/3] DL predictions ...", flush=True)
    dl = dl_predictions(daily, wins)
    assert all(len(v) == len(actuals) for v in dl.values()), "length mismatch"

    preds = {"ARIMA": stat["ARIMA"], "Holt-Winters": stat["Holt-Winters"], **ml, **dl}
    df_pred = pd.DataFrame(preds, index=dates)
    df_pred.insert(0, "Actual", actuals)
    df_pred.to_csv("/tmp/per_day_predictions.csv")
    print(f"Saved per-day predictions: /tmp/per_day_predictions.csv "
          f"({df_pred.shape[0]} rows × {df_pred.shape[1]} cols)")

    # Forecast errors
    errors = {name: actuals - preds[name] for name in preds}

    # MAE per model (for ranking and reference)
    mae = {name: float(np.mean(np.abs(e))) for name, e in errors.items()}
    print("\nMAE across all 180 test days:")
    for name, m in sorted(mae.items(), key=lambda kv: kv[1]):
        print(f"  {name:<18} {m:.4f}")

    model_names = list(preds.keys())
    M = len(model_names)
    dm_mat = np.full((M, M), np.nan)
    p1_mat = np.full((M, M), np.nan)   # one-sided p-value: row better than col
    for i, ni in enumerate(model_names):
        for j, nj in enumerate(model_names):
            if i == j: continue
            stat_ij, p_two, p_one = dm_test(errors[ni], errors[nj], h=1, loss="abs")
            dm_mat[i, j] = stat_ij
            p1_mat[i, j] = p_one
    # Holm correction across the upper triangle (M*(M-1)/2 pairs)
    raw_p_two = []
    pair_idx = []
    for i in range(M):
        for j in range(i + 1, M):
            _, p_two, _ = dm_test(errors[model_names[i]], errors[model_names[j]], h=1, loss="abs")
            raw_p_two.append(p_two)
            pair_idx.append((i, j))
    holm_p = holm_correction(raw_p_two)

    dm_df = pd.DataFrame(dm_mat, index=model_names, columns=model_names)
    p1_df = pd.DataFrame(p1_mat, index=model_names, columns=model_names)
    holm_df = pd.DataFrame(np.full((M, M), np.nan), index=model_names, columns=model_names)
    for (i, j), p in zip(pair_idx, holm_p):
        holm_df.iloc[i, j] = p
        holm_df.iloc[j, i] = p

    dm_df.round(3).to_csv("/tmp/dm_statistic_matrix.csv")
    p1_df.round(4).to_csv("/tmp/dm_pvalue_matrix.csv")
    holm_df.round(4).to_csv("/tmp/dm_holm_pvalue_matrix.csv")

    print("\nDM statistic matrix (row better than column when negative):")
    print(dm_df.round(2).to_string())
    print("\nOne-sided DM p-value (P[row has SMALLER absolute error than col], small => row better):")
    print(p1_df.round(3).to_string())
    print("\nHolm-adjusted two-sided p-values (symmetric matrix):")
    print(holm_df.round(3).to_string())


if __name__ == "__main__":
    main()
