"""Rolling-origin (expanding-window) backtest for the statistical and
machine-learning pipelines on BNBL.

Setup:
- K = 6 non-overlapping test windows of H = 30 days, ending at the most
  recent date and walking backward.
- For each window k, train on every observation BEFORE the window's start
  (expanding-window), evaluate on the 30 days using a one-step-ahead
  rolling-forecast protocol consistent with the patched notebooks.
- Reports per-window MAE / RMSE / MAPE / DA, plus mean ± std across
  windows for every model.
"""
import warnings
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

from pathlib import Path as _Path
CSV = str(_Path(__file__).resolve().parent.parent / "BNBL_price_report_All.csv")
K = 6           # number of test windows
H = 30          # length of each window in days
P, D, Q = 1, 1, 1


# ----------------------------------------------------------------------
# Data + features
# ----------------------------------------------------------------------
def load_daily():
    df = pd.read_csv(CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    d = (
        df.groupby(df["Date"].dt.date)["Close Price"].last().reset_index()
    )
    d.rename(columns={"Date": "Date", "Close Price": "Close"}, inplace=True)
    d["Date"] = pd.to_datetime(d["Date"])
    d.set_index("Date", inplace=True)
    return d


def create_features(data, lags=7, rolling_windows=(3, 5, 10, 20)):
    """Leak-fixed feature engineering: pct_change features lagged by 1."""
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


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def metrics(y_true, y_pred, prev_actual=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nz = y_true != 0
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100
    if prev_actual is not None:
        prev_actual = np.asarray(prev_actual, dtype=float)
        actual_move = y_true - prev_actual
        pred_move = y_pred - prev_actual
        mask = np.abs(actual_move) > 1e-9
        if mask.sum() == 0:
            da = float("nan")
        else:
            da = (np.sign(actual_move[mask]) == np.sign(pred_move[mask])).mean() * 100
    else:
        da = (np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))).mean() * 100
    return mae, rmse, mape, da


# ----------------------------------------------------------------------
# Window definition (most-recent first; non-overlapping; walking backward)
# ----------------------------------------------------------------------
def window_slices(n, k, h):
    """Yield (train_end_idx, test_start_idx, test_end_idx) for each window."""
    out = []
    end = n
    for _ in range(k):
        start = end - h
        if start - 1 < 30:                 # need enough warm-up
            break
        out.append((start, end))
        end = start
    return list(reversed(out))             # report oldest -> newest


# ----------------------------------------------------------------------
# Statistical models with rolling 1-step inside each window
# ----------------------------------------------------------------------
def stat_eval_window(series, train_idx_end, test_start, test_end):
    train = series.iloc[:test_start]
    test = series.iloc[test_start:test_end]

    # ARIMA rolling 1-step
    history = train.copy()
    arima_pred = []
    for t in range(len(test)):
        fit = ARIMA(history, order=(P, D, Q)).fit()
        arima_pred.append(float(fit.forecast(steps=1).iloc[0]))
        history = pd.concat([history, pd.Series([test.iloc[t]], index=[test.index[t]])])

    # Holt-Winters rolling 1-step
    history = train.copy()
    hw_pred = []
    for t in range(len(test)):
        fit = ExponentialSmoothing(
            history, trend="add", seasonal=None, initialization_method="estimated"
        ).fit()
        hw_pred.append(float(fit.forecast(steps=1).iloc[0]))
        history = pd.concat([history, pd.Series([test.iloc[t]], index=[test.index[t]])])

    # DA anchor: yesterday's actual = day immediately before each test point.
    prev_actual = np.concatenate(
        [[float(series.iloc[test_start - 1])], np.asarray(test.values[:-1], float)]
    )
    return {
        "ARIMA": metrics(test.values, arima_pred, prev_actual),
        "Holt-Winters": metrics(test.values, hw_pred, prev_actual),
    }


# ----------------------------------------------------------------------
# ML models: GridSearch on training portion, predict the test window
# ----------------------------------------------------------------------
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
            {"n_estimators": [200], "max_depth": [5, 10],
             "learning_rate": [0.05, 0.1], "num_leaves": [31, 63]},
        ),
    }


def ml_eval_window(df_feat, test_start_date, test_end_date, prev_actual_close):
    X = df_feat.drop("Close", axis=1)
    y = df_feat["Close"]

    # By date so feature dropna does not misalign indices
    train_mask = df_feat.index < test_start_date
    test_mask = (df_feat.index >= test_start_date) & (df_feat.index < test_end_date)
    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]

    if len(X_te) == 0:
        return {}

    tscv = TimeSeriesSplit(n_splits=3)
    out = {}
    for name, (model, params) in ml_models().items():
        gs = GridSearchCV(model, params, cv=tscv,
                          scoring="neg_mean_squared_error", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        pred = gs.best_estimator_.predict(X_te)
        prev = np.concatenate([[prev_actual_close], np.asarray(y_te.values[:-1], float)])
        out[name] = metrics(y_te.values, pred, prev)
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    daily = load_daily()
    series = daily["Close"].astype(float)
    df_feat = create_features(daily)

    n = len(series)
    windows = window_slices(n, K, H)
    print(f"Series length: {n}, K={K} windows of H={H}, total test days: {sum(b-a for a,b in windows)}")
    for i, (a, b) in enumerate(windows):
        print(f"  window {i+1}: {series.index[a].date()} .. {series.index[b-1].date()}")

    rows = []  # (model, window_idx, MAE, RMSE, MAPE, DA, train_end_date, test_start_date, test_end_date)
    for wi, (a, b) in enumerate(windows, start=1):
        print(f"\n=== Window {wi}/{len(windows)} : {series.index[a].date()} -> {series.index[b-1].date()} ===")

        # Statistical
        stat_out = stat_eval_window(series, a, a, b)
        for name, (mae, rmse, mape, da) in stat_out.items():
            rows.append((name, wi, mae, rmse, mape, da))
            print(f"  {name:<18} MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}% DA={da:.2f}%")

        # ML
        prev_close = float(series.iloc[a - 1])
        ml_out = ml_eval_window(df_feat,
                                test_start_date=series.index[a],
                                test_end_date=series.index[b - 1] + pd.Timedelta(days=1),
                                prev_actual_close=prev_close)
        for name, (mae, rmse, mape, da) in ml_out.items():
            rows.append((name, wi, mae, rmse, mape, da))
            print(f"  {name:<18} MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}% DA={da:.2f}%")

    # Aggregate
    res = pd.DataFrame(rows, columns=["model", "window", "MAE", "RMSE", "MAPE", "DA"])
    print("\n\n=== Per-window results (long form) ===")
    print(res.to_string(index=False))

    agg = res.groupby("model").agg(
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAPE_mean=("MAPE", "mean"), MAPE_std=("MAPE", "std"),
        DA_mean=("DA", "mean"), DA_std=("DA", "std"),
    ).round(4)
    print("\n=== Mean ± std across windows ===")
    print(agg.to_string())

    res.to_csv("/tmp/multiwindow_per_window.csv", index=False)
    agg.to_csv("/tmp/multiwindow_aggregate.csv")
    print("\nSaved: /tmp/multiwindow_per_window.csv  /tmp/multiwindow_aggregate.csv")


if __name__ == "__main__":
    main()
