"""Feature-group ablation across the same K=6 expanding-window backtest
used for Table 1.

Feature groups (defined in Step 3 of the ML notebook, leak-fixed):
  - lags         : lag_1 .. lag_7
  - roll_mean    : roll_mean_{3,5,10,20}
  - roll_std     : roll_std_{3,5,10,20}
  - return       : pct_change shifted by 1
  - volatility   : 5-day std of returns shifted by 1

For each model and each ablation we report MAE / RMSE / MAPE / DA mean
across windows AND the delta vs the all-features baseline. Positive
delta MAE => removing the group HURT the model => the group was useful.
"""
import warnings
import numpy as np
import pandas as pd
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
K = 6
H = 30


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


def da_anchored(y_true, y_pred, prev_actual):
    actual_move = np.asarray(y_true) - np.asarray(prev_actual)
    pred_move = np.asarray(y_pred) - np.asarray(prev_actual)
    mask = np.abs(actual_move) > 1e-9
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(actual_move[mask]) == np.sign(pred_move[mask])) * 100)


def feature_groups():
    return {
        "lags":      [f"lag_{i}" for i in range(1, 8)],
        "roll_mean": [f"roll_mean_{w}" for w in (3, 5, 10, 20)],
        "roll_std":  [f"roll_std_{w}"  for w in (3, 5, 10, 20)],
        "return":    ["return"],
        "volatility":["volatility"],
    }


def models_grid():
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


def windows(n, k, h):
    out, end = [], n
    for _ in range(k):
        start = end - h
        if start - 30 < 0:
            break
        out.append((start, end))
        end = start
    return list(reversed(out))


def evaluate(df_feat, drop_cols, models):
    """Return mean ± std MAE/RMSE/MAPE/DA across K windows after dropping
    `drop_cols` from the feature matrix."""
    X = df_feat.drop(columns=["Close"] + list(drop_cols))
    y = df_feat["Close"]
    n = len(df_feat)
    wins = windows(n, K, H)
    tscv = TimeSeriesSplit(n_splits=3)

    results = {name: [] for name in models}
    for a, b in wins:
        X_tr, X_te = X.iloc[:a], X.iloc[a:b]
        y_tr, y_te = y.iloc[:a], y.iloc[a:b]
        prev = np.concatenate([[float(y.iloc[a - 1])], np.asarray(y_te.values[:-1], float)])
        for name, (model, params) in models.items():
            gs = GridSearchCV(model, params, cv=tscv,
                              scoring="neg_mean_squared_error", n_jobs=-1)
            gs.fit(X_tr, y_tr)
            pred = gs.best_estimator_.predict(X_te)
            yt = y_te.values.astype(float); nz = yt != 0
            mae = mean_absolute_error(yt, pred)
            rmse = float(np.sqrt(mean_squared_error(yt, pred)))
            mape = float(np.mean(np.abs((yt[nz] - pred[nz]) / yt[nz])) * 100)
            da = da_anchored(yt, pred, prev)
            results[name].append((mae, rmse, mape, da))

    summary = {}
    for name, lst in results.items():
        arr = np.array(lst)  # rows = windows, cols = mae,rmse,mape,da
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros_like(mean)
        summary[name] = {
            "MAE_mean": mean[0], "MAE_std": std[0],
            "RMSE_mean": mean[1], "RMSE_std": std[1],
            "MAPE_mean": mean[2], "MAPE_std": std[2],
            "DA_mean":  mean[3], "DA_std":  std[3],
        }
    return summary


def main():
    daily = load_daily()
    df_feat = create_features(daily)
    models = models_grid()
    groups = feature_groups()

    rows = []  # ablation, model, MAE_mean, MAE_std, ..., dMAE
    print("Computing baseline (all features)...", flush=True)
    base = evaluate(df_feat, drop_cols=[], models=models)
    for name, m in base.items():
        rows.append(("All features", name, m["MAE_mean"], m["MAE_std"],
                     m["RMSE_mean"], m["RMSE_std"], m["DA_mean"], m["DA_std"], 0.0))

    for g_name, cols in groups.items():
        print(f"Computing ablation: drop {g_name} ({len(cols)} cols)...", flush=True)
        out = evaluate(df_feat, drop_cols=cols, models=models)
        for name, m in out.items():
            d_mae = m["MAE_mean"] - base[name]["MAE_mean"]
            rows.append((f"–{g_name}", name, m["MAE_mean"], m["MAE_std"],
                         m["RMSE_mean"], m["RMSE_std"], m["DA_mean"], m["DA_std"], d_mae))

    # Lags-only minimal baseline (drop everything but lags)
    keep = set([f"lag_{i}" for i in range(1, 8)] + ["Close"])
    drop_cols = [c for c in df_feat.columns if c not in keep]
    print(f"Computing 'lags only' minimal baseline (drop {len(drop_cols)} cols)...", flush=True)
    out = evaluate(df_feat, drop_cols=drop_cols, models=models)
    for name, m in out.items():
        d_mae = m["MAE_mean"] - base[name]["MAE_mean"]
        rows.append(("Lags only", name, m["MAE_mean"], m["MAE_std"],
                     m["RMSE_mean"], m["RMSE_std"], m["DA_mean"], m["DA_std"], d_mae))

    df = pd.DataFrame(rows, columns=["ablation", "model",
                                     "MAE_mean", "MAE_std",
                                     "RMSE_mean", "RMSE_std",
                                     "DA_mean", "DA_std", "delta_MAE_vs_baseline"])
    df.to_csv("/tmp/ablation_results.csv", index=False)

    print("\n=== Ablation summary ===")
    pivot_mae = df.pivot(index="ablation", columns="model", values="MAE_mean").round(4)
    pivot_dmae = df.pivot(index="ablation", columns="model", values="delta_MAE_vs_baseline").round(4)
    pivot_da = df.pivot(index="ablation", columns="model", values="DA_mean").round(2)
    print("\nMAE mean across K windows (lower is better):")
    print(pivot_mae.to_string())
    print("\nDelta MAE vs all-features baseline (positive = group was useful):")
    print(pivot_dmae.to_string())
    print("\nDirectional Accuracy mean across K windows:")
    print(pivot_da.to_string())


if __name__ == "__main__":
    main()
