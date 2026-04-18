# %% [markdown]
# # 03 — ARIMA & SARIMAX forecasting
#
# For each target: pick best ARIMA(p,d,q) by AIC on the training series, then
# fit SARIMAX with the same orders plus a weekly seasonal block (period = 52).
# Forecast the 20% test horizon one step at a time (rolling via `append`).

# %%
import sys
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (  # noqa: E402
    load_clean_resample,
    train_test_split_ts,
    plot_pred,
    save_metrics,
    naive_persistence,
    seasonal_naive,
    TARGETS,
    FIG_DIR,
    METRICS_DIR,
    WEEKLY_PARQUET,
)

FIG_OUT = FIG_DIR / "03_arima_sarimax"
FIG_OUT.mkdir(parents=True, exist_ok=True)
METRICS_CSV = METRICS_DIR / "03_arima_sarimax.csv"
if METRICS_CSV.exists():
    METRICS_CSV.unlink()

# %% [markdown]
# ## Load weekly frame & split

# %%
if WEEKLY_PARQUET.exists():
    df = pd.read_parquet(WEEKLY_PARQUET)
else:
    df = load_clean_resample()

train_df, test_df = train_test_split_ts(df, test_frac=0.2)
print("train", train_df.shape, "test", test_df.shape)

# %% [markdown]
# ## Baselines (logged first so every target has a reference row)
#
# - `naive_persistence`: predict last observed value (y_hat_t = y_{t-1}).
# - `seasonal_naive_52`: predict the value from 52 weeks ago.

# %%
baseline_rows = []
for target in TARGETS:
    y_train = train_df[target]
    y_test = test_df[target]
    np_pred = naive_persistence(y_train, y_test)
    sn_pred = seasonal_naive(y_train, y_test, period=52)
    m_np = plot_pred(y_train, y_test, np_pred, target, "naive_persistence", FIG_OUT)
    m_sn = plot_pred(y_train, y_test, sn_pred, target, "seasonal_naive_52", FIG_OUT)
    baseline_rows.append({"target": target, "model": "naive_persistence",
                          "order": "-", "seasonal_order": "-", **m_np})
    baseline_rows.append({"target": target, "model": "seasonal_naive_52",
                          "order": "-", "seasonal_order": "-", **m_sn})

save_metrics(baseline_rows, METRICS_CSV)

# %% [markdown]
# ## ARIMA grid search on each target (select by AIC on training data)

# %%
P_RANGE = [0, 1, 2]
D_RANGE = [0, 1]
Q_RANGE = [0, 1, 2]

best_arima_orders = {}
arima_search_rows = []
for target in TARGETS:
    y = train_df[target].astype(float)
    best = None
    for p, d, q in itertools.product(P_RANGE, D_RANGE, Q_RANGE):
        try:
            res = ARIMA(y, order=(p, d, q)).fit()
            aic = res.aic
        except Exception:
            continue
        arima_search_rows.append({"target": target, "p": p, "d": d, "q": q, "aic": aic})
        if best is None or aic < best[1]:
            best = ((p, d, q), aic)
    best_arima_orders[target] = best[0]
    print(f"{target}: best ARIMA order = {best[0]}  AIC = {best[1]:.2f}")

pd.DataFrame(arima_search_rows).sort_values(["target", "aic"]).groupby("target").head(3)

# %% [markdown]
# ## Rolling one-step-ahead forecast — ARIMA

# %%
def rolling_forecast(fitted_res, y_test: pd.Series) -> np.ndarray:
    """One-step-ahead forecasts by appending each true observation as we go."""
    preds = []
    res = fitted_res
    for t in range(len(y_test)):
        yhat = float(res.forecast(steps=1).iloc[0])
        preds.append(yhat)
        res = res.append([y_test.iloc[t]], refit=False)
    return np.asarray(preds, dtype=float)


metric_rows = []
for target in TARGETS:
    order = best_arima_orders[target]
    y_train = train_df[target].astype(float)
    y_test = test_df[target].astype(float)

    res = ARIMA(y_train, order=order).fit()
    preds = rolling_forecast(res, y_test)
    y_pred = pd.Series(preds, index=y_test.index)

    m = plot_pred(y_train, y_test, y_pred, target, f"ARIMA{order}", FIG_OUT)
    metric_rows.append({
        "target": target, "model": "ARIMA", "order": str(order), "seasonal_order": "-",
        **m,
    })

# %% [markdown]
# ## SARIMAX grid — same (p,d,q) with seasonal block period = 52
#
# Seasonal candidates: `(0,0,0,0)` (identical to ARIMA) and `(1,0,1,52)`.
# Select best by AIC on training data.

# %%
SEASONAL_CANDIDATES = [(0, 0, 0, 0), (1, 0, 1, 52)]
best_sarimax = {}
for target in TARGETS:
    y = train_df[target].astype(float)
    best = None
    for (p, d, q), seas in itertools.product(
        list(itertools.product(P_RANGE, D_RANGE, Q_RANGE)), SEASONAL_CANDIDATES
    ):
        try:
            res = SARIMAX(y, order=(p, d, q), seasonal_order=seas,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            aic = res.aic
        except Exception:
            continue
        if best is None or aic < best[2]:
            best = ((p, d, q), seas, aic)
    best_sarimax[target] = best
    print(f"{target}: SARIMAX order={best[0]} seasonal={best[1]} AIC={best[2]:.2f}")

# %% [markdown]
# ## Rolling one-step-ahead forecast — SARIMAX

# %%
def sarimax_rolling(y_train: pd.Series, y_test: pd.Series, order, seasonal_order) -> np.ndarray:
    res = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    preds = []
    for t in range(len(y_test)):
        yhat = float(res.forecast(steps=1).iloc[0])
        preds.append(yhat)
        res = res.append([y_test.iloc[t]], refit=False)
    return np.asarray(preds, dtype=float)


for target in TARGETS:
    order, seas, _ = best_sarimax[target]
    y_train = train_df[target].astype(float)
    y_test = test_df[target].astype(float)

    preds = sarimax_rolling(y_train, y_test, order, seas)
    y_pred = pd.Series(preds, index=y_test.index)

    name = f"SARIMAX{order}x{seas}"
    m = plot_pred(y_train, y_test, y_pred, target, name, FIG_OUT)
    metric_rows.append({
        "target": target, "model": "SARIMAX", "order": str(order), "seasonal_order": str(seas),
        **m,
    })

# %% [markdown]
# ## Save metrics

# %%
save_metrics(metric_rows, METRICS_CSV)
print("Saved ->", METRICS_CSV)

# %%
pd.DataFrame(metric_rows)
