# %% [markdown]
# # 04 — Markov Chain forecasting (three variants)
#
# Lowest sensible k = 2 states (Low / High). Two discretizations per target:
# quantile (median split) and k-means (k=2). Three prediction variants:
#
# 1. **Argmax** — next value = representative of `argmax(P[current, :])` state.
# 2. **Expected value** — next value = `sum(P[current, :] * reps)`. Continuous.
# 3. **Markov-regression hybrid** — Markov predicts the next state; a per-state
#    linear regression on the last 4 lags then produces a continuous value.
#
# All variants are one-step-ahead (re-anchor state on the true observation
# each step). Metrics include `MASE` vs in-sample persistence.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

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

sns.set_theme(style="whitegrid", context="notebook")

FIG_OUT = FIG_DIR / "04_markov_chain"
FIG_OUT.mkdir(parents=True, exist_ok=True)
METRICS_CSV = METRICS_DIR / "04_markov_chain.csv"
if METRICS_CSV.exists():
    METRICS_CSV.unlink()

K = 2
HYBRID_LAGS = 4

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
# ## Baselines

# %%
baseline_rows = []
for target in TARGETS:
    np_pred = naive_persistence(train_df[target], test_df[target])
    sn_pred = seasonal_naive(train_df[target], test_df[target], period=52)
    m_np = plot_pred(train_df[target], test_df[target], np_pred, target,
                     "naive_persistence", FIG_OUT)
    m_sn = plot_pred(train_df[target], test_df[target], sn_pred, target,
                     "seasonal_naive_52", FIG_OUT)
    baseline_rows.append({"target": target, "method": "-", "variant": "naive_persistence", **m_np})
    baseline_rows.append({"target": target, "method": "-", "variant": "seasonal_naive_52", **m_sn})

save_metrics(baseline_rows, METRICS_CSV)

# %% [markdown]
# ## Helpers

# %%
def quantile_discretize(train_values: np.ndarray):
    thresh = float(np.median(train_values))
    low_rep = float(np.mean(train_values[train_values <= thresh]))
    high_rep = float(np.mean(train_values[train_values > thresh]))
    reps = np.array([low_rep, high_rep], dtype=float)

    def assign(x):
        return (np.asarray(x, dtype=float) > thresh).astype(int)

    return assign, reps, {"threshold": thresh}


def kmeans_discretize(train_values: np.ndarray):
    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    km.fit(train_values.reshape(-1, 1))
    centers = km.cluster_centers_.ravel()
    order = np.argsort(centers)
    remap = {old: new for new, old in enumerate(order)}
    reps = centers[order]

    def assign(x):
        raw = km.predict(np.asarray(x, dtype=float).reshape(-1, 1))
        return np.array([remap[v] for v in raw])

    return assign, reps, {"centers_sorted": reps.tolist()}


def transition_matrix(states: np.ndarray, k: int = K) -> np.ndarray:
    P = np.zeros((k, k), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        P[a, b] += 1.0
    rs = P.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return P / rs


def plot_transition(P: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    sns.heatmap(P, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=["Low", "High"], yticklabels=["Low", "High"], ax=ax)
    ax.set_xlabel("Next state"); ax.set_ylabel("Current state")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.show()

# %% [markdown]
# ## Variant A — argmax (discrete back-mapping)

# %%
rows = []
for target in TARGETS:
    print(f"\n--- {target} (argmax) ---")
    y_train = train_df[target].astype(float)
    y_test = test_df[target].astype(float)

    for method_name, disc in [("quantile", quantile_discretize), ("kmeans", kmeans_discretize)]:
        assign, reps, _ = disc(y_train.values)
        train_states = assign(y_train.values)
        test_states_true = assign(y_test.values)
        P = transition_matrix(train_states)

        preds_state = []
        cur = int(train_states[-1])
        for t in range(len(test_states_true)):
            nxt = int(np.argmax(P[cur]))
            preds_state.append(nxt)
            cur = int(test_states_true[t])
        y_pred = pd.Series(reps[np.asarray(preds_state)], index=y_test.index)

        model_name = f"Markov-argmax-{method_name}-k{K}"
        m = plot_pred(y_train, y_test, y_pred, target, model_name, FIG_OUT)
        plot_transition(P, f"{target} — {method_name} (argmax)",
                        FIG_OUT / f"transition_{target}_{method_name}.png")
        rows.append({"target": target, "method": method_name, "variant": "argmax",
                     "rep_low": float(reps[0]), "rep_high": float(reps[1]),
                     "P00": float(P[0, 0]), "P01": float(P[0, 1]),
                     "P10": float(P[1, 0]), "P11": float(P[1, 1]), **m})

# %% [markdown]
# ## Variant B — expected value (continuous interpolation)

# %%
for target in TARGETS:
    print(f"\n--- {target} (expected) ---")
    y_train = train_df[target].astype(float)
    y_test = test_df[target].astype(float)

    for method_name, disc in [("quantile", quantile_discretize), ("kmeans", kmeans_discretize)]:
        assign, reps, _ = disc(y_train.values)
        train_states = assign(y_train.values)
        test_states_true = assign(y_test.values)
        P = transition_matrix(train_states)

        preds_value = []
        cur = int(train_states[-1])
        for t in range(len(test_states_true)):
            val = float(np.sum(P[cur] * reps))
            preds_value.append(val)
            cur = int(test_states_true[t])
        y_pred = pd.Series(preds_value, index=y_test.index)

        model_name = f"Markov-Evalue-{method_name}-k{K}"
        m = plot_pred(y_train, y_test, y_pred, target, model_name, FIG_OUT)
        rows.append({"target": target, "method": method_name, "variant": "expected_value",
                     "rep_low": float(reps[0]), "rep_high": float(reps[1]),
                     "P00": float(P[0, 0]), "P01": float(P[0, 1]),
                     "P10": float(P[1, 0]), "P11": float(P[1, 1]), **m})

# %% [markdown]
# ## Variant C — Markov-regression hybrid
#
# Fit one `LinearRegression` per state on the 4 most recent lags (from training
# observations only). At test time:
#
# 1. Predict the next state using `argmax(P[current, :])`.
# 2. Apply that state's regressor to the last 4 observed values.
# 3. Re-anchor the current state on the true observation.

# %%
def fit_state_regressors(y_values: np.ndarray, states: np.ndarray, lags: int = HYBRID_LAGS):
    """Return {state: LinearRegression} trained on (last `lags` values -> next value)."""
    X_all, y_all, s_all = [], [], []
    for i in range(lags, len(y_values)):
        X_all.append(y_values[i - lags : i])
        y_all.append(y_values[i])
        s_all.append(states[i])
    X_all = np.asarray(X_all, dtype=float)
    y_all = np.asarray(y_all, dtype=float)
    s_all = np.asarray(s_all, dtype=int)

    models = {}
    overall = LinearRegression().fit(X_all, y_all) if len(X_all) > 0 else None
    for s in np.unique(s_all):
        m = (s_all == s)
        if m.sum() >= 3:
            models[int(s)] = LinearRegression().fit(X_all[m], y_all[m])
        else:
            models[int(s)] = overall
    return models, overall


for target in TARGETS:
    print(f"\n--- {target} (hybrid) ---")
    y_train = train_df[target].astype(float)
    y_test = test_df[target].astype(float)

    for method_name, disc in [("quantile", quantile_discretize), ("kmeans", kmeans_discretize)]:
        assign, reps, _ = disc(y_train.values)
        train_states = assign(y_train.values)
        test_states_true = assign(y_test.values)
        P = transition_matrix(train_states)

        state_models, fallback = fit_state_regressors(y_train.values, train_states, HYBRID_LAGS)

        hist = list(y_train.values)
        preds_value = []
        cur = int(train_states[-1])
        for t in range(len(test_states_true)):
            pred_state = int(np.argmax(P[cur]))
            last_lags = np.asarray(hist[-HYBRID_LAGS:], dtype=float).reshape(1, -1)
            reg = state_models.get(pred_state, fallback)
            yhat = float(reg.predict(last_lags)[0]) if reg is not None else float(reps[pred_state])
            preds_value.append(yhat)
            hist.append(float(y_test.values[t]))
            cur = int(test_states_true[t])
        y_pred = pd.Series(preds_value, index=y_test.index)

        model_name = f"Markov-hybrid-{method_name}-k{K}"
        m = plot_pred(y_train, y_test, y_pred, target, model_name, FIG_OUT)
        rows.append({"target": target, "method": method_name, "variant": "hybrid_regression",
                     "rep_low": float(reps[0]), "rep_high": float(reps[1]),
                     "P00": float(P[0, 0]), "P01": float(P[0, 1]),
                     "P10": float(P[1, 0]), "P11": float(P[1, 1]), **m})

# %% [markdown]
# ## Save + summary

# %%
save_metrics(rows, METRICS_CSV)
print("Saved ->", METRICS_CSV)

# %%
res = pd.DataFrame(rows)
summary = (res
           .sort_values(["target", "MASE"])
           .groupby("target")
           .head(3)
           [["target", "method", "variant", "MASE", "RMSE", "MAE", "MAPE", "sMAPE", "R2",
             "beats_naive"]])
summary
